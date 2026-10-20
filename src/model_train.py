"""Train and evaluate classical ML baselines for AMP classification.

Recommended workflow:
  1. Run src/load_final_data.py to create data/processed/final_dataset_v3.csv.
  2. Install CD-HIT to enable the homology-aware split:
       conda install -c bioconda -c conda-forge cd-hit
  3. Run from any directory:
       python /path/to/project/src/model_train.py

The default split clusters sequences at 80% identity with coverage controls, then
keeps each cluster entirely in either the training or test fold. If CD-HIT is not
available, the default run stops instead of silently reporting random-split results.
An explicitly labelled exploratory random split is available with --split-mode random.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "processed" / "final_dataset_v3.csv"
DEFAULT_OUTDIR = ROOT / "results" / "ml_baseline_v3"
SEED = 42
AA = list("ACDEFGHIKLMNPQRSTVWY")
VALID_AA = set(AA)

HYDROPHOBICITY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
CHARGE = {"K": 1.0, "R": 1.0, "H": 0.1, "D": -1.0, "E": -1.0}
DIPEPTIDES = [a + b for a in AA for b in AA]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--split-mode", choices=["cluster", "random"], default="cluster",
        help="cluster = CD-HIT homology-aware split (recommended); random = exploratory only",
    )
    parser.add_argument(
        "--identity", type=float, default=0.80,
        help="CD-HIT sequence identity threshold used for grouping (default: 0.80)",
    )
    return parser.parse_args()


def load_and_validate_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "First run: python src/load_final_data.py\n"
            "Or pass an explicit path with --dataset."
        )

    df = pd.read_csv(path)
    required = {"id", "sequence", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["sequence"] = df["sequence"].fillna("").astype(str).str.strip().str.upper()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    if not set(df["label"].unique()).issubset({0, 1}) or df["label"].nunique() != 2:
        raise ValueError("The label column must contain both binary labels 0 and 1 only.")
    if (df["sequence"] == "").any():
        raise ValueError("Empty sequences found. Rebuild/clean the dataset before training.")
    invalid = df["sequence"].apply(lambda s: any(aa not in VALID_AA for aa in s))
    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum())} sequences contain noncanonical amino-acid symbols. "
            "Rebuild the v3 dataset before training."
        )

    label_counts = df.groupby("sequence")["label"].nunique()
    conflicts = label_counts[label_counts > 1]
    if len(conflicts):
        raise ValueError(
            f"Found {len(conflicts)} exact sequences carrying conflicting labels. "
            "Run src/load_final_data.py to rebuild the cleaned v3 dataset."
        )

    duplicate_rows = int(df["sequence"].duplicated().sum())
    if duplicate_rows:
        print(f"Removing {duplicate_rows} same-label exact duplicate rows before training.")
        df = df.drop_duplicates(subset="sequence", keep="first").copy()

    df["length"] = df["sequence"].str.len()
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df)} unique sequences from {path}")
    print("Class counts (0 = negative candidate, 1 = AMP):")
    print(df["label"].value_counts().sort_index())
    return df


def compute_features(sequence: str) -> dict[str, float]:
    """Return 422 AAC/physicochemical/dipeptide features, as in the original baseline."""
    length = len(sequence)
    if length == 0:
        raise ValueError("Cannot featurize an empty sequence.")

    features: dict[str, float] = {}
    for aa in AA:
        features[f"aac_{aa}"] = sequence.count(aa) / length

    features["mean_net_charge"] = sum(CHARGE.get(aa, 0.0) for aa in sequence) / length
    features["mean_hydrophobicity"] = sum(HYDROPHOBICITY[aa] for aa in sequence) / length

    dipeptide_counts = dict.fromkeys(DIPEPTIDES, 0)
    denominator = max(length - 1, 1)
    for index in range(length - 1):
        dipeptide_counts[sequence[index:index + 2]] += 1
    for dipeptide in DIPEPTIDES:
        features[f"di_{dipeptide}"] = dipeptide_counts[dipeptide] / denominator

    return features


def cluster_sequences_with_cdhit(
    df: pd.DataFrame, outdir: Path, identity: float
) -> np.ndarray:
    """Run CD-HIT and return a cluster ID for every row in df."""
    executable = shutil.which("cd-hit")
    if executable is None:
        raise RuntimeError(
            "CD-HIT was not found, but --split-mode cluster is the default.\n"
            "Install it in your Linux/conda environment with:\n"
            "  conda install -c bioconda -c conda-forge cd-hit\n"
            "Then rerun the model script. Use --split-mode random only for an explicitly "
            "exploratory result, not as your final generalization estimate."
        )
    if not (0.4 <= identity <= 1.0):
        raise ValueError("--identity must be between 0.40 and 1.00 for this CD-HIT setup.")

    cluster_dir = outdir / "clustering"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    fasta_in = cluster_dir / "sequences_for_cdhit.fasta"
    fasta_out = cluster_dir / "cdhit_representatives.fasta"
    clusters_file = Path(str(fasta_out) + ".clstr")

    # Use synthetic IDs so source accession punctuation cannot break .clstr parsing.
    synthetic_to_row = {}
    with fasta_in.open("w", encoding="utf-8") as handle:
        for row_index, sequence in enumerate(df["sequence"]):
            seq_id = f"seq{row_index:08d}"
            synthetic_to_row[seq_id] = row_index
            handle.write(f">{seq_id}\n{sequence}\n")

    # For thresholds >= 0.7 CD-HIT recommends word length 5.
    # The threshold is fixed to 0.80 by default, so reject unsupported low settings
    # rather than silently choosing an inappropriate word length.
    if identity < 0.70:
        raise ValueError(
            "This script currently configures CD-HIT for identity thresholds >= 0.70. "
            "Use the default --identity 0.80 for the initial study."
        )

    command = [
    executable, "-i", str(fasta_in), "-o", str(fasta_out),
    "-c", str(identity), "-n", "5",
    "-l","9",
    "-aS", "0.8", "-aL", "0.8",
    "-d", "0", "-T", "0", "-M", "0",
]
    print("\nClustering sequences to reduce train/test homology leakage:")
    print(" ".join(command))
    completed = subprocess.run(command, text=True, capture_output=True)
    (cluster_dir / "cdhit_stdout_stderr.txt").write_text(
        completed.stdout + "\n" + completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0 or not clusters_file.exists():
        raise RuntimeError(
            f"CD-HIT failed (exit code {completed.returncode}). See "
            f"{cluster_dir / 'cdhit_stdout_stderr.txt'}"
        )

    row_to_cluster: dict[int, str] = {}
    current_cluster = None
    with clusters_file.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">Cluster"):
                current_cluster = line.split()[-1]
            elif line and current_cluster is not None:
                match = re.search(r">(seq\d+)\.\.\.", line)
                if match is None:
                    raise ValueError(f"Could not parse CD-HIT cluster line: {line}")
                seq_id = match.group(1)
                if seq_id not in synthetic_to_row:
                    raise ValueError(f"Unexpected CD-HIT sequence ID: {seq_id}")
                row_to_cluster[synthetic_to_row[seq_id]] = current_cluster

    if len(row_to_cluster) != len(df):
        missing_count = len(df) - len(row_to_cluster)
        raise RuntimeError(f"CD-HIT cluster file did not account for {missing_count} sequences.")

    groups = np.array([row_to_cluster[i] for i in range(len(df))], dtype=object)
    pd.DataFrame({"id": df["id"], "sequence": df["sequence"], "label": df["label"], "cluster": groups}).to_csv(
        cluster_dir / "sequence_clusters.csv", index=False
    )
    print(f"Sequences: {len(df)} | CD-HIT clusters: {len(set(groups))} | identity: {identity:.2f}")
    return groups


def make_models() -> dict[str, object]:
    models: dict[str, object] = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, random_state=SEED)),
        ]),
        "svc_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=SEED)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=400, random_state=SEED, n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    }

    if XGBClassifier is None:
        print("WARNING: xgboost is not installed; XGBoost and ensemble components will be skipped.")
    else:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        )

    return models


def get_positive_probability(model, X_test: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"Model {type(model).__name__} does not implement predict_proba.")
    classes = list(model.classes_)
    positive_index = classes.index(1)
    return model.predict_proba(X_test)[:, positive_index]


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "models").mkdir(exist_ok=True)

    df = load_and_validate_dataset(dataset_path)
    print("\nExtracting 422 sequence-derived features...")
    feature_rows = [compute_features(seq) for seq in df["sequence"]]
    X = pd.DataFrame(feature_rows, index=df.index)
    y = df["label"].astype(int)
    print(f"Feature matrix: {X.shape[0]} sequences x {X.shape[1]} features")

    if args.split_mode == "cluster":
        groups = cluster_sequences_with_cdhit(df, outdir, args.identity)
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        split_note = (
            f"StratifiedGroupKFold first fold; CD-HIT identity={args.identity}; "
            "coverage >= 80% of both sequences"
        )
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        if train_groups & test_groups:
            raise RuntimeError("Leakage guard failed: a CD-HIT cluster occurs in both splits.")
    else:
        print(
            "WARNING: using a random stratified split. This can overestimate "
            "generalization when similar peptides occur in both sets."
        )
        all_idx = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            all_idx, test_size=0.20, random_state=SEED, stratify=y
        )
        groups = np.array([f"row_{i}" for i in range(len(df))], dtype=object)
        split_note = "Random stratified 80:20 split (exploratory only; not homology-aware)"

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print(f"\nSplit: {split_note}")
    print(f"Train size: {len(train_idx)} | Test size: {len(test_idx)}")
    print("Train class counts:", y_train.value_counts().sort_index().to_dict())
    print("Test class counts:", y_test.value_counts().sort_index().to_dict())
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise RuntimeError(
            "The selected split does not contain both labels in train and test. "
            "Inspect the cluster distribution or try another split/fold."
        )

    base_models = make_models()
    models = dict(base_models)
    if len(base_models) >= 2:
        models["soft_voting_ensemble"] = VotingClassifier(
            estimators=[(name, clone(model)) for name, model in base_models.items()],
            voting="soft",
            n_jobs=1,
        )

    metric_rows = []
    prediction_df = df.iloc[test_idx][[c for c in ["id", "sequence", "length", "source"] if c in df.columns]].copy()
    prediction_df.insert(0, "dataset_row_index", test_idx)
    prediction_df["true_label"] = y_test.to_numpy()
    reports = []
    roc_plot_data = []
    pr_plot_data = []

    for name, model in models.items():
        print(f"\nTraining: {name}")
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        y_pred = model.predict(X_test)
        y_prob = get_positive_probability(model, X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        metrics = {
            "model": name,
            "split_mode": args.split_mode,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "pr_auc_average_precision": average_precision_score(y_test, y_prob),
            "mcc": matthews_corrcoef(y_test, y_pred),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "training_seconds": round(elapsed, 3),
        }
        metric_rows.append(metrics)
        prediction_df[f"{name}_prob_amp"] = y_prob
        prediction_df[f"{name}_predicted_label"] = y_pred
        reports.append(
            f"{'=' * 72}\nMODEL: {name}\n"
            f"Training time (s): {elapsed:.3f}\n"
            f"{classification_report(y_test, y_pred, labels=[0, 1], target_names=['negative_candidate', 'AMP'], zero_division=0)}\n"
            f"Confusion matrix [rows=true 0/1, columns=predicted 0/1]:\n"
            f"{confusion_matrix(y_test, y_pred, labels=[0, 1])}\n"
        )
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        roc_plot_data.append((name, fpr, tpr, metrics["roc_auc"]))
        pr_plot_data.append((name, recall, precision, metrics["pr_auc_average_precision"]))

        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", name)
        joblib.dump(model, outdir / "models" / f"{safe_name}.joblib")
        print(
            f"  accuracy={metrics['accuracy']:.3f} | F1={metrics['f1']:.3f} | "
            f"ROC-AUC={metrics['roc_auc']:.3f} | PR-AUC(AP)={metrics['pr_auc_average_precision']:.3f} | "
            f"MCC={metrics['mcc']:.3f}"
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values("mcc", ascending=False)
    metrics_df.to_csv(outdir / "model_metrics.csv", index=False)
    prediction_df.to_csv(outdir / "test_predictions.csv", index=False)
    (outdir / "classification_reports.txt").write_text("\n".join(reports), encoding="utf-8")
    (outdir / "split_info.json").write_text(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "split_note": split_note,
                "seed": SEED,
                "feature_count": int(X.shape[1]),
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
                "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
                "cdhit_identity": args.identity if args.split_mode == "cluster" else None,
                "models": list(models.keys()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (outdir / "feature_columns.json").write_text(json.dumps(list(X.columns), indent=2), encoding="utf-8")

    # Save poster-friendly comparisons using matplotlib's default color cycle.
    plt.figure(figsize=(8, 6))
    for name, fpr, tpr, auc_value in roc_plot_data:
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_value:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves — AMP classification")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(outdir / "roc_curves.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for name, recall, precision, ap_value in pr_plot_data:
        plt.plot(recall, precision, label=f"{name} (AP={ap_value:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–recall curves — AMP classification")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(outdir / "precision_recall_curves.png", dpi=300)
    plt.close()

    print("\nModel comparison (sorted by MCC; higher is better):")
    print(metrics_df[["model", "accuracy", "f1", "roc_auc", "pr_auc_average_precision", "mcc"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved outputs to: {outdir}")
    print("Important: results are a baseline, not proof that predicted peptides are experimentally antimicrobial.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
