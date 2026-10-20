#!/usr/bin/env python3
"""Train an ESM-2 embedding + MLP AMP classifier with group-aware evaluation.

Expected project layout:
  data/processed/final_dataset_v3.csv
  results/ml_baseline_v3/clustering/sequences_for_cdhit.fasta
  results/ml_baseline_v3/clustering/cdhit_representatives.fasta.clstr

Run from the project root:
  python src/train_esm_mlp.py --seed 42

Embeddings are cached; subsequent runs skip the ESM forward pass when the
sequences and model name match the cache.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


DEFAULT_MODEL = "facebook/esm2_t6_8M_UR50D"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/processed/final_dataset_v3.csv")
    parser.add_argument("--cluster-fasta", type=Path, default=root / "results/ml_baseline_v3/clustering/sequences_for_cdhit.fasta")
    parser.add_argument("--cluster-file", type=Path, default=root / "results/ml_baseline_v3/clustering/cdhit_representatives.fasta.clstr")
    parser.add_argument("--baseline-results", type=Path, default=root / "results/ml_baseline_v3")
    parser.add_argument("--output-dir", type=Path, default=root / "results/esm_mlp_v1")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face ESM model ID")
    parser.add_argument("--seed", type=int, default=42, help="Use the same seed as the classical baseline split")
    parser.add_argument("--n-splits", type=int, default=5, help="StratifiedGroupKFold folds; baseline used 5 folds")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40, help="Maximum early-stopping search epochs")
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--rebuild-split", action="store_true", help="Ignore a previously saved split manifest and recreate it")
    parser.add_argument("--force-embeddings", action="store_true", help="Regenerate the embedding cache")
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    required = {"sequence", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset must contain columns {sorted(required)}; missing {sorted(missing)}")
    df = df.copy()
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    if not set(df["label"].unique()).issubset({0, 1}) or df["label"].nunique() != 2:
        raise ValueError("Expected binary labels 0 (negative candidate) and 1 (AMP).")
    if df["sequence"].isna().any() or (df["sequence"] == "").any():
        raise ValueError("Empty sequences found in dataset.")
    if df["sequence"].duplicated().any():
        raise ValueError("Expected final_dataset_v3.csv to have unique sequences. Resolve duplicates first.")
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    invalid = [s for s in df["sequence"] if not set(s).issubset(valid_aa)]
    if invalid:
        raise ValueError(f"Found {len(invalid)} sequences containing noncanonical amino-acid symbols.")
    return df


def read_cdhit_groups(
    df: pd.DataFrame, fasta_path: Path, cluster_path: Path
) -> Tuple[np.ndarray, Dict[str, str]]:
    """Map each dataset row to its CD-HIT cluster using sequence content, not row order."""
    if not fasta_path.exists():
        raise FileNotFoundError(f"CD-HIT input FASTA not found: {fasta_path}")
    if not cluster_path.exists():
        raise FileNotFoundError(
            f"CD-HIT cluster file not found: {cluster_path}. Run the ML baseline first."
        )

    fasta_records = list(SeqIO.parse(str(fasta_path), "fasta"))
    sequence_to_dataset_index = {seq: i for i, seq in enumerate(df["sequence"].tolist())}
    fasta_id_to_sequence: Dict[str, str] = {}
    for record in fasta_records:
        seq = str(record.seq).upper()
        if record.id in fasta_id_to_sequence:
            raise ValueError(f"Duplicate FASTA ID in CD-HIT input: {record.id}")
        fasta_id_to_sequence[record.id] = seq
        if seq not in sequence_to_dataset_index:
            raise ValueError(
                f"CD-HIT input sequence {record.id} is not present in the dataset. "
                "Ensure clustering was generated from final_dataset_v3.csv."
            )

    if len(fasta_id_to_sequence) != len(df):
        raise ValueError(
            f"Dataset has {len(df)} rows but CD-HIT FASTA has {len(fasta_id_to_sequence)} records. "
            "The clustering FASTA must come from this exact dataset."
        )

    current_cluster = None
    id_to_cluster: Dict[str, str] = {}
    for line in cluster_path.read_text().splitlines():
        if line.startswith(">Cluster"):
            current_cluster = line[1:].strip().replace(" ", "_")
            continue
        match = re.search(r">([^\s]+?)\.\.\.", line)
        if match:
            if current_cluster is None:
                raise ValueError("Malformed CD-HIT .clstr file: sequence found before a cluster header.")
            seq_id = match.group(1)
            if seq_id in id_to_cluster:
                raise ValueError(f"Sequence {seq_id} appears more than once in the cluster file.")
            id_to_cluster[seq_id] = current_cluster

    missing_ids = sorted(set(fasta_id_to_sequence) - set(id_to_cluster))
    extra_ids = sorted(set(id_to_cluster) - set(fasta_id_to_sequence))
    if missing_ids or extra_ids:
        raise ValueError(
            f"CD-HIT membership mismatch: {len(missing_ids)} FASTA IDs missing from clusters; "
            f"{len(extra_ids)} unknown cluster IDs. Check CD-HIT's -l setting and output files."
        )

    groups = np.empty(len(df), dtype=object)
    for seq_id, seq in fasta_id_to_sequence.items():
        idx = sequence_to_dataset_index[seq]
        groups[idx] = id_to_cluster[seq_id]
    if any(g is None for g in groups):
        raise ValueError("Some dataset rows were not assigned a CD-HIT cluster.")
    id_to_sequence = {seq_id: fasta_id_to_sequence[seq_id] for seq_id in fasta_id_to_sequence}
    print(f"CD-HIT groups loaded: {len(set(groups))} clusters for {len(groups)} sequences")
    return groups, id_to_cluster


def find_existing_split(output_dir: Path, baseline_dir: Path, explicit: Path | None = None) -> Path | None:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend([
        output_dir / "split_assignments.csv",
        baseline_dir / "split_assignments.csv",
        baseline_dir / "split_assignment.csv",
        baseline_dir / "train_test_split.csv",
        baseline_dir / "split_manifest.csv",
    ])
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def make_or_load_split(
    df: pd.DataFrame,
    groups: np.ndarray,
    output_dir: Path,
    baseline_dir: Path,
    seed: int,
    n_splits: int,
    rebuild: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assignment_path = output_dir / "split_assignments.csv"
    existing = None if rebuild else find_existing_split(output_dir, baseline_dir)
    if existing is not None:
        print(f"Loading saved train/test split: {existing}")
        split_df = pd.read_csv(existing)
        if not {"sequence", "split"}.issubset(split_df.columns):
            raise ValueError(f"Split file {existing} must contain 'sequence' and 'split' columns.")
        split_df["sequence"] = split_df["sequence"].astype(str).str.upper()
        if split_df["sequence"].duplicated().any() or set(split_df["sequence"]) != set(df["sequence"]):
            raise ValueError("Saved split does not match this dataset's exact sequence set.")
        split_map = dict(zip(split_df["sequence"], split_df["split"].astype(str).str.lower()))
        normalized = {k: ("train" if v in {"train", "tr", "0"} else "test" if v in {"test", "te", "1"} else v) for k, v in split_map.items()}
        if not set(normalized.values()).issubset({"train", "test"}) or set(normalized.values()) != {"train", "test"}:
            raise ValueError("Split values must identify train and test (or 0=train, 1=test).")
        split = np.array([normalized[s] for s in df["sequence"]])
    else:
        print(f"Creating first StratifiedGroupKFold split (n_splits={n_splits}, seed={seed})")
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, test_idx = next(splitter.split(np.zeros(len(df)), df["label"].to_numpy(), groups))
        split = np.full(len(df), "train", dtype=object)
        split[test_idx] = "test"
        # Save the split before training so all later models can reuse the exact assignments.
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "sequence": df["sequence"],
            "label": df["label"],
            "split": split,
            "cluster_group": groups,
        }).to_csv(assignment_path, index=False)
        print(f"Saved reusable split manifest: {assignment_path}")

    train_idx = np.flatnonzero(split == "train")
    test_idx = np.flatnonzero(split == "test")
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Split assignment produced an empty train or test partition.")
    overlap = set(groups[train_idx]) & set(groups[test_idx])
    if overlap:
        raise ValueError(f"Train/test cluster leakage detected: {len(overlap)} clusters overlap.")

    y = df["label"].to_numpy()
    print(f"Split sizes: train={len(train_idx)}, test={len(test_idx)}")
    print(f"Train class counts: {pd.Series(y[train_idx]).value_counts().sort_index().to_dict()}")
    print(f"Test class counts: {pd.Series(y[test_idx]).value_counts().sort_index().to_dict()}")
    if len(df) == 5446 and (len(train_idx), len(test_idx)) != (4357, 1089):
        print(
            "WARNING: split sizes differ from the previously reported baseline (4357/1089). "
            "Use the same random_state and fold settings as src/model_train.py before making a direct metric comparison.",
            file=sys.stderr,
        )
    return train_idx, test_idx, groups


def generate_embeddings(
    sequences: List[str], model_name: str, cache_path: Path, batch_size: int,
    device_choice: str, threads: int, force: bool,
) -> np.ndarray:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "ESM dependencies are missing. Install CPU PyTorch and Transformers using the README commands."
        ) from exc

    if device_choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available. Use --device cpu.")
        device = torch.device("cuda")
    elif device_choice == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, threads))
    print(f"Embedding device: {device}")
    print(f"ESM model: {model_name}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force:
        try:
            cached = np.load(cache_path, allow_pickle=False)
            cached_model = str(cached["model_name"].item())
            cached_sequences = cached["sequences"].astype(str).tolist()
            cached_embeddings = cached["embeddings"].astype(np.float32)
            if cached_model == model_name and cached_sequences == sequences and cached_embeddings.shape[0] == len(sequences):
                print(f"Loaded cached embeddings: {cached_embeddings.shape} from {cache_path}")
                return cached_embeddings
            print("Embedding cache does not match model/sequences; regenerating it.")
        except Exception as exc:
            print(f"Could not read embedding cache ({exc}); regenerating it.")

    print("Loading tokenizer and pretrained model (first run downloads model files)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    batches = []
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    special_ids = {
        token_id for token_id in [tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]
        if token_id is not None
    }
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            tokenized = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
            residue_mask = attention_mask.bool()
            for token_id in special_ids:
                residue_mask = residue_mask & (input_ids != token_id)
            mask = residue_mask.unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            pooled = (summed / counts).detach().cpu().float().numpy()
            batches.append(pooled)
            if (start // batch_size + 1) % 20 == 0 or start + batch_size >= len(sequences):
                print(f"  Embedded {min(start + batch_size, len(sequences))}/{len(sequences)} sequences "
                      f"(batch {start // batch_size + 1}/{total_batches})")

    embeddings = np.concatenate(batches, axis=0).astype(np.float32)
    if embeddings.shape[0] != len(sequences) or not np.isfinite(embeddings).all():
        raise ValueError(f"Invalid embedding matrix: {embeddings.shape}")
    np.savez_compressed(
        cache_path,
        model_name=np.array(model_name),
        sequences=np.asarray(sequences, dtype=str),
        embeddings=embeddings,
    )
    print(f"Saved embedding cache: {cache_path} ({embeddings.shape})")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return embeddings


def build_mlp(input_dim: int, torch):
    from torch import nn
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.35),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.20),
        nn.Linear(64, 2),
    )


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    device,
    seed: int,
) -> Tuple[object, StandardScaler, int, List[dict]]:
    """Search for an epoch count using a group-held-out validation set."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    train_ds = TensorDataset(torch.from_numpy(X_train_scaled), torch.from_numpy(y_train.astype(np.int64)))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_x = torch.from_numpy(X_val_scaled).to(device)
    val_y_t = torch.from_numpy(y_val.astype(np.int64)).to(device)

    torch.manual_seed(seed)
    model = build_mlp(X_train_scaled.shape[1], torch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_state = None
    best_epoch = 1
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.inference_mode():
            val_logits = model(val_x)
            val_loss = float(criterion(val_logits, val_y_t).cpu())
            val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
            val_pred = (val_probs >= 0.5).astype(int)
            val_f1 = float(f1_score(y_val, val_pred, zero_division=0))
        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_f1_at_0_5": val_f1})
        print(f"  epoch {epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_F1={val_f1:.4f}")

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}; best validation loss was at epoch {best_epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, best_epoch, history


def fit_final_model(
    X_train: np.ndarray, y_train: np.ndarray, epochs: int, args: argparse.Namespace,
    device, seed: int,
):
    """Refit scaler and MLP on all outer-training data for the selected epoch count."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(X_scaled), torch.from_numpy(y_train.astype(np.int64)))
    generator = torch.Generator().manual_seed(seed + 100)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=generator)
    torch.manual_seed(seed + 100)
    model = build_mlp(X_scaled.shape[1], torch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"  final fit epoch {epoch:02d}/{epochs}: loss={np.mean(losses):.4f}")
    return model, scaler


def main() -> int:
    args = parse_args()
    set_seeds(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except ImportError as exc:
        print("ERROR: PyTorch is not installed. See README_ESM_MLP.md for CPU install steps.", file=sys.stderr)
        raise exc

    df = load_data(args.data)
    print(f"Loaded {len(df)} sequences from {args.data}")
    print(f"Class counts: {df['label'].value_counts().sort_index().to_dict()}")

    groups, _ = read_cdhit_groups(df, args.cluster_fasta, args.cluster_file)
    train_idx, test_idx, groups = make_or_load_split(
        df, groups, args.output_dir, args.baseline_results, args.seed, args.n_splits, args.rebuild_split
    )
    sequences = df["sequence"].tolist()
    cache_filename = args.model.replace("/", "__").replace(":", "_")
    cache_path = args.output_dir / f"embeddings_{cache_filename}.npz"
    embeddings = generate_embeddings(
        sequences=sequences,
        model_name=args.model,
        cache_path=cache_path,
        batch_size=args.batch_size,
        device_choice=args.device,
        threads=args.threads,
        force=args.force_embeddings,
    )

    y = df["label"].to_numpy()
    outer_train_groups = groups[train_idx]
    inner_splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed + 1)
    inner_train_rel, val_rel = next(
        inner_splitter.split(embeddings[train_idx], y[train_idx], outer_train_groups)
    )
    inner_train_idx = train_idx[inner_train_rel]
    val_idx = train_idx[val_rel]
    print(f"Validation tuning: train={len(inner_train_idx)}, validation={len(val_idx)} (within outer training set)")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    # If CUDA was explicitly requested, catch availability cleanly before training.
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Re-run with --device cpu.")
    torch.set_num_threads(max(1, args.threads))

    print("\nSearching for a training duration with a cluster-held-out validation fold...")
    _, _, best_epoch, history = train_model(
        embeddings[inner_train_idx], y[inner_train_idx], embeddings[val_idx], y[val_idx],
        args, device, args.seed + 2,
    )
    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    print(f"Selected epoch count: {best_epoch}")

    print("\nRefitting scaler and MLP on the full outer training set...")
    final_model, final_scaler = fit_final_model(
        embeddings[train_idx], y[train_idx], best_epoch, args, device, args.seed + 3
    )

    X_test = final_scaler.transform(embeddings[test_idx]).astype(np.float32)
    final_model.eval()
    with torch.inference_mode():
        logits = final_model(torch.from_numpy(X_test).to(device))
        probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    y_test = y[test_idx]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": "ESM-2 + MLP",
        "esm_model": args.model,
        "seed": args.seed,
        "n_splits": args.n_splits,
        "n_sequences": int(len(df)),
        "n_clusters": int(len(set(groups))),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "selected_epochs": int(best_epoch),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc_average_precision": float(average_precision_score(y_test, probabilities)),
        "mcc": float(matthews_corrcoef(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_test, predictions, zero_division=0)),
        "specificity": float(recall_score(y_test, predictions, pos_label=0, zero_division=0)),
        "threshold": 0.5,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame([metrics]).to_csv(args.output_dir / "metrics.csv", index=False)

    labels = [0, 1]
    cm = confusion_matrix(y_test, predictions, labels=labels)
    pd.DataFrame(cm, index=["true_0_negative_candidate", "true_1_AMP"],
                 columns=["pred_0_negative_candidate", "pred_1_AMP"]).to_csv(args.output_dir / "confusion_matrix.csv")
    report = classification_report(y_test, predictions, labels=labels,
                                   target_names=["negative_candidate", "AMP"], digits=4, zero_division=0)
    (args.output_dir / "classification_report.txt").write_text(report + "\n")
    prediction_df = pd.DataFrame({
        "sequence": df.iloc[test_idx]["sequence"].to_numpy(),
        "true_label": y_test,
        "predicted_label": predictions,
        "amp_probability": probabilities,
        "cluster_group": groups[test_idx],
    })
    prediction_df.to_csv(args.output_dir / "test_predictions.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm_fig, ax = plt.subplots(figsize=(5.2, 4.3))
        image = ax.imshow(cm, interpolation="nearest")
        cm_fig.colorbar(image, ax=ax)
        ax.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=["Predicted 0", "Predicted 1"],
               yticklabels=["True 0", "True 1"],
               xlabel="Predicted label", ylabel="True label", title="ESM-2 + MLP confusion matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        cm_fig.tight_layout()
        cm_fig.savefig(args.output_dir / "confusion_matrix.png", dpi=160)
        plt.close(cm_fig)

        hist_df = pd.DataFrame(history)
        hist_fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.plot(hist_df["epoch"], hist_df["train_loss"], label="Train loss")
        ax.plot(hist_df["epoch"], hist_df["val_loss"], label="Validation loss")
        ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="ESM-2 + MLP tuning history")
        ax.legend()
        hist_fig.tight_layout()
        hist_fig.savefig(args.output_dir / "training_curve.png", dpi=160)
        plt.close(hist_fig)
    except ImportError:
        print("matplotlib not available; skipped PNG plots (CSV outputs were saved).")

    # Save a portable model checkpoint with architecture metadata and the fitted scaler.
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in final_model.state_dict().items()},
        "input_dim": int(embeddings.shape[1]),
        "hidden_dims": [256, 64],
        "dropout": [0.35, 0.20],
        "esm_model": args.model,
        "selected_epochs": int(best_epoch),
        "threshold": 0.5,
        "seed": int(args.seed),
    }, args.output_dir / "esm_mlp_model.pt")
    joblib.dump(final_scaler, args.output_dir / "embedding_scaler.joblib")

    print("\nFinal test metrics (AMP is positive class 1):")
    for key in ["accuracy", "f1", "roc_auc", "pr_auc_average_precision", "mcc", "precision", "recall_sensitivity", "specificity"]:
        print(f"  {key}={metrics[key]:.4f}")
    print("\nClassification report:")
    print(report)
    print(f"Saved results to: {args.output_dir.resolve()}")
    print("Important: verify the split seed matches src/model_train.py before comparing metrics directly.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
