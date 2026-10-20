# AMP classification project — updated baseline scripts

These replacements preserve the original data files and write a new cleaned dataset called `data/processed/final_dataset_v3.csv`.

## What changed

- Paths are resolved relative to the project root, so scripts can be launched from any working directory.
- The data loader filters to 10–50 residues and canonical amino acids, removes exact sequences that appear with conflicting labels, and keeps only one row per exact sequence.
- It removes invalid sequences, cross-label conflicts, and exact duplicates first, then balances the cleaned classes by random undersampling (seed 42) to preserve your original balanced-class study design. Use `--keep-imbalance` to retain all cleaned sequences instead.
- The old `final_dataset.csv` and `final_dataset_v2.csv` files are not overwritten.
- The ML script uses the original 422 feature types: 20 amino-acid composition, mean charge, mean hydrophobicity, and 400 dipeptide frequencies.
- Scaling happens inside pipelines for Logistic Regression and SVC, avoiding fitting a scaler on the test data.
- The models are evaluated individually. A real soft-voting ensemble is also trained when at least two base models are available.
- Reported metrics include accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, average precision (PR-AUC-style summary), and MCC.
- ROC and precision–recall plots, model files, reports, predictions, and split metadata are saved under `results/ml_baseline_v3/`.
- The default evaluation split uses CD-HIT sequence clusters so a cluster does not cross from train to test. A random split is available for exploratory comparison only.

## Install / run

Run from the repository root:

```bash
# 1) Build the cleaned v3 dataset (does not overwrite v1/v2)
python src/load_final_data.py

# To retain all unique cleaned sequences without balancing instead:
# python src/load_final_data.py --keep-imbalance

# 2) Install CD-HIT in the active conda environment (Linux)
conda install -c bioconda -c conda-forge cd-hit

# 3) Train and evaluate baseline models with a sequence-cluster split
python src/model_train.py
```

If `xgboost` is missing from the Python environment, install it with `conda install -c conda-forge xgboost` or `pip install xgboost`. The script will otherwise warn and continue without XGBoost.

For an exploratory random split only:

```bash
python src/model_train.py --split-mode random
```

Do **not** present random-split results as the final estimate of performance on novel peptide families. The default CD-HIT split uses 80% identity and at least 80% alignment coverage of both sequences; clearly state this setting in your report.

## Output files

- `data/processed/final_dataset_v3.csv`: cleaned data used for training.
- `data/processed/final_dataset_v3_audit.txt`: source filtering and class-count audit.
- `results/ml_baseline_v3/model_metrics.csv`: model comparison table.
- `results/ml_baseline_v3/classification_reports.txt`: precision/recall/F1 reports and confusion matrices.
- `results/ml_baseline_v3/test_predictions.csv`: held-out predictions and probabilities.
- `results/ml_baseline_v3/roc_curves.png` and `precision_recall_curves.png`: poster-ready baseline plots.
- `results/ml_baseline_v3/models/`: serialized fitted models.
- `results/ml_baseline_v3/clustering/sequence_clusters.csv`: row-to-cluster assignments used for splitting.
- `results/ml_baseline_v3/split_info.json`: split and model metadata.

## Scientific caveats to retain in the report

1. Label 0 is a *negative candidate* class drawn from the configured UniProt FASTA source. The file name and filtering code do not establish that every sequence has been experimentally proven non-antimicrobial.
2. CD-HIT clustering at 80% identity reduces similarity leakage but is not a guarantee that every train/test peptide is biologically unrelated. Report the threshold and coverage settings.
3. The positive source is `naturalAMPs_APD2024a.fasta`, matching the old loader. It is not `plantAMPs_APD2024.fasta`; change the source intentionally if the project is meant to focus specifically on plant AMPs.
4. A high classifier score is a computational prediction, not experimental proof that a candidate is an active, safe, or novel antibiotic.
5. The script uses a fixed baseline configuration, not a separate validation set for hyperparameter tuning. Avoid tuning against the final test fold. If performing extensive tuning, use grouped cross-validation on training data and keep the test fold untouched.
