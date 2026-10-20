# ESM-2 + MLP next stage

This adds a pretrained protein-language-model embedding baseline on top of the existing AMP project. It does not overwrite or modify `src/model_train.py` or `results/ml_baseline_v3/`.

## 1. Install dependencies in `amp_env`

From the project root:

```bash
conda activate amp_env

# CPU-only PyTorch (recommended for the current machine)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Hugging Face model/tokenizer
pip install 'transformers>=4.44,<5' safetensors
```

The first run needs internet access to download `facebook/esm2_t6_8M_UR50D` from Hugging Face. It is the small 8M-parameter ESM-2 variant, chosen to keep CPU inference practical. Hugging Face's ESM model docs describe the token-level hidden states used here: https://huggingface.co/docs/transformers/model_doc/esm

## 2. Check the baseline split seed

The ESM script defaults to `--seed 42` and `--n-splits 5`. Confirm those match the values in `src/model_train.py` (search for `StratifiedGroupKFold`, `random_state`, and `n_splits`). They must match to reproduce the same first outer fold, assuming the dataset order and CD-HIT parameters are unchanged.

```bash
grep -nE 'StratifiedGroupKFold|random_state|n_splits' src/model_train.py
```

If the baseline used a different seed, pass that seed to the ESM command below. The first ESM run writes `results/esm_mlp_v1/split_assignments.csv`; future runs reuse that manifest so the split remains frozen. The script also accepts a saved baseline split manifest if one exists in `results/ml_baseline_v3/` under a recognized filename. If you have the actual baseline split assignments, place/copy them as `results/esm_mlp_v1/split_assignments.csv` with at least `sequence` and `split` columns, where split is `train` or `test`, before first running ESM.

The script verifies that no CD-HIT cluster crosses train and test. It will warn if the split sizes differ from the baseline's reported 4357/1089 split.

## 3. Run ESM-2 embedding + MLP

```bash
python src/train_esm_mlp.py --seed 42 --device cpu --batch-size 16 --epochs 40
```

The script:

1. Loads `data/processed/final_dataset_v3.csv` and the saved CD-HIT `.clstr` file.
2. Computes mean-pooled ESM-2 residue embeddings (excluding special tokens), and caches them in `results/esm_mlp_v1/`.
3. Creates or loads a `StratifiedGroupKFold` outer split that keeps CD-HIT clusters separate.
4. Uses a second group-held-out fold inside the outer training set to pick an epoch count with early stopping.
5. Refits the MLP on the whole outer training set for the selected number of epochs, then evaluates the untouched outer test fold.

The `--batch-size 16` setting is deliberately conservative for CPU use. If RAM use is comfortable, you can try 32. Use `--force-embeddings` to rebuild cached embeddings; use `--rebuild-split` only when you intentionally want to recreate the split.

## 4. Outputs

Under `results/esm_mlp_v1/`:

- `embeddings_facebook__esm2_t6_8M_UR50D.npz` — cached embeddings
- `split_assignments.csv` — frozen train/test assignments and cluster groups
- `training_history.csv`, `training_curve.png` — epoch-selection diagnostics
- `metrics.json`, `metrics.csv` — test metrics
- `classification_report.txt`, `confusion_matrix.csv`, `confusion_matrix.png`
- `test_predictions.csv` — sequence-level test predictions and AMP probabilities
- `esm_mlp_model.pt`, `embedding_scaler.joblib` — checkpoint and scaler

## 5. Evaluation caution

The existing classical run's reported soft-voting metrics were accuracy 0.915, F1 0.915, ROC-AUC 0.974, PR-AUC (average precision) 0.973, and MCC 0.829. Only compare ESM-2 + MLP numbers directly to those results after confirming the exact same outer test membership is used. Matching split sizes and class counts alone is not proof that the test sequences match.
