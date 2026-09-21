# Ensemble Implementation Spec — ExciPick (Api-Excipient-Compatibility-Model)

**Audience:** an AI coding agent with full read/write access to this repository.
**Goal:** (1) fix two real bugs found in this repo during the audit for this task,
then (2) build 6 ensemble models (2 base-model pairs × 3 ensemble methods), each with
its own saved checkpoint/artifact and its own separately-stored metrics, in a new
top-level `ensemble/` folder.

Read this entire document before writing any code. Do Part A (fixes) first — Part B
(the ensembles) depends on the fixed loading path being in place, and every path/name
below must match this repo exactly or predictions will misalign or checkpoints will
fail to load.

---

# PART A — Fix the two real bugs found during this audit

## Bug 1 (root cause): model identity is guessed from folder name instead of read from the checkpoint

`inference.py` contains a hand-maintained `MODEL_REGISTRY` dict mapping checkpoint
folder name → `(encoder, fusion, pooling, loss)`. This is a second, independent source
of truth that has already drifted from `src/config.py`'s own naming logic: for
`molformer_cross_attn_asl` and `molformer_cross_attn_bce`, `Config.resolve_paths()`
would compute `checkpoints/molformer_cross_attn_global_gated_<loss>` (folding the
pooling mode into the name), but the actual folders on disk are named without the
pooling segment — an older naming convention from before pooling was added to
`resolve_paths()`. `MODEL_REGISTRY` papers over this by hardcoding the correct tuple
per folder name. That's a workaround, not a fix — it depends on a human keeping a
side-table in sync with training runs forever, and it already required tribal
knowledge to notice.

**The actual fix:** `src/train.py`'s `train()` function already saves the *entire*
`Config` object used for that run inside the checkpoint itself:

```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_pr_auc": val_pr_auc,
    "config": config,          # <-- the ground truth is already sitting right here
}, best_ckpt_path)
```

So no folder-name parsing or hardcoded table is needed at all — load the checkpoint,
read `checkpoint["config"]`, and use its `encoder` / `fusion` / `pooling` / `loss` /
`encoder_output_dim` fields directly. This is correct by construction for every
checkpoint that exists, regardless of what its folder happens to be named, and it
cannot drift out of sync the way a hardcoded table can.

**Write this helper in `ensemble/_common.py`** (used everywhere a base model needs to
be loaded, both in Part A verification and Part B):

```python
import os
import torch
from src.config import Config

# Fallback only for checkpoints saved before "config" was embedded (should not be
# needed for this repo's current checkpoints, but keep it so old checkpoints don't
# hard-fail).
LEGACY_MODEL_REGISTRY = {
    "molformer_concat_asl":                  ("molformer", "concat",     "global_gated_attention", "asl"),
    "molformer_cross_attn_asl":              ("molformer", "cross_attn", "global_gated_attention", "asl"),
    "molformer_cross_attn_bce":              ("molformer", "cross_attn", "global_gated_attention", "bce"),
    "molformer_cross_attn_focal":            ("molformer", "cross_attn", "global_gated_attention", "focal"),
    "molformer_cross_attn_weighted_bce":     ("molformer", "cross_attn", "global_gated_attention", "weighted_bce"),
    "molformer_cross_attn_global_gated_asl": ("molformer", "cross_attn", "global_gated_attention", "asl"),
    "molformer_cross_attn_global_gated_bce": ("molformer", "cross_attn", "global_gated_attention", "bce"),
    "molformer_cross_attn_pairwise_asl":     ("molformer", "cross_attn", "explicit_pairwise",      "asl"),
    "molformer_cross_attn_pairwise_bce":     ("molformer", "cross_attn", "explicit_pairwise",      "bce"),
    "chemberta_cross_attn_asl":              ("chemberta", "cross_attn", "global_gated_attention", "asl"),
    "pretrained_gin_cross_attn_asl":         ("pretrained_gin", "cross_attn", "global_gated_attention", "asl"),
    "pretrained_gin_concat_asl":             ("pretrained_gin", "concat",     "global_gated_attention", "asl"),
    "fixed_vector_pubchemfp_concat_asl":     ("fixed_vector",   "concat",     "global_gated_attention", "asl"),
}


def load_config_for_checkpoint(model_name: str, device) -> Config:
    """Build the Config for `model_name` by reading it out of the checkpoint itself.

    This is the fix for the folder-name-guessing bug: do not infer encoder/fusion/
    pooling/loss from `model_name` or from `Config.resolve_paths()`. Read them from
    `checkpoint["config"]`, which `src/train.py` already saves for every run.
    """
    ckpt_path = os.path.join("checkpoints", model_name, "best_model.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "config" in checkpoint and isinstance(checkpoint["config"], Config):
        config = checkpoint["config"]
        print(f"[{model_name}] loaded config from checkpoint: "
              f"encoder={config.encoder}, fusion={config.fusion}, "
              f"pooling={config.pooling}, loss={config.loss}")
    else:
        # Old-style checkpoint with no embedded config — fall back, but say so loudly.
        if model_name not in LEGACY_MODEL_REGISTRY:
            raise ValueError(
                f"Checkpoint '{model_name}' has no embedded config and is not in "
                f"LEGACY_MODEL_REGISTRY — cannot determine its architecture safely."
            )
        encoder, fusion, pooling, loss = LEGACY_MODEL_REGISTRY[model_name]
        print(f"[{model_name}] WARNING: no embedded config in checkpoint, "
              f"using LEGACY_MODEL_REGISTRY fallback: "
              f"encoder={encoder}, fusion={fusion}, pooling={pooling}, loss={loss}")
        config = Config()
        config.encoder = encoder
        config.fusion = fusion
        config.pooling = pooling
        config.loss = loss

    # Sanity check: this repo's ensemble task only touches molformer checkpoints.
    assert config.encoder == "molformer", (
        f"Expected a molformer checkpoint for '{model_name}', got encoder={config.encoder}"
    )

    # Now, and ONLY now, set the I/O paths this task needs (never touch the
    # architecture fields above). Do this directly — do NOT call
    # config.resolve_paths() or src.dataset.build_dataloaders() anywhere in this
    # pipeline (see Bug 2 below for why).
    config.checkpoint_dir = os.path.join("checkpoints", model_name)
    config.metrics_dir = os.path.join("metrics", model_name)
    config.train_csv = "data/train.csv"
    config.val_csv = "data/val.csv"
    config.test_csv = "data/test.csv"

    return config, checkpoint
```

Then build the encoder/model exactly as `inference.py` already does, using this
`config` — no other change needed:

```python
from src.encoders import ENCODER_REGISTRY
from src.model import CompatibilityModel

def build_model_from_checkpoint(model_name, device):
    config, checkpoint = load_config_for_checkpoint(model_name, device)

    encoder_cls = ENCODER_REGISTRY[config.encoder]
    encoder = encoder_cls(config.molformer_model_path, device=str(device))
    encoder.to(device)
    config.encoder_output_dim = encoder.output_dim

    model = CompatibilityModel(config, encoder)
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config
```

**Verification step (do this before Part B):** call `load_config_for_checkpoint` for
all 3 models below and print the resolved `(encoder, fusion, pooling, loss)`. Confirm
they match the table in Part B, Section 0. If any of them do NOT match (i.e. the
embedded config disagrees with what the folder name implies), STOP and report this to
the user before proceeding — that would mean a checkpoint's actual trained
architecture differs from what its folder name suggests, which is exactly the kind of
silent mismatch this fix exists to catch, and the ensemble results would be
meaningless if built on the wrong model.

Optional but recommended, separately from this task: apply the same fix to
`inference.py` itself — replace its per-model `Config()` construction (which currently
sets `encoder/fusion/pooling/loss` from `MODEL_REGISTRY`) with a call to
`load_config_for_checkpoint`, keeping `MODEL_REGISTRY` only as the `LEGACY_MODEL_REGISTRY`
fallback above. This removes the second source of truth for good rather than just
avoiding it in this task's new code.

## Bug 2: `build_dataloaders()` silently overwrites `checkpoint_dir` / `metrics_dir`

`src/dataset.py`:

```python
def build_dataloaders(config):
    if hasattr(config, "resolve_paths"):
        config.resolve_paths()
    ...
```

`Config.resolve_paths()` recomputes `checkpoint_dir` and `metrics_dir` from
`encoder`/`fusion`/`pooling`/`loss` every time it's called. Any code that sets
`checkpoint_dir`/`metrics_dir` explicitly (as `inference.py` does, and as this task's
code does above) and *then* calls `build_dataloaders(config)` will have those fields
silently reset — for the folder-naming reasons in Bug 1, this can point at the wrong
directory with no error raised. This is why the previous version of this spec had to
tell you to avoid `build_dataloaders()` entirely and build `CompatibilityDataset`
directly instead.

**The actual fix (apply to `src/config.py` and `src/dataset.py`):** split
`resolve_paths()` into two independent methods so CSV-path resolution and
checkpoint/metrics-path resolution can't clobber each other:

```python
# src/config.py — inside class Config

def resolve_csv_paths(self):
    """Only fills in train/val/test CSV defaults. Safe to call any number of times."""
    if self.train_csv is None:
        self.train_csv = f"{self.data_dir}/train.csv"
    if self.val_csv is None:
        self.val_csv = f"{self.data_dir}/val.csv"
    if self.test_csv is None:
        self.test_csv = f"{self.data_dir}/test.csv"

def resolve_checkpoint_paths(self):
    """Derives checkpoint_dir/metrics_dir from encoder/fusion/pooling/loss.
    Call this once, right after building a fresh Config for training a NEW run —
    never call it on a config whose checkpoint_dir/metrics_dir were already set
    explicitly (e.g. when loading an existing checkpoint), or it will overwrite them.
    """
    if self.fusion == "cross_attn":
        if self.pooling == "explicit_pairwise":
            pool_str = "pairwise"
        elif self.pooling == "global_gated_attention":
            pool_str = "global_gated"
        else:
            pool_str = self.pooling
        combo = f"{self.encoder}_{self.fusion}_{pool_str}_{self.loss}"
    else:
        combo = f"{self.encoder}_{self.fusion}_{self.loss}"
    self.checkpoint_dir = f"checkpoints/{combo}"
    self.metrics_dir = f"metrics/{combo}"

def resolve_paths(self):
    """Back-compat wrapper: old callers that expect resolve_paths() to do both
    still work. New code should call the two methods above individually instead,
    so it can resolve CSV paths without clobbering an explicitly-set checkpoint_dir.
    """
    self.resolve_checkpoint_paths()
    self.resolve_csv_paths()
```

Then update `build_dataloaders` in `src/dataset.py` to only resolve CSV paths, never
checkpoint paths:

```python
def build_dataloaders(config):
    if hasattr(config, "resolve_csv_paths"):
        config.resolve_csv_paths()   # was: config.resolve_paths()
    ...
```

And update `main.py` (the training entry point) to call `config.resolve_checkpoint_paths()`
explicitly, once, right after constructing a fresh `Config()` for a new training run
(this preserves today's training behavior exactly — it's the same computation, just no
longer bundled with CSV resolution or callable in a place that can clobber an
already-loaded checkpoint's paths).

This is a small, backward-compatible, non-destructive patch: existing behavior for
training a new run is unchanged (`resolve_checkpoint_paths()` computes exactly what
`resolve_paths()` used to), and `__post_init__` still calls `resolve_paths()` so any
code relying on the old combined behavior keeps working — the only new capability is
that code can now resolve CSV paths alone without touching checkpoint paths, which is
what this task's loading code (and `inference.py`) actually needs.

**Do not skip Bug 2's patch and rely only on Bug 1's fix.** Bug 1's fix makes model
*identity* correct (which architecture to build). Bug 2's patch is what makes it safe
to later call `build_dataloaders()` or any future code that does, without silently
losing the checkpoint/metrics paths Bug 1 just set correctly. They fix two different
failure modes and both matter.

---

# PART B — Build the 6 ensemble models

Everything below is unchanged in substance from the original spec, except that all
model/config loading now goes through `load_config_for_checkpoint` /
`build_model_from_checkpoint` from Part A instead of a hardcoded table.

## 0. What exists already (do not recompute, reuse it)

- Model class: `src.model.CompatibilityModel`
- Encoder registry: `src.encoders.ENCODER_REGISTRY`
- Dataset: `src.dataset.CompatibilityDataset`, collate function `src.dataset.collate_fn`
- Config: `src.config.Config` (as patched in Part A)
- Evaluation helpers (**reuse these exactly, do not reimplement**):
  `src.evaluate.collect_predictions(model, loader, device) -> (probs, labels)`
  `src.evaluate.tune_threshold(probs, labels, step) -> (best_thresh, best_f1)`
  `src.evaluate.compute_metrics(probs, labels, threshold) -> dict` (pr_auc, f1, mcc,
  precision, recall, accuracy, threshold, confusion_matrix)
  `src.evaluate.save_metrics(metrics, filepath)`
- Descriptors: `src.descriptors.DESCRIPTOR_NAMES` (21 names), `src.descriptors.normalize_descriptors(values, mean, std)`
- Data splits (columns: `API_CID,Excipient_CID,Outcome1,API_Smiles,Excipient_Smiles`):
  `data/train.csv`, `data/val.csv`, `data/test.csv`
- Held-out set (columns: `Api_name,Excipient_name,API_CID,Excipient_CID,API_Smiles,Excipient_Smiles,ground_truth`):
  `held_out_testset/held_out_test_set.csv`, with its own descriptor tables
  `held_out_testset/api_descriptors.csv`, `held_out_testset/excipient_descriptors.csv`
- Descriptor tables (full dataset, used for val/test): `data/api_descriptors.csv`,
  `data/excipient_descriptors.csv`, keyed by `API_CID` / `Excipient_CID`
- Normalization stats (mean/std computed on train split, used everywhere): `models/descriptor_norm_stats.json`
  → keys `api_mean`, `api_std`, `exc_mean`, `exc_std`, each length 21, same order as `DESCRIPTOR_NAMES`
- Existing checkpoints: `checkpoints/<model_name>/best_model.pt`, each containing
  `model_state_dict` and (per Part A) `config`
- Existing metrics: `metrics/<model_name>/val_metrics.json`, `test_metrics.json`,
  `training_history.json`. `val_metrics.json["threshold"]` is that base model's own
  val-tuned decision threshold (not used by the ensembles except for reference).

### The 3 base models involved in this task

Expected `(encoder, fusion, pooling, loss)` for each — **verify these against what
`load_config_for_checkpoint` actually reads out of each checkpoint (Part A's
verification step) before proceeding; this table is the expectation to check against,
not the source of truth anymore:**

| model_name (= checkpoint & metrics folder name) | encoder | fusion | pooling | loss |
|---|---|---|---|---|
| `molformer_concat_asl` | `molformer` | `concat` | `global_gated_attention` | `asl` |
| `molformer_cross_attn_asl` | `molformer` | `cross_attn` | `global_gated_attention` | `asl` |
| `molformer_cross_attn_bce` | `molformer` | `cross_attn` | `global_gated_attention` | `bce` |

### The 2 ensemble pairs to build (as explicitly requested)

- **Pair 1:** `molformer_concat_asl` + `molformer_cross_attn_asl`
- **Pair 2:** `molformer_concat_asl` + `molformer_cross_attn_bce`

### The 3 ensemble methods to apply to each pair

1. Simple average of probabilities
2. Weighted average of probabilities (weight tuned on val)
3. Logistic-regression stacker

→ 2 pairs × 3 methods = **6 ensemble models**, each stored separately.

---

## 1. Building datasets/loaders for the 3 base models

Using the `config` returned by `load_config_for_checkpoint` (Section/Part A), build
datasets **directly** with `CompatibilityDataset` — still do not call
`build_dataloaders()` here even after Bug 2's patch, since it would needlessly rebuild
things this task already controls explicitly:

```python
from src.dataset import CompatibilityDataset, collate_fn
from torch.utils.data import DataLoader

val_ds = CompatibilityDataset(
    config.val_csv, config.api_descriptors_path,
    config.excipient_descriptors_path, config.descriptor_norm_stats_path,
)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0)
# same pattern for test_csv, and for held_out_testset/held_out_test_set.csv
# (held-out set needs the HeldOutDataset class already defined in inference.py — reuse it,
#  import it directly: `from inference import HeldOutDataset`)
```

`shuffle=False` is mandatory — the ensemble alignment step in Section 3 depends on
val/test row order being identical to the row order in `data/val.csv` / `data/test.csv`,
and held-out row order identical to `held_out_testset/held_out_test_set.csv`.

---

## 2. Step 1 — Cache base-model predictions (do this once per base model, reuse for both pairs)

For each of the 3 base models, and for each of the 3 splits (`val`, `test`, `held_out`),
build the model via `build_model_from_checkpoint(model_name, device)` (Part A), then
run `src.evaluate.collect_predictions(model, loader, device)` to get `(probs, labels)`,
then save a CSV with **explicit key columns** so later alignment cannot go wrong even if
row order assumptions are ever violated:

Output path: `ensemble/_base_predictions/<model_name>/<split>.csv`

Columns: `row_index, API_CID, Excipient_CID, label, prob`

- `row_index` = the 0-based index into that split's dataset (same order the dataloader
  produced, since `shuffle=False`)
- `API_CID`, `Excipient_CID` = read directly from the split's source CSV at that row_index
  (`data/val.csv` / `data/test.csv` / `held_out_testset/held_out_test_set.csv`)
- `label` = ground truth (`Outcome1` for val/test, `ground_truth` for held_out) as int 0/1
- `prob` = sigmoid probability from `collect_predictions`

This gives 3 models × 3 splits = 9 cached CSVs under `ensemble/_base_predictions/`.

---

## 3. Step 2 — Align pairs

For each of the 2 pairs and each split, load the two base models' cached CSVs from
Step 1 and merge them on **`["row_index", "API_CID", "Excipient_CID"]`** (inner join;
assert the merged row count equals the original split row count and that the two
`label` columns are identical after the join — raise an error if not, do not silently
proceed on a mismatch). Result per split: a dataframe with columns
`row_index, API_CID, Excipient_CID, label, prob_a, prob_b` where `prob_a` is the
`concat_asl` probability and `prob_b` is the cross-attn variant's probability.

---

## 4. Step 3 — Build the 3 ensembles per pair

For each of the 2 pairs, create this folder layout under `ensemble/`:

```
ensemble/
  concat_asl__cross_attn_asl/
    avg/
      checkpoint/ensemble_config.json
      metrics/val_metrics.json
      metrics/test_metrics.json
      metrics/held_out_metrics.json
      held_out_predictions.csv
    weighted_avg/
      checkpoint/ensemble_config.json
      checkpoint/weight_search.csv
      metrics/val_metrics.json
      metrics/test_metrics.json
      metrics/held_out_metrics.json
      held_out_predictions.csv
    lr_stacker/
      checkpoint/lr_model.joblib
      checkpoint/cv_results.csv
      metrics/val_metrics.json
      metrics/test_metrics.json
      metrics/held_out_metrics.json
      held_out_predictions.csv
  concat_asl__cross_attn_bce/
    avg/ ...
    weighted_avg/ ...
    lr_stacker/ ...
  comparison_summary.csv
```

All metrics JSON files must use `src.evaluate.compute_metrics` and
`src.evaluate.save_metrics` exactly (same schema as the rest of the repo:
`pr_auc, f1, mcc, precision, recall, accuracy, threshold, confusion_matrix`), so the
new numbers are directly comparable to existing `metrics/<model>/test_metrics.json` files.

### 4a. Method 1 — Simple average

```
ensemble_prob = (prob_a + prob_b) / 2
```
computed on all 3 splits.
Tune the decision threshold on **val only** using `src.evaluate.tune_threshold(val_probs, val_labels, step=0.001)`.
Apply that fixed threshold to test and held-out. Compute and save metrics for all 3 splits.
Save `checkpoint/ensemble_config.json`:
```json
{"method": "avg", "base_model_a": "molformer_concat_asl", "base_model_b": "<other>",
 "weight_a": 0.5, "weight_b": 0.5, "threshold": <val-tuned threshold>}
```

### 4b. Method 2 — Weighted average

**Weight selection metric: val PR-AUC (`sklearn.metrics.average_precision_score`), not F1.**
Rationale (do not change this without re-reading the project spec): the dataset is
~90/10 imbalanced (`master_experiment_spec.md`), and this repo's own convention already
treats PR-AUC as the primary threshold-independent ranking metric — every existing
`metrics/<model>/test_metrics.json` lists `pr_auc` first, and F1 in this codebase is
only ever a *threshold-tuning* target (see `tune_threshold`), not a model-selection
target. A weight chosen by F1 would be entangled with an arbitrary threshold; PR-AUC
lets you pick the weight independent of thresholding, then tune the threshold
afterward exactly like the rest of the repo does.

Procedure:
1. On the **val split only**, sweep `w = 0.00, 0.05, 0.10, ..., 1.00` (21 values).
   For each `w`: `blended = w * prob_a + (1 - w) * prob_b`; compute
   `val_pr_auc = average_precision_score(val_labels, blended)`.
2. Save the full sweep to `checkpoint/weight_search.csv` (columns: `weight_a, val_pr_auc`).
3. Pick `w* = argmax(val_pr_auc)`. If there is a tie, break it by higher
   `f1_score` at that `w` using a threshold from `tune_threshold` computed for
   each tied candidate, and pick the one with higher resulting F1.
4. With `w*` fixed, tune the decision threshold on val via `tune_threshold`
   (same as Method 1).
5. Apply `w*` and the threshold to test and held-out. Compute and save metrics for all 3 splits.
6. Save `checkpoint/ensemble_config.json`:
```json
{"method": "weighted_avg", "base_model_a": "molformer_concat_asl", "base_model_b": "<other>",
 "weight_a": <w*>, "weight_b": <1-w*>, "selection_metric": "val_pr_auc",
 "threshold": <val-tuned threshold>}
```

### 4c. Method 3 — Logistic regression stacker

**Features (finalized — use all of these, 44 total):**
- `prob_a`, `prob_b` (the two base models' probabilities) — 2 features
- The 21 normalized API descriptors + 21 normalized excipient descriptor values for
  that pair, taken from the **same** normalized descriptor vectors the models
  themselves are trained on: load raw values from `data/api_descriptors.csv` /
  `data/excipient_descriptors.csv` (or the `held_out_testset/` versions for the
  held-out split) keyed by `API_CID` / `Excipient_CID`, then normalize with
  `src.descriptors.normalize_descriptors(values, mean, std)` using the means/stds from
  `models/descriptor_norm_stats.json` — do not fit new normalization stats, reuse the
  existing ones so the features are on the same scale the base models saw. Replace any
  resulting NaN with 0.0, matching `CompatibilityDataset.__getitem__`.
- Total feature vector per pair: `[prob_a, prob_b, *api_desc_norm (21), *exc_desc_norm (21)]` = 44 dims.

**Explicit overfitting caveat (do not skip this step):** the val split has ~842 rows
and only ~10% positive class (~84 positives). 44 features on that little data is a
real overfitting risk for an unregularized model, so counter it with:
- `StandardScaler` fit on val features only, applied to val/test/held-out.
- `sklearn.linear_model.LogisticRegression(penalty="l2", class_weight="balanced", solver="liblinear")`
- Tune `C` via `sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
  cross-validation **on the val split only**, grid `C in [0.001, 0.01, 0.1, 1, 10, 100]`,
  scored by mean `average_precision_score` (PR-AUC) across folds — consistent with the
  PR-AUC selection rationale in 4b. Save all fold scores to `checkpoint/cv_results.csv`
  (columns: `C, fold, val_pr_auc`).
- Refit on the **full val split** with the best `C`.
- Tune the decision threshold on val predictions via `tune_threshold` (same as
  Methods 1 and 2, for consistency).
- Apply to test and held-out. Compute and save metrics for all 3 splits.
- Save the fitted scaler + model + feature order + threshold together with `joblib.dump`:
```python
joblib.dump(
    {"scaler": scaler, "model": lr_model,
     "feature_names": ["prob_a", "prob_b"] + [f"api_{n}" for n in DESCRIPTOR_NAMES] + [f"exc_{n}" for n in DESCRIPTOR_NAMES],
     "best_C": best_C, "threshold": threshold},
    "ensemble/<pair>/lr_stacker/checkpoint/lr_model.joblib",
)
```

---

## 5. Held-out predictions CSV (per ensemble, all 6)

For every one of the 6 ensembles, also save `held_out_predictions.csv` with columns:
`Api_name, Excipient_name, API_CID, Excipient_CID, ground_truth, ensemble_probability, ensemble_prediction`
(`ensemble_prediction` = `(ensemble_probability >= threshold).astype(int)` using that
ensemble's own val-tuned threshold).

---

## 6. Final comparison table

Write `ensemble/comparison_summary.csv` with one row per model: the 3 base models
(`molformer_concat_asl`, `molformer_cross_attn_asl`, `molformer_cross_attn_bce`, reading
their existing `metrics/<name>/test_metrics.json`) plus the 6 new ensembles. Columns:
`model_name, split, pr_auc, f1, mcc, precision, recall, accuracy, threshold`, with one
row per `(model, split)` for `split in ["val", "test", "held_out"]` — so 9 models × 3
splits = 27 rows total. This lets a human immediately see whether any ensemble beats
the best individual base model on test and held-out, and whether the held-out ranking
agrees with the (much more reliable) test-set ranking.

---

## 7. Implementation notes / order of operations

1. **Apply Part A's two patches first** (`src/config.py` split of `resolve_paths()`,
   `src/dataset.py`'s `build_dataloaders` calling `resolve_csv_paths()` instead) and
   run the Part A verification step for all 3 base models before writing any ensemble
   code. If verification fails for any model, stop and report it — do not proceed on a
   guess.
2. Write a shared module `ensemble/_common.py` with: `load_config_for_checkpoint`,
   `build_model_from_checkpoint` (Part A), the prediction-caching function (Section 2),
   and the metrics/threshold helpers imported from `src.evaluate` (do not duplicate
   their logic).
3. Run the caching step once (Section 2) — it is the expensive part (loads MoLFormer
   3 times). Cache to disk so re-running the ensembling logic later doesn't require
   reloading models.
4. Implement Sections 4a/4b/4c as pure-Python/pandas/sklearn functions operating on the
   cached CSVs — no GPU/model loading needed for these.
5. Run both pairs through all 3 methods, writing to the folder layout in Section 4.
6. Build `comparison_summary.csv` last, reading from the freshly written metrics files.
7. Do not overwrite anything under `checkpoints/`, `metrics/`, or `held_out_testset/`
   — all new output goes under `ensemble/` only. The only files this task modifies
   outside `ensemble/` are `src/config.py`, `src/dataset.py`, and (optionally)
   `inference.py`, per Part A — and those are additive/backward-compatible patches,
   not rewrites.
