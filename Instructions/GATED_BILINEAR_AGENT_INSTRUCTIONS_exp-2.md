# AI Agent Instructions — Gated Bilinear PGB Implementation

**Project:** `Api-Excipient-Compatibility-Model-main`  
**Goal:** Rewrite the 9 group-split model pipelines so they match the *Prior-Gated Bilinear (PGB)* architecture described in `Gated_Bilinear_PairNet_Architecture_and_Results.docx` and in `Instructions/Gated_Bilinear_Architectures_All_Group_Split_Models.md`. The result must score close to the doc's numbers on the 24-pair held-out set: PR-AUC ≥ 0.80, ROC-AUC ≥ 0.85 at a 0.60 screen cut, and 0 false positives at the default cut.

---

## 0. Read first — what you are working with

### File map (all paths relative to project root)
```
src/
  config.py          195 lines   — Config dataclass, all hyperparameters
  model.py           573 lines   — CompatibilityModel, ProjectionHead, GatedAttentionPooling
  train.py           148 lines   — training loop, AdamW, ReduceLROnPlateau, early stop
  evaluate.py        119 lines   — collect_predictions, tune_threshold (val F1), compute_metrics
  cross_validate.py  281 lines   — 5-fold CV harness, build_encoder, WeightedRandomSampler
  dataset.py         173 lines   — CompatibilityDataset, collate_fn
  descriptors.py     132 lines   — DESCRIPTOR_NAMES (21), _compute_descriptors, normalize
  lookup.py           71 lines   — REACTION_TABLE (5 SMARTS rules), get_mechanism_matches
  encoders/
    fixed_vector_encoder.py      — FixedVectorEncoder (mol2vec / pubchemfp / maccs / morgan)
    molformer_encoder.py         — MoLFormerEncoder
    chemberta_encoder.py         — ChemBERTaEncoder
    gin_encoder.py               — PretrainedGINEncoder
    stanford_gat_encoder.py      — StanfordPretrainedGATEncoder
    dmpnn_encoder.py             — DMPNNEncoder (CheMeleon)
```

### The 9 encoder families (group-split only, do not touch random-split runs)
| # | Family key | Encoder class | is_sequence_capable |
|---|---|---|---|
| 1 | `fixed_vector_maccs` | `FixedVectorEncoder(source="maccs")` | False |
| 2 | `fixed_vector_morgan` | `FixedVectorEncoder(source="morgan")` | False |
| 3 | `fixed_vector_pubchemfp` | `FixedVectorEncoder(source="pubchemfp")` | False |
| 4 | `fixed_vector_mol2vec` | `FixedVectorEncoder(source="mol2vec")` | False |
| 5 | `molformer` | `MoLFormerEncoder` | True |
| 6 | `chemberta` | `ChemBERTaEncoder` | True |
| 7 | `pretrained_gin` | `PretrainedGINEncoder` | True |
| 8 | `pretrained_gat` | `StanfordPretrainedGATEncoder` | True |
| 9 | `dmpnn_chemprop` | `DMPNNEncoder` | True |

### Current problems (verified from code + predictions)
1. **No excipient prior** — `init_bias()` sets a global bias, same for every excipient.
2. **Double class-weighting** — `use_balanced_sampler=True` default + ASL/focal loss both upweight positives.
3. **No rank-16 bilinear, no gate** — pair vector is only `[h_api, h_exc, product, diff]` (609-d).
4. **No chemistry flags, no salt-aware parsing** — `fr_*` fragment counts used instead of yes/no flags.
5. **Val-F1 threshold is unstable** — ranges 0.042 to 0.998 across models; many thresholds above 0.80 kill recall.
6. **Fixed-vector models return zero vectors** for unknown molecules on the 24-pair set (15/24 APIs are zeros).
7. **Descriptors include 11 `fr_*` counts** that the doc replaced with 12 physics-based descriptors.

---

## 1. New files to create

### 1.1 `src/features.py` — chemistry flags + on-the-fly fingerprints + salt parsing

Create this file from scratch. It must export four public functions.

```python
"""
src/features.py
On-the-fly molecular feature extraction for the PGB head.
All functions accept a single SMILES string and return numpy arrays.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, rdMolDescriptors
from rdkit.Chem.rdchem import Mol

# ── helpers ────────────────────────────────────────────────────────────────

def _parse_salt_aware(smiles: str) -> list[Mol]:
    """
    Split on '.' and return a list of valid RDKit Mol objects.
    If the whole SMILES parses, return [mol].  If not, try fragments.
    Never returns an empty list (falls back to []).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return [mol]
    frags = smiles.split(".")
    mols = [Chem.MolFromSmiles(f) for f in frags]
    return [m for m in mols if m is not None]


def _or_bits(arrs: list[np.ndarray]) -> np.ndarray:
    """Element-wise OR over a list of uint8 arrays. Returns zeros if list is empty."""
    if not arrs:
        return np.array([], dtype=np.uint8)
    result = arrs[0].copy()
    for a in arrs[1:]:
        result = result | a
    return result


# ── public API ─────────────────────────────────────────────────────────────

def maccs_bits(smiles: str) -> np.ndarray:
    """
    167-bit MACCS key vector, salt-aware (bits are OR-ed across fragments).
    Returns float32 array of shape (167,).
    Bit 0 is kept to match RDKit's indexing (indices 0-166).
    """
    mols = _parse_salt_aware(smiles)
    if not mols:
        return np.zeros(167, dtype=np.float32)
    arrays = [np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.uint8) for m in mols]
    return _or_bits(arrays).astype(np.float32)


def morgan_bits(smiles: str, radius: int = 2, n_bits: int = 512) -> np.ndarray:
    """
    Morgan fingerprint, salt-aware OR-fold. Returns float32 (n_bits,).
    """
    mols = _parse_salt_aware(smiles)
    if not mols:
        return np.zeros(n_bits, dtype=np.float32)
    arrays = []
    for mol in mols:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arrays.append(np.array(list(fp), dtype=np.uint8))
    return _or_bits(arrays).astype(np.float32)


# Chemistry flag SMARTS — 18 flags total, in this exact order
_FLAG_SMARTS = [
    ("no_carbons",        None),          # handled specially: len(mol.GetAtoms() with atomic num 6) == 0
    ("oxidizer",          "[#8-,#8+0;!$([OH])][#6,#7,#16]"),   # peroxide-like or O attached to heteroatom
    ("strong_acid",       "[SX4](=O)(=O)[OH]"),                 # sulfonic acid
    ("strong_base",       "[OH-,O-;!$(OC=O)]"),                 # free oxide/hydroxide ion
    ("has_metal",         "[#3,#11,#12,#13,#19,#20,#22,#23,#25,#26,#27,#28,#29,#30]"),
    ("has_Mg",            "[#12]"),
    ("has_Ca",            "[#20]"),
    ("has_Al",            "[#13]"),
    ("has_Si",            "[#14]"),
    ("has_Ti_Fe",         "[#22,#26]"),
    ("has_Mn_Cr",         "[#25,#24]"),
    ("is_salt",           None),          # handled specially: '.' in SMILES or fragment count > 1
    ("very_short",        None),          # handled specially: heavy atom count <= 3
    ("many_OH",           "[OX2H]"),      # flag fires when count >= 2
    ("has_amine",         "[NX3;H1,H2;!$(NC=O)]"),
    ("carboxylic_acid",   "[CX3](=O)[OX2H1]"),
    ("ester",             "[CX3](=O)[OX2H0][#6]"),
    ("phenol",            "[OX2H][cX3]"),
]

_FLAG_PATTERNS = None  # lazily compiled


def _compile_flag_patterns():
    global _FLAG_PATTERNS
    if _FLAG_PATTERNS is not None:
        return
    _FLAG_PATTERNS = []
    for name, smarts in _FLAG_SMARTS:
        if smarts is not None:
            pat = Chem.MolFromSmarts(smarts)
        else:
            pat = None
        _FLAG_PATTERNS.append((name, pat))


def chemistry_flags(smiles: str) -> np.ndarray:
    """
    18-bit chemistry flag vector, salt-aware.
    Returns float32 array of shape (18,).
    Each flag is 1.0 if ANY fragment matches.
    """
    _compile_flag_patterns()
    mols = _parse_salt_aware(smiles)
    flags = np.zeros(18, dtype=np.float32)
    if not mols:
        return flags

    for i, (name, pat) in enumerate(_FLAG_PATTERNS):
        if name == "no_carbons":
            # True if no fragment has a carbon atom
            has_carbon = any(
                any(a.GetAtomicNum() == 6 for a in m.GetAtoms())
                for m in mols
            )
            flags[i] = 0.0 if has_carbon else 1.0
        elif name == "is_salt":
            # True if original SMILES has a dot OR more than one fragment parsed
            flags[i] = 1.0 if ("." in smiles or len(mols) > 1) else 0.0
        elif name == "very_short":
            # True if every fragment has <= 3 heavy atoms
            all_short = all(m.GetNumHeavyAtoms() <= 3 for m in mols)
            flags[i] = 1.0 if all_short else 0.0
        elif name == "many_OH":
            # True if any fragment has >= 2 hydroxyl matches
            total = sum(len(m.GetSubstructMatches(pat)) for m in mols if pat)
            flags[i] = 1.0 if total >= 2 else 0.0
        else:
            if pat is not None:
                match = any(m.HasSubstructMatch(pat) for m in mols)
                flags[i] = 1.0 if match else 0.0

    return flags


# 12 RDKit descriptors (replaces the 21 currently in descriptors.py for PGB models)
PGB_DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "NumHeteroatoms",
    "qed",
]


def pgb_descriptors(smiles: str) -> np.ndarray:
    """
    12 physicochemical descriptors for PGB.
    Returns float32 (12,). NaN on parse failure — caller must nan_to_num.
    """
    from rdkit.Chem import QED
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full(12, float("nan"), dtype=np.float32)

    vals = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        mol.GetNumHeavyAtoms(),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6)),
        QED.qed(mol),
    ]
    return np.array(vals, dtype=np.float32)
```

**Verification:** after writing the file, run:
```bash
python3 -c "
from src.features import maccs_bits, morgan_bits, chemistry_flags, pgb_descriptors
import numpy as np
m = maccs_bits('CC(=O)Nc1ccc(O)cc1')   # paracetamol
print('MACCS shape:', m.shape, 'sum:', m.sum())
mo = morgan_bits('CC(=O)Nc1ccc(O)cc1')
print('Morgan shape:', mo.shape, 'sum:', mo.sum())
f = chemistry_flags('[Mg+2]')
print('Flags shape:', f.shape, 'metal:', f[4], 'Mg:', f[5])
d = pgb_descriptors('CC(=O)Nc1ccc(O)cc1')
print('Desc shape:', d.shape, 'any nan:', np.isnan(d).any())
"
```
All four asserts must pass before proceeding.

---

### 1.2 `src/pgb_prior.py` — per-excipient prior table

Create this file from scratch. It is used by the training harness to attach `p_excipient` to each batch row, and by the PGB head to compute the logit offset.

```python
"""
src/pgb_prior.py
Laplace-smoothed per-excipient incompatibility rate computed from training rows.
Used by PGBCompatibilityModel to provide the logit offset.
"""

import json
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

GLOBAL_RATE = 0.094          # global incompatibility rate across the full dataset
LAPLACE_ALPHA = 4            # Laplace smoothing count
LAPLACE_PRIOR = GLOBAL_RATE  # pseudocount prior


def _canonical_stereo_blind(smiles: str) -> str | None:
    """
    Canonical SMILES with stereo info stripped.
    Returns None if RDKit cannot parse the string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def _strip_hydrates_solvents(smiles: str) -> str:
    """
    Remove water and common solvent fragments from a SMILES.
    Returns the heaviest remaining fragment, stereo-blind canonical.
    Solvent/water patterns: O (water), CO (methanol), CCO (ethanol), OCC (same).
    """
    STRIP = {"O", "[H]O[H]", "CO", "CCO", "OCC", "CC(O)=O"}
    frags = smiles.split(".")
    # Remove known solvents
    keep = [f for f in frags if f.strip() not in STRIP]
    if not keep:
        keep = frags  # nothing to strip, keep original
    # Return canonical of heaviest fragment
    best = sorted(keep, key=len, reverse=True)[0]
    return _canonical_stereo_blind(best) or best


def _morgan512(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)


class ExcipientPriorTable:
    """
    Holds a train-derived per-excipient incompatibility rate.
    Three lookup keys per excipient (exact → family → nearest-neighbour → global).
    """

    def __init__(self):
        self._exact: dict[str, tuple[float, int]] = {}   # key → (p, n)
        self._family: dict[str, float] = {}               # stripped key → p_family
        self._fps: dict[str, object] = {}                 # exact_key → Morgan fp
        self._fitted = False

    def fit(self, train_df: pd.DataFrame, leave_out_cluster_id=None):
        """
        Compute the prior table from train_df rows.
        train_df must have columns: Excipient_Smiles, Outcome1, [cluster_id optional].

        If leave_out_cluster_id is given, rows whose API belongs to that cluster
        are excluded from the prior computation (leave-own-cluster-out for training rows).
        """
        df = train_df.copy()
        if leave_out_cluster_id is not None and "cluster_id" in df.columns:
            df = df[df["cluster_id"] != leave_out_cluster_id]

        for smi, group in df.groupby("Excipient_Smiles"):
            key = _canonical_stereo_blind(smi)
            if key is None:
                key = smi
            n = len(group)
            pos = group["Outcome1"].sum()
            p = (pos + LAPLACE_ALPHA * LAPLACE_PRIOR) / (n + LAPLACE_ALPHA)
            self._exact[key] = (float(p), int(n))

            fam_key = _strip_hydrates_solvents(smi)
            if fam_key not in self._family:
                self._family[fam_key] = p
            else:
                # blend toward the new value with running mean
                self._family[fam_key] = (self._family[fam_key] + p) / 2.0

            fp = _morgan512(smi)
            if fp is not None:
                self._fps[key] = fp

        self._fitted = True

    def lookup(self, exc_smiles: str) -> dict:
        """
        Return a dict with keys:
            p_exact       float  Laplace-smoothed incompatibility rate
            p_delta       float  p_exact - GLOBAL_RATE
            p_family      float  hydrate-stripped family rate
            log_n         float  log(1 + n)
            unseen        float  1.0 if unseen, 0.0 otherwise

        Call this at inference time (for every batch row).
        p_exact is the value used as the logit offset.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before lookup().")

        key = _canonical_stereo_blind(exc_smiles)
        if key is None:
            key = exc_smiles

        # Exact match
        if key in self._exact:
            p, n = self._exact[key]
            fam_key = _strip_hydrates_solvents(exc_smiles)
            p_fam = self._family.get(fam_key, p)
            return {
                "p_exact": p,
                "p_delta": p - GLOBAL_RATE,
                "p_family": p_fam,
                "log_n": float(np.log1p(n)),
                "unseen": 0.0,
            }

        # Family key fallback
        fam_key = _strip_hydrates_solvents(exc_smiles)
        if fam_key in self._family:
            p_fam = self._family[fam_key]
            return {
                "p_exact": p_fam,
                "p_delta": p_fam - GLOBAL_RATE,
                "p_family": p_fam,
                "log_n": 0.0,
                "unseen": 0.0,
            }

        # Nearest-neighbour fallback (Tanimoto >= 0.6)
        fp_query = _morgan512(exc_smiles)
        if fp_query is not None and self._fps:
            from rdkit import DataStructs
            best_sim = 0.0
            best_p = GLOBAL_RATE
            for stored_key, stored_fp in self._fps.items():
                sim = DataStructs.TanimotoSimilarity(fp_query, stored_fp)
                if sim > best_sim:
                    best_sim = sim
                    best_p, _ = self._exact[stored_key]
            if best_sim >= 0.6:
                return {
                    "p_exact": best_p,
                    "p_delta": best_p - GLOBAL_RATE,
                    "p_family": best_p,
                    "log_n": 0.0,
                    "unseen": 0.0,
                }

        # Global fallback
        return {
            "p_exact": GLOBAL_RATE,
            "p_delta": 0.0,
            "p_family": GLOBAL_RATE,
            "log_n": 0.0,
            "unseen": 1.0,
        }

    def save(self, path: str):
        data = {
            "exact": {k: list(v) for k, v in self._exact.items()},
            "family": self._family,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExcipientPriorTable":
        t = cls()
        with open(path) as f:
            data = json.load(f)
        t._exact = {k: tuple(v) for k, v in data["exact"].items()}
        t._family = data["family"]
        t._fitted = True
        return t
```

---

## 2. Files to modify

### 2.1 `src/config.py` — add PGB hyperparameters

Add the following fields to the `Config` dataclass. Insert them as a new section **after** the existing `# --- Descriptors ---` block and before `# --- Axis C: loss ---`.

```python
    # --- PGB (Prior-Gated Bilinear) head ---
    use_pgb_head: bool = False              # enable PGB head and all associated features
    pgb_tower_dim: int = 96                 # output dimension of each molecule tower
    pgb_bilinear_rank: int = 16             # rank of the low-rank bilinear term
    pgb_head_hidden_1: int = 128            # MLP layer 1 width
    pgb_head_hidden_2: int = 48             # MLP layer 2 width
    pgb_head_dropout_1: float = 0.30        # dropout after layer 1 (was 0.50)
    pgb_head_dropout_2: float = 0.20        # dropout after layer 2 (was 0.40)
    pgb_prior_scale_init: float = 0.8       # initial value of learned prior scale s
    pgb_prior_offset: float = 0.2           # fraction of global log-odds added to bias init
    pgb_num_flags: int = 18                 # number of chemistry flags (do not change)
    pgb_num_mechanisms: int = 5             # number of SMARTS mechanism rules (do not change)
    pgb_num_desc: int = 12                  # number of PGB descriptors (do not change)
    pgb_prior_vec_dim: int = 5              # [p_exact, p_delta, p_family, log_n, unseen]

    # --- PGB training overrides (only active when use_pgb_head=True) ---
    pgb_lr: float = 1.2e-3                  # LR for new PGB layers (tower, head)
    pgb_encoder_lr: float = 3e-4            # LR for kept encoder layers (proj, cross-attn)
    pgb_weight_decay: float = 2e-3
    pgb_max_epochs: int = 60                # 40 for fixed-vector, 60 for seq. encoders
    pgb_early_stop_patience: int = 7
    pgb_lr_patience: int = 4
    pgb_batch_size: int = 64
    pgb_use_balanced_sampler: bool = False  # MUST stay False — no double weighting
    pgb_loss: str = "asym_focal"            # "asym_focal" is the only valid PGB loss
    pgb_focal_gamma_pos: float = 1.2
    pgb_focal_gamma_neg: float = 2.5
    pgb_tnr_floor: float = 0.97             # min true-negative-rate for threshold selection
    pgb_prior_table_path: str = ""          # set at runtime per fold
```

Also add a new resolver method on `Config`. At the end of `resolve_checkpoint_paths`, insert:

```python
        # PGB runs get their own prefix so they don't overwrite the baseline runs
        if getattr(self, "use_pgb_head", False):
            self.checkpoint_dir = self.checkpoint_dir.replace("checkpoints/", "checkpoints/pgb_")
            self.metrics_dir = self.metrics_dir.replace("metrics/", "metrics/pgb_")
```

### 2.2 `src/loss.py` — add asymmetric focal loss

At the end of the existing `get_loss_fn` function, add a new branch before the final `else`:

```python
    elif config.loss == "asym_focal":
        # Asymmetric focal BCE: gamma_pos on false negatives, gamma_neg on false positives
        # pos_weight is passed in at training time, same as weighted_bce
        def _asym_focal(logits, targets, sw=None, pw=None):
            p = torch.sigmoid(logits)
            # Positive part: down-weight easy positives
            loss_pos = targets * (1 - p) ** config.pgb_focal_gamma_pos \
                       * F.binary_cross_entropy_with_logits(
                           logits, targets,
                           pos_weight=pw,
                           reduction="none"
                       )
            # Negative part: heavily penalize confident false alarms
            loss_neg = (1 - targets) * p ** config.pgb_focal_gamma_neg \
                       * F.binary_cross_entropy_with_logits(
                           logits, targets,
                           reduction="none"
                       )
            loss = loss_pos + loss_neg
            if sw is not None:
                loss = loss * sw
            return loss.mean()
        return _asym_focal
```

Also update `get_loss_fn` to read attributes from `config` safely (some configs won't have `pgb_focal_*`):

```python
    elif config.loss == "asym_focal":
        gp = getattr(config, "pgb_focal_gamma_pos", 1.2)
        gn = getattr(config, "pgb_focal_gamma_neg", 2.5)
        # ... use gp, gn in the closure instead of config.pgb_focal_gamma_*
```

### 2.3 `src/evaluate.py` — replace `tune_threshold` with MCC+TNR version

Replace the entire `tune_threshold` function (lines 48–62 of the current file) with:

```python
def tune_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    step: float = 0.001,
    tnr_floor: float = 0.0,   # 0.0 means no floor (old behaviour, backward-compat)
):
    """
    Sweep thresholds on a validation set.

    If tnr_floor > 0 (e.g. 0.97):
        - Candidate thresholds are those where TNR >= tnr_floor.
        - Among candidates, pick the one that maximises val MCC.
        - If no threshold satisfies the floor, relax to the threshold
          with the highest TNR (most conservative available).
    If tnr_floor == 0:
        - Pick the threshold that maximises val F1  (old behaviour).
    """
    thresholds = np.arange(step, 1.0, step)
    labels_int = labels.astype(int)

    if tnr_floor == 0.0:
        # ── legacy F1 mode ───────────────────────────────────────────────
        best_f1 = -1.0
        best_thresh = 0.5
        for t in thresholds:
            preds = (probs >= t).astype(int)
            f1 = f1_score(labels_int, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        return best_thresh, best_f1

    # ── MCC + TNR-floor mode ─────────────────────────────────────────────
    negatives = (labels_int == 0).sum()

    best_mcc = -2.0
    best_thresh = 0.5
    best_tnr_fallback = -1.0
    best_thresh_fallback = 0.5

    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn = int(((preds == 0) & (labels_int == 0)).sum())
        tnr = tn / negatives if negatives > 0 else 1.0

        # Track best TNR seen (for the fallback case)
        if tnr > best_tnr_fallback:
            best_tnr_fallback = tnr
            best_thresh_fallback = t

        if tnr < tnr_floor:
            continue  # does not satisfy the floor

        mcc = matthews_corrcoef(labels_int, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t

    if best_mcc == -2.0:
        # No threshold satisfied the floor — use the most conservative one
        return best_thresh_fallback, 0.0

    return best_thresh, best_mcc
```

Update `evaluate_model` to accept and pass through `tnr_floor`:

```python
def evaluate_model(model, val_loader, test_loader, device, config):
    val_probs, val_labels = collect_predictions(model, val_loader, device)
    tnr_floor = getattr(config, "pgb_tnr_floor", 0.0) if getattr(config, "use_pgb_head", False) else 0.0
    best_thresh, _ = tune_threshold(val_probs, val_labels, config.threshold_step, tnr_floor=tnr_floor)
    val_metrics = compute_metrics(val_probs, val_labels, best_thresh)
    test_probs, test_labels = collect_predictions(model, test_loader, device)
    test_metrics = compute_metrics(test_probs, test_labels, best_thresh)
    return val_metrics, test_metrics, best_thresh
```

### 2.4 `src/model.py` — add `PGBCompatibilityModel`

Do **not** modify `CompatibilityModel`. Add a completely new class below it.

#### 2.4.1 `MoleculeTower` — per-family encoder tower → 96-d

```python
class MoleculeTower(nn.Module):
    """
    Turns a single molecule's features into a 96-d vector h.
    Input: struct_vec [B, struct_dim], maccs [B,167], morgan [B,512],
           pgb_desc [B,12], flags [B,18].
    For fixed-vector families struct_dim == 0 (no structural vector).
    """

    def __init__(
        self,
        struct_dim: int,       # 128 for seq. encoders, 0 for fixed-vector families
        primary_dim: int,      # MACCS=167, Morgan=1024, PubChem=881, Mol2vec=300
        secondary_dim: int,    # 512 (Morgan secondary) or 166 (MACCS secondary) or 0
        pgb_dim: int = 96,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.has_struct = struct_dim > 0
        self.has_secondary = secondary_dim > 0

        # Structural branch (from sequence encoder)
        if self.has_struct:
            self.struct_block = nn.Identity()   # already 128-d, pass through
            struct_out = struct_dim
        else:
            struct_out = 0

        # Primary fingerprint branch
        self.primary_block = nn.Sequential(
            nn.Linear(primary_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Secondary fingerprint branch (optional)
        if self.has_secondary:
            secondary_out = 32
            self.secondary_block = nn.Sequential(
                nn.Linear(secondary_dim, secondary_out),
                nn.LayerNorm(secondary_out),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )
        else:
            secondary_out = 0

        # Descriptor block: 12 → 24
        self.desc_block = nn.Sequential(
            nn.Linear(12, 24),
            nn.LayerNorm(24),
            nn.GELU(),
        )

        # Flag block: 18 → 16
        self.flag_block = nn.Sequential(
            nn.Linear(18, 16),
            nn.GELU(),
        )

        # Mix: concat all branches → 96
        mix_in = struct_out + 64 + secondary_out + 24 + 16
        self.mix = nn.Sequential(
            nn.Linear(mix_in, pgb_dim),
            nn.LayerNorm(pgb_dim),
            nn.GELU(),
        )

    def forward(
        self,
        struct_vec,     # [B, struct_dim] or None
        primary,        # [B, primary_dim]
        secondary,      # [B, secondary_dim] or None
        pgb_desc,       # [B, 12]
        flags,          # [B, 18]
    ):
        parts = []
        if self.has_struct and struct_vec is not None:
            parts.append(struct_vec)
        parts.append(self.primary_block(primary))
        if self.has_secondary and secondary is not None:
            parts.append(self.secondary_block(secondary))
        parts.append(self.desc_block(pgb_desc))
        parts.append(self.flag_block(flags))
        return self.mix(torch.cat(parts, dim=-1))
```

#### 2.4.2 `PGBHead` — shared bilinear + gate + MLP

```python
class PGBHead(nn.Module):
    """
    Prior-Gated Bilinear head.
    Input: h_api [B,96], h_exc [B,96], prior_vec [B,5], mech_flags [B,5], exc_flags [B,18]
    Output: logit [B]
    """

    def __init__(self, config):
        super().__init__()
        d = config.pgb_tower_dim       # 96
        r = config.pgb_bilinear_rank   # 16

        # Bilinear maps (no bias — each is a thin projection)
        self.U = nn.Linear(d, r, bias=False)
        self.V = nn.Linear(d, r, bias=False)

        # Gate network: [prior_vec(5) + exc_flags(18) + mech_flags(5)] → r
        gate_in = config.pgb_prior_vec_dim + config.pgb_num_flags + config.pgb_num_mechanisms
        self.gate = nn.Sequential(
            nn.Linear(gate_in, r),
            nn.GELU(),
            nn.Linear(r, r),
            nn.Sigmoid(),
        )

        # MLP: [h_api(d) + h_exc(d) + product(d) + diff(d) + bilinear(r) + gated(r) + prior(5) + mech(5)]
        concat_dim = d + d + d + d + r + r + config.pgb_prior_vec_dim + config.pgb_num_mechanisms  # 426
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, config.pgb_head_hidden_1),
            nn.GELU(),
            nn.Dropout(config.pgb_head_dropout_1),
            nn.Linear(config.pgb_head_hidden_1, config.pgb_head_hidden_2),
            nn.GELU(),
            nn.Dropout(config.pgb_head_dropout_2),
            nn.Linear(config.pgb_head_hidden_2, 1),
        )

        # Learned prior scale and bias offset
        self.prior_scale = nn.Parameter(torch.tensor(config.pgb_prior_scale_init))
        self.prior_bias = nn.Parameter(torch.zeros(1))

    def forward(self, h_api, h_exc, prior_vec, mech_flags, exc_flags):
        """
        h_api, h_exc : [B, 96]
        prior_vec    : [B, 5]  — [p_exact, p_delta, p_family, log_n, unseen]
        mech_flags   : [B, 5]
        exc_flags    : [B, 18]
        Returns      : logit [B]
        """
        product = h_api * h_exc
        diff    = torch.abs(h_api - h_exc)

        bilinear = self.U(h_api) * self.V(h_exc)              # [B, r]

        gate_input = torch.cat([prior_vec, exc_flags, mech_flags], dim=-1)  # [B, 28]
        gate       = self.gate(gate_input)                     # [B, r]
        gated      = bilinear * gate                           # [B, r]

        concat = torch.cat(
            [h_api, h_exc, product, diff, bilinear, gated, prior_vec, mech_flags],
            dim=-1,
        )  # [B, 426]

        residual = self.mlp(concat).squeeze(-1)  # [B]

        # Prior logit offset: s * log(p / (1 - p)) + b
        p = prior_vec[:, 0].clamp(1e-6, 1 - 1e-6)   # p_exact
        logit_prior = torch.log(p / (1 - p))
        logit = residual + self.prior_scale * logit_prior + self.prior_bias

        return logit   # [B]

    def init_bias(self, global_rate: float, pgb_prior_offset: float = 0.2):
        """
        Set the prior_bias so that at zero residual and a never-seen excipient
        (p_exact = global_rate) the output is (1 + pgb_prior_offset) * log(p/(1-p)).
        """
        if 0 < global_rate < 1:
            log_odds = np.log(global_rate / (1 - global_rate))
            self.prior_bias.data.fill_(pgb_prior_offset * log_odds)
```

#### 2.4.3 `PGBCompatibilityModel` — top-level model

```python
class PGBCompatibilityModel(nn.Module):
    """
    Full Prior-Gated Bilinear model for one encoder family.

    Keeps the existing encoder (frozen or trainable) and the existing
    cross-attention/pooling layers from CompatibilityModel.
    Replaces the 609-d pair head with MoleculeTower + PGBHead.

    Instantiate by calling pgb_model_from_config(config, encoder) — do not
    call this class directly unless you know exactly what tower_spec to pass.
    """

    def __init__(self, config, encoder, api_tower: MoleculeTower, exc_tower: MoleculeTower):
        super().__init__()
        self.config  = config
        self.encoder = encoder
        self.api_tower = api_tower
        self.exc_tower = exc_tower
        self.head = PGBHead(config)

        enc_dim  = encoder.output_dim
        proj_dim = config.proj_dim   # 128 — kept from existing config

        self.use_cross_attn = (
            config.fusion == "cross_attn" and encoder.is_sequence_capable
        )

        if self.use_cross_attn:
            self.api_proj = ProjectionHead(enc_dim, proj_dim, config.proj_dropout)
            self.exc_proj = ProjectionHead(enc_dim, proj_dim, config.proj_dropout)
            self.exc_placeholder = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.01)
            self.exc_global_placeholder = nn.Parameter(torch.randn(1, proj_dim) * 0.01)
            self.cross_attn_exc = nn.MultiheadAttention(
                proj_dim, config.num_heads, config.attn_dropout, batch_first=True
            )
            self.cross_attn_api = nn.MultiheadAttention(
                proj_dim, config.num_heads, config.attn_dropout, batch_first=True
            )
            self.ln_exc = nn.LayerNorm(proj_dim)
            self.ln_api = nn.LayerNorm(proj_dim)

            pooling = getattr(config, "pooling", "global_gated_attention")
            self.pooling = pooling
            if pooling == "global_gated_attention":
                pool_hidden = getattr(config, "pool_hidden_dim", proj_dim)
                self.api_pool = GatedAttentionPooling(proj_dim, pool_hidden)
                self.exc_pool = GatedAttentionPooling(proj_dim, pool_hidden)
                self.api_pool_fusion = nn.Sequential(
                    nn.Linear(proj_dim * 2, proj_dim), nn.LayerNorm(proj_dim),
                    nn.GELU(), nn.Dropout(config.proj_dropout),
                )
                self.exc_pool_fusion = nn.Sequential(
                    nn.Linear(proj_dim * 2, proj_dim), nn.LayerNorm(proj_dim),
                    nn.GELU(), nn.Dropout(config.proj_dropout),
                )
        else:
            self.struct_dim = 0   # fixed-vector: no structural branch

    def _encode_struct(self, batch, device):
        """
        Returns h_api_struct [B,128] and h_exc_struct [B,128] for seq. encoders,
        or None, None for fixed-vector encoders (struct_dim == 0).
        Mirrors CompatibilityModel.forward logic for cross-attn + gated pooling / CLS.
        """
        if not self.use_cross_attn:
            return None, None

        api_smiles = batch["api_smiles"]
        exc_smiles = batch["exc_smiles"]
        exc_available = batch["exc_available"]

        api_tok, api_pool, api_mask = self.encoder.encode(api_smiles)
        exc_tok, exc_pool, exc_mask = self.encoder.encode(exc_smiles)

        api_tok  = api_tok.to(device);  api_pool  = api_pool.to(device);  api_mask  = api_mask.to(device)
        exc_tok  = exc_tok.to(device);  exc_pool  = exc_pool.to(device);  exc_mask  = exc_mask.to(device)

        B = api_tok.shape[0]

        api_tok_p  = self.api_proj(api_tok)
        api_pool_p = self.api_proj(api_pool)
        exc_tok_p  = self.exc_proj(exc_tok)
        exc_pool_p = self.exc_proj(exc_pool)

        api_cls = api_pool_p.unsqueeze(1)
        api_seq = torch.cat([api_cls, api_tok_p], dim=1)
        api_mask_ext = torch.cat([torch.zeros(B,1,dtype=torch.bool,device=device), api_mask], dim=1)

        exc_cls = exc_pool_p.unsqueeze(1)
        exc_seq = torch.cat([exc_cls, exc_tok_p], dim=1)
        exc_mask_ext = torch.cat([torch.zeros(B,1,dtype=torch.bool,device=device), exc_mask], dim=1)

        # Missing-excipient placeholder logic (same as CompatibilityModel)
        exc_avail_mask = exc_available.unsqueeze(1).unsqueeze(2)
        exc_seq = exc_seq * exc_avail_mask
        inv_mask = (1 - exc_available).unsqueeze(1).unsqueeze(2)
        exc_seq_cls  = exc_seq[:, :1, :] + inv_mask * self.exc_global_placeholder.expand(B,-1).unsqueeze(1)
        if exc_seq.shape[1] > 1:
            exc_seq_tok1 = exc_seq[:, 1:2, :] + inv_mask * self.exc_placeholder.expand(B,1,-1)
            exc_seq = torch.cat([exc_seq_cls, exc_seq_tok1, exc_seq[:, 2:, :]], dim=1)
        else:
            exc_seq = exc_seq_cls

        exc_mask_ext = exc_mask_ext & (exc_available.unsqueeze(1).bool())
        for i in range(B):
            if exc_available[i].item() == 0.0:
                exc_mask_ext[i, :] = True;  exc_mask_ext[i, 0] = False

        refined_exc, _ = self.cross_attn_exc(query=exc_seq, key=api_seq, value=api_seq,
                                              key_padding_mask=api_mask_ext)
        refined_exc = torch.nan_to_num(refined_exc)
        refined_exc = self.ln_exc(exc_seq + refined_exc)

        refined_api, _ = self.cross_attn_api(query=api_seq, key=exc_seq, value=exc_seq,
                                              key_padding_mask=exc_mask_ext)
        refined_api = torch.nan_to_num(refined_api)
        refined_api = self.ln_api(api_seq + refined_api)

        pooling = getattr(self, "pooling", "global_gated_attention")
        if pooling == "global_gated_attention":
            api_global  = refined_api[:, 0, :]
            exc_global  = refined_exc[:, 0, :]
            api_tokens  = refined_api[:, 1:, :]
            exc_tokens  = refined_exc[:, 1:, :]
            api_token_mask = api_mask_ext[:, 1:]
            exc_token_mask = exc_mask_ext[:, 1:]

            api_token_pool, _ = self.api_pool(api_tokens, api_token_mask)
            exc_token_pool, _ = self.exc_pool(exc_tokens, exc_token_mask)

            missing_exc = (exc_available == 0.0)
            exc_token_pool = torch.where(
                missing_exc.unsqueeze(1),
                self.exc_global_placeholder.expand(exc_tokens.size(0), -1),
                exc_token_pool,
            )
            api_struct = self.api_pool_fusion(torch.cat([api_global, api_token_pool], dim=-1))
            exc_struct = self.exc_pool_fusion(torch.cat([exc_global, exc_token_pool], dim=-1))
        else:
            # CLS pooling (ChemBERTa, D-MPNN)
            api_struct = refined_api[:, 0, :]
            exc_struct = refined_exc[:, 0, :]

        return api_struct, exc_struct

    def forward(self, batch: dict) -> torch.Tensor:
        device = batch["api_desc"].device

        # Structural vectors (from encoder + cross-attention)
        api_struct, exc_struct = self._encode_struct(batch, device)

        # On-the-fly feature tensors (pre-computed by PGBDataset collate)
        api_primary   = batch["api_primary"].to(device)    # [B, primary_dim]
        exc_primary   = batch["exc_primary"].to(device)
        api_secondary = batch.get("api_secondary")
        exc_secondary = batch.get("exc_secondary")
        if api_secondary is not None:
            api_secondary = api_secondary.to(device)
            exc_secondary = exc_secondary.to(device)
        api_pgb_desc  = batch["api_pgb_desc"].to(device)   # [B, 12]
        exc_pgb_desc  = batch["exc_pgb_desc"].to(device)
        api_flags     = batch["api_flags"].to(device)      # [B, 18]
        exc_flags     = batch["exc_flags"].to(device)
        prior_vec     = batch["prior_vec"].to(device)      # [B, 5]
        mech_flags    = batch["mech_flags"].to(device)     # [B, 5]

        # Normalize PGB descriptors (using batch-level stats stored in batch)
        api_pgb_desc_norm = batch["api_pgb_desc_norm"].to(device)
        exc_pgb_desc_norm = batch["exc_pgb_desc_norm"].to(device)

        h_api = self.api_tower(api_struct, api_primary, api_secondary, api_pgb_desc_norm, api_flags)
        h_exc = self.exc_tower(exc_struct, exc_primary, exc_secondary, exc_pgb_desc_norm, exc_flags)

        return self.head(h_api, h_exc, prior_vec, mech_flags, exc_flags)
```

#### 2.4.4 `pgb_model_from_config` — factory function

```python
def pgb_model_from_config(config, encoder) -> "PGBCompatibilityModel":
    """
    Build a PGBCompatibilityModel appropriate for `config.encoder`.

    Tower specs per family:
      maccs      : primary=167 MACCS, secondary=512 Morgan
      morgan     : primary=1024 Morgan, secondary=167 MACCS
      pubchemfp  : primary=881 PubChem, secondary=512 Morgan
      mol2vec    : primary=300 Mol2vec, secondary=512 Morgan (+ MACCS side, but 512 here)
      molformer  : struct=128, primary=167 MACCS, secondary=0
      chemberta  : struct=128, primary=512 Morgan, secondary=0
      pretrained_gin : struct=128, primary=167 MACCS, secondary=0
      pretrained_gat : struct=128, primary=167 MACCS, secondary=0
      dmpnn_chemprop : struct=128, primary=167 MACCS, secondary=0
    """
    enc = config.encoder
    fvs = getattr(config, "fixed_vector_source", "mol2vec")

    _tower_specs = {
        # (struct_dim, primary_dim, secondary_dim)
        "fixed_vector": {
            "maccs":     (0, 167, 512),
            "morgan":    (0, 1024, 167),
            "pubchemfp": (0, 881,  512),
            "mol2vec":   (0, 300,  512),
        },
        "seq": (128, 167, 0),
    }

    if enc == "fixed_vector":
        sd, pd_, sec = _tower_specs["fixed_vector"][fvs]
    else:
        sd, pd_, sec = _tower_specs["seq"]

    api_tower = MoleculeTower(sd, pd_, sec, config.pgb_tower_dim, config.proj_dropout)
    exc_tower = MoleculeTower(sd, pd_, sec, config.pgb_tower_dim, config.proj_dropout)

    model = PGBCompatibilityModel(config, encoder, api_tower, exc_tower)
    model.head.init_bias(
        global_rate=getattr(config, "positive_prior", 0.094),
        pgb_prior_offset=config.pgb_prior_offset,
    )
    return model
```

### 2.5 `src/dataset.py` — add `PGBDataset` and `pgb_collate_fn`

Add a new class and collate function at the bottom of the file. Do not modify `CompatibilityDataset` or `collate_fn`.

```python
# ── PGB dataset ────────────────────────────────────────────────────────────

from src.features import maccs_bits, morgan_bits, chemistry_flags, pgb_descriptors
from src.pgb_prior import ExcipientPriorTable
from src.lookup import get_mechanism_matches

# PGB descriptor normalization stats — computed once per fold from training rows
class PGBNormStats:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std  = std

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-8)

    @classmethod
    def from_smiles_list(cls, smiles_list: list[str]) -> "PGBNormStats":
        from src.features import pgb_descriptors
        vecs = [pgb_descriptors(s) for s in smiles_list]
        vecs = [v for v in vecs if not np.isnan(v).all()]
        mat  = np.stack(vecs)
        mean = np.nanmean(mat, axis=0)
        std  = np.nanstd(mat, axis=0)
        return cls(mean, std)


class PGBDataset(Dataset):
    """
    Dataset for PGBCompatibilityModel.
    Pre-computes all on-the-fly features in __getitem__:
        primary fingerprint (MACCS or Morgan etc.)
        secondary fingerprint (Morgan or MACCS)
        18 chemistry flags
        12 PGB descriptors (normalized)
        5-dim excipient prior vector
        5-dim mechanism flags
    """

    def __init__(
        self,
        csv_path: str,
        encoder_name: str,           # e.g. "fixed_vector", "molformer"
        fixed_vector_source: str,    # e.g. "maccs", "morgan", "pubchemfp", "mol2vec"
        prior_table: ExcipientPriorTable,
        pgb_norm_stats: PGBNormStats,
        fixed_vector_encoder=None,   # FixedVectorEncoder instance for lookup-based encoders
        sample_weight: float = 1.0,
    ):
        self.df = pd.read_csv(csv_path)
        self.enc_name = encoder_name
        self.fvs = fixed_vector_source
        self.prior_table = prior_table
        self.norm = pgb_norm_stats
        self.fv_enc = fixed_vector_encoder
        self.sample_weight = sample_weight

    def _primary(self, smiles: str) -> np.ndarray:
        """Primary fingerprint block for this family."""
        if self.enc_name == "fixed_vector":
            if self.fvs == "maccs":
                return maccs_bits(smiles)                # [167]
            elif self.fvs == "morgan":
                return morgan_bits(smiles, n_bits=1024)  # [1024]
            elif self.fvs == "pubchemfp":
                # Use the lookup-table vector; fall back to zero-vector as before
                if self.fv_enc is not None:
                    v = self.fv_enc.encode([smiles]).numpy()[0]
                    # If all-zero (not in table), return zeros — caller handles this
                    return v
                return np.zeros(881, dtype=np.float32)
            elif self.fvs == "mol2vec":
                if self.fv_enc is not None:
                    v = self.fv_enc.encode([smiles]).numpy()[0]
                    return v
                return np.zeros(300, dtype=np.float32)
        # Seq encoders: MACCS as primary side branch
        return maccs_bits(smiles)   # [167]

    def _secondary(self, smiles: str) -> np.ndarray | None:
        """Secondary fingerprint block (may be None)."""
        if self.enc_name == "fixed_vector":
            if self.fvs == "maccs":
                return morgan_bits(smiles, n_bits=512)  # [512]
            elif self.fvs == "morgan":
                return maccs_bits(smiles)               # [167]
            elif self.fvs == "pubchemfp":
                return morgan_bits(smiles, n_bits=512)  # [512]
            elif self.fvs == "mol2vec":
                return morgan_bits(smiles, n_bits=512)  # [512]
        return None  # seq encoders have no secondary

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        api_smi = str(row["API_Smiles"])
        exc_smi = str(row["Excipient_Smiles"]) if pd.notna(row.get("Excipient_Smiles")) else ""
        exc_available = 1.0 if exc_smi else 0.0
        label = float(row["Outcome1"])

        api_prim = self._primary(api_smi)
        exc_prim = self._primary(exc_smi) if exc_smi else np.zeros_like(api_prim)

        api_sec  = self._secondary(api_smi)
        exc_sec  = self._secondary(exc_smi) if exc_smi else (
            np.zeros_like(api_sec) if api_sec is not None else None
        )

        api_flags = chemistry_flags(api_smi)
        exc_flags = chemistry_flags(exc_smi) if exc_smi else np.zeros(18, dtype=np.float32)

        api_desc_raw = pgb_descriptors(api_smi)
        exc_desc_raw = pgb_descriptors(exc_smi) if exc_smi else np.full(12, 0.0, dtype=np.float32)
        api_desc_raw = np.nan_to_num(api_desc_raw, nan=0.0)
        exc_desc_raw = np.nan_to_num(exc_desc_raw, nan=0.0)
        api_desc_norm = np.nan_to_num(self.norm.normalize(api_desc_raw), nan=0.0)
        exc_desc_norm = np.nan_to_num(self.norm.normalize(exc_desc_raw), nan=0.0)

        prior_info = self.prior_table.lookup(exc_smi) if exc_smi else {
            "p_exact": 0.094, "p_delta": 0.0, "p_family": 0.094, "log_n": 0.0, "unseen": 1.0,
        }
        prior_vec = np.array([
            prior_info["p_exact"],
            prior_info["p_delta"],
            prior_info["p_family"],
            prior_info["log_n"],
            prior_info["unseen"],
        ], dtype=np.float32)

        mech = get_mechanism_matches(api_smi, exc_smi) if exc_smi else [0] * 5
        mech_flags = np.array(mech, dtype=np.float32)

        item = {
            "api_smiles":       api_smi,
            "exc_smiles":       exc_smi,
            "api_primary":      torch.tensor(api_prim,      dtype=torch.float32),
            "exc_primary":      torch.tensor(exc_prim,      dtype=torch.float32),
            "api_pgb_desc":     torch.tensor(api_desc_raw,  dtype=torch.float32),
            "exc_pgb_desc":     torch.tensor(exc_desc_raw,  dtype=torch.float32),
            "api_pgb_desc_norm":torch.tensor(api_desc_norm, dtype=torch.float32),
            "exc_pgb_desc_norm":torch.tensor(exc_desc_norm, dtype=torch.float32),
            "api_flags":        torch.tensor(api_flags,     dtype=torch.float32),
            "exc_flags":        torch.tensor(exc_flags,     dtype=torch.float32),
            "prior_vec":        torch.tensor(prior_vec,     dtype=torch.float32),
            "mech_flags":       torch.tensor(mech_flags,    dtype=torch.float32),
            "exc_available":    torch.tensor(exc_available, dtype=torch.float32),
            "label":            torch.tensor(label,         dtype=torch.float32),
            "sample_weight":    torch.tensor(self.sample_weight, dtype=torch.float32),
        }
        if api_sec is not None:
            item["api_secondary"] = torch.tensor(api_sec, dtype=torch.float32)
            item["exc_secondary"] = torch.tensor(exc_sec, dtype=torch.float32)

        return item


def pgb_collate_fn(batch: list[dict]) -> dict:
    """Collate for PGBDataset — handles optional secondary fingerprint."""
    keys = list(batch[0].keys())
    out = {}
    for k in keys:
        if k in ("api_smiles", "exc_smiles"):
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out
```

### 2.6 `src/train.py` — add `pgb_train` function

Do not modify `train`. Add a new function below it:

```python
def pgb_train(model, train_loader, val_loader, config, device):
    """
    Training loop for PGBCompatibilityModel.
    Uses two LR groups: new PGB layers at pgb_lr, kept encoder layers at pgb_encoder_lr.
    Loss: asym_focal with single pos_weight. No balanced sampler.
    Returns: model (best ckpt restored), history dict.
    """
    import math
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Identify parameter groups by name
    pgb_params   = []
    enc_params   = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Kept layers: cross-attention, projection, pooling (carried from CompatibilityModel)
        if any(k in name for k in [
            "api_proj", "exc_proj", "cross_attn", "ln_exc", "ln_api",
            "api_pool", "exc_pool", "api_pool_fusion", "exc_pool_fusion",
            "exc_placeholder", "exc_global_placeholder",
        ]):
            enc_params.append(param)
        else:
            pgb_params.append(param)

    optimizer = AdamW([
        {"params": pgb_params, "lr": config.pgb_lr},
        {"params": enc_params, "lr": config.pgb_encoder_lr if enc_params else config.pgb_lr},
    ], weight_decay=config.pgb_weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=config.pgb_lr_patience)

    # Asymmetric focal loss with single pos_weight
    from src.loss import get_loss_fn
    loss_fn = get_loss_fn(config)
    pos_weight = compute_pos_weight(train_loader)

    best_pr_auc = -1.0
    best_epoch  = 0
    patience_counter = 0
    best_ckpt_path = os.path.join(config.checkpoint_dir, "best_model.pt")
    history = {"train_loss": [], "val_pr_auc": [], "lr": []}

    for epoch in range(1, config.pgb_max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            batch_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                            for k, v in batch.items()}
            labels = batch_device.pop("label")
            sw     = batch_device.pop("sample_weight")

            optimizer.zero_grad()
            logits = model(batch_device)
            loss   = loss_fn(logits, labels, sw, pw=pos_weight.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        avg_loss   = total_loss / max(n_batches, 1)
        val_pr_auc = get_val_pr_auc(model, val_loader, device)
        scheduler.step(val_pr_auc)
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(avg_loss)
        history["val_pr_auc"].append(val_pr_auc)
        history["lr"].append(current_lr)

        print(f"PGB Epoch {epoch:3d}/{config.pgb_max_epochs} | "
              f"Loss: {avg_loss:.4f} | Val PR-AUC: {val_pr_auc:.4f} | LR: {current_lr:.2e}")

        if val_pr_auc > best_pr_auc + config.early_stop_min_delta:
            best_pr_auc = val_pr_auc
            best_epoch  = epoch
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_pr_auc": val_pr_auc, "config": config}, best_ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= config.pgb_early_stop_patience:
                print(f"\nPGB early stopping at epoch {epoch}")
                break

    print(f"\nPGB restoring best model from epoch {best_epoch} (PR-AUC={best_pr_auc:.4f})")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, history
```

### 2.7 `src/cross_validate.py` — add `pgb_cross_validate` function

Do not modify `cross_validate`. Add a new function at the end of the file:

```python
def pgb_cross_validate(config: Config, n_folds: int = 5):
    """
    5-fold CV for PGBCompatibilityModel.
    Mirrors cross_validate() logic but:
    - Uses PGBDataset + pgb_collate_fn instead of CompatibilityDataset
    - Builds PGBCompatibilityModel via pgb_model_from_config()
    - Computes ExcipientPriorTable and PGBNormStats per fold (train-only)
    - Uses pgb_train() instead of train()
    - Calls tune_threshold with tnr_floor=config.pgb_tnr_floor
    """
    from src.dataset import PGBDataset, pgb_collate_fn, PGBNormStats
    from src.pgb_prior import ExcipientPriorTable
    from src.model import pgb_model_from_config
    from src.train import pgb_train

    seed_everything(config.seed)
    device = get_device(config.device)

    raw = pd.read_csv(os.path.join(config.data_dir, "start_dataset.csv"))

    if getattr(config, "split_type", "cluster") == "random":
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.seed)
        splits = kf.split(raw, y=raw["Outcome1"])
    else:
        unique_apis = raw["API_Smiles"].unique().tolist()
        smiles_to_cluster = butina_cluster(unique_apis)
        raw["cluster_id"] = raw["API_Smiles"].map(smiles_to_cluster)
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=config.seed)
        splits = sgkf.split(raw, y=raw["Outcome1"], groups=raw["cluster_id"])

    all_val_metrics  = []
    all_test_metrics = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        print(f"\n{'='*60}\nPGB FOLD {fold_i+1}/{n_folds}\n{'='*60}")

        train_fold = raw.iloc[train_idx].reset_index(drop=True)
        test_fold  = raw.iloc[test_idx].reset_index(drop=True)

        if getattr(config, "split_type", "cluster") == "random":
            inner_kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=config.seed+fold_i)
            inner_train_idx, inner_val_idx = next(inner_kf.split(train_fold, y=train_fold["Outcome1"]))
        else:
            inner_sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=config.seed+fold_i)
            inner_train_idx, inner_val_idx = next(inner_sgkf.split(
                train_fold, y=train_fold["Outcome1"], groups=train_fold["cluster_id"]))

        actual_train = train_fold.iloc[inner_train_idx].drop(columns=["cluster_id"], errors="ignore").reset_index(drop=True)
        actual_val   = train_fold.iloc[inner_val_idx].drop(columns=["cluster_id"],   errors="ignore").reset_index(drop=True)
        actual_test  = test_fold.drop(columns=["cluster_id"], errors="ignore").reset_index(drop=True)

        fold_dir = os.path.join(config.metrics_dir, f"fold_{fold_i}")
        os.makedirs(fold_dir, exist_ok=True)
        fold_train_csv = os.path.join(fold_dir, "train.csv")
        fold_val_csv   = os.path.join(fold_dir, "val.csv")
        fold_test_csv  = os.path.join(fold_dir, "test.csv")
        actual_train.to_csv(fold_train_csv, index=False)
        actual_val.to_csv(fold_val_csv,   index=False)
        actual_test.to_csv(fold_test_csv,  index=False)

        fold_config = copy.deepcopy(config)
        fold_config.train_csv = fold_train_csv
        fold_config.val_csv   = fold_val_csv
        fold_config.test_csv  = fold_test_csv
        fold_config.checkpoint_dir = os.path.join(config.checkpoint_dir, f"fold_{fold_i}")
        fold_config.positive_prior = float(actual_train["Outcome1"].mean())
        fold_config.use_balanced_sampler = False  # always off for PGB
        fold_config.loss = "asym_focal"

        # Build encoder
        encoder = build_encoder(fold_config, device)
        if not encoder.is_sequence_capable:
            fold_config.fusion = "concat"

        # Build PGB norm stats from training SMILES only
        all_train_smiles = (
            actual_train["API_Smiles"].tolist() +
            actual_train["Excipient_Smiles"].dropna().tolist()
        )
        pgb_norm = PGBNormStats.from_smiles_list(all_train_smiles)

        # Build excipient prior from training rows only
        prior_table = ExcipientPriorTable()
        prior_table.fit(
            actual_train,
            leave_out_cluster_id=None,   # for training rows: leave-cluster-out done implicitly by the fold
        )

        # Get fixed-vector encoder if applicable
        fv_enc = encoder if fold_config.encoder == "fixed_vector" else None

        train_ds = PGBDataset(fold_train_csv, fold_config.encoder,
                              getattr(fold_config, "fixed_vector_source", "maccs"),
                              prior_table, pgb_norm, fixed_vector_encoder=fv_enc)
        val_ds   = PGBDataset(fold_val_csv,   fold_config.encoder,
                              getattr(fold_config, "fixed_vector_source", "maccs"),
                              prior_table, pgb_norm, fixed_vector_encoder=fv_enc)
        test_ds  = PGBDataset(fold_test_csv,  fold_config.encoder,
                              getattr(fold_config, "fixed_vector_source", "maccs"),
                              prior_table, pgb_norm, fixed_vector_encoder=fv_enc)

        train_loader = DataLoader(train_ds, batch_size=fold_config.pgb_batch_size,
                                  shuffle=True, collate_fn=pgb_collate_fn, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=fold_config.pgb_batch_size,
                                  shuffle=False, collate_fn=pgb_collate_fn, num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=fold_config.pgb_batch_size,
                                  shuffle=False, collate_fn=pgb_collate_fn, num_workers=0)

        # Build model
        model = pgb_model_from_config(fold_config, encoder)
        model.to(device)

        # Train
        model, history = pgb_train(model, train_loader, val_loader, fold_config, device)

        # Evaluate
        val_metrics, test_metrics, threshold = evaluate_model(
            model, val_loader, test_loader, device, fold_config
        )

        print(f"\nPGB Fold {fold_i+1} Val:  PR-AUC={val_metrics['pr_auc']:.4f}, F1={val_metrics['f1']:.4f}, MCC={val_metrics['mcc']:.4f}")
        print(f"PGB Fold {fold_i+1} Test: PR-AUC={test_metrics['pr_auc']:.4f}, F1={test_metrics['f1']:.4f}, MCC={test_metrics['mcc']:.4f}")
        print(f"  threshold={threshold:.3f}")

        all_val_metrics.append(val_metrics)
        all_test_metrics.append(test_metrics)

        save_metrics(val_metrics,  os.path.join(fold_dir, "val_metrics.json"))
        save_metrics(test_metrics, os.path.join(fold_dir, "test_metrics.json"))
        json.dump(history, open(os.path.join(fold_dir, "training_history.json"), "w"), indent=2)

        del model, encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate
    cv_summary = {}
    for key in ["pr_auc", "f1", "mcc", "precision", "recall", "accuracy"]:
        vals      = [m[key] for m in all_val_metrics]
        test_vals = [m[key] for m in all_test_metrics]
        cv_summary[f"val_{key}_mean"]  = float(np.mean(vals))
        cv_summary[f"val_{key}_std"]   = float(np.std(vals))
        cv_summary[f"test_{key}_mean"] = float(np.mean(test_vals))
        cv_summary[f"test_{key}_std"]  = float(np.std(test_vals))

    save_metrics(cv_summary, os.path.join(config.metrics_dir, "cv_metrics.json"))
    return cv_summary
```

---

## 3. Entry-point script to create

Create a new file `run_pgb.py` in the project root:

```python
"""
run_pgb.py
Entry point for training the 9 Prior-Gated Bilinear (PGB) models.
Run one family at a time:
    python run_pgb.py --family maccs
    python run_pgb.py --family molformer
    python run_pgb.py --family all
"""

import argparse
from src.config import Config
from src.cross_validate import pgb_cross_validate

FAMILIES = {
    "maccs":      {"encoder": "fixed_vector", "fixed_vector_source": "maccs",     "fusion": "concat", "pgb_max_epochs": 40},
    "morgan":     {"encoder": "fixed_vector", "fixed_vector_source": "morgan",    "fusion": "concat", "pgb_max_epochs": 40},
    "pubchemfp":  {"encoder": "fixed_vector", "fixed_vector_source": "pubchemfp", "fusion": "concat", "pgb_max_epochs": 40},
    "mol2vec":    {"encoder": "fixed_vector", "fixed_vector_source": "mol2vec",   "fusion": "concat", "pgb_max_epochs": 40},
    "molformer":  {"encoder": "molformer",  "fusion": "cross_attn", "pooling": "global_gated_attention", "pgb_max_epochs": 60},
    "chemberta":  {"encoder": "chemberta",  "fusion": "cross_attn", "pooling": "cls",                    "pgb_max_epochs": 60},
    "gin":        {"encoder": "pretrained_gin", "fusion": "cross_attn", "pooling": "global_gated_attention", "pgb_max_epochs": 60},
    "gat":        {"encoder": "pretrained_gat", "fusion": "cross_attn", "pooling": "global_gated_attention", "pgb_max_epochs": 60},
    "dmpnn":      {"encoder": "dmpnn_chemprop", "fusion": "cross_attn", "pooling": "cls",                "pgb_max_epochs": 60},
}

def run_family(name: str, args):
    spec = FAMILIES[name]
    config = Config()
    config.use_pgb_head = True
    config.split_type   = "cluster"
    config.seed         = args.seed

    for k, v in spec.items():
        setattr(config, k, v)

    if args.epochs:
        config.pgb_max_epochs = args.epochs
    if args.device:
        config.device = args.device

    config.resolve_paths()

    print(f"\n{'='*60}")
    print(f"Training PGB model: {name}")
    print(f"{'='*60}")
    summary = pgb_cross_validate(config)

    print(f"\nPGB {name} CV summary:")
    for k in ["pr_auc", "f1", "mcc"]:
        print(f"  val  {k}: {summary[f'val_{k}_mean']:.4f} ± {summary[f'val_{k}_std']:.4f}")
        print(f"  test {k}: {summary[f'test_{k}_mean']:.4f} ± {summary[f'test_{k}_std']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="maccs",
                        choices=list(FAMILIES.keys()) + ["all"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    if args.family == "all":
        for name in FAMILIES:
            run_family(name, args)
    else:
        run_family(args.family, args)
```

---

## 4. Inference script update

Update `Inference/run_inference.py` to support PGB models. At the top of the file, add:

```python
PGB_MODELS = [
    "pgb_maccs",
    "pgb_morgan",
    "pgb_pubchemfp",
    "pgb_mol2vec",
    "pgb_molformer",
    "pgb_chemberta",
    "pgb_gin",
    "pgb_gat",
    "pgb_dmpnn",
]
```

When running inference for PGB models, use `PGBDataset` and `pgb_collate_fn` instead of `CompatibilityDataset` and `collate_fn`. The prior table must be loaded from the saved checkpoint's associated `prior_table.json` (save it alongside `best_model.pt` in `pgb_train`). Norm stats must also be saved and reloaded.

---

## 5. Validation protocol before training each family

Run this quick smoke test after writing all files, before spending GPU time:

```bash
python3 -c "
import torch
import numpy as np
from src.config import Config
from src.features import maccs_bits, morgan_bits, chemistry_flags, pgb_descriptors
from src.pgb_prior import ExcipientPriorTable
from src.dataset import PGBDataset, pgb_collate_fn, PGBNormStats
from src.model import pgb_model_from_config
from src.encoders.fixed_vector_encoder import FixedVectorEncoder

# 1. Feature shapes
assert maccs_bits('CC(=O)O').shape == (167,), 'MACCS shape wrong'
assert morgan_bits('CC(=O)O').shape == (512,), 'Morgan shape wrong'
assert chemistry_flags('CC(=O)O').shape == (18,), 'Flags shape wrong'
assert pgb_descriptors('CC(=O)O').shape == (12,), 'PGB desc shape wrong'

# 2. Prior table
import pandas as pd
mock_train = pd.DataFrame({
    'Excipient_Smiles': ['OC[C@H](O)C(=O)[C@@H](O)[C@H](O)CO'] * 5,  # mannitol
    'Outcome1': [0, 0, 1, 0, 0],
})
pt = ExcipientPriorTable()
pt.fit(mock_train)
r = pt.lookup('OC[C@H](O)C(=O)[C@@H](O)[C@H](O)CO')
assert 0 < r['p_exact'] < 1, 'Prior out of range'
assert r['unseen'] == 0.0, 'Should be seen'

# 3. Model forward pass (MACCS family, CPU)
config = Config()
config.use_pgb_head = True
config.encoder = 'fixed_vector'
config.fixed_vector_source = 'maccs'
config.fusion = 'concat'
config.positive_prior = 0.094
config.resolve_paths()

enc = FixedVectorEncoder(source='maccs', vector_path=None, device='cpu')
model = pgb_model_from_config(config, enc)
model.eval()

B = 4
mock_batch = {
    'api_smiles': ['CC(=O)Nc1ccc(O)cc1'] * B,
    'exc_smiles': ['OC[C@H](O)C(=O)[C@@H](O)[C@H](O)CO'] * B,
    'api_primary':       torch.zeros(B, 167),
    'exc_primary':       torch.zeros(B, 167),
    'api_secondary':     torch.zeros(B, 512),
    'exc_secondary':     torch.zeros(B, 512),
    'api_pgb_desc':      torch.zeros(B, 12),
    'exc_pgb_desc':      torch.zeros(B, 12),
    'api_pgb_desc_norm': torch.zeros(B, 12),
    'exc_pgb_desc_norm': torch.zeros(B, 12),
    'api_flags':         torch.zeros(B, 18),
    'exc_flags':         torch.zeros(B, 18),
    'prior_vec':         torch.tensor([[0.094, 0.0, 0.094, 0.0, 1.0]] * B),
    'mech_flags':        torch.zeros(B, 5),
    'exc_available':     torch.ones(B),
    'label':             torch.zeros(B),
    'sample_weight':     torch.ones(B),
}
with torch.no_grad():
    logits = model(mock_batch)
assert logits.shape == (B,), f'Logit shape wrong: {logits.shape}'
assert not torch.isnan(logits).any(), 'NaN in logits'
print('All smoke tests passed.')
"
```

All assertions must pass before training.

---

## 6. Build order and acceptance criteria

Train in this order. Stop at the first family that fails the gate criteria and debug before proceeding.

| Order | Family | Expected val PR-AUC | Expected test false alarms | 24-pair recall @ 0.60 |
|---|---|---|---|---|
| 1 | **MACCS** | ≥ 0.62 | ≤ 30 | ≥ 7/12 |
| 2 | Morgan | ≥ 0.57 | ≤ 25 | ≥ 7/12 |
| 3 | PubChem | ≥ 0.62 | ≤ 25 | ≥ 8/12 |
| 4 | MoLFormer | ≥ 0.70 | ≤ 25 | ≥ 8/12 |
| 5 | D-MPNN | ≥ 0.67 | ≤ 25 | ≥ 8/12 |
| 6 | GAT | ≥ 0.58 | ≤ 40 | ≥ 9/12 |
| 7 | GIN | ≥ 0.57 | ≤ 35 | ≥ 7/12 |
| 8 | ChemBERTa | ≥ 0.62 | ≤ 30 | ≥ 7/12 |
| 9 | Mol2vec | ≥ 0.55 | ≤ 40 | ≥ 6/12 |

**Acceptance rule for a single family run:**
- Val PR-AUC is not worse than the corresponding baseline in `metrics/` (±0.02 tolerance)
- Test false alarms (FP on compatible pairs) are lower than baseline
- 24-pair PR-AUC ≥ 0.78 and ROC-AUC ≥ 0.82

**Hardest pair — expect a miss at any threshold:**
`Acetazolamide × Mannitol` — Mannitol's prior is ~0.04, so the model will score this pair below the screen cut. This is a known limitation of any prior-based design when the excipient has a low base rate. Do not retune the threshold to catch it.

---

## 7. What NOT to do

- Do not modify `CompatibilityModel`, `collate_fn`, `CompatibilityDataset`, or `cross_validate`. The existing models must continue to work.
- Do not enable `random_split` runs. All PGB training is group-split (API-cluster) only.
- Do not tune the threshold on the 24-pair held-out set. Tune only on the validation fold, using `pgb_tnr_floor=0.97`.
- Do not set `use_balanced_sampler=True` in any PGB config. The asymmetric focal loss handles class imbalance.
- Do not lower `pgb_tnr_floor` below 0.90 to chase recall on the 24-pair set. The floor exists to protect compatible pairs (the false alarm problem).
- Do not add cross-attention to the GIN family. Your own ablation showed concat and cross-attn score the same for GIN; the bilinear head now does the interaction.
- Do not use the explicit pairwise pooling mode (`pooling="explicit_pairwise"`) in any PGB run — it scored lower than gated pooling in your baselines.
