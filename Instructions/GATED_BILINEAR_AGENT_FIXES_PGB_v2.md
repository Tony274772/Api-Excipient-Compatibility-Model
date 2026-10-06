# AI Agent Fix Instructions — PGB / Gated Bilinear v2

## Purpose

You have **already implemented** the previous `GATED_BILINEAR_AGENT_INSTRUCTIONS_exp-2.md` changes in the project.

Do **not** start the PGB implementation again from scratch.
Do **not** discard the current working implementation.

This document is a **correction pass** over the implementation that now exists.

The goal is to fix the architectural and data-pipeline issues that can prevent the current 9 PGB/gated-bilinear models from reproducing the intended Prior-Gated Bilinear (PGB) behavior.

The reference design is based on:
- `Gated_Bilinear_PairNet_Architecture_and_Results.docx`
- `Instructions/Gated_Bilinear_Architectures_All_Group_Split_Models.md`
- the already-implemented PGB code from the previous agent instructions.

The most important principle is:

> **Do not interpret poor results from the current 9 models as evidence that gated bilinear is ineffective until the fixes below are implemented and verified.**

---

# 1. CRITICAL FIX #1 — TRUE LEAVE-API-CLUSTER-OUT EXCIPIENT PRIOR

## Problem

The current implementation is intended to use a leave-own-API-cluster-out prior during training, but the current wiring can call:

```python
prior_table.fit(
    actual_train,
    leave_out_cluster_id=None,
)
```

with the comment that cluster exclusion is "implicit".

That is not sufficient.

If the prior table is fitted once over all training rows and then reused by every training example, a training row can see its own label through the excipient prior.

That creates a target-dependent prior signal during training.

## Required behavior

### Training rows

For every training row `i`:

```text
prior(i) = prior calculated from training rows
           excluding ALL rows belonging to the same API cluster as row i
```

This is **leave-own-API-cluster-out**, not merely leave-one-row-out.

If the current row belongs to cluster `C`, exclude every training row with:

```python
cluster_id == C
```

from the prior calculation.

### Validation / official test / held-out inference

Use the complete available training set to build the prior:

```text
train rows only
    ↓
full prior table
    ↓
validation/test/held-out lookup
```

Do NOT use validation/test/held-out labels in any prior.

## Required implementation

### 1. Preserve cluster IDs

When creating the fold dataframes, do not permanently remove `cluster_id` until after the per-row training priors have been computed.

If the current group-split code already has the API cluster assignment, reuse that exact assignment.

Do not independently re-cluster the training data for this purpose unless absolutely necessary.

### 2. Add a function similar to

```python
def build_leave_cluster_out_prior_vectors(train_df, cluster_col="cluster_id"):
    """
    For every training row, compute its excipient prior using
    all training rows except rows belonging to the same API cluster.
    Returns one 5-d prior vector per training row.
    """
```

The exact function name is flexible, but behavior is mandatory.

### 3. PGBDataset must support per-row priors

The training dataset must NOT perform:

```python
self.prior_table.lookup(exc_smi)
```

using one universal prior table for every training row.

Instead, training examples must receive their already-computed row-specific prior vector.

For validation/test datasets, normal full-training prior lookup is correct.

## Verification

Add a test that proves no training row sees its own API cluster in its prior source.

Also print or assert a few examples:

```text
row index
cluster id
excipient key
prior with cluster excluded
prior with full train
```

At least one example must show that the two values differ when that cluster contributes evidence.

---

# 2. CRITICAL FIX #2 — DO NOT EXPOSE THE UNGATED BILINEAR TO THE FINAL MLP

## Problem

The current PGB head may contain:

```python
concat = torch.cat(
    [h_api, h_exc, product, diff, bilinear, gated, prior_vec, mech_flags],
    dim=-1,
)
```

This makes the gate bypassable.

The MLP can simply learn to use `bilinear` and ignore `gated`.

Then the model is not truly using the gate as an interaction controller.

## Required change

Remove the raw ungated `bilinear` from the final MLP input.

Use:

```python
bilinear = self.U(h_api) * self.V(h_exc)
gate = self.gate(gate_input)
gated = bilinear * gate
```

Then only expose `gated` to the final MLP.

For example:

```python
concat = torch.cat(
    [
        h_api,
        h_exc,
        product,
        diff,
        gated,
        prior_vec,
        mech_flags,
    ],
    dim=-1,
)
```

The exact inclusion of other contextual features is allowed, but **raw `bilinear` must not be another bypass path**.

## Important

Keep the actual bilinear operation:

```python
bilinear = self.U(h_api) * self.V(h_exc)
```

The change is only that the final decision path receives the gated version, not both versions.

## Required shape assertion

For rank `r=16`:

```text
h_api        [B,96]
h_exc        [B,96]
bilinear     [B,16]
gate         [B,16]
gated        [B,16]
```

---

# 3. CRITICAL FIX #3 — THE GATE MUST SEE BOTH API AND EXCIPIENT CHEMISTRY

## Problem

The current gate is approximately:

```python
gate_input = torch.cat([
    prior_vec,
    exc_flags,
    mech_flags,
], dim=-1)
```

So the gate sees excipient chemistry but not the API chemistry flags.

That is too asymmetric for a pairwise compatibility problem.

## Required change

Pass API chemistry flags into `PGBHead.forward()`.

Change the interface from something like:

```python
forward(h_api, h_exc, prior_vec, mech_flags, exc_flags)
```

to:

```python
forward(h_api, h_exc, prior_vec, api_flags, exc_flags, mech_flags)
```

Then construct:

```python
gate_input = torch.cat([
    prior_vec,      # 5
    api_flags,      # 18
    exc_flags,      # 18
    mech_flags,     # 5
], dim=-1)
```

Total:

```text
46 input features
```

Recommended gate:

```text
46 → 32 → 16 → 16 sigmoid
```

or another small MLP of comparable size.

Do not make the gate unnecessarily large.

## Why

The gate should be able to learn patterns such as:

```text
API chemistry
      +
excipient chemistry
      +
prior/reliability context
      ↓
interaction trust
```

The five hand-written mechanism flags may remain as auxiliary hints, but they must not be the only way API chemistry reaches the gate.

---

# 4. FIX #4 — CANONICAL EXCIPIENT PRIOR MUST AGGREGATE, NOT OVERWRITE

## Problem

The current prior-building code can do:

```python
for smi, group in df.groupby("Excipient_Smiles"):
    key = _canonical_stereo_blind(smi)
    self._exact[key] = (p, n)
```

If multiple raw SMILES canonicalize to the same stereo-blind key, later entries overwrite earlier entries.

That means the final prior may depend on row ordering.

## Required change

First canonicalize every row:

```python
df["exc_key"] = df["Excipient_Smiles"].map(_canonical_stereo_blind)
```

Then aggregate by canonical key:

```python
grouped = (
    df.groupby("exc_key")
      .agg(
          n=("Outcome1", "size"),
          pos=("Outcome1", "sum"),
      )
)
```

Then calculate:

```python
p = (pos + alpha * GLOBAL_RATE) / (n + alpha)
```

Store:

```text
key → (p, n)
```

Unknown/unparseable rows can fall back safely, but valid canonical duplicates must aggregate.

---

# 5. FIX #5 — FAMILY PRIOR MUST USE COUNTS, NOT RUNNING MEAN OF PRIORS

## Problem

The current implementation can contain:

```python
self._family[fam_key] = (self._family[fam_key] + p) / 2.0
```

This incorrectly gives equal influence to groups with very different sample counts.

For example:

```text
family A: 1 observation
family B: 50 observations
```

must not have equal weight simply because their smoothed priors are updated sequentially.

## Required change

Aggregate family-level counts first:

```text
family positive count
family total count
       ↓
Laplace-smoothed family prior
```

For example:

```python
family_p = (
    family_pos + alpha * GLOBAL_RATE
) / (
    family_n + alpha
)
```

Do not use an iterative arithmetic mean of individual priors.

---

# 6. FIX #6 — SAVE/LOAD MUST PRESERVE NEAREST-NEIGHBOUR PRIOR DATA

## Problem

The prior table contains:

```python
self._fps
```

for Morgan nearest-neighbour lookup.

But the current save format only stores exact/family prior dictionaries.

After loading:

```text
_exact  → present
_family → present
_fps    → empty
```

which silently disables the nearest-neighbour fallback.

## Required change

Choose one of these implementations:

### Preferred

Save the data needed to rebuild `_fps`, then reconstruct `_fps` during `load()`.

For example, save the exact canonical keys and recreate their Morgan fingerprints.

### Alternative

Serialize the fingerprint representation directly.

Either is acceptable.

## Required verification

Before save:

```text
NN fallback available = True
```

After load:

```text
NN fallback available = True
```

Run an inference lookup for an unseen-but-similar excipient and verify that the nearest-neighbour path is actually exercised when Tanimoto ≥ threshold.

---

# 7. CRITICAL FIX #7 — REPLACE ZERO-VECTOR UNKNOWN FIXED-VECTOR REPRESENTATIONS

## Problem

The current PGB dataset may still do:

```python
return np.zeros(881)
```

for missing PubChem vectors or:

```python
return np.zeros(300)
```

for missing Mol2vec vectors.

That treats:

```text
unknown molecule
```

as:

```text
real molecule whose embedding happens to be all zeros
```

This is incorrect and loses information about availability.

## Required behavior

For lookup-based primary features:

```text
known vector
    ↓
normal projection

unknown vector
    ↓
learned missing embedding
    +
availability flag
```

## Implementation

Add learned parameters in `MoleculeTower` or the appropriate feature branch:

```python
self.missing_primary_embedding = nn.Parameter(...)
```

The learned missing vector should be in the projected branch dimension, not necessarily the original raw feature dimension.

Add:

```text
primary_available ∈ {0,1}
```

to the tower input.

Recommended example:

```text
primary projected feature [64]
primary availability      [1]
                         ↓
                  rest of tower
```

Do this for:

- PubChem fixed-vector family
- Mol2vec fixed-vector family

## Also preserve on-the-fly fingerprints

For MACCS and Morgan, continue computing them from SMILES directly.

Do not replace those with lookup-zero behavior.

---

# 8. FIX #8 — MAKE SALT/IONIC DESCRIPTORS CONSISTENT WITH SALT-AWARE FINGERPRINTS

## Problem

MACCS/Morgan/flags may process all fragments of a salt, while the descriptor branch can be based on only one parsed molecule or otherwise treat the salt inconsistently.

For this task, inorganic and salt components matter.

## Required change

Keep the existing 12 main physicochemical descriptors, but add a small explicit salt/inorganic context branch.

Recommended additional features:

```text
num_fragments
num_metal_atoms
num_inorganic_fragments
formal_charge_total
max_abs_formal_charge
organic_heavy_atoms
inorganic_heavy_atoms
has_counterion
```

Do not force these into the original 12-descriptor definition if that would break the documented 12-vector interface.

Instead:

```text
12 normalized PGB descriptors
+
small salt-context branch
```

The salt context should be computed from all valid fragments.

## Important

Do not silently use only the largest fragment for every piece of information.

---

# 9. RECOMMENDED ARCHITECTURE CHANGE FOR SEQUENCE / GRAPH MODELS — FIRST PGB PASS SHOULD USE LATE FUSION

## Current concern

The current PGB wrapper can preserve the previous cross-attention block for sequence-capable models.

That means the model can already perform pair interaction through cross-attention and then perform another explicit bilinear interaction.

With only about 2k training pairs, this may make optimization harder to interpret.

## Required first-pass architecture

For the first corrected PGB experiment, use:

```text
API encoder
    ↓
API projection/tower
    ↓
h_api [96]

EXC encoder
    ↓
EXC projection/tower
    ↓
h_exc [96]

h_api + h_exc
    ↓
PGB head
```

Do not use cross-attention as the main pair representation in this first corrected architecture.

This applies initially to:

- MoLFormer
- ChemBERTa
- Pretrained GAT
- D-MPNN / CheMeleon

GIN should remain late-fusion as well.

## Do not delete old baseline code

This is a PGB experimental path.

Keep the original baseline `CompatibilityModel` untouched.

Optionally keep a config switch such as:

```python
pgb_use_cross_attn = False
```

for the corrected PGB implementation.

Later, cross-attention can be tested as an explicit ablation.

---

# 10. FIX #9 — PRIOR TRUST SHOULD DEPEND ON PRIOR RELIABILITY

## Problem

The current final logit is approximately:

```python
logit = residual + prior_scale * logit_prior + prior_bias
```

with a single unconstrained learned scalar `prior_scale`.

This means the same prior trust is applied to:

```text
many observations
```

and

```text
one observation / unseen
```

which is not ideal.

## Recommended change

Replace the unrestricted global prior scale with a bounded, evidence-conditioned trust value.

For example:

```text
prior_vec
    ↓
small prior-trust MLP
    ↓
sigmoid
    ↓
prior_weight ∈ (0,1)
```

Then:

```python
logit = (
    residual
    + prior_weight * logit_prior
    + prior_bias
)
```

The prior-trust network should use information such as:

```text
p_exact
p_delta
p_family
log_n
unseen
```

Optionally include nearest-neighbour confidence if available.

## Desired behavior

```text
strong historical evidence
    → prior can be trusted more

weak evidence / unseen excipient
    → prior influence reduced

strong pair evidence
    → residual can override prior
```

This is especially important for rare positives involving historically "safe" excipients.

---

# 11. DO NOT TREAT HARD-CODED MECHANISM FLAGS AS THE MAIN DECISION MAKER

The existing five SMARTS mechanism flags may remain.

However:

```text
mechanism rule == 1
```

must not force the model toward incompatible, nor should:

```text
mechanism rule == 0
```

mean that no chemistry evidence exists.

Use them as auxiliary features.

The learned bilinear interaction should remain capable of discovering interactions beyond the five manually encoded rules.

The gate should receive both learned chemistry representations and the auxiliary rule features.

---

# 12. FIX #10 — CORRECT THE 24-PAIR EVALUATION PROTOCOL

## Important metric rule

Do not write or interpret:

```text
ROC-AUC at threshold 0.60
```

or:

```text
PR-AUC at threshold 0.60
```

ROC-AUC and PR-AUC are ranking metrics and do not depend on one binary threshold.

## Report separately

### Ranking metrics

```text
24-pair ROC-AUC
24-pair PR-AUC
```

### Thresholded metrics at 0.60

```text
TP / 12 incompatible
FN / 12 incompatible
TN / 12 compatible
FP / 12 compatible
precision
recall
MCC
```

### Thresholded metrics at default threshold

Report the same confusion counts.

## Most important rule

The 24-pair held-out set is **report-only**.

Do NOT choose:

- architecture
- feature set
- threshold
- rank
- loss
- seed
- ensemble

because it improves the 24-pair result.

Use validation to choose the model/configuration.
Use official test only as a final evaluation.
Use the 24-pair set for final held-out reporting.

---

# 13. REMOVE THE "HARDEST PAIR MUST FAIL" ASSUMPTION

The previous instructions said that:

```text
Acetazolamide × Mannitol
```

should be expected to fail because mannitol has a low prior.

Do not hard-code this failure into the architecture or evaluation logic.

It is a useful known failure case, but the corrected architecture should be allowed to override a low prior when strong pair evidence exists.

The point of the residual architecture is precisely to let pair evidence override historical prevalence when justified.

Therefore:

```text
low prior ≠ forced compatible prediction
```

---

# 14. UPDATE THE PGB HEAD SHAPE AFTER REMOVING THE RAW BILINEAR BYPASS

The current head may say:

```text
426-dimensional input
```

because it contains both:

```text
bilinear [16]
gated [16]
```

After removing the raw bilinear path, recalculate the exact input dimension from the actual tensors.

For example, if using:

```text
h_api        96
h_exc        96
product      96
diff         96
gated        16
prior        5
mechanisms   5
```

the concatenation is:

```text
96 + 96 + 96 + 96 + 16 + 5 + 5 = 410
```

Do not leave a stale comment or hard-coded `426`.

Use a computed dimension where possible.

Example:

```python
concat_dim = (
    d + d + d + d
    + r
    + prior_dim
    + mechanism_dim
)
```

---

# 15. UPDATE CONFIGURATION

Add or modify the following settings as appropriate:

```python
pgb_use_cross_attn: bool = False
pgb_gate_use_api_flags: bool = True
pgb_gate_use_exc_flags: bool = True
pgb_gate_use_mechanism_flags: bool = True
pgb_use_missing_embedding: bool = True
pgb_use_salt_context: bool = True
pgb_use_adaptive_prior_trust: bool = True
pgb_bilinear_rank: int = 16
```

Keep defaults conservative.

Do not change the original baseline configuration used outside PGB.

---

# 16. TRAINING RULES

Keep the following rules from the previous implementation:

```text
NO WeightedRandomSampler for PGB
NO double class weighting
NO SMOTE
Use one pos_weight
Use asymmetric focal / the existing PGB loss
Use validation PR-AUC for early stopping
Use validation-only threshold selection
```

Do not reintroduce balanced sampling while fixing the architecture.

---

# 17. REQUIRED UNIT TESTS BEFORE ANY GPU TRAINING

Add and run all of these.

## A. Feature shape tests

```text
MACCS = 167
Morgan = 512
Chem flags = 18
PGB descriptors = 12
```

## B. Salt tests

Test at minimum:

```text
[Mg+2]
[Mg+2].[O-]C(=O)C(=O)[O-]
```

Verify:

```text
metal flag == 1
Mg flag == 1
salt detection behaves correctly
MACCS/Morgan include all valid fragments
salt context is nonzero where appropriate
```

## C. Prior aggregation test

Create several raw SMILES that canonicalize to the same stereo-blind key.

Verify all observations are aggregated instead of overwritten.

## D. Family prior test

Create two family variants with very different sample counts.

Verify family prior is count-weighted, not a simple mean of variant priors.

## E. Leave-cluster-out test

Construct a tiny dataframe with:

```text
cluster A: positive excipient X
cluster B: compatible excipient X
```

Verify the prior for a row in cluster A does not use cluster A evidence.

## F. Save/load prior test

Before save:

```text
exact lookup works
family lookup works
nearest-neighbour lookup works
```

After load:

```text
exact lookup works
family lookup works
nearest-neighbour lookup works
```

## G. Gate bypass test

Inspect the forward graph or model source and assert that raw `bilinear` is not concatenated into the final MLP input.

The only bilinear representation exposed to the final head must be the gated version.

## H. Gate API-input test

Verify the gate input dimension includes:

```text
API flags
EXC flags
prior
mechanism flags
```

## I. Missing-vector test

For PubChem and Mol2vec:

```text
known molecule → availability = 1
unknown molecule → availability = 0
unknown molecule → NOT all-zero projected feature
```

## J. Model forward tests

Run a batch through all nine PGB families.

Assert:

```text
logits shape = [B]
no NaNs
no infs
```

---

# 18. REQUIRED ABLATION CHECKS

After implementation, before training all nine models, run the corrected MACCS family first.

Produce three versions:

### A — Current implementation

The already-existing PGB implementation before these fixes.

### B — Corrected implementation

All fixes in this document.

### C — Corrected implementation without adaptive prior trust

This isolates whether prior reliability is helping.

The key comparison is:

```text
A vs B
```

Do not choose based on the 24-pair set alone.

Primary selection should use validation metrics and official-test performance.

---

# 19. FIRST TRAINING ORDER AFTER FIXES

Do NOT immediately train all nine.

Use this order:

```text
1. MACCS
2. PubChem
3. MoLFormer
4. D-MPNN
5. GAT
6. ChemBERTa
7. GIN
8. Morgan
9. Mol2vec
```

After MACCS, inspect:

```text
validation PR-AUC
validation MCC
validation TNR
official-test PR-AUC
official-test MCC
official-test FP
```

Then inspect 24-pair ranking and thresholded results only as a diagnostic/final held-out check.

---

# 20. REQUIRED DEBUG INFORMATION DURING TRAINING

For every family, print once per run:

```text
family
encoder
fusion mode
h_api shape
h_exc shape
bilinear shape
gate shape
prior shape
final MLP input shape
number of trainable parameters
number of frozen parameters
```

Also print:

```text
train positive rate
number of unique excipients
number of unseen validation excipients
number of unseen test excipients
number of unknown PubChem/Mol2vec vectors
```

For training priors, print:

```text
fraction of train rows using leave-cluster-out prior
fraction using global fallback
mean prior
std prior
min prior
max prior
```

---

# 21. REQUIRED GATE DIAGNOSTICS

The model must expose or log statistics for:

```text
gate mean
 gate std
 gate min
 gate max
```

for each training epoch or at least for selected validation epochs.

Watch for pathological behavior such as:

```text
all gate ≈ 0
all gate ≈ 1
no variation across pairs
```

The gate should be capable of differentiating contexts.

Also inspect gate values for a few representative pairs:

```text
API + reactive excipient
API + inert excipient
unseen excipient
known high-prior excipient
known low-prior excipient
```

---

# 22. REQUIRED PRIOR DIAGNOSTICS

Inspect at least:

```text
prior
prior_delta
p_family
log_n
unseen
prior_weight
```

for examples such as:

```text
lactose
mannitol
MgO
CaSO4
Mg stearate
unseen polymer
```

The model must not be hard-coded to classify these one way or another.

These are diagnostic examples only.

---

# 23. DO NOT CHANGE THE DATA SPLIT

Do not change:

- train rows
- validation rows
- official test rows
- API-cluster grouping
- held-out file contents

Do not introduce random-split leakage.

Do not use the 24-pair labels during training or architecture selection.

---

# 24. DO NOT MAKE THESE CHANGES

Do NOT:

- add SMOTE
- restore WeightedRandomSampler
- add tree ensembles
- tune the model on the 24-pair labels
- hard-code special cases for named APIs/excipients
- force acetazolamide × mannitol to be positive
- force lactose to be positive
- force metal oxides to be positive
- make the gate depend on the label
- compute priors from validation/test labels
- silently fall back to zero vectors for missing lookup features
- expose raw ungated bilinear features to the final MLP
- change the original baseline model

---

# 25. FINAL ARCHITECTURE TARGET

The corrected PGB architecture should conceptually be:

```text
                         API SMILES
                            │
                    family-specific encoder
                            │
                     API feature tower
                            │
                       h_api [96]
                            │
                            │
                            ├──────────────┐
                            │              │
                            ▼              │
                       Bilinear U         │
                            │              │
                            ▼              │
                         z_api             │
                                           │
                                           ×
                                           │
                         z_exc ◄───────────┘
                           ▲
                           │
                    Bilinear V
                           │
                           │
                    h_exc [96]
                           ▲
                           │
                    EXC feature tower
                           ▲
                           │
                   family-specific encoder
                           ▲
                           │
                       EXC SMILES

Pair chemistry context:

API flags [18] ─────┐
EXC flags [18] ─────┼──→ Gate MLP → gate [16]
Prior [5] ──────────┤
Mechanisms [5] ─────┘

bilinear [16] × gate [16]
          ↓
   gated bilinear [16]
          ↓
final pair head
          ↓
residual
          ↓
adaptive prior trust × prior log-odds
          ↓
       final logit
```

The final MLP should receive the gated bilinear interaction, not an ungated copy.

---

# 26. FINAL VERIFICATION CHECKLIST

Before declaring the implementation ready, confirm every item below:

```text
[ ] Training prior is true leave-API-cluster-out
[ ] Validation/test prior uses train rows only
[ ] Canonical duplicate excipients aggregate correctly
[ ] Family prior aggregates counts correctly
[ ] Nearest-neighbour prior survives save/load
[ ] PubChem/Mol2vec unknowns use learned missing embeddings
[ ] Availability masks are present
[ ] Salt-aware fingerprints remain enabled
[ ] Salt/ionic context is represented in descriptors
[ ] API chemistry flags reach the gate
[ ] EXC chemistry flags reach the gate
[ ] Mechanism flags remain auxiliary
[ ] Raw ungated bilinear is NOT passed to final MLP
[ ] Gated bilinear is passed to final MLP
[ ] Sequence/graph models first use late fusion
[ ] Adaptive prior trust is implemented or explicitly ablated
[ ] No balanced sampler
[ ] No SMOTE
[ ] No label leakage
[ ] 24-pair set is report-only
[ ] AUC metrics are reported independently of threshold
[ ] All 9 PGB forward passes are valid
[ ] No NaNs / infs
[ ] Gate statistics are non-degenerate
```

---

# 27. REPORT BACK TO THE USER

After implementation, do NOT merely say "done".

Report:

1. Exactly which files were changed.
2. Which of the above fixes were already present and which were newly changed.
3. The final PGB head tensor dimensions.
4. Whether true leave-cluster-out priors are now actually used for every training row.
5. Whether the ungated bilinear bypass is removed.
6. Whether API flags reach the gate.
7. Whether missing-vector handling is fixed.
8. Whether prior save/load preserves NN fallback.
9. Whether deep PGB models use late fusion in the first corrected run.
10. The unit-test results.
11. Only then begin training the corrected models.

Do not claim performance improvement until the corrected models have actually been trained and evaluated.
