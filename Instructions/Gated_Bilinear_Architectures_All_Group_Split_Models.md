# Prior-Gated Bilinear Architectures for Every Group-Split Model

**Project:** API–Excipient incompatibility prediction
**Scope:** group-split (API-cluster) models only. Random-split models are not touched. The 193-pair dump is ignored.
**Targets:** validation set (842 pairs), official test set (672 pairs), and the 24-pair named held-out set.
**Source of the idea:** *Gated Bilinear PairNet – Architecture and Results* (the docx you shared).

This file contains architecture only: layer sizes, data flow and flow charts. It has no code. (The text boxes below sit inside code fences purely so they stay aligned; they are diagrams, not code.)

---

## 0. Read this first

1. **Nothing here has been trained.** I read your code, your metric files and the doc. The "current" numbers below are computed from your own `metrics/*/val_metrics.json`, `test_metrics.json` and `held_out_testset/held_out_predictions.csv`. The "design targets" in section 9 are hypotheses scaled from what the doc reported, not promises. Section 9 gives a protocol that tells you quickly whether a family is really improving.
2. **The doc's gain did not come from one trick.** It came from removing double class-weighting, giving each excipient a historical-rate prior, adding chemistry flags, and adding a gated bilinear interaction. I carry all of these into every family and keep each family's own encoder as the "eyes" of the model.
3. **One shared head, nine different towers.** Every family ends in the same *Prior-Gated Bilinear (PGB) head* (section 3). What differs is the tower that turns each molecule into a 96-number vector (sections 4 to 6). Each tower keeps what already works in that family and adds only what that family is missing.
4. **Strong families may not gain on validation.** The doc's model scored val PR-AUC 0.69. Your MoLFormer and D-MPNN already score about 0.70 to 0.72 there. For those, the gain to look for is fewer false alarms and higher MCC, not higher val PR-AUC.

**Legend for the flow charts.** A box is a layer or block. `[n]` is a vector length. `LN` is LayerNorm. `Drop .15` is dropout 0.15. `*` is element-wise product. `|a-b|` is element-wise absolute difference. Arrows run top to bottom.

---

## 1. What made the doc's model better, and what I found in your code

### 1.1 What transfers

| # | Idea in the doc | Why it helped there | How it is used in your models |
|---|---|---|---|
| 1 | Remove double class-weighting (sampler **and** weighted loss) | False alarms on compatible pairs fell from 56 to 14 on the official test | Balanced sampler off in every family. One `pos_weight`. Asymmetric focal loss. |
| 2 | Excipient prior used as a starting log-odds, with the network learning a residual | Each excipient starts from its own historical incompatibility rate | Same block in every family (section 3.3) |
| 3 | 18 chemistry flags and salt-aware parsing | Fingerprints and encoders see almost nothing in `O=[Mg]` or `[Mg+2]` salts | Added to every tower. Matters most for the string and graph encoders. |
| 4 | Two untied towers, several feature blocks mixed to 96-d | An API and a filler are different kinds of molecule | Each family gets its own multi-block tower (sections 4 to 6) |
| 5 | Rank-16 bilinear interaction with a gate | "Amine x reducing sugar" is an interaction, not a sum. The gate shuts it for inert excipients. | Same head in every family |
| 6 | Threshold from validation MCC with a true-negative-rate floor | A val-F1 threshold landed at 0.957 and hid the false alarms | Same rule in every family (section 8) |

### 1.2 Four additions that are mine, not the doc's

| # | Addition | Why |
|---|---|---|
| A1 | **Leave-own-API-cluster-out prior for training rows.** Validation, test and 24-pair rows use the prior from all train rows. | In your train file, 180 of 254 excipient keys have 3 rows or fewer. A prior computed on all train rows therefore contains each row's own label, which can make the network over-trust it in training and under-deliver on new APIs. |
| A2 | **Stereo-blind excipient key, plus a hydrate-stripped family key and a nearest-neighbour key as extra inputs.** | I checked your 24-pair file: with exact SMILES keys only 13 of 24 excipients match train; with a stereo-blind key 21 of 24 match. On val and test the match rate stays at 94% and 95%. Lactose, mannitol, sucrose and beta-cyclodextrin are the ones recovered. |
| A3 | **Mechanism flags.** Your `src/lookup.py` already defines 5 reaction rules (Maillard, ester/amide hydrolysis, metal-catalysed oxidation, acid–base salt displacement, Schiff base). They are off by default. I turn them on as 5 inputs to the head and the gate. | They are the "amine x sugar" knowledge the bilinear term is meant to learn, given as a hint instead of left to be discovered from about 190 positives. |
| A4 | **A "vector missing" mask for fixed-vector models, and on-the-fly MACCS and Morgan.** | See finding F3 below. |

### 1.3 Findings from your code and files (verified by reading them)

* **F1. Double weighting is on by default.** `use_balanced_sampler` is true in your config, so unless a run was launched with `--no_balanced_sampler`, it uses the balanced sampler on top of its loss (ASL, focal, weighted BCE alike). Your own ablation on MoLFormer with global gated pooling supports the doc only partly. Turning the sampler off raised val PR-AUC in all four loss pairs. Test PR-AUC moved: weighted BCE 0.501 to 0.666 (large gain), ASL 0.625 to 0.639, focal 0.701 to 0.683, plain BCE 0.721 to 0.647 (loss). So removing the sampler is **necessary but not sufficient**. The prior offset is what takes over the job of calibrating the base rate.
* **F2. The val-F1 threshold is unstable.** The thresholds chosen across your 54 group-split runs range from 0.042 to 0.998.
* **F3. Fixed-vector models see zero vectors on the 24-pair set.** The encoder looks molecules up in precomputed tables of 850 molecules and silently returns all zeros for anything missing. Only 9 of the 24 APIs and 21 of the 24 excipients are in those tables. So 15 of 24 API vectors are all zeros in all 16 fixed-vector runs, and their 24-pair numbers are partly meaningless. MACCS and Morgan can be computed from SMILES on the fly. PubChem and Mol2vec tables must be extended for any new molecule.
* **F4. Token-pair pooling hurt.** Your explicit pairwise variants score lower than gated pooling (MoLFormer val PR-AUC 0.66 vs 0.72; ChemBERTa 0.54 to 0.56 vs 0.57 to 0.63). I therefore do **not** add token-pair features. The bilinear term works on whole-molecule vectors instead.
* **F5. Four pairs of your runs (eight runs) are identical.** With concat fusion the pooling option is unused, so `dmpnn_chemprop_concat_cls_*` and `..._global_gated_*` have identical metrics (ASL and BCE), and the same holds for `pretrained_gat_concat_*`.
* **F6. Excipients repeat across splits, APIs do not.** 0% of val and test APIs appear in train, but 94% (val) and 95% (test) of rows have an excipient that appears in train. This is why the excipient prior is expected to help val and test strongly. On the 24-pair set only 2 of 24 APIs are in train.

---

## 2. Where each family stands today, and the pipeline I chose

For each family I picked the existing pipeline with the best mean of (val PR-AUC, test PR-AUC, 24-pair ROC-AUC). Numbers are at each run's own val-F1 threshold, from your files. 24-pair numbers for the four fixed-vector families are affected by finding F3.

| Family | Chosen current pipeline | Val PR | Val MCC | Test PR | Test MCC | Test FP | 24-pair PR | 24-pair ROC |
|---|---|---|---|---|---|---|---|---|
| MACCS fixed vector | concat, focal | 0.619 | 0.561 | 0.572 | 0.457 | 42 | 0.780 | 0.750 |
| Morgan fixed vector | concat, ASL | 0.548 | 0.493 | 0.549 | 0.402 | 35 | 0.800 | 0.812 |
| PubChem fixed vector | concat, focal | 0.594 | 0.520 | 0.597 | 0.454 | 33 | 0.775 | 0.785 |
| Mol2vec fixed vector | concat, ASL | 0.569 | 0.492 | 0.498 | 0.360 | 58 | 0.687 | 0.618 |
| MoLFormer | cross-attn, global gated pooling, ASL | 0.721 | 0.641 | 0.687 | 0.562 | 38 | 0.718 | 0.611 |
| ChemBERTa | cross-attn, CLS pooling, BCE | 0.626 | 0.548 | 0.655 | 0.533 | 53 | 0.695 | 0.632 |
| Pretrained GIN | concat, ASL | 0.517 | 0.501 | 0.563 | 0.417 | 53 | 0.701 | 0.701 |
| Pretrained GAT (Stanford) | cross-attn, global gated pooling, ASL | 0.577 | 0.511 | 0.575 | 0.395 | 77 | 0.859 | 0.868 |
| D-MPNN (CheMeleon) | cross-attn, CLS pooling, ASL | 0.703 | 0.587 | 0.620 | 0.517 | 35 | 0.855 | 0.812 |
| *Doc's Gated Bilinear PairNet, for reference* | | 0.69 | 0.64 | 0.73 | 0.61 | 14 | 0.86 | 0.94 |

Reading the table: your transformers are best on val and test, your graph models are best on the 24-pair ranking, and every family has a high false-positive count on test. The doc's model beats every row on test false alarms and on 24-pair ROC.

---

## 3. The shared building blocks

### 3.1 The whole model, in one picture

```
┌───────────────────┐  ┌───────────────────┐  ┌──────────────────────┐  ┌───────────────────┐
│     API tower     │  │  EXCIPIENT tower  │  │   Excipient prior    │  │  Mechanism flags  │
│ (family specific) │  │ (family specific, │  │      block [5]       │  │ [5] (SMARTS rules │
│   out h_API [96]  │  │    own weights)   │  │ (train-only history) │  │     API x EXC)    │
│                   │  │   out h_EXC [96]  │  │                      │  │                   │
└───────────────────┘  └───────────────────┘  └──────────────────────┘  └───────────────────┘
          │                      │                        │                       │
          └──────────────────────┴────────────┬───────────┴───────────────────────┘
                                              │
                                              ▼
                           ┌─────────────────────────────────────┐
                           │      PRIOR-GATED BILINEAR HEAD      │
                           │     interactions -> gate -> MLP     │
                           │ residual + excipient-prior log-odds │
                           └─────────────────────────────────────┘
                                              │
                                              ▼
                        ┌───────────────────────────────────────────┐
                        │ P(incompatible)  ->  decision threshold t │
                        │  t picked on VAL by MCC with a TNR floor  │
                        └───────────────────────────────────────────┘
```

Every family produces `h_API [96]` and `h_EXC [96]` with its own tower. The prior block and mechanism flags are identical in every family.

### 3.2 Per-molecule feature blocks

| Block | Size | What it is | Used by |
|---|---|---|---|
| MACCS keys | 166 | Functional-group and ring checklist. Your table has 166 columns (bit 0 dropped). On-the-fly RDKit output must drop bit 0 to match. | MACCS family (primary); side branch in Mol2vec, Morgan, MoLFormer, GIN, GAT, D-MPNN |
| Morgan fingerprint, radius 2 | 512 (side branch) or 1024 (Morgan family primary) | Local circular atom neighbourhoods | Side branch in MACCS, PubChem, Mol2vec, ChemBERTa; primary in Morgan |
| PubChem fingerprint | 881 | Substructure keys from the PubChem scheme (precomputed table) | PubChem family |
| Mol2vec | 300 | Dense substructure embedding (precomputed table) | Mol2vec family |
| RDKit descriptors | 12, z-scored on train only | MolWt, MolLogP, TPSA, H-donors, H-acceptors, rotatable bonds, aromatic rings, ring count, FractionCSP3, heavy atoms, heteroatoms, QED | All families. **Replaces** your current 21 (the 11 `fr_*` counts leave; heteroatoms and QED join). |
| Chemistry flags | 18 yes/no | No carbons, oxidizer, strong acid, strong base, metal, Mg, Ca, Al, Si, Ti/Fe, Mn/Cr, is a salt, very short string, many –OH, amine, carboxylic acid, ester, phenol | All families |
| Mechanism flags | 5 yes/no per pair | The 5 rules already in `src/lookup.py`, true when the API side and the excipient side both match | Head and gate in all families |

**Salt-aware parsing.** If a SMILES contains a dot, every fragment is processed and the bits are OR-ed, so `[Mg+2]` and `[Al+3]` never disappear inside a salt string.

### 3.3 The excipient prior block

```
                  ┌───────────────────────────────────────┐
                  │            TRAIN rows only            │
                  │        (API, excipient, label)        │
                  │ train rows: leave-own-API-cluster-out │
                  │  val / test / 24-pair: all train rows │
                  └───────────────────────────────────────┘
                                      │
             ┌────────────────────────┼─────────────────────────┐
             ▼                        ▼                         ▼
┌────────────────────────┐  ┌───────────────────┐  ┌────────────────────────┐
│       Exact key        │  │     Family key    │  │     Neighbour key      │
│ stereo-blind canonical │  │ hydrate / solvent │  │  Morgan-512 Tanimoto   │
│  SMILES, salt pieces   │  │   stripped, same  │  │  >= 0.6, similarity-   │
│  sorted, hydrate kept  │  │     smoothing     │  │        weighted        │
│    Laplace alpha=4     │  │    -> p_family    │  │ -> p_nn  (else global) │
│     -> p_exact , n     │  │                   │  │                        │
└────────────────────────┘  └───────────────────┘  └────────────────────────┘
             │                        │                         │
             └────────────────────────┼─────────────────────────┘
                                      │
                                      ▼
      ┌───────────────────────────────────────────────────────────────┐
      │                        PRIOR BLOCK [5]                        │
      │ p_exact | p_exact - 0.094 | p_family | log(1+n) | unseen flag │
      └───────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
        ┌──────────────────────────────────────────────────────────┐
        │     Logit offset uses p_exact only (as in the doc).      │
        │ p_family and p_nn go to the head and the gate as inputs. │
        └──────────────────────────────────────────────────────────┘
```

| Output | Meaning |
|---|---|
| `p_exact` | Laplace-smoothed incompatible rate for this excipient, `(positives + 4 x 0.094) / (rows + 4)`. Global rate 0.094 if unseen. |
| `p_exact - 0.094` | How unusual this excipient is |
| `p_family` | Same statistic after stripping hydrate and solvent pieces, smoothed toward the global rate |
| `log(1 + n)` | How much evidence stands behind `p_exact` |
| `unseen flag` | 1 if the excipient (and its neighbours) was never seen |

Why the offset uses `p_exact` only: lactose without water has prior about 0.26 in your train file and lactose monohydrate about 0.017. The doc treats this as a flaw. On your 24-pair set it happens to separate three incompatible lactose pairs from one compatible monohydrate pair correctly. I keep the sharp key for the offset and give the head the merged family value as extra information, so the network can learn how far to trust each. I would not merge them blindly.

**Honest caution.** Three of the 24 excipients (magnesium aluminum silicate, calcium phosphate, talc) stay unseen under any key, and one of them is a true incompatible. The gate stays half-open for those and the score then depends on the pair evidence.

### 3.4 The Prior-Gated Bilinear head (identical in all nine families)

```
                         ┌──────────────────────────────┐
                         │ h_API [96]        h_EXC [96] │
                         └──────────────────────────────┘
                                         │
        ┌───────────────────┬────────────┴─────────┬───────────────────────┐
        ▼                   ▼                      ▼                       ▼
┌───────────────┐  ┌─────────────────┐  ┌─────────────────────┐  ┌───────────────────┐
│ h_API * h_EXC │  │ |h_API - h_EXC| │  │   bilinear rank-16  │  │   gated bilinear  │
│      [96]     │  │       [96]      │  │ (h_API.U)*(h_EXC.V) │  │ bilinear x gate g │
│               │  │                 │  │         [16]        │  │        [16]       │
└───────────────┘  └─────────────────┘  └─────────────────────┘  └───────────────────┘
        │                   │                      │                       │
        └───────────────────┴────────────┬─────────┴───────────────────────┘
                                         │
                                         ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │   CONCAT [426] = h_API 96 + h_EXC 96 + product 96 + |diff| 96   │
        │ + bilinear 16 + gated bilinear 16 + prior block 5 + mechanism 5 │
        └─────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                  ┌─────────────────────────────────────────────┐
                  │ Linear 426 -> 128  |  GELU  |  Dropout 0.30 │
                  └─────────────────────────────────────────────┘
                                         │
                                         ▼
                  ┌─────────────────────────────────────────────┐
                  │ Linear 128 -> 48   |  GELU  |  Dropout 0.20 │
                  └─────────────────────────────────────────────┘
                                         │
                                         ▼
                        ┌─────────────────────────────────┐
                        │ Linear 48 -> 1   =   residual r │
                        └─────────────────────────────────┘
                                         │
                                         ▼
         ┌──────────────────────────────────────────────────────────────┐
         │               logit = r + s * log(p/(1-p)) + b               │
         │              p = excipient prior (offset only)               │
         │ s learned, starts 0.8  |  b starts at 0.2*log(base/(1-base)) │
         └──────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │ probability = sigmoid(logit) │
                         └──────────────────────────────┘
```

```
      ┌────────────────────────────────────┐
      │ gate inputs (all already computed) │
      └────────────────────────────────────┘
                         │
          ┌──────────────┴┬──────────────┐
          ▼               ▼              ▼
   ┌─────────────┐  ┌───────────┐  ┌───────────┐
   │ prior block │  │ EXC flags │  │ mechanism │
   │     [5]     │  │    [18]   │  │    [5]    │
   └─────────────┘  └───────────┘  └───────────┘
          │               │              │
          └──────────────┬┴──────────────┘
                         │
                         ▼
                  ┌─────────────┐
                  │ concat [28] │
                  └─────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │ Linear 28 -> 16 | GELU │
            └────────────────────────┘
                         │
                         ▼
           ┌───────────────────────────┐
           │ Linear 16 -> 16 | sigmoid │
           └───────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────┐
│                  gate g [16]                   │
│ multiplies the 16 bilinear channels one-to-one │
└────────────────────────────────────────────────┘
```

| Step | Layer | In to Out | Parameters |
|---|---|---|---|
| Product | `h_API * h_EXC` | 96 to 96 | 0 |
| Difference | `abs(h_API - h_EXC)` | 96 to 96 | 0 |
| Bilinear, left map U | Linear, no bias | 96 to 16 | 1,536 |
| Bilinear, right map V | Linear, no bias | 96 to 16 | 1,536 |
| Bilinear output | `(h_API U) * (h_EXC V)` | 16 | 0 |
| Gate layer 1 | Linear + GELU on `[prior 5, EXC flags 18, mechanism 5]` | 28 to 16 | 464 |
| Gate layer 2 | Linear + sigmoid | 16 to 16 | 272 |
| Gated bilinear | bilinear x gate, channel by channel | 16 | 0 |
| Concat | 96 + 96 + 96 + 96 + 16 + 16 + 5 + 5 | 426 | 0 |
| MLP layer 1 | Linear + GELU + Dropout 0.30 | 426 to 128 | 54,656 |
| MLP layer 2 | Linear + GELU + Dropout 0.20 | 128 to 48 | 6,192 |
| MLP layer 3 | Linear | 48 to 1 | 49 |
| Prior scale `s` and bias `b` | learned scalars | 1 + 1 | 2 |
| **Head total** | | | **about 64,700** |

The gate input is 28 numbers (the doc leaves the gate's exact size open). A per-channel 16-wide gate is my choice, so lactose-type contexts can open some channels and keep others shut.

**Final score.** `logit = residual + s x log(p / (1 - p)) + b`, where `p` is `p_exact`. `s` is learned and starts at 0.8. `b` starts at `0.2 x log(base / (1 - base))` so that with a zero residual and a never-seen excipient the model starts at the base rate 0.094. This replaces your current "last-layer bias = log prior" initialisation.

**Dropouts.** The head uses 0.30 and 0.20 instead of your current 0.5 and 0.4. The 96-d bottleneck, the gate and the prior already regularise. These are starting values to tune on validation.

### 3.5 Loss and training (same rule in every family)

```
         ┌───────────────────────────────────────────────────────┐
         │ Existing group split (API-cluster) train / val / test │
         │           random-split files are not touched          │
         └───────────────────────────────────────────────────────┘
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │         Feature build (train-only statistics)          │
        │ flags, MACCS, Morgan, 12 descriptors z-scored on train │
        └────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                 ┌──────────────────────────────────────┐
                 │        Excipient prior table         │
                 │ leave-API-cluster-out for train rows │
                 └──────────────────────────────────────┘
                                     │
                                     ▼
              ┌─────────────────────────────────────────────┐
              │ Model forward (family tower x2 -> PGB head) │
              └─────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Loss: asymmetric focal, gamma+ 1.2, gamma- 2.5, pos_weight n_neg/n_pos │
│                  NO balanced sampler (plain shuffle)                   │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                ┌─────────────────────────────────────────┐
                │ AdamW + ReduceLROnPlateau on VAL PR-AUC │
                │      early stop, restore best epoch     │
                └─────────────────────────────────────────┘
                                     │
                                     ▼
       ┌───────────────────────────────────────────────────────────┐
       │ Threshold t from VAL: maximise MCC subject to TNR >= 0.97 │
       └───────────────────────────────────────────────────────────┘
                                     │
                                     ▼
             ┌──────────────────────────────────────────────┐
             │    Freeze t.  Report VAL | TEST | 24-pair    │
             │ PR-AUC, ROC-AUC, MCC, FP, TP at the frozen t │
             └──────────────────────────────────────────────┘
```

| Setting | Value | Replaces |
|---|---|---|
| Loss | Asymmetric focal BCE, gamma+ = 1.2, gamma- = 2.5 | ASL / focal / weighted BCE / BCE |
| Class weight | one `pos_weight` = n_neg / n_pos | sampler **and** weighted loss together |
| Sampler | none, plain shuffle | `WeightedRandomSampler` (on by default today) |
| Threshold | val MCC subject to a true-negative-rate floor (I suggest 0.97; the doc does not give the value) | val F1 sweep |
| Selection and early stop | val PR-AUC, scheduler on plateau | unchanged idea |

---

## 4. Fixed-vector families

These four models currently share one structure: a 1-vector-per-molecule encoder, a 256-to-128 projection, a 21-descriptor head and a 609-d pair vector. For all four, the new tower mixes four blocks into 96 numbers, then the shared head runs.

**Pipeline change for all four (finding F3).** Compute MACCS and Morgan from SMILES at run time. For PubChem and Mol2vec, extend the lookup tables to every molecule in val, test and the 24-pair file. If a table vector is still missing, the primary block's output is replaced by a learned "missing" vector (64 numbers) instead of an all-zero input. Tower sizes do not change.

### 4.1 MACCS family (closest to the doc: the reference build)

```
                       ┌───────────────────────────────────────┐
                       │     one SMILES (API or excipient)     │
                       │ salt-aware: split on '.', OR the bits │
                       └───────────────────────────────────────┘
                                           │
          ┌───────────────────────┬────────┴─────────────┬──────────────────┐
          ▼                       ▼                      ▼                  ▼
┌──────────────────┐  ┌───────────────────────┐  ┌───────────────┐  ┌───────────────┐
│      MACCS       │  │       Morgan r2       │  │   RDKit desc  │  │   Chem flags  │
│     166 bits     │  │ 512 bits (on the fly) │  │  12 z-scored  │  │   18 yes/no   │
│  Linear 166->64  │  │     Linear 512->64    │  │ Linear 12->24 │  │ Linear 18->16 │
│ LN|GELU|Drop .15 │  │    LN|GELU|Drop .15   │  │   LN | GELU   │  │      GELU     │
└──────────────────┘  └───────────────────────┘  └───────────────┘  └───────────────┘
          │                       │                      │                  │
          └───────────────────────┴────────┬─────────────┴──────────────────┘
                                           │
                                           ▼
                                   ┌──────────────┐
                                   │ CONCAT [168] │
                                   └──────────────┘
                                           │
                                           ▼
                          ┌────────────────────────────────┐
                          │ Linear 168 -> 96  |  LN | GELU │
                          └────────────────────────────────┘
                                           │
                                           ▼
               ┌───────────────────────────────────────────────────────┐
               │                        h  [96]                        │
               │ API tower and EXC tower: same shape, separate weights │
               └───────────────────────────────────────────────────────┘
```

| # | Layer | In to Out | Notes |
|---|---|---|---|
| 1 | MACCS block: Linear, LN, GELU, Drop .15 | 166 to 64 | primary |
| 2 | Morgan block: Linear, LN, GELU, Drop .15 | 512 to 64 | **new** secondary |
| 3 | Descriptor block: Linear, LN, GELU | 12 to 24 | replaces 21-to-32-to-24 head |
| 4 | Flag block: Linear, GELU | 18 to 16 | **new** |
| 5 | Concat | 64 + 64 + 24 + 16 = 168 | |
| 6 | Mix: Linear, LN, GELU | 168 to 96 | `h` |

* **Parameters:** about 60,900 per tower, about 186,000 for the whole model. This matches the doc's "about 185,000".
* **Keeps:** MACCS as the main input, separate API and excipient weights, concat + product + |diff| operators.
* **Removes:** the 256-wide projection, the 21 descriptors with `fr_*` counts, the 609-d flat head, the sampler.
* **Why it should help:** this is the doc's model with two extras (A1, A3). The main fault of the current MACCS runs is 40 to 56 test false alarms, which the doc cut to 14.

### 4.2 Morgan family

```
                       ┌───────────────────────────────────────┐
                       │     one SMILES (API or excipient)     │
                       │ salt-aware: split on '.', OR the bits │
                       └───────────────────────────────────────┘
                                           │
          ┌───────────────────────┬────────┴─────────────┬──────────────────┐
          ▼                       ▼                      ▼                  ▼
┌──────────────────┐  ┌───────────────────────┐  ┌───────────────┐  ┌───────────────┐
│    Morgan r2     │  │         MACCS         │  │   RDKit desc  │  │   Chem flags  │
│    1024 bits     │  │ 166 bits (on the fly) │  │  12 z-scored  │  │   18 yes/no   │
│ Linear 1024->64  │  │     Linear 166->64    │  │ Linear 12->24 │  │ Linear 18->16 │
│ LN|GELU|Drop .25 │  │    LN|GELU|Drop .15   │  │   LN | GELU   │  │      GELU     │
└──────────────────┘  └───────────────────────┘  └───────────────┘  └───────────────┘
          │                       │                      │                  │
          └───────────────────────┴────────┬─────────────┴──────────────────┘
                                           │
                                           ▼
                                   ┌──────────────┐
                                   │ CONCAT [168] │
                                   └──────────────┘
                                           │
                                           ▼
                          ┌────────────────────────────────┐
                          │ Linear 168 -> 96  |  LN | GELU │
                          └────────────────────────────────┘
                                           │
                                           ▼
                         ┌──────────────────────────────────┐
                         │             h  [96]              │
                         │ separate weights for API and EXC │
                         └──────────────────────────────────┘
```

| # | Layer | In to Out | Notes |
|---|---|---|---|
| 1 | Morgan block: Linear, LN, GELU, Drop **.25** | 1024 to 64 | primary. Dropout is higher because 1024 sparse bits overfit on about 2,000 rows. |
| 2 | MACCS block: Linear, LN, GELU, Drop .15 | 166 to 64 | **new** secondary. Gives named functional groups that hashed bits cannot. |
| 3 | Descriptor block | 12 to 24 | |
| 4 | Flag block | 18 to 16 | |
| 5 | Concat, then Mix: Linear, LN, GELU | 168 to 96 | `h` |

* **Parameters:** about 93,600 per tower, about 252,000 total.
* **Why it should help:** Morgan bits are hashed, so one bit can mean different substructures. The MACCS side branch and the flags give the network named chemistry to lean on. Morgan and PubChem are already the two best fixed-vector families on the 24-pair ranking (ROC about 0.8), and the aim is to keep that while cutting test false alarms.

### 4.3 PubChem fingerprint family

```
                       ┌───────────────────────────────────────┐
                       │     one SMILES (API or excipient)     │
                       │ salt-aware: split on '.', OR the bits │
                       └───────────────────────────────────────┘
                                           │
          ┌───────────────────────┬────────┴─────────────┬──────────────────┐
          ▼                       ▼                      ▼                  ▼
┌──────────────────┐  ┌───────────────────────┐  ┌───────────────┐  ┌───────────────┐
│    PubChem FP    │  │       Morgan r2       │  │   RDKit desc  │  │   Chem flags  │
│     881 bits     │  │ 512 bits (on the fly) │  │  12 z-scored  │  │   18 yes/no   │
│  Linear 881->64  │  │     Linear 512->64    │  │ Linear 12->24 │  │ Linear 18->16 │
│ LN|GELU|Drop .25 │  │    LN|GELU|Drop .15   │  │   LN | GELU   │  │      GELU     │
└──────────────────┘  └───────────────────────┘  └───────────────┘  └───────────────┘
          │                       │                      │                  │
          └───────────────────────┴────────┬─────────────┴──────────────────┘
                                           │
                                           ▼
                                   ┌──────────────┐
                                   │ CONCAT [168] │
                                   └──────────────┘
                                           │
                                           ▼
                          ┌────────────────────────────────┐
                          │ Linear 168 -> 96  |  LN | GELU │
                          └────────────────────────────────┘
                                           │
                                           ▼
                         ┌──────────────────────────────────┐
                         │             h  [96]              │
                         │ separate weights for API and EXC │
                         └──────────────────────────────────┘
```

| # | Layer | In to Out | Notes |
|---|---|---|---|
| 1 | PubChem block: Linear, LN, GELU, Drop **.25** | 881 to 64 | primary |
| 2 | Morgan block: Linear, LN, GELU, Drop .15 | 512 to 64 | **new** secondary. PubChem keys already overlap MACCS, so the complement is local atom environments. |
| 3 | Descriptor block | 12 to 24 | |
| 4 | Flag block | 18 to 16 | |
| 5 | Concat, then Mix: Linear, LN, GELU | 168 to 96 | `h` |

* **Parameters:** about 106,600 per tower, about 278,000 total.
* **Extra risk:** PubChem fingerprints cannot be computed with RDKit alone. The lookup table must be extended for new molecules, or the "missing" vector fires.

### 4.4 Mol2vec family (lowest expected return)

```
                             ┌───────────────────────────────────────┐
                             │     one SMILES (API or excipient)     │
                             │ salt-aware: split on '.', OR the bits │
                             └───────────────────────────────────────┘
                                                 │
          ┌────────────────────┬─────────────────┴─┬──────────────────┬──────────────────┐
          ▼                    ▼                   ▼                  ▼                  ▼
┌──────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐  ┌───────────────┐
│     Mol2vec      │  │     MACCS      │  │   Morgan r2    │  │   RDKit desc  │  │   Chem flags  │
│    300 dense     │  │    166 bits    │  │    512 bits    │  │  12 z-scored  │  │   18 yes/no   │
│  Linear 300->64  │  │ Linear 166->32 │  │ Linear 512->32 │  │ Linear 12->24 │  │ Linear 18->16 │
│ LN|GELU|Drop .15 │  │    LN|GELU     │  │    LN|GELU     │  │   LN | GELU   │  │      GELU     │
└──────────────────┘  └────────────────┘  └────────────────┘  └───────────────┘  └───────────────┘
          │                    │                   │                  │                  │
          └────────────────────┴─────────────────┬─┴──────────────────┴──────────────────┘
                                                 │
                                                 ▼
                                         ┌──────────────┐
                                         │ CONCAT [168] │
                                         └──────────────┘
                                                 │
                                                 ▼
                                ┌────────────────────────────────┐
                                │ Linear 168 -> 96  |  LN | GELU │
                                └────────────────────────────────┘
                                                 │
                                                 ▼
                               ┌──────────────────────────────────┐
                               │             h  [96]              │
                               │ separate weights for API and EXC │
                               └──────────────────────────────────┘
```

| # | Layer | In to Out | Notes |
|---|---|---|---|
| 1 | Mol2vec block: Linear, LN, GELU, Drop .15 | 300 to 64 | primary (dense embedding) |
| 2 | MACCS block: Linear, LN, GELU | 166 to 32 | **new** |
| 3 | Morgan block: Linear, LN, GELU | 512 to 32 | **new** |
| 4 | Descriptor block | 12 to 24 | |
| 5 | Flag block | 18 to 16 | |
| 6 | Concat, then Mix: Linear, LN, GELU | 64 + 32 + 32 + 24 + 16 = 168 to 96 | `h` |

* **Parameters:** about 58,400 per tower, about 181,000 total.
* **Honest view:** Mol2vec is your weakest family (test PR-AUC 0.40 to 0.50). With two fingerprint branches added, the model will largely behave like the MACCS family. Build it last, and drop it from any final blend if it does not beat the MACCS family on validation.

---

## 5. Transformer families

For both transformers, the encoder stays **frozen**. The projection, cross-attention and pooling layers are kept exactly as they are. What is added is the tower mix after pooling, and the shared head in place of your 609-d classifier.

### 5.1 MoLFormer-XL

```
            ┌────────────┐                ┌──────────────────┐
            │ API SMILES │                │ EXCIPIENT SMILES │
            └────────────┘                └──────────────────┘
                   │                                │
                   ▼                                ▼
         ┌──────────────────┐            ┌─────────────────────┐
         │   MoLFormer-XL   │            │     MoLFormer-XL    │
         │ (frozen, shared) │            │ (same frozen model) │
         │  tokens [L,768]  │            │    tokens [L,768]   │
         └──────────────────┘            └─────────────────────┘
                   │                                │
                   ▼                                ▼
     ┌───────────────────────────┐    ┌───────────────────────────┐
     │       API projection      │    │       EXC projection      │
     │ 768->256 LN GELU Drop .15 │    │ 768->256 LN GELU Drop .15 │
     │      256->128 LN GELU     │    │      256->128 LN GELU     │
     └───────────────────────────┘    └───────────────────────────┘
                   │                                │
                   └───────────────┬────────────────┘
                                   │
                                   ▼
       ┌───────────────────────────────────────────────────────┐
       │  Prepend CLS slot, then bidirectional cross-attention │
       │   8 heads, d=128, dropout .15, residual + LayerNorm   │
       │ API queries -> EXC tokens ; EXC queries -> API tokens │
       └───────────────────────────────────────────────────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
 ┌──────────────────────────────┐    ┌──────────────────────────────┐
 │     API: gated attention     │    │     EXC: gated attention     │
 │ pool over atoms/tokens [128] │    │ pool over atoms/tokens [128] │
 │     + global slot [128]      │    │     + global slot [128]      │
 │    concat 256->Linear 128    │    │    concat 256->Linear 128    │
 │       LN|GELU|Drop .15       │    │       LN|GELU|Drop .15       │
 └──────────────────────────────┘    └──────────────────────────────┘
                 │                                   │
                 └─────────────────┬─────────────────┘
                                   │
                                   ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ TOWER MIX  (drawn once; API and EXC each have their own weights) │
 └──────────────────────────────────────────────────────────────────┘
                                   │
       ┌──────────────────┬────────┴─────────┬──────────────────┐
       ▼                  ▼                  ▼                  ▼
┌─────────────┐  ┌────────────────┐  ┌───────────────┐  ┌───────────────┐
│    struct   │  │     MACCS      │  │   RDKit desc  │  │   Chem flags  │
│    [128]    │  │    166 bits    │  │  12 z-scored  │  │   18 yes/no   │
│ from fusion │  │ Linear 166->32 │  │ Linear 12->24 │  │ Linear 18->16 │
│             │  │   LN | GELU    │  │   LN | GELU   │  │      GELU     │
└─────────────┘  └────────────────┘  └───────────────┘  └───────────────┘
       │                  │                  │                  │
       └──────────────────┴────────┬─────────┴──────────────────┘
                                   │
                                   ▼
                           ┌──────────────┐
                           │ CONCAT [200] │
                           └──────────────┘
                                   │
                                   ▼
                  ┌────────────────────────────────┐
                  │ Linear 200 -> 96  |  LN | GELU │
                  └────────────────────────────────┘
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │             h  [96]             │
                  │ feeds the head as h_API / h_EXC │
                  └─────────────────────────────────┘
```

| # | Layer | In to Out | Status |
|---|---|---|---|
| 1 | MoLFormer-XL, frozen, shared by API and excipient | SMILES to tokens [L, 768] | KEEP |
| 2 | Projection (separate weights per side): Linear, LN, GELU, Drop .15 | 768 to 256 | KEEP |
| 3 | Projection: Linear, LN, GELU | 256 to 128 | KEEP |
| 4 | CLS slot from pooled vector, then bidirectional cross-attention, 8 heads, residual + LN | [L+1, 128] each side | KEEP |
| 5 | Gated attention pooling over tokens (128 to 128 tanh x sigmoid, then score) | [L, 128] to 128 | KEEP |
| 6 | Concat global slot + pooled, Linear, LN, GELU, Drop .15 | 256 to 128 | KEEP |
| 7 | MACCS side block: Linear, LN, GELU | 166 to 32 | **NEW** |
| 8 | Descriptor block: Linear, LN, GELU | 12 to 24 | **CHANGED** (was 21 to 32 to 24) |
| 9 | Flag block: Linear, GELU | 18 to 16 | **NEW** |
| 10 | Concat | 128 + 32 + 24 + 16 = 200 | **NEW** |
| 11 | Mix: Linear, LN, GELU | 200 to 96 | **NEW** |
| 12 | PGB head | 426 to 1 | **REPLACES** 609 to 128 to 64 to 1 |

* **Trainable parameters:** about 726,000 existing + about 116,000 new = about 842,000.
* **Why MACCS on the side:** MoLFormer reads SMILES characters. Its 24-pair ROC is 0.47 to 0.71 across your runs (0.61 for the chosen one), well below your D-MPNN and GAT runs (0.76 to 0.87). That suggests it struggles with salts and unusual excipients, though 24 pairs is a small sample. Flags and MACCS give it functional-group evidence in a different form.
* **What to watch:** val PR-AUC is already about 0.72, so look for test false alarms (38 now, doc 14), test MCC (0.56 now, doc 0.61) and 24-pair ROC.

### 5.2 ChemBERTa-77M-MTR

```
            ┌────────────┐                ┌──────────────────┐
            │ API SMILES │                │ EXCIPIENT SMILES │
            └────────────┘                └──────────────────┘
                   │                                │
                   ▼                                ▼
         ┌───────────────────┐           ┌─────────────────────┐
         │ ChemBERTa-77M-MTR │           │  ChemBERTa-77M-MTR  │
         │  (frozen, shared) │           │ (same frozen model) │
         │   tokens [L,384]  │           │    tokens [L,384]   │
         └───────────────────┘           └─────────────────────┘
                   │                                │
                   ▼                                ▼
     ┌───────────────────────────┐    ┌───────────────────────────┐
     │       API projection      │    │       EXC projection      │
     │ 384->256 LN GELU Drop .15 │    │ 384->256 LN GELU Drop .15 │
     │      256->128 LN GELU     │    │      256->128 LN GELU     │
     └───────────────────────────┘    └───────────────────────────┘
                   │                                │
                   └───────────────┬────────────────┘
                                   │
                                   ▼
       ┌───────────────────────────────────────────────────────┐
       │  Prepend CLS slot, then bidirectional cross-attention │
       │   8 heads, d=128, dropout .15, residual + LayerNorm   │
       │ API queries -> EXC tokens ; EXC queries -> API tokens │
       └───────────────────────────────────────────────────────┘
                                   │
                   ┌───────────────┴────────────────┐
                   ▼                                ▼
     ┌───────────────────────────┐    ┌───────────────────────────┐
     │       API: CLS slot       │    │       EXC: CLS slot       │
     │ (index 0 after attention) │    │ (index 0 after attention) │
     │           [128]           │    │           [128]           │
     └───────────────────────────┘    └───────────────────────────┘
                   │                                │
                   └───────────────┬────────────────┘
                                   │
                                   ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ TOWER MIX  (drawn once; API and EXC each have their own weights) │
 └──────────────────────────────────────────────────────────────────┘
                                   │
       ┌──────────────────┬────────┴─────────┬──────────────────┐
       ▼                  ▼                  ▼                  ▼
┌─────────────┐  ┌────────────────┐  ┌───────────────┐  ┌───────────────┐
│    struct   │  │   Morgan r2    │  │   RDKit desc  │  │   Chem flags  │
│    [128]    │  │    512 bits    │  │  12 z-scored  │  │   18 yes/no   │
│ from fusion │  │ Linear 512->32 │  │ Linear 12->24 │  │ Linear 18->16 │
│             │  │   LN | GELU    │  │   LN | GELU   │  │      GELU     │
└─────────────┘  └────────────────┘  └───────────────┘  └───────────────┘
       │                  │                  │                  │
       └──────────────────┴────────┬─────────┴──────────────────┘
                                   │
                                   ▼
                           ┌──────────────┐
                           │ CONCAT [200] │
                           └──────────────┘
                                   │
                                   ▼
                  ┌────────────────────────────────┐
                  │ Linear 200 -> 96  |  LN | GELU │
                  └────────────────────────────────┘
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │             h  [96]             │
                  │ feeds the head as h_API / h_EXC │
                  └─────────────────────────────────┘
```

| # | Layer | In to Out | Status |
|---|---|---|---|
| 1 | ChemBERTa, frozen, shared | SMILES to tokens [L, 384] | KEEP |
| 2 | Projection (per side): Linear, LN, GELU, Drop .15 then Linear, LN, GELU | 384 to 256 to 128 | KEEP |
| 3 | CLS slot + bidirectional cross-attention, 8 heads, residual + LN | [L+1, 128] | KEEP |
| 4 | CLS slot (index 0) read out | 128 | KEEP |
| 5 | Morgan side block: Linear, LN, GELU | 512 to 32 | **NEW** |
| 6 | Descriptor block | 12 to 24 | **CHANGED** |
| 7 | Flag block | 18 to 16 | **NEW** |
| 8 | Concat, then Mix: Linear, LN, GELU | 200 to 96 | **NEW** |
| 9 | PGB head | 426 to 1 | **REPLACES** classifier |

* **Trainable parameters:** about 397,000 existing + about 138,000 new = about 535,000.
* **Why CLS pooling and Morgan:** in your runs CLS pooling beat gated pooling for ChemBERTa on test and was about equal on val. This model is trained on physicochemical targets, so its vectors describe bulk properties. Morgan bits add local substructure information it lacks. Gated pooling stays as an option if CLS is not the best on validation after the head is added.

---

## 6. Graph-neural-network families

All three keep their encoders **frozen**. The graph encoders return one vector per atom, and the pipeline builds a padded atom sequence for cross-attention. A MACCS side block is added to each because a graph with 2 atoms and no bonds (`O=[Mg]`) carries almost no structure. The flags cover that case.

### 6.1 Pretrained GIN (Hu et al., 5 layers, 300-d)

```
            ┌────────────┐                ┌──────────────────┐
            │ API SMILES │                │ EXCIPIENT SMILES │
            └────────────┘                └──────────────────┘
                   │                                │
                   ▼                                ▼
    ┌────────────────────────────┐   ┌────────────────────────────┐
    │       Pretrained GIN       │   │       Pretrained GIN       │
    │     (frozen, 5 layers)     │   │     (frozen, 5 layers)     │
    │ atoms [L,300] + mean [300] │   │ atoms [L,300] + mean [300] │
    └────────────────────────────┘   └────────────────────────────┘
                   │                                │
                   ▼                                ▼
     ┌───────────────────────────┐    ┌───────────────────────────┐
     │         projection        │    │         projection        │
     │ 300->256 LN GELU Drop .15 │    │ 300->256 LN GELU Drop .15 │
     │      256->128 LN GELU     │    │      256->128 LN GELU     │
     └───────────────────────────┘    └───────────────────────────┘
                   │                                │
                   ▼                                ▼
     ┌──────────────────────────┐     ┌──────────────────────────┐
     │  gated attn pool [128]   │     │  gated attn pool [128]   │
     │  + projected mean [128]  │     │  + projected mean [128]  │
     │ concat 256 -> Linear 128 │     │ concat 256 -> Linear 128 │
     │   LN | GELU | Drop .15   │     │   LN | GELU | Drop .15   │
     └──────────────────────────┘     └──────────────────────────┘
                   │                                │
                   └───────────────┬────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ NO cross-attention: the two molecules only meet in the head │
    └─────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ TOWER MIX  (drawn once; API and EXC each have their own weights) │
 └──────────────────────────────────────────────────────────────────┘
                                   │
       ┌──────────────────┬────────┴─────────┬──────────────────┐
       ▼                  ▼                  ▼                  ▼
┌─────────────┐  ┌────────────────┐  ┌───────────────┐  ┌───────────────┐
│    struct   │  │     MACCS      │  │   RDKit desc  │  │   Chem flags  │
│    [128]    │  │    166 bits    │  │  12 z-scored  │  │   18 yes/no   │
│ from fusion │  │ Linear 166->32 │  │ Linear 12->24 │  │ Linear 18->16 │
│             │  │   LN | GELU    │  │   LN | GELU   │  │      GELU     │
└─────────────┘  └────────────────┘  └───────────────┘  └───────────────┘
       │                  │                  │                  │
       └──────────────────┴────────┬─────────┴──────────────────┘
                                   │
                                   ▼
                           ┌──────────────┐
                           │ CONCAT [200] │
                           └──────────────┘
                                   │
                                   ▼
                  ┌────────────────────────────────┐
                  │ Linear 200 -> 96  |  LN | GELU │
                  └────────────────────────────────┘
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │             h  [96]             │
                  │ feeds the head as h_API / h_EXC │
                  └─────────────────────────────────┘
```

| # | Layer | In to Out | Status |
|---|---|---|---|
| 1 | Pretrained GIN, frozen, shared | SMILES to atoms [L, 300] + mean [300] | KEEP |
| 2 | Projection (per side): Linear, LN, GELU, Drop .15 then Linear, LN, GELU | 300 to 256 to 128 | KEEP |
| 3 | Gated attention pooling over atoms | [L, 128] to 128 | **NEW wiring** (module exists; concat fusion does not use it today) |
| 4 | Concat pooled + projected mean, Linear, LN, GELU, Drop .15 | 256 to 128 | **NEW wiring** |
| 5 | MACCS side block | 166 to 32 | **NEW** |
| 6 | Descriptor block, flag block | 12 to 24, 18 to 16 | **CHANGED / NEW** |
| 7 | Concat, then Mix: Linear, LN, GELU | 200 to 96 | **NEW** |
| 8 | PGB head | 426 to 1 | **REPLACES** classifier |

* **Trainable parameters:** about 354,000 existing + about 116,000 new = about 470,000.
* **Why late fusion (no cross-attention):** your GIN concat and cross-attention runs score the same on val, so cross-attention added parameters without a gain. The bilinear term in the head now does the interaction. Plain mean pooling of atoms is replaced by gated attention pooling, which lets the model weight reactive atoms.

### 6.2 Pretrained GAT (Stanford, context-prediction, 5 layers, 300-d)

```
            ┌────────────┐                ┌──────────────────┐
            │ API SMILES │                │ EXCIPIENT SMILES │
            └────────────┘                └──────────────────┘
                   │                                │
                   ▼                                ▼
      ┌─────────────────────────┐      ┌─────────────────────────┐
      │ Stanford GAT (ctx-pred) │      │ Stanford GAT (ctx-pred) │
      │     (frozen, shared)    │      │   (same frozen model)   │
      │      tokens [L,300]     │      │      tokens [L,300]     │
      └─────────────────────────┘      └─────────────────────────┘
                   │                                │
                   ▼                                ▼
     ┌───────────────────────────┐    ┌───────────────────────────┐
     │       API projection      │    │       EXC projection      │
     │ 300->256 LN GELU Drop .15 │    │ 300->256 LN GELU Drop .15 │
     │      256->128 LN GELU     │    │      256->128 LN GELU     │
     └───────────────────────────┘    └───────────────────────────┘
                   │                                │
                   └───────────────┬────────────────┘
                                   │
                                   ▼
       ┌───────────────────────────────────────────────────────┐
       │  Prepend CLS slot, then bidirectional cross-attention │
       │   8 heads, d=128, dropout .15, residual + LayerNorm   │
       │ API queries -> EXC tokens ; EXC queries -> API tokens │
       └───────────────────────────────────────────────────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
 ┌──────────────────────────────┐    ┌──────────────────────────────┐
 │     API: gated attention     │    │     EXC: gated attention     │
 │ pool over atoms/tokens [128] │    │ pool over atoms/tokens [128] │
 │     + global slot [128]      │    │     + global slot [128]      │
 │    concat 256->Linear 128    │    │    concat 256->Linear 128    │
 │       LN|GELU|Drop .15       │    │       LN|GELU|Drop .15       │
 └──────────────────────────────┘    └──────────────────────────────┘
                 │                                   │
                 └─────────────────┬─────────────────┘
                                   │
                                   ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ TOWER MIX  (drawn once; API and EXC each have their own weights) │
 └──────────────────────────────────────────────────────────────────┘
                                   │
       ┌──────────────────┬────────┴─────────┬──────────────────┐
       ▼                  ▼                  ▼                  ▼
┌─────────────┐  ┌────────────────┐  ┌───────────────┐  ┌───────────────┐
│    struct   │  │     MACCS      │  │   RDKit desc  │  │   Chem flags  │
│    [128]    │  │    166 bits    │  │  12 z-scored  │  │   18 yes/no   │
│ from fusion │  │ Linear 166->32 │  │ Linear 12->24 │  │ Linear 18->16 │
│             │  │   LN | GELU    │  │   LN | GELU   │  │      GELU     │
└─────────────┘  └────────────────┘  └───────────────┘  └───────────────┘
       │                  │                  │                  │
       └──────────────────┴────────┬─────────┴──────────────────┘
                                   │
                                   ▼
                           ┌──────────────┐
                           │ CONCAT [200] │
                           └──────────────┘
                                   │
                                   ▼
                  ┌────────────────────────────────┐
                  │ Linear 200 -> 96  |  LN | GELU │
                  └────────────────────────────────┘
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │             h  [96]             │
                  │ feeds the head as h_API / h_EXC │
                  └─────────────────────────────────┘
```

| # | Layer | In to Out | Status |
|---|---|---|---|
| 1 | Pretrained GAT, frozen, shared | SMILES to atoms [L, 300] | KEEP |
| 2 | Projection (per side) | 300 to 256 to 128 | KEEP |
| 3 | CLS slot + bidirectional cross-attention, 8 heads, residual + LN | [L+1, 128] | KEEP |
| 4 | Gated attention pooling + global slot, Linear, LN, GELU, Drop .15 | 256 to 128 | KEEP |
| 5 | MACCS side block, descriptor block, flag block | 166 to 32, 12 to 24, 18 to 16 | **NEW / CHANGED** |
| 6 | Concat, then Mix: Linear, LN, GELU | 200 to 96 | **NEW** |
| 7 | PGB head | 426 to 1 | **REPLACES** classifier |

* **Trainable parameters:** about 487,000 existing + about 116,000 new = about 602,000.
* **Priority for this family:** it has the best 24-pair ranking you have (ROC 0.87) and among the highest test false-alarm counts (77 of 595 compatible pairs for the chosen run, 91 for a concat run, at their val-F1 thresholds). Sampler removal, asymmetric focal loss and the prior offset all act directly on that fault. The pipeline itself is not changed, so the ranking strength should be preserved.

### 6.3 D-MPNN (CheMeleon)

```
             ┌────────────┐              ┌──────────────────┐
             │ API SMILES │              │ EXCIPIENT SMILES │
             └────────────┘              └──────────────────┘
                    │                              │
                    ▼                              ▼
        ┌───────────────────────┐      ┌───────────────────────┐
        │    CheMeleon D-MPNN   │      │    CheMeleon D-MPNN   │
        │    (frozen, shared)   │      │  (same frozen model)  │
        │      tokens [L,D]     │      │      tokens [L,D]     │
        │ + feature dropout .20 │      │ + feature dropout .20 │
        └───────────────────────┘      └───────────────────────┘
                    │                              │
                    ▼                              ▼
       ┌─────────────────────────┐    ┌─────────────────────────┐
       │      API projection     │    │      EXC projection     │
       │ D->256 LN GELU Drop .15 │    │ D->256 LN GELU Drop .15 │
       │     256->128 LN GELU    │    │     256->128 LN GELU    │
       └─────────────────────────┘    └─────────────────────────┘
                    │                              │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
       ┌───────────────────────────────────────────────────────┐
       │  Prepend CLS slot, then bidirectional cross-attention │
       │   8 heads, d=128, dropout .15, residual + LayerNorm   │
       │ API queries -> EXC tokens ; EXC queries -> API tokens │
       └───────────────────────────────────────────────────────┘
                                   │
                   ┌───────────────┴────────────────┐
                   ▼                                ▼
     ┌───────────────────────────┐    ┌───────────────────────────┐
     │       API: CLS slot       │    │       EXC: CLS slot       │
     │ (index 0 after attention) │    │ (index 0 after attention) │
     │           [128]           │    │           [128]           │
     └───────────────────────────┘    └───────────────────────────┘
                   │                                │
                   └───────────────┬────────────────┘
                                   │
                                   ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ TOWER MIX  (drawn once; API and EXC each have their own weights) │
 └──────────────────────────────────────────────────────────────────┘
                                   │
       ┌──────────────────┬────────┴─────────┬──────────────────┐
       ▼                  ▼                  ▼                  ▼
┌─────────────┐  ┌────────────────┐  ┌───────────────┐  ┌───────────────┐
│    struct   │  │     MACCS      │  │   RDKit desc  │  │   Chem flags  │
│    [128]    │  │    166 bits    │  │  12 z-scored  │  │   18 yes/no   │
│ from fusion │  │ Linear 166->32 │  │ Linear 12->24 │  │ Linear 18->16 │
│             │  │   LN | GELU    │  │   LN | GELU   │  │      GELU     │
└─────────────┘  └────────────────┘  └───────────────┘  └───────────────┘
       │                  │                  │                  │
       └──────────────────┴────────┬─────────┴──────────────────┘
                                   │
                                   ▼
                           ┌──────────────┐
                           │ CONCAT [200] │
                           └──────────────┘
                                   │
                                   ▼
                  ┌────────────────────────────────┐
                  │ Linear 200 -> 96  |  LN | GELU │
                  └────────────────────────────────┘
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │             h  [96]             │
                  │ feeds the head as h_API / h_EXC │
                  └─────────────────────────────────┘
```

| # | Layer | In to Out | Status |
|---|---|---|---|
| 1 | CheMeleon message passing, frozen, shared | SMILES to atoms [L, D] | KEEP |
| 2 | **Feature dropout 0.20 on the encoder output** | D to D | **NEW** |
| 3 | Projection (per side): Linear, LN, GELU, Drop .15 then Linear, LN, GELU | D to 256 to 128 | KEEP |
| 4 | CLS slot + bidirectional cross-attention, 8 heads, residual + LN | [L+1, 128] | KEEP |
| 5 | CLS slot (index 0) read out | 128 | KEEP |
| 6 | MACCS side block, descriptor block, flag block | 166 to 32, 12 to 24, 18 to 16 | **NEW / CHANGED** |
| 7 | Concat, then Mix: Linear, LN, GELU | 200 to 96 | **NEW** |
| 8 | PGB head | 426 to 1 | **REPLACES** classifier |

* **D** is the native CheMeleon width. The code reads it from the checkpoint; I believe it is 2048 in the public release, but I could not open your checkpoint, so confirm it. All dimension counts below the projection do not depend on it.
* **Trainable parameters:** about 1,249,000 existing (if D = 2048, the two 2048-to-256 projections alone are about 1.05 million) + about 116,000 new = about 1.36 million. This is the largest model here, and the one most at risk of overfitting about 2,000 rows. The extra feature dropout and the 96-d bottleneck are there for that reason.
* **Why CLS pooling:** your CLS runs scored higher than gated pooling on test (0.620 to 0.625 vs 0.593 to 0.608).
* **Duplicates:** the concat variants of this encoder are identical to each other (finding F5). Do not rebuild them.

### 6.4 Encoders in your code with no results yet

AttentiveFP, GINE, GATv2, PNA and the from-scratch D-MPNN all emit 300-wide atom vectors (`gnn_hidden_dim` = 300). They have no metric folders, so I have no baseline for them. If you run them, use the **GAT template in section 6.2 exactly** (cross-attention, gated pooling, MACCS side block, PGB head), changing only the encoder box. Do this only after the nine families above are validated.

---

## 7. All architectures at a glance

| Family | Tower input | Mix input | Tower output | Pair vector | New trainable parameters (whole model, approx.) |
|---|---|---|---|---|---|
| MACCS | MACCS 166, Morgan 512, desc 12, flags 18 | 168 | 96 | 426 | 186,000 |
| Morgan | Morgan 1024, MACCS 166, desc 12, flags 18 | 168 | 96 | 426 | 252,000 |
| PubChem | PubChem 881, Morgan 512, desc 12, flags 18 | 168 | 96 | 426 | 278,000 |
| Mol2vec | Mol2vec 300, MACCS 166, Morgan 512, desc 12, flags 18 | 168 | 96 | 426 | 181,000 |
| MoLFormer | tokens 768 to struct 128, MACCS 166, desc 12, flags 18 | 200 | 96 | 426 | 842,000 (incl. existing layers) |
| ChemBERTa | tokens 384 to struct 128, Morgan 512, desc 12, flags 18 | 200 | 96 | 426 | 535,000 |
| GIN | atoms 300 to struct 128, MACCS 166, desc 12, flags 18 | 200 | 96 | 426 | 470,000 |
| GAT | atoms 300 to struct 128, MACCS 166, desc 12, flags 18 | 200 | 96 | 426 | 602,000 |
| D-MPNN | atoms D to struct 128, MACCS 166, desc 12, flags 18 | 200 | 96 | 426 | 1,365,000 (if D = 2048) |

The parameter counts are hand-computed from the layer tables above (frozen encoders excluded) and are approximate.

---

## 8. Training recipe

| Item | Fixed-vector families | Transformer and graph families |
|---|---|---|
| Optimiser | AdamW, lr 1.2e-3, weight decay 2e-3 | AdamW with two learning rates: **new layers** (tower mix, side blocks, head) 1e-3; **kept layers** (projection, cross-attention, pooling) 3e-4. Weight decay 2e-3. |
| Batch size | 64 | 64 |
| Max epochs | 40 | 60 |
| Scheduler | halve lr after 4 epochs without val PR-AUC gain | same |
| Early stop | 7 epochs without +0.002 val PR-AUC | same |
| Gradient clip | 1.0 | 1.0 |
| Loss | asymmetric focal, gamma+ 1.2, gamma- 2.5, one `pos_weight` | same |
| Sampler | none | none |
| Encoders | n/a | frozen |
| Seeds | at least 3, report mean and spread | at least 3 |
| Threshold | val MCC with TNR floor 0.97, frozen before looking at test or 24-pair | same |

Starting values for the transformer and graph learning rates are my choice; the doc only gives 1.2e-3 for its small all-new network. Tune them on validation only.

**A second, a-priori operating point.** The doc used a "screen" cut of 0.60 picked while looking at the 24 pairs. To avoid that, define the screen cut on validation: the highest threshold whose val recall is at least 0.75. Both cuts are then fixed before any test or 24-pair number is read.

---

## 9. How to judge whether it worked

**9.1 What each set can and cannot tell you**

| Set | Positives / negatives | What one flipped pair does | Use |
|---|---|---|---|
| Validation | 76 / 766 | small | Selection only: architecture choices, early stop, threshold |
| Official test | 77 / 595 | small, but your own ablation moves test PR-AUC between 0.50 and 0.72 for one architecture across 8 loss/sampler settings | Report. Compare means across seeds. |
| 24-pair named set | 12 / 12 | accuracy moves about 4 points and MCC about 0.08 per pair | Report PR-AUC and ROC-AUC first. Never tune on it. |

**9.2 Design targets (hypotheses, not measured)**

These come from the size of the gain the doc reported, discounted for families that are weaker or already strong.

| Family | Test PR-AUC now to target | Test false alarms now to target | 24-pair ROC now to target |
|---|---|---|---|
| MACCS | 0.57 to 0.65–0.73 | 42 to 25 or fewer | 0.75 to 0.85 or more (part of the gain is fixing finding F3) |
| Morgan | 0.55 to 0.62–0.70 | 35 to 25 or fewer | 0.81 hold at 0.85 or more |
| PubChem | 0.60 to 0.64–0.70 | 33 to 25 or fewer | 0.79 to 0.85 or more |
| Mol2vec | 0.50 to 0.55–0.63 | 58 to 40 or fewer | 0.62 to 0.70 or more |
| MoLFormer | 0.69 to 0.72–0.76 | 38 to 25 or fewer | 0.61 to 0.75 or more |
| ChemBERTa | 0.66 to 0.66–0.71 | 53 to 30 or fewer | 0.63 to 0.70 or more |
| GIN | 0.56 to 0.60–0.67 | 53 to 35 or fewer | 0.70 to 0.75 or more |
| GAT | 0.58 to 0.62–0.70 | 77 to 40 or fewer | 0.87 hold at 0.85 or more |
| D-MPNN | 0.62 to 0.66–0.72 | 35 to 25 or fewer | 0.81 hold at 0.85 or more |

**9.3 Acceptance rule.** A family counts as improved when, averaged over seeds, (a) val PR-AUC or val MCC is not worse than today, (b) test PR-AUC and test MCC are higher and test false alarms are lower, and (c) 24-pair PR-AUC and ROC-AUC are not lower. A gain smaller than about 0.03 PR-AUC on test is within what a different seed or loss setting can produce in your own ablation, so I would not read it as a real win.

---

## 10. Build order and ablation ladder

Build the shared head once, check it on the MACCS family (the closest to the doc), then reuse it. Suggested order: MACCS, Morgan, PubChem, MoLFormer, D-MPNN, GAT, GIN, ChemBERTa, Mol2vec.

For each family, climb this ladder and stop at the first step where validation stops improving. This shows which ingredient helps and which does not.

```
             ┌───────────────────────────────────────────────┐
             │ S0  your current best pipeline for the family │
             │             (numbers in section 2)            │
             └───────────────────────────────────────────────┘
                                     │
                                     ▼
     ┌──────────────────────────────────────────────────────────────┐
     │ S1  remove sampler, use asymmetric focal + single pos_weight │
     │                        keep old head                         │
     └──────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
      ┌────────────────────────────────────────────────────────────┐
      │ S2  + excipient prior as logit offset (OOF for train rows) │
      └────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │ S3  + family tower (extra fingerprint, 12 desc, 18 flags, 96-d) │
    └─────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
  ┌────────────────────────────────────────────────────────────────────┐
  │ S4  + bilinear rank-16 + gate + mechanism flags   = full PGB model │
  └────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Compare each step on VAL first; read TEST and 24-pair only as a report │
└────────────────────────────────────────────────────────────────────────┘
```

| If this happens | Likely cause | Action |
|---|---|---|
| S1 raises val PR-AUC but test PR-AUC falls | Sampler removal alone, as in your BCE ablation | Continue to S2; the prior offset should recover calibration. If not, try a milder 1:3 sampler once, still with a single `pos_weight`. |
| S2 helps val and test but 24-pair falls | Half of 24-pair excipients lack an exact prior | Check the stereo-blind key (A2) is in use; check the gate is half-open for unseen excipients |
| S3 helps nothing for string or graph encoders | Side blocks are redundant with the encoder | Keep flags only, drop the fingerprint side block |
| S4 helps val but not test | Bilinear over-fits about 190 positives | Lower the rank from 16 to 8; raise head dropout |

---

## 11. Optional last step: blending

Your repo already has average, weighted-average and logistic-stacker ensembling. After the families are rebuilt, a blend of one transformer (MoLFormer), one graph model (D-MPNN or GAT) and one fingerprint model (MACCS) is the most promising mix: transformers lead on val and test, graph models lead on the 24-pair ranking, and fingerprints are the cheapest. Pick blend weights on validation only.

---

## 12. Known weak spots (carried over from the doc, plus mine)

* **Unseen excipients** (magnesium aluminum silicate, calcium phosphate, talc in the 24-pair file): the prior is the global rate and the gate is half-open. Expect scores near the middle.
* **A rare mechanism with a "safe" excipient** (acetazolamide with mannitol in the doc): the prior fights the pair evidence. This is an honest limit of any prior-based design.
* **A hydrate-sharp prior** (lactose vs lactose monohydrate): it helped on the 24 pairs but may not generalise to a new hydrate pair. The family prior is given to the head so the network can learn to rely on it.
* **Fixed-vector 24-pair numbers** stay unreliable for PubChem and Mol2vec until their tables cover the new molecules (finding F3).
* **Only about 190 positives in train.** Every extra learned interaction is a chance to over-fit. This is why the rank is 16, the tower output is 96, the encoders are frozen, and the ladder in section 10 exists.
