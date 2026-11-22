# Data audit and curation of the source dataset

Source: `data/raw/start_dataset.csv`, byte-identical to `dataset.csv` of the *Towards safer…* repository
(github.com/phuongnvp/drug_exp_compatibility; sha256 `583f19c4…f8cdb`). It has 3544 rows (533 API CIDs, 336 excipient
CIDs) and 344 incompatible rows (9.7 %). The raw file is never modified.

Reproduce everything here with:

```bash
python scripts/data/curate_dataset.py
```

| Input | Role |
|---|---|
| `data/curation/molecule_curation.csv` | hand-curated decision per molecule and role: `remove`, `repair` or `flag`, with a category and a one-line rationale |
| `data/curation/pubchem_snapshot_2026-11-11.csv` | frozen PubChem title / formula / SMILES / InChIKey for all 850 CIDs, so the run needs no network |

| Output | Content |
|---|---|
| `data/processed/clean.csv` | **default curated set**: 3396 rows, 246 incompatible (7.2 %), 508 APIs, 270 excipients |
| `data/processed/clean_strict.csv` | also drops non-(drug, excipient) pairs and wrong structures: 3061 rows, 206 incompatible (6.7 %) |
| `data/curation/audit/audit_log.txt` | every number in this document |
| `data/curation/audit/*.csv` | removed, repaired and collapsed rows, parent-level label conflicts, and a per-molecule table |

Both outputs keep `row_id` (the raw row index) and the `API_flags` / `Excipient_flags` columns, so you can trace any
row back to the raw file and filter it further.

## 1. Duplicates

| Check | Result |
|---|---|
| Exact duplicate rows | 0 |
| Duplicate (API_CID, Excipient_CID) pairs | 0 |
| Self pairs (API = excipient) | 0 |
| Pairs in both orders | 1: ethylene oxide + alumina, with each molecule in both roles (removed: ethylene oxide is not a drug) |
| CIDs used as both API and excipient | 19 |
| **Pairs with identical structures under different CIDs** | **26 groups, 53 rows. 27 rows collapsed, no label conflicts** |
| Parent-level duplicates (same pair up to salt / hydrate) | 10 pairs with **conflicting labels**. Kept and reported, see §4 |

There are no duplicates at the CID level, but there are duplicates at the structure level. Every SMILES in the file
is the PubChem *connectivity* SMILES: only 1 of 838 has a stereo mark. So enantiomers, epimers and anomers get the same
input. Mannitol = sorbitol; β-lactose = α-D-lactose = maltose = cellobiose ("DEAE-cellulose"); lansoprazole =
dexlansoprazole; cetirizine = levocetirizine; and omeprazole magnesium appears under two CIDs. A pair that appears twice
under different CIDs is the same training example twice, so we keep one row of each group.

## 2. Format and identifier errors

* **Whitespace.** 4 rows have a leading or trailing space in the excipient SMILES (phenol, MgCO₃, Mg stearate,
  NaCMC). The same CID then has two different SMILES strings. Stripped.
* **CID ↔ SMILES consistency.** Every SMILES matches the current PubChem connectivity SMILES of its CID. The one
  exception is the only stereo SMILES in the file (CID 214350).
* **Wrong structures from name → CID lookup.** These are repaired:
  * *Povidone* (CID 131751496, 86 rows). PubChem's "Tolpovidone I 131" record writes the polymer ends as **phosphorus
    atoms** (`C1CC(=O)N(C1)C(CP)P`), so every fingerprint model saw P in povidone. We replace it with the H-capped
    repeat unit `CCN1CCCC1=O`. We deliberately do *not* merge it with N-vinylpyrrolidone (CID 6917, used for the other
    PVP grade): clenbuterol, indomethacin, ketoprofen and ranitidine are each tested against both, with opposite
    labels. Merging the two creates 4 label conflicts.
  * *Hydroxypropyl cellulose* (CID 123706, 55 rows). The lookup matched a PubChem synonym ("2-Hydroxypropyl
    cellulose") on an unrelated aniline, **4-(1-aminopropyl)-N,N,3-trimethylaniline**. We replace it with the
    hydroxypropyl-glucose oligomer already in the data (CID 71306830).
  * *Lactose, anhydrous* (CID 3037558, 12 rows) is encoded as the open-chain form. We replace it with the ring form.
* **Unidentifiable structures** (removed): the mercapto radical `[SH]`, methylidynephosphine `C#P` / `[C-]#P` (most
  likely a misread "CP"), and 2-(2,4-diaminophenoxy)ethanol (a hair-dye intermediate in 7 FDA-label rows).

## 3. Entities that are not an API or an excipient

**Removed** (121 rows, 98 incompatible):

| Category | Rows (incompatible) | Examples |
|---|---|---|
| Laboratory reagent | 67 (61) | KMnO₄, ozone, H₂O₂, H₂SO₄, KClO₃, CrO₃, PbO₂, nitrates, LiAlH₄, KOtBu, formaldehyde, glyceraldehyde, nitroalkanes, ethylene oxide, **diazomethane as an "API"** |
| Counter-ion-free ion | 23 (21) | phosphate, sulfate, carbonate, citrate, Fe²⁺/Fe³⁺, Cu²⁺, Ag⁺, iodide; maleate(2−) and salicylate as "APIs" |
| Elemental substance | 13 (12) | Al, Zn, Li, K, Ag, Hg, Cu, P |
| Invalid / unidentifiable structure | 11 (1) | see §2 |
| Research chemical | 5 (1) | BANA, a nitroacridine, a netropsin analogue, a cyanine dye |
| Degradant | 2 (2) | 5-hydroxymethylfurfural |

These rows are 3.4 % of the data but **28.5 % of all incompatible rows**. Most are forced-degradation or reactivity
tests reported in the literature, not formulation compatibility. A model trained with them partly learns
"strong oxidiser → incompatible".

**Flagged** (kept in `clean.csv`; the first five are dropped in `clean_strict.csv`):

| Flag | Rows | Meaning |
|---|---|---|
| `non_drug_api` | 36 | excipient or chemical in the API column (benzalkonium, bronopol, BHT, glycerol, aspartame, CaCl₂, Ca(OH)₂, digitonin…) |
| `multi_api` | 56 | fixed-dose combination encoded as one mixture SMILES (sacubitril/valsartan, ivacaftor/tezacaftor, a 4-drug "Quad tablet"…) |
| `drug_as_excipient` | 18 | a drug in the excipient column (mitomycin C, warfarin Na, chloramphenicol, kanamycin…): really a drug–drug pair |
| `approximate_structure` | 228 | structure is chemically wrong for the material: iron oxides as neutral `[Fe]` + water, NaCMC as open-chain glucose + acetic acid + neutral Na, Cremophor EL as a C₅ polyol, HEC with hydroxypropyl groups |
| `colourant_uncertain` | 9 | stains or hair dyes (brilliant green, acridine orange, HC Blue 1…), probably name-lookup errors for a lake colour |
| `polymer_surrogate` | 838 | polymer given as a monomer or oligomer (PVP → NVP, PVA → vinyl alcohol, PEG → ethylene glycol, cellulose → cellobiose…). Inherent to SMILES models; reported, not dropped |
| `strong_base`, `processing_solvent` | 29, 40 | NaOH / KOH / NH₃ (pharmacopoeial pH adjusters) and granulation solvents: legitimate but borderline |

## 4. Label conflicts across salt and hydrate forms

There are 10 pairs that are the same at the parent level but have opposite labels. Seven of them come from one
artifact: **lactose**. "β-Lactose" (CID 6134) is the lactose of the literature-derived rows: 59 rows, 22
incompatible. "α-Lactose monohydrate" (CID 104938) is the lactose of the FDA-label rows: 142 rows, 3 incompatible. The
two differ only by a hydrate water, so the same API (ketoprofen, ranitidine, fluoxetine, primaquine, clenbuterol,
aminophylline) is labelled incompatible with one and compatible with the other. Either the source or a
physical-form difference could explain this; the structure cannot. The other three are salt forms (promethazine vs its
HCl, diclofenac vs diclofenac Na with Mg stearate, levothyroxine with two CMC CIDs). We keep these rows and list them in
`parent_label_conflicts.csv`.

## 5. Source confounding (the most important finding for the paper)

The positives and negatives come from different sources. The incompatible pairs come from the literature. The
compatible pairs come from FDA-approved formulations. You can see the split in who appears in each class:

| | raw | clean | clean_strict |
|---|---|---|---|
| APIs with only compatible rows (rows, median per API) | 374 (2965, 8) | 375 (2928, 8) | 367 (2641, 7) |
| APIs with only incompatible rows (rows, median per API) | 122 (260, **1**) | 97 (167, 1) | 77 (129, 1) |
| APIs with both labels | 37 | 36 | 36 |
| ROC-AUC, label from the API's leave-one-out incompatible rate | **0.952** | 0.933 | 0.922 |
| ROC-AUC, label from excipient's leave-one-out rate | 0.850 | 0.814 | 0.795 |
| ROC-AUC, **smaller API molecular weight → incompatible** | **0.759** | 0.728 | 0.709 |
| ROC-AUC, fewer rows per API → incompatible | 0.819 | 0.807 | 0.766 |

Only 37 of the 533 APIs have both labels. A recent, large drug with ~8 FDA-label excipients is almost always "compatible".
An older, small drug with one literature report is almost always "incompatible". API molecular weight on its own reaches
ROC-AUC 0.76, and an API-cluster split does not remove this signal (0.735 on the current `api_cluster` val set). Any
reported ROC-AUC should therefore be compared against these trivial baselines, not against 0.5. Curation reduces the
confounding but cannot remove it, because it comes from how the data was sampled.

## 6. Limitations not fixed

* Stereo is missing everywhere. Restoring PubChem isomeric SMILES would separate mannitol / sorbitol and α / β
  lactose, but it changes every featurisation and only matters to models that use chirality.
* Polymers are approximated by monomers or oligomers; there is no better SMILES-level fix.
* Every label is a single binary call. Experimental method, conditions and physical form are not recorded.
* The 22 held-out pairs do not overlap `clean.csv` at the parent-pair level, so curation does not touch the
  held-out evaluation.

## 7. Relation to earlier work in this repository

`experiments/reagent_ablation/` removed 14 reagent CIDs (72 rows) and found no benefit, possibly a small loss, on the
22 held-out pairs. The curation here is broader (121 rows removed, 153 repaired, 27 collapsed). No model has been
retrained on `clean.csv` yet. The existing splits, checkpoints and results still come from the raw file.
