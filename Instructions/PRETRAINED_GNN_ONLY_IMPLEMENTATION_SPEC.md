# PRETRAINED GNN EMBEDDINGS — IMPLEMENTATION SPECIFICATION
## API–Excipient Compatibility Model

**Target repository:** `Api-Excipient-Compatibility-Model-main`

**Purpose:** replace the previously proposed randomly initialized PyG GNN experiments with a small set of **genuinely pretrained molecular GNN encoders** that fit the project's existing encoder interface and are appropriate for a small API–excipient compatibility dataset.

---

# 0. READ THIS FIRST

This document is intended to be given directly to an AI coding agent.

The agent must:

1. Read this entire specification before changing code.
2. Inspect the complete current repository and its current code state.
3. Preserve the existing MoLFormer, ChemBERTa, fixed-vector, and existing pretrained GIN functionality.
4. **Preserve the already-added CheMeleon implementation if it is correct.** Do not create a second conflicting CheMeleon implementation.
5. Remove/disable the previously proposed random-init versions of AttentiveFP, GINE, GATv2, and PNA if the current repository contains them.
6. Add only the pretrained GNN encoders specified in this document.
7. Do not train a randomly initialized GNN from scratch on the small API–excipient compatibility dataset as a substitute for a pretrained embedding model.
8. Do not silently download model weights during normal training or inference.
9. Download model resources once, verify them, store them locally, and use the local files thereafter.
10. Integrate the pretrained GNNs into the **existing** `concat` and `cross_attn` fusion paths and all four existing losses.

The final pretrained-GNN set is:

```text
1. CheMeleon / D-MPNN        -> already added by the agent; preserve/fix as needed
2. Existing pretrained GIN   -> already present in the project; preserve/fix only as needed
3. Stanford pretrained GAT   -> add as the only new GNN encoder
```

Do **not** add these as the requested primary embedding encoders:

```text
GCN
GraphSAGE
AttentiveFP
GINE
GATv2
PNA
```

Reason: the current task is specifically to use **pretrained molecular representations**, not to initialize generic PyG layers randomly and learn molecular representations from the small compatibility dataset.

---

# 1. CORE OBJECTIVE

The objective is to obtain strong molecular embeddings before exposing the model to the small API–excipient compatibility dataset.

Use this general pipeline:

```text
SMILES
   |
   v
Pretrained molecular GNN
   |
   +---- atom/node embeddings [N_atoms, D]
   |
   +---- pooled molecular embedding [D]
   |
   v
Existing encoder ProjectionHead
   |
   +---- concat fusion
   |          OR
   +---- existing bidirectional cross-attention
   |
   v
Existing descriptor pathway
   |
   v
Existing pair interaction / absolute difference
   |
   v
Existing classifier
   |
   v
Compatibility logit
   |
   v
Existing loss + evaluation
```

The pretrained GNN is primarily an **embedding generator**.

For the first/default experiments:

```text
PRETRAINED GNN = FROZEN
DOWNSTREAM COMPATIBILITY MODEL = TRAINABLE
```

This is the default because the compatibility dataset is small and the project's purpose is to benefit from knowledge learned on large molecular corpora rather than asking a small labeled dataset to learn a useful molecular representation from scratch.

A separate optional fine-tuning experiment can be supported later, but it must not replace the frozen-pretrained baseline.

---

# 2. FINAL MODEL SET

## 2.1 Model A — CheMeleon D-MPNN

The current repository has already been modified by another agent to add CheMeleon.

The new agent must:

- inspect the current CheMeleon implementation first;
- preserve it if correct;
- repair it if necessary;
- not duplicate it under another class/path unnecessarily.

Required behavior:

```text
CheMeleon pretrained message-passing weights
-> native Chemprop/CheMeleon graph representation
-> atom/node embeddings
-> pooled embedding
-> existing compatibility model
```

CheMeleon is a pretrained foundation model for molecular message passing. Use the native checkpoint-compatible architecture and feature representation. Do not force the checkpoint through the unrelated PyG feature schema.

Official resources:

- GitHub: https://github.com/JacksonBurns/chemeleon
- Zenodo: https://zenodo.org/records/15460715
- Checkpoint: https://zenodo.org/records/15460715/files/chemeleon_mp.pt

The checkpoint must be downloaded once and stored locally.

Recommended:

```text
models/
  pretrained/
    chemeleon_mp.pt
```

Normal training/inference must use the local file.

Do not automatically access the internet when the local checkpoint exists.

If the local file is missing, stop with a clear error telling the user how to perform the one-time download.

Do not silently substitute a randomly initialized D-MPNN.

## 2.2 Model B — existing pretrained DGL-LifeSci GIN

Preserve the project's current `pretrained_gin` encoder.

The project already uses a pretrained GIN from the Hu et al. pretraining work through DGL-LifeSci, with:

```text
output_dim = 300
frozen = True
node-level embeddings available
pooled embedding available
```

The current source code already contains a local checkpoint mechanism:

```text
models/pretrained/gin/gin_supervised_contextpred_pre_trained.pth
```

and uses the `gin_supervised_contextpred` architecture.

Preserve this functionality.

Do not rewrite the GIN using modern PyG layers simply because PyG is used for GAT.

The official DGL-LifeSci path and local checkpoint should remain isolated from the new Stanford GAT implementation.

The current project behavior is intentionally useful here:

```text
DGL-LifeSci pretrained GIN
-> node embeddings [N_atoms, 300]
-> mean pool [300]
-> frozen representation
```

Do not turn this model into a randomly initialized GIN.

## 2.3 Model C — Stanford pretrained chemistry GAT

Add the pretrained GAT from:

https://github.com/snap-stanford/pretrain-gnns

The upstream project explicitly provides pretrained GNN models for chemistry and states that saved pretrained models are released for chemistry and biology applications. Its chemistry implementation includes GIN, GCN, GAT, and GraphSAGE variants. [Stanford pretrain-gnns README](https://github.com/snap-stanford/pretrain-gnns/) [Stanford chemistry model implementation](https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/model.py)

For this project, use **the chemistry pretrained GAT checkpoint** from that repository.

Do not use a random PyG `GATConv` or `GATv2Conv` as a substitute.

The upstream chemistry GAT implementation is a custom edge-aware GAT and is **not equivalent** to simply constructing a modern PyG `GATConv`/`GATv2Conv` and loading an unrelated state dict. The original implementation includes its own atom/bond embeddings and attention mechanism. [Stanford chemistry model implementation](https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/model.py)

The upstream chemistry pretraining code uses, by default:

```text
num_layer = 5
emb_dim   = 300
JK        = last
```

and supports `gnn_type="gat"`. [Stanford chemistry pretraining script](https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/pretrain_masking.py)

Use the **exact checkpoint-compatible architecture and feature processing** for the released GAT weights. If the selected checkpoint was generated with a different exact `num_layer`, `JK`, or related configuration, infer/use the checkpoint's actual architecture instead of guessing.

Output requirement:

```text
node embeddings [N_atoms, 300]
pooled embedding [300]
```

Default:

```text
FROZEN
```

Do not fine-tune it in the default compatibility experiments.

---

# 3. WHY THESE THREE AND NOT THE OTHER FOUR GENERIC GNNs

The final requested GNN embedding set is deliberately small:

```text
CheMeleon D-MPNN
Pretrained GIN
Pretrained GAT
```

The following are explicitly out of scope for the main pretrained-embedding experiment:

```text
GCN
GraphSAGE
AttentiveFP
GINE
GATv2
PNA
```

Do not add them merely because PyG exposes these layers.

The Stanford repository does have pretrained GCN and GraphSAGE chemistry variants, but adding every available architecture is not necessary for this experiment. The repository itself documents the four architectures under a common older chemistry implementation. [Stanford pretrain-gnns](https://github.com/snap-stanford/pretrain-gnns/)

The purpose here is to get **useful pretrained molecular representations**, not to create the largest possible architecture grid.

A smaller set also reduces:

- dependency conflicts,
- duplicated representation families,
- experiment count,
- checkpoint management,
- unnecessary training/inference complexity.

---

# 4. IMPORTANT — PRETRAINED MEANS PRETRAINED

The following is forbidden for the primary implementation:

```text
random initialization
-> API/Excipient compatibility dataset
-> train GNN
-> call output an "embedding"
```

That would be a task-trained GNN, not a pretrained molecular embedding model.

Instead:

```text
large-scale molecular pretraining checkpoint
-> load checkpoint
-> freeze encoder
-> generate molecular embeddings
-> train compatibility layers on the small dataset
```

The pretrained encoder can later have an explicit optional fine-tuning mode, but:

```text
frozen_pretrained = DEFAULT
fine_tuned_pretrained = OPTIONAL
random_from_scratch = NOT PART OF THIS TASK
```

---

# 5. CURRENT PROJECT ARCHITECTURE TO PRESERVE

The existing project was intentionally designed around a config-selectable encoder.

The important encoder contract is:

```python
class Encoder(nn.Module):
    is_sequence_capable = True
    output_dim: int

    def encode(self, smiles_batch: list[str]):
        """
        Returns:
            token_embeddings: [B, L, D]
            pooled_embedding: [B, D]
            token_mask: [B, L] bool
                         True  = padding/invalid
                         False = real atom/token
        """
```

This contract is the most important integration boundary.

Do not redesign `CompatibilityModel` merely to accommodate a GNN.

---

# 6. TOKEN SEMANTICS FOR GNNs

For pretrained molecular GNNs:

```text
one atom = one token
```

The sequence order should be the molecular graph's atom ordering used by the source checkpoint's featurization.

For the Stanford GAT specifically, preserve its original atom ordering and molecular feature encoding.

For GIN, preserve the existing DGL-LifeSci molecular graph behavior.

For CheMeleon, preserve Chemprop's native graph ordering and featurization.

Do not re-tokenize graphs using a generic SMILES tokenizer.

The resulting token sequence is fed to the project's existing cross-attention block.

---

# 7. FROZEN EMBEDDING POLICY

## 7.1 Default

All three pretrained GNN encoders should be frozen for the main experiment:

```python
for p in encoder.parameters():
    p.requires_grad_(False)
```

The downstream layers remain trainable:

```text
ProjectionHead
cross-attention
pooling
descriptor projection
pair interaction
classifier
```

## 7.2 Why frozen

The current compatibility dataset is small. The purpose of these models is to transfer molecular structural information learned from much larger pretraining corpora into the compatibility model.

Do not interpret “frozen” as “weak”. It means the learned molecular representation is kept intact while the task-specific layers learn compatibility.

## 7.3 Optional fine-tuning mode

Implement an optional configuration flag only if the existing code architecture supports it cleanly:

```text
pretrained_gnn_finetune = False
```

When false:

```text
encoder frozen
```

When true:

```text
encoder trainable with a deliberately small LR
```

Do not make fine-tuning the default.

A future fine-tuning run can use a smaller learning rate such as:

```text
5e-6 to 2e-5
```

for the pretrained encoder and the existing task LR for downstream layers.

But this is a secondary experiment, not the primary requirement.

---

# 8. CRITICAL: LOCAL MODEL DOWNLOADS

All pretrained model weights must be downloaded **once** and stored locally.

After setup, the project must be runnable without internet access.

Recommended layout:

```text
models/
  pretrained/
    chemeleon_mp.pt
    gin/
      gin_supervised_contextpred_pre_trained.pth
    stanford_gat/
      <exact chemistry GAT checkpoint>.pth
```

The exact Stanford GAT filename must be obtained from the upstream repository's released chemistry pretrained model files. Do not invent a filename.

## 8.1 One-time download utility

Create or extend:

```text
scripts/download_gnn_resources.py
```

It must support at least:

```text
--resource chemeleon
--resource stanford_gat
--resource all
```

For each resource it must:

1. Download from the official project source when explicitly requested.
2. Store the file under `models/pretrained/`.
3. Verify the downloaded file exists and is readable.
4. Verify checksum when an authoritative checksum is available.
5. Refuse to overwrite a valid existing file unless `--force` is supplied.
6. Print the final local path.
7. Record source URL and checksum metadata where practical.
8. Never run automatically from the training loop.

Example:

```bash
python scripts/download_gnn_resources.py --resource chemeleon
python scripts/download_gnn_resources.py --resource stanford_gat
```

or:

```bash
python scripts/download_gnn_resources.py --resource all
```

## 8.2 Local-only runtime

During normal:

```text
python main.py ...
python inference.py ...
python -m src.cross_validate ...
```

there must be **zero implicit HTTP/download calls**.

If a required local model is missing:

```text
FAIL FAST
print exact missing path
print one-time setup command
exit
```

Do not silently download.

## 8.3 Optional offline environment flag

Support:

```text
GNN_OFFLINE=1
```

When enabled, any attempt to access the network for model weights should raise a clear configuration error.

---

# 9. STANFORD PRETRAINED GAT — INTEGRATION STRATEGY

This is the most important new encoder implementation detail.

The Stanford repository was developed against a much older PyTorch/PyG stack. Its README lists, for example:

```text
Python 3.7
PyTorch 1.0.1
PyTorch Geometric 1.0.3
```

so do **not** downgrade the entire current project to that environment merely to run the GAT. [Stanford pretrain-gnns README](https://github.com/snap-stanford/pretrain-gnns/)

Instead:

```text
current project environment
        |
        +---- existing DGL-LifeSci GIN path
        |
        +---- current CheMeleon/Chemprop path
        |
        +---- locally vendored Stanford GAT architecture + checkpoint-compatible adapter
```

## 9.1 Vendor only what is necessary

Add a local implementation module such as:

```text
src/encoders/stanford_gat_encoder.py
```

and, if needed:

```text
src/encoders/stanford_gat_legacy.py
```

The implementation must preserve the mathematical structure and parameter naming needed by the checkpoint.

Do not simply instantiate modern `torch_geometric.nn.GATConv` and hope the checkpoint loads.

The original Stanford chemistry model defines its own `GATConv` and embeds bond type/direction into the attention/message path. [Stanford chemistry model](https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/model.py)

## 9.2 Source files to use as the compatibility reference

Use these official sources:

- Repository: https://github.com/snap-stanford/pretrain-gnns
- Chemistry GAT/GIN/GCN/GraphSAGE implementation: https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/model.py
- Chemistry molecular loader/feature encoding: https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/loader.py
- Chemistry pretraining scripts: https://github.com/snap-stanford/pretrain-gnns/tree/master/chem

The upstream model uses:

```text
num_atom_type = 120
num_chirality_tag = 3
num_bond_type = 6
num_bond_direction = 3
```

and the chemistry loader constructs categorical atom/bond features accordingly. [Stanford chemistry model](https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/model.py) [Stanford chemistry loader](https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/loader.py)

Preserve this source-checkpoint relationship.

## 9.3 Node-level output

The Stanford `GNN` module returns node representations.

The project adapter must expose:

```text
node_representation [N_atoms, 300]
```

then:

```text
token_embeddings [B, L_max, 300]
pooled_embedding [B, 300]
token_mask [B, L_max]
```

Do not use the Stanford graph-level prediction head.

The compatibility project needs **node embeddings**, not the original downstream task classifier.

## 9.4 Pooling

Use the existing project-compatible pooling at the adapter boundary:

```text
node embeddings -> mean over real atoms -> pooled embedding
```

Do not use the Stanford graph prediction head for the pooled vector.

---

# 10. STANFORD GAT FEATURE PIPELINE

Do not force the Stanford GAT through the 39/10 feature schema used in the previous random-init PyG design.

The Stanford GAT checkpoint expects the feature representation used by its own chemistry loader/model.

Therefore:

```text
SMILES
  -> Stanford chemistry loader-compatible RDKit conversion
  -> categorical atom/bond features
  -> Stanford GAT
  -> node embeddings
```

The project adapter should convert these exact native outputs into the common encoder contract.

This is essential for checkpoint correctness.

---

# 11. STANFORD GAT CHECKPOINT VALIDATION

When adding the GAT checkpoint, the agent must inspect the actual released chemistry checkpoint and validate:

```text
1. file exists
2. checkpoint can be torch.load()'d
3. checkpoint state_dict keys match the vendored architecture
4. embedding dimension is 300
5. number of layers matches the checkpoint
6. JK mode matches the checkpoint
7. atom/bond embedding dimensions match
8. one known chemistry SMILES produces finite node embeddings
```

Do not silently load with:

```python
strict=False
```

if doing so would hide missing or mismatched learned parameters.

Use strict loading or an explicit key-remapping procedure whose mapping is asserted and tested.

If the checkpoint is incompatible, fail clearly rather than using a randomly initialized model.

---

# 12. EXISTING PRETRAINED GIN — DO NOT BREAK IT

Inspect:

```text
src/encoders/gin_encoder.py
```

The current implementation has several project-specific workarounds, including DGL/GraphBolt handling and local checkpoint support.

Preserve those unless there is a demonstrated bug.

The current encoder behavior is approximately:

```text
SMILES
 -> RDKit
 -> DGL molecular graph
 -> PretrainAtomFeaturizer / PretrainBondFeaturizer
 -> pretrained `gin_supervised_contextpred`
 -> 300-dim node embeddings
 -> mean pooling
 -> frozen cache
```

That is already well aligned with the desired pretrained embedding architecture.

Do not replace this with a randomly initialized PyG GIN.

---

# 13. CHEMELEON — KEEP CURRENT IMPLEMENTATION IF CORRECT

Because another agent has already added CheMeleon, the first action is to inspect it.

Check:

```text
- where checkpoint is loaded
- whether it uses a local path
- whether local loading happens before any network fallback
- whether the exact native Chemprop architecture is used
- whether node/vertex embeddings are exposed
- whether graph pooling is done correctly
- whether it is frozen by default
- whether `output_dim` is determined from the actual model
- whether the compatibility model receives `[B,L,D]` and `[B,D]`
```

If the current implementation instead fine-tunes CheMeleon by default, modify it so that the new primary mode is:

```text
frozen pretrained CheMeleon
```

with optional explicit fine-tuning only if supported cleanly.

Do not add a second independent CheMeleon class if one already exists and is functioning.

---

# 14. COMMON OUTPUT CONTRACT

All three pretrained encoders must expose:

```python
token_embeddings, pooled_embedding, token_mask = encoder.encode(smiles_batch)
```

For GIN and Stanford GAT:

```text
token_embeddings: [B, L, 300]
pooled_embedding: [B, 300]
token_mask:       [B, L]
```

For CheMeleon:

```text
D = actual native message-passing width
```

Do not hard-code CheMeleon as 300 unless its actual loaded checkpoint reports 300.

The existing downstream projection layer handles arbitrary encoder width.

---

# 15. PADDING AND MASKING

For a batch of different molecule sizes:

```text
molecule 0 -> N0 atoms
molecule 1 -> N1 atoms
...
```

pad node embeddings to:

```text
L_max = max(N0, N1, ...)
```

Set:

```text
False = valid atom
True  = padding
```

The project already follows this mask convention.

Test it explicitly.

Do not let padded zeros participate in:

- cross-attention
- gated pooling
- mean pooling

except where the existing project intentionally uses a masked operation.

---

# 16. MISSING EXCIPIENT HANDLING

The existing project supports unavailable excipients.

For an empty/missing excipient string:

```text
pooled embedding = zeros
one-token embedding = zeros
mask = [True]
```

Do not pass empty strings through model-specific graph parsing.

The existing compatibility model must retain its existing missing-excipient handling.

---

# 17. CONCAT FUSION

All three pretrained encoders must support:

```text
fusion = concat
```

The pipeline is:

```text
pretrained node embeddings
      |
      v
pooled molecular embedding
      |
      v
existing ProjectionHead
      |
      v
existing descriptor + pair feature pipeline
      |
      v
classifier
```

No special GNN-specific classifier should be introduced.

---

# 18. CROSS-ATTENTION FUSION

All three pretrained encoders must support:

```text
fusion = cross_attn
```

Pipeline:

```text
API pretrained node embeddings
            |
            v
     existing ProjectionHead
            |
            +-------------------+
                                |
Excipient pretrained nodes      |
            |                   |
            v                   |
     existing ProjectionHead    |
            |                   |
            +--------+----------+
                     |
                     v
       existing bidirectional cross-attention
                     |
                     v
       existing global_gated_attention pooling
                     |
                     v
       existing descriptor/pairwise classifier
```

Do not create another graph-level cross-attention mechanism inside the GNN encoder.

---

# 19. ALL FOUR EXISTING LOSSES

Every pretrained encoder must work with:

```text
bce
weighted_bce
focal
asl
```

Therefore the minimum matrix is:

```text
3 encoders
x
2 fusion modes
x
4 losses
=
24 primary combinations
```

The loss code should remain shared.

Do not introduce encoder-specific loss implementations.

---

# 20. TRAINING SETTINGS

For the default frozen-pretrained experiment, keep the task-level settings aligned with the existing project:

```text
seed                  = 42
batch_size            = 64
effective_batch_size  = 64
optimizer             = AdamW
lr                    = existing project LR
weight_decay          = existing project weight decay
max_epochs            = existing project max epochs
early stopping        = existing project policy
balanced sampler      = existing project setting
descriptors           = ON
```

The primary point is that the GNN encoder itself is pretrained and frozen. There is no reason to introduce architecture-specific task-training tricks unless they are necessary for compatibility.

Do not make the new pretrained GNN experiments incomparable to the existing models by changing every downstream hyperparameter.

---

# 21. OPTIMIZER RULES

For default frozen pretrained encoders:

```text
GNN parameters = requires_grad False
```

Only downstream parameters enter the optimizer.

Before training, print:

```text
trainable encoder parameters = 0
trainable downstream parameters = ...
```

For the optional fine-tuning mode:

```text
pretrained encoder LR = 5e-6 to 2e-5
new downstream layers = existing task LR
```

Use separate parameter groups.

Do not accidentally give a frozen encoder gradients.

---

# 22. EMBEDDING CACHING

Because the default pretrained encoders are frozen, it is legitimate to cache their learned outputs.

There are two supported caching levels.

## 22.1 In-memory cache

Cache by exact SMILES string.

```python
cache[smiles] = (node_embeddings, pooled_embedding)
```

Only for frozen encoders.

## 22.2 Optional disk cache

For repeated fusion/loss experiments, optionally create:

```text
models/embedding_cache/
  chemeleon/
  pretrained_gin/
  stanford_gat/
```

A cache file should include:

```text
encoder name
checkpoint hash
encoder architecture/version
SMILES hash
node embedding
pooled embedding
```

If any encoder or checkpoint version changes, invalidate the cache.

Do not reuse stale embeddings from another encoder configuration.

For optional fine-tuning mode:

```text
DO NOT USE DISK-CACHED LEARNED EMBEDDINGS
```

because the representation changes during training.

---

# 23. IMPORTANT — DISK CACHE MUST NOT BECOME A LEAK

Caching pretrained embeddings is not data leakage when:

```text
embedding model is pretrained independently
encoder is frozen
cache contains only deterministic representations of the molecule
```

But do not cache anything derived from compatibility labels.

The embedding cache must depend only on:

```text
SMILES
pretrained checkpoint
encoder configuration
```

not:

```text
Outcome1
train/val/test label information
```

---

# 24. CONFIGURATION CHANGES

Extend `src/config.py` so the encoder selection can include:

```python
Literal[
    "molformer",
    "pretrained_gin",
    "chemberta",
    "fixed_vector",
    "dmpnn_chemprop",
    "pretrained_gat",
]
```

If the current CheMeleon implementation already uses a different key, preserve the current working key or provide a backward-compatible alias.

Recommended canonical keys:

```text
dmpnn_chemprop
pretrained_gin
pretrained_gat
```

Add fields similar to:

```python
gnn_pretrained_frozen: bool = True

chemeleon_checkpoint_path: str = "models/pretrained/chemeleon_mp.pt"

stanford_gat_checkpoint_path: str = "models/pretrained/stanford_gat/<resolved_checkpoint>.pth"
stanford_gat_source_repo: str = "https://github.com/snap-stanford/pretrain-gnns"

embedding_cache_dir: str = "models/embedding_cache"

gnn_offline: bool = False
```

Do not hard-code an unverified Stanford checkpoint filename. The implementation should resolve/validate the actual local file and store the final path in the run configuration.

---

# 25. ENCODER REGISTRY

Extend `src/encoders/__init__.py` lazily.

Add:

```python
def _get_pretrained_gat():
    from src.encoders.stanford_gat_encoder import StanfordPretrainedGATEncoder
    return StanfordPretrainedGATEncoder
```

and register:

```python
"pretrained_gat": _get_pretrained_gat
```

Do not remove:

```text
molformer
pretrained_gin
chemberta
fixed_vector
```

Do not register random:

```text
attentivefp
gine
gatv2
pna
```

unless they are kept as disabled legacy aliases for backward compatibility, and even then they must not silently construct a random model when the user expects a pretrained encoder.

---

# 26. MODEL.PY INTEGRATION

The existing `CompatibilityModel` should continue to use:

```python
enc_dim = encoder.output_dim
```

and:

```python
self.encoder.is_sequence_capable
```

The new GAT should require no GAT-specific branch in the compatibility classifier.

The desired path is:

```text
encoder.encode()
-> generic structural projection
-> generic fusion
-> generic descriptors
-> generic pairwise interaction
-> generic classifier
```

Do not hard-code:

```python
if encoder == "pretrained_gat":
```

inside the structural classifier unless required for model-loading compatibility.

---

# 27. MAIN.PY INTEGRATION

Extend the existing encoder factory.

Do not build a second training system.

The main flow should remain:

```text
parse config
-> set split paths
-> validate pretrained local checkpoint
-> build encoder
-> build CompatibilityModel
-> build loaders
-> train/evaluate
```

For the pretrained frozen GNNs, checkpoint validation should happen before training begins.

---

# 28. INFERENCE.PY INTEGRATION

`inference.py` must be able to reconstruct all three pretrained GNN encoders from a saved run configuration.

At load time:

```text
checkpoint config
-> encoder key
-> local pretrained checkpoint
-> recreate encoder
-> recreate CompatibilityModel
-> load compatibility checkpoint
-> inference
```

Do not download model weights automatically.

If the local pretrained file is unavailable, print:

```text
Required local pretrained model not found: <path>
Run the one-time GNN resource download/setup command.
```

---

# 29. CROSS-VALIDATION

The pretrained GNN encoder itself should remain frozen across folds.

This is different from the downstream compatibility parameters, which are retrained independently in each fold.

For each fold:

```text
same pretrained checkpoint
same frozen encoder
new compatibility model
new train/val fold split
```

Do not train the GNN differently per fold unless explicitly running the optional fine-tuning ablation.

The current descriptor normalization must continue using only training-fold statistics as required by the existing project.

---

# 30. EVALUATION

Continue using the project's existing metrics:

```text
PR-AUC
F1
MCC
Precision
Recall
Accuracy
confusion matrix
```

Do not choose the classification threshold on test data.

Keep the existing validation threshold procedure.

The pretrained encoder is frozen, so model-selection differences primarily come from the interaction between the pretrained representation and the existing compatibility architecture.

---

# 31. REQUIRED TESTS

Create or extend tests for:

## 31.1 Pretrained GAT

Test:

```text
local checkpoint exists
checkpoint loads
checkpoint architecture is correct
node embeddings are returned
output_dim == 300
outputs are finite
```

## 31.2 Existing GIN regression

Test:

```text
pretrained GIN still loads
output shape unchanged
existing DGL workaround still works
frozen parameters remain frozen
```

## 31.3 CheMeleon regression

Test the current implementation:

```text
local checkpoint loads
node embeddings available
native feature path preserved
frozen by default
```

## 31.4 Graph edge cases

Every encoder should handle:

```text
single atom
ordinary molecule
multi-fragment salt
empty/missing excipient
```

## 31.5 Batch semantics

Check:

```text
batch ordering preserved
mask semantics correct
padding does not contribute to pooled representations
```

## 31.6 Gradient policy

For frozen pretrained runs:

```text
encoder.grad == None
```

and downstream trainable parameters should receive gradients.

For optional fine-tuning mode:

```text
encoder receives finite gradients
```

---

# 32. SMOKE TEST MATRIX

At minimum run:

```text
pretrained_gin + concat + bce
pretrained_gin + cross_attn + asl

pretrained_gat + concat + bce
pretrained_gat + cross_attn + asl

dmpnn_chemprop + concat + bce
dmpnn_chemprop + cross_attn + asl
```

Each smoke test must perform:

```text
forward
loss computation
backward
optimizer step
validation forward
checkpoint save
checkpoint reload
```

For frozen encoders, verify the encoder itself was not changed by the optimizer step.

---

# 33. FULL PRIMARY EXPERIMENT MATRIX

Run the 24 primary combinations:

```text
Encoders:
    dmpnn_chemprop
    pretrained_gin
    pretrained_gat

Fusion:
    concat
    cross_attn

Loss:
    bce
    weighted_bce
    focal
    asl
```

Total:

```text
3 × 2 × 4 = 24
```

Do not expand this matrix with random-init GCN/GINE/GATv2/PNA/AttentiveFP models.

---

# 34. RESOURCE DOCUMENTATION TO PUT IN THE REPOSITORY

Add a short file such as:

```text
models/pretrained/README.md
```

containing:

```text
CheMeleon
- source repository
- Zenodo URL
- local filename
- checksum if available

Pretrained GIN
- DGL-LifeSci source
- local filename
- pretrained model name

Stanford GAT
- source repository
- exact chemistry checkpoint filename
- exact local filename
- architecture configuration used
- checkpoint checksum if available
```

The Stanford repository is the authoritative source for the chemistry pretrained GAT implementation and released pretrained model family. [Stanford pretrain-gnns README](https://github.com/snap-stanford/pretrain-gnns/)

---

# 35. CHECKPOINT / EXPERIMENT REPRODUCIBILITY

Every downstream compatibility checkpoint should save enough information to reconstruct the model:

```text
encoder
fusion
pooling
loss
seed
batch_size
encoder frozen/fine-tuned flag
encoder output dimension
pretrained checkpoint path
pretrained checkpoint checksum
source repository
model architecture version
```

For Stanford GAT also record:

```text
num_layer
emb_dim
JK
gnn_type
feature schema/version
```

Do not store only the final compatibility checkpoint without its pretrained-resource provenance.

---

# 36. NO SILENT FALLBACKS

These situations must FAIL LOUDLY:

```text
CheMeleon file missing
Stanford GAT checkpoint missing
Stanford GAT checkpoint architecture mismatch
Stanford GAT checkpoint keys mismatch
pretrained GIN checkpoint missing and offline mode is enabled
```

Forbidden behavior:

```text
pretrained checkpoint missing
      ↓
random initialization
      ↓
continue training
```

That would invalidate the purpose of the experiment.

---

# 37. DO NOT DOWNGRADE THE WHOLE PROJECT FOR STANFORD GAT

The Stanford repository is old and lists an old PyTorch/PyG environment. [Stanford pretrain-gnns README](https://github.com/snap-stanford/pretrain-gnns/)

Do not change the project's global PyTorch/PyG environment to the 2020 versions merely to make the old implementation run.

Instead:

```text
preserve current project environment
+
vendor/adapt exact pretrained GAT architecture
+
load exact released GAT checkpoint
```

Use modern compatibility shims only where they do not change the model mathematics or checkpoint semantics.

If a dependency conflict cannot be solved safely, isolate the Stanford GAT code in a small compatibility module rather than downgrading the entire project.

---

# 38. OPTIONAL FINE-TUNING ABLATION

After the frozen pretrained experiments work, optionally support:

```text
pretrained_gnn_finetune=True
```

Use:

```text
small encoder LR
existing downstream LR
```

and compare against the frozen model.

This is optional and should not replace the frozen pretrained experiment.

Do not start by fine-tuning all three models at once. First prove that frozen pretrained embeddings work correctly.

---

# 39. REMOVAL / CLEANUP REQUIREMENTS

If the current branch contains the previously proposed random-init implementations:

```text
src/encoders/attentivefp_encoder.py
src/encoders/gine_encoder.py
src/encoders/gatv2_encoder.py
src/encoders/pna_encoder.py
```

then:

1. Remove them if they were added only for the previous GNN experiment.
2. Remove their registry entries.
3. Remove their configuration fields if those fields are no longer needed.
4. Remove unused PyG-only dependencies introduced solely for these random models.
5. Do not remove dependencies required by the existing project or the Stanford GAT adapter.

Do not remove any existing pretrained model implementation merely because a replacement is being added.

---

# 40. FINAL ARCHITECTURE

The final encoder choice should be:

```text
                     API SMILES
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
    CheMeleon       Pretrained GIN   Pretrained GAT
     D-MPNN            DGL-LifeSci    Stanford
          |              |              |
          +--------------+--------------+
                         |
                choose ONE encoder
                         |
                         v
                node-level embeddings
                         |
                         v
                pooled molecule vector
                         |
              +----------+----------+
              |                     |
           CONCAT              CROSS-ATTN
              |                     |
              +----------+----------+
                         |
                  descriptors
                         |
                  pair features
                         |
                    classifier
                         |
                       logit
                         |
                BCE / WBCE / Focal / ASL
```

---

# 41. ACCEPTANCE CRITERIA

The implementation is complete only when ALL of the following are true.

## Encoder set

```text
[PASS] CheMeleon pretrained encoder works
[PASS] existing pretrained GIN works
[PASS] Stanford pretrained GAT works
[PASS] random-init AttentiveFP is not used
[PASS] random-init GINE is not used
[PASS] random-init GATv2 is not used
[PASS] random-init PNA is not used
```

## Pretraining

```text
[PASS] each of the three main encoders loads an actual pretrained checkpoint
[PASS] pretrained checkpoints are stored locally
[PASS] normal runtime does not download weights
[PASS] missing checkpoints fail clearly
[PASS] checkpoint provenance is recorded
```

## Architecture

```text
[PASS] all three encoders implement the common sequence interface
[PASS] node embeddings are available
[PASS] pooled embeddings are available
[PASS] token masks are correct
[PASS] concat works
[PASS] cross-attention works
```

## Losses

```text
[PASS] BCE works
[PASS] weighted BCE works
[PASS] focal works
[PASS] ASL works
```

## Frozen behavior

```text
[PASS] pretrained encoder parameters are frozen by default
[PASS] downstream parameters train
[PASS] optimizer does not contain frozen encoder parameters
```

## Regression

```text
[PASS] MoLFormer still works
[PASS] ChemBERTa still works
[PASS] fixed vectors still work
[PASS] existing pretrained GIN still works
```

## Tests

```text
[PASS] single-molecule forward
[PASS] batched forward
[PASS] salt/multi-fragment molecule
[PASS] missing excipient
[PASS] checkpoint reload
[PASS] inference
```

---

# 42. REQUIRED FINAL REPORT FROM THE CODING AGENT

After implementation, report:

1. Exact files changed.
2. Exact pretrained models now available.
3. Exact local checkpoint paths.
4. Exact one-time download/setup commands.
5. Exact Stanford GAT checkpoint filename and source location.
6. Exact architecture configuration loaded for Stanford GAT.
7. Exact CheMeleon checkpoint/version used.
8. Whether each encoder is frozen by default.
9. Parameter counts for encoder and downstream model.
10. Smoke tests executed.
11. Full 24-combination matrix configuration support.
12. Confirmation that concat works for all 3 pretrained encoders.
13. Confirmation that cross-attention works for all 3 pretrained encoders.
14. Confirmation that all 4 losses work.
15. Confirmation that normal training/inference uses local model weights only.
16. Any incompatibility encountered and exactly how it was resolved.
17. Confirmation that the old random-init GNN experiment code was removed or disabled.
18. Confirmation that existing MoLFormer/ChemBERTa/fixed-vector/pretrained-GIN paths still work.

---

# 43. DO NOT MISINTERPRET THIS SPECIFICATION

The following interpretation is WRONG:

```text
"Use the GAT architecture but initialize its weights randomly."
```

The correct interpretation is:

```text
"Use the released pretrained chemistry GAT weights and exact checkpoint-compatible architecture as a frozen molecular embedding extractor."
```

The same principle applies to:

```text
CheMeleon
pretrained GIN
```

The goal of this task is **pretrained molecular representation transfer**, not learning a new molecular encoder from the small compatibility dataset.

---

# 44. OFFICIAL REFERENCES

## CheMeleon / Chemprop

- https://github.com/JacksonBurns/chemeleon
- https://zenodo.org/records/15460715
- https://chemprop.readthedocs.io/

## Stanford pretrained GNNs

- https://github.com/snap-stanford/pretrain-gnns/
- https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/model.py
- https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/loader.py

The Stanford project describes self-supervised pretraining, supervised pretraining, and fine-tuning workflows, and explicitly states that pretrained models are released for chemistry and biology applications. Its chemistry `GNN` implementation supports `gin`, `gcn`, `gat`, and `graphsage` model types and returns node representations. [Stanford pretrain-gnns README](https://github.com/snap-stanford/pretrain-gnns/) [Stanford chemistry model](https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/model.py)

---

# 45. FINAL IMPLEMENTATION PRIORITY

When choices arise, use this order:

```text
1. Preserve actual pretrained checkpoint semantics
2. Produce genuine pretrained molecular embeddings
3. Keep encoders frozen by default
4. Integrate cleanly with existing concat/cross-attention
5. Preserve all four losses
6. Keep normal runtime offline/local
7. Preserve existing project models
8. Only then optimize code/performance
```

Do not sacrifice pretrained-model correctness for architectural symmetry.

Do not replace pretrained weights with random initialization.

Do not silently train a new GNN on the small API–excipient dataset and call it a pretrained embedding model.

**END OF SPECIFICATION**
