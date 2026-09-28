# GNN Encoder Expansion Specification
## API–Excipient Compatibility Model — Five Molecular GNN Encoders

**Target repository:** `Api-Excipient-Compatibility-Model-main`

**Requested encoder set:**
1. D-MPNN / Chemprop-style directed bond message passing
2. AttentiveFP
3. GINE
4. GATv2
5. PNA (Principal Neighbourhood Aggregation)

**Integration target:** every new encoder must work with the existing compatibility model in both:
- `concat` fusion
- `cross_attn` fusion (using the existing `global_gated_attention` pooling path by default)

and with all four existing losses:
- `bce`
- `weighted_bce`
- `focal`
- `asl`

This document is intended to be given directly to an AI coding agent. The agent should implement the changes described here rather than redesigning the downstream compatibility architecture.

---

# 0. READ THIS FIRST — critical implementation decisions

## 0.1 What was audited

The supplied repository was statically inspected before writing this specification. The project currently contains the following important pieces:

- `main.py`
- `inference.py`
- `src/config.py`
- `src/dataset.py`
- `src/model.py`
- `src/train.py`
- `src/loss.py`
- `src/evaluate.py`
- `src/cross_validate.py`
- `src/encoders/__init__.py`
- existing encoders for MoLFormer, ChemBERTa, fixed vectors, and a pretrained DGL-LifeSci GIN
- `master_experiment_spec.md`
- `Instructions/pre_global+gated_attt.md`
- `Instructions/pre_pairwise.md`
- existing tests for gated pooling and explicit pairwise interaction

The repository contains 3544 rows in `data/start_dataset.csv`, with 344 positive rows and 3200 negative rows. The existing cluster split is 2030/842/672 for train/validation/test, and the random split is 2126/709/709. The current SMILES columns have no missing values in those split files, but the code explicitly supports missing excipients and this behavior must remain supported.

The existing compatibility model is already substantially encoder-agnostic: it consumes sequence-capable encoders through the contract

```text
token_embeddings : [B, L, D]
pooled_embedding  : [B, D]
token_mask        : [B, L]   # True = padding / invalid
```

and then projects the encoder output into the existing 128-dimensional structural space. This is exactly the contract the five new GNNs should implement.

## 0.2 Metric-first policy: use the strongest practical GNN implementations

The project's objective is **to improve compatibility metrics**, not to artificially equalize the representation strength of the five GNNs. Therefore, do **not** nerf, downgrade, freeze, or replace a strong pretrained representation merely to make the five models look more statistically symmetric.

For the requested D-MPNN/Chemprop model, the preferred Tier-1 implementation is:

```text
D-MPNN / Chemprop = CheMeleon-pretrained message-passing weights + end-to-end fine-tuning
```

The other four requested architectures remain independently trainable GNN encoders. The experiments are therefore **performance-oriented**: each architecture may use the strongest appropriate practical configuration and initialization. Common task-level controls should remain fixed where they are not part of the architecture itself, but **do not reduce model capacity or remove useful mechanisms simply for symmetry**.

Chemprop officially supports foundation-model initialization through `--from-foundation`, and current Chemprop documentation exposes pretrained foundation loading for CheMeleon. The CheMeleon project itself recommends Chemprop 2.2.0+ and the `--from-foundation CheMeleon` route. [Chemprop transfer-learning CLI](https://chemprop.readthedocs.io/en/main/cmd.html) [CheMeleon README](https://github.com/JacksonBurns/chemeleon)

Important implementation rule: **load CheMeleon with the native Chemprop graph featurization and native message-passing implementation**. Do not take the pretrained checkpoint and force it through the project's separate 39/10 PyG feature schema. That would destroy the semantics learned by the pretrained weights.

Also implement a scratch Chemprop-style D-MPNN mode as a secondary diagnostic (`dmpnn_scratch`) so the project can distinguish the value of the architecture from the value of the CheMeleon initialization. This scratch mode is not the main metric-seeking D-MPNN run.

## 0.3 Feature schema: common where it is technically appropriate

Use one fixed 39-dimensional atom / 10-dimensional bond feature schema for the four PyG GNNs:

- AttentiveFP
- GINE
- GATv2
- PNA

For the **CheMeleon-pretrained D-MPNN**, use Chemprop's exact native molecular featurizer and feature dimensions required by the checkpoint. This is a deliberate exception because pretrained weights only make sense when their input representation is preserved.

This gives two levels of control:

```text
PyG GNNs -> same 39/10 graph representation
CheMeleon -> exact pretrained Chemprop representation
Downstream -> same compatibility model / fusion / losses / metrics
```

The existing downstream model is already designed to accept arbitrary encoder output dimensions, so the CheMeleon representation may remain at its native hidden width (commonly much larger than 300) and be projected by the existing encoder projection head before cross-attention. Do **not** force CheMeleon down to 300 before the compatibility model merely for symmetry.

## 0.4 New GNNs are trainable encoders, not frozen precomputed embeddings

For these five models, the word “embedding” means **the node/atom hidden representation produced by the GNN**, not a static CSV file.

The encoder must remain inside `CompatibilityModel` so its parameters receive gradients from the compatibility loss:

```text
SMILES
  -> RDKit molecular graph
  -> trainable GNN
  -> node embeddings [N_atoms, 300]
  -> mean pool [300]
  -> existing projection / cross-attention / descriptor / classifier pipeline
  -> compatibility logit
```

Do NOT precompute the 300-dimensional GNN outputs once and save them as fixed CSV embeddings for the training experiment. That would detach the GNN from the compatibility loss and turn the experiment into a fixed-vector experiment.

It is fine, and strongly recommended, to cache the **RDKit/PyG graph objects** because they contain no learned outputs and therefore do not break gradient flow.

## 0.5 Preserve the existing downstream model

Do not rewrite the existing fusion/classifier architecture merely because the encoder changes.

The target data path is:

```text
GNN node embeddings [B, L, 300]
          |
          +--> pooled mean [B, D]
          |
          +--> token mask [B, L]

existing CompatibilityModel
          |
          +--> ProjectionHead D -> 128
          |
          +--> cross-attention OR concat
          |
          +--> existing global + gated attention pooling when cross_attn
          |
          +--> existing 21 descriptors
          |
          +--> existing interaction + absolute difference features
          |
          +--> existing classifier
          |
          +--> raw logit [B]
```

No encoder-specific branch should be added to `src/model.py` beyond the existing generic `encoder.is_sequence_capable` contract.

---

# 1. Current project contract that the agent MUST preserve

## 1.1 Current sequence encoder interface

Each sequence-capable encoder is expected to expose:

```python
class Encoder(nn.Module):
    is_sequence_capable = True
    # Encoder-specific width. CheMeleon keeps its native width; the four PyG GNNs use 300 by default.
    output_dim: int

    def encode(self, smiles_batch: list[str]):
        """
        Returns:
            token_embeddings: [B, L, D]
            pooled_embedding: [B, D]
            token_mask: [B, L] bool, True = padding / invalid
        """
```

For the four PyG GNN encoders:

```text
D = 300 by default
```

For CheMeleon D-MPNN:

```text
D = native checkpoint message-passing width
```

The existing downstream ProjectionHead performs the common 128-dimensional alignment.

## 1.2 Existing projection dimensions

The current `src/model.py` uses:

```text
encoder output -> ProjectionHead -> 128
```

with the existing projection block:

```text
Linear(D -> 256)
LayerNorm(256)
GELU
Dropout(0.15)
Linear(256 -> 128)
LayerNorm(128)
GELU
```

Do not replace this with a GNN-specific projection.

## 1.3 Existing descriptor path

The project currently uses 21 normalized descriptors per molecule:

```text
21 -> 32 -> 24
```

with LayerNorm/GELU/Dropout in the existing `DescriptorProjectionHead`.

Do not change descriptor dimensionality as part of this task.

## 1.4 Existing cross-attention path

The current cross-attention implementation:

1. gets token embeddings and pooled embeddings from the encoder
2. projects tokens and pooled vectors to 128
3. prepends the pooled vector as a learned-structure/global token
4. performs bidirectional cross-attention with 8 heads
5. uses residual + LayerNorm
6. uses `global_gated_attention` by default
7. concatenates global + gated token pooling into 256 and fuses back to 128
8. combines the structural representation with descriptors, availability, interaction, and absolute difference features

This already works for variable sequence lengths, which maps naturally to variable atom counts.

## 1.5 Existing concat path

For `fusion="concat"`, the model ignores node-level token embeddings and uses the encoder's pooled vector:

```text
pooled [B, 300]
  -> existing ProjectionHead
  -> [B, 128]
```

Then it uses the exact same descriptor + interaction + difference + classifier path as all other encoders.

## 1.6 Existing losses

All four losses operate on raw logits:

```text
bce
weighted_bce
focal
asl
```

Do not add a GNN-specific loss. The same `src/loss.py` and `src/train.py` path must work unchanged.

---

# 2. Requested experiment matrix

The minimum requested matrix is:

```text
5 requested GNN families
x
2 fusion modes
x
4 losses
=
40 primary models

Plus optional ablation:
`dmpnn_scratch` vs `dmpnn_chemprop`/CheMeleon
```

The five encoders are:

| Encoder key | Human-readable name | Sequence-capable | Output dim |
|---|---|---:|---:|
| `dmpnn_chemprop` | D-MPNN / Chemprop + CheMeleon pretrained | Yes | native checkpoint width |
| `attentivefp` | AttentiveFP | Yes | 300 |
| `gine` | GINE | Yes | 300 |
| `gatv2` | GATv2 | Yes | 300 |
| `pna` | PNA | Yes | 300 |

Fusion values:

```text
cross_attn
concat
```

Loss values:

```text
bce
weighted_bce
focal
asl
```

For the first controlled grid, use:

```text
pooling = global_gated_attention
use_descriptors = True
use_balanced_sampler = True
seed = 42
effective_batch_size = 64
base_lr = 1.5e-4
weight_decay = 8e-4
max_epochs = 150
```

For CheMeleon-pretrained D-MPNN use a separate encoder parameter group:

```text
CheMeleon encoder lr = 1.0e-5
downstream/new-head lr = 1.5e-4
```

This is intentional: preserve the pretrained chemical representation while allowing the compatibility head to adapt aggressively. For the other four GNNs, all trainable groups may use the common `1.5e-4` base LR.

All runs should keep the same effective batch size (64), number of epochs, sampler, split, descriptors, threshold-selection protocol, and evaluation metrics. If a physical batch-size reduction is required for a larger encoder, compensate with gradient accumulation to preserve the effective batch size.

Do not change these between encoders merely because one encoder is easier to train. The goal here is a controlled architecture comparison.

---

# 3. Resource / source / download links

## 3.1 D-MPNN / Chemprop

Official Chemprop repository:

- https://github.com/chemprop/chemprop

Official `BondMessagePassing` documentation:

- https://chemprop.readthedocs.io/en/latest/autoapi/chemprop/nn/message_passing/base/index.html

Official Python message-passing tutorial:

- https://chemprop.readthedocs.io/en/main/tutorial/python/models/message_passing.html

Chemprop documentation shows `BondMessagePassing` as a directed-bond message-passing model and documents a default hidden size of 300 and depth 3. The official implementation produces vertex-level hidden representations that can be used as node embeddings. [Chemprop BondMessagePassing](https://chemprop.readthedocs.io/en/latest/autoapi/chemprop/nn/message_passing/base/index.html)

### Optional pretrained D-MPNN resource: CheMeleon

GitHub:

- https://github.com/JacksonBurns/chemeleon

Zenodo model record:

- https://zenodo.org/records/15460715

Direct checkpoint file:

- https://zenodo.org/records/15460715/files/chemeleon_mp.pt

### Mandatory local-download policy

**Do not depend on the internet during normal training or inference.** The CheMeleon checkpoint must be downloaded once and stored inside the repository or a configured local model directory. Subsequent runs must load the local file directly.

Recommended repository layout:

```text
models/
  pretrained/
    chemeleon_mp.pt
```

Add a one-time setup utility, for example:

```text
scripts/download_gnn_resources.py
```

It must:

1. download `chemeleon_mp.pt` from the official Zenodo URL when explicitly requested;
2. save it to `models/pretrained/chemeleon_mp.pt`;
3. verify the official MD5 checksum `6a80b54fdb7de37ef0374d302f01e8ce`;
4. refuse to overwrite a valid local checkpoint unless `--force` is passed;
5. print the final local path;
6. never download automatically during `main.py`, `cross_validate.py`, or inference once a local checkpoint exists.

Example one-time setup:

```bash
python scripts/download_gnn_resources.py --resource chemeleon --output models/pretrained/chemeleon_mp.pt
```

Then configure:

```text
dmpnn_pretrained_checkpoint = models/pretrained/chemeleon_mp.pt
```

The default runtime policy is **local-only**. If the local checkpoint is missing, fail with a clear message telling the user to run the one-time download utility. Do not silently fetch from the internet.

The four PyG architectures do not require external pretrained weight downloads in the baseline implementation; their parameters are initialized and trained locally. The only external model weight in this requested five-model set is the CheMeleon foundation checkpoint. PyG itself should likewise be installed once into the local Python environment, following the official Torch/CUDA-compatible installation instructions. [CheMeleon Zenodo model record](https://zenodo.org/records/15460715)

The Zenodo record currently contains `chemeleon_mp.pt` (~34.9 MB). In this project it is the **primary/default D-MPNN foundation checkpoint**, not an optional experiment. [CheMeleon model record](https://zenodo.org/records/15460715)

## 3.2 PyTorch Geometric

Official repository:

- https://github.com/pyg-team/pytorch_geometric

Installation:

- https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

Use PyG for the four architectures below. This avoids adding another DGL-specific runtime path for the new models while leaving the existing `pretrained_gin` encoder untouched.

## 3.3 AttentiveFP

Official API:

- https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.AttentiveFP.html

Official source/example:

- https://github.com/pyg-team/pytorch_geometric/blob/master/examples/attentive_fp.py

The official PyG example uses 39 atom features and 10 bond features for AttentiveFP. The public API supports configurable hidden channels, layers, timesteps, edge dimensions, and dropout. [AttentiveFP API](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.models.AttentiveFP.html)

Original paper:

- https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959

## 3.4 GINE

Official PyG API:

- https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html

Original/pretraining paper containing the GINE operator:

- https://arxiv.org/abs/1905.12265

The PyG GINE implementation incorporates edge features into the neighborhood aggregation and is directly designed as the edge-aware variant associated with the pretraining work. [GINE API](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html)

## 3.5 GATv2

Official PyG API:

- https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html

Original paper:

- https://arxiv.org/abs/2105.14491

The current PyG `GATv2Conv` supports multi-head attention and optional edge features through `edge_dim`, making it appropriate for the common 10-dimensional bond feature representation. [GATv2 API](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html)

## 3.6 PNA

Official PyG API:

- https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.PNAConv.html

Official example:

- https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pna.py

Original paper:

- https://arxiv.org/abs/2004.05718

Important: PyG's PNA operator requires a histogram of training-set node in-degrees for its degree scalers. This must be computed from the training split only, including a separate histogram for each CV fold. [PNA API](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.PNAConv.html)

---

# 4. Dependency changes

## 4.1 Required dependencies

Add both PyG and Chemprop because the requested D-MPNN is now allowed to use the pretrained CheMeleon message-passing weights.

Recommended ranges:

```text
torch-geometric>=2.9,<2.10
chemprop==2.3.1
```

The CheMeleon project documents compatibility with Chemprop 2.2.0 or newer, and Chemprop 2.3.1 is the current release at the time of writing. Pin 2.3.1 for reproducibility, then run the mandatory CheMeleon smoke test before training. If the checkpoint is incompatible with that exact version, pin the first known-compatible Chemprop 2.x version and record the reason rather than silently changing architecture semantics. [CheMeleon README](https://github.com/JacksonBurns/chemeleon) [Chemprop CLI](https://chemprop.readthedocs.io/en/main/cmd.html)

Do not pin a separate `torch` version in this task. Use the project's existing Torch installation and follow the official PyG installation instructions when a Torch/CUDA-specific wheel is required.

The current repository already contains:

```text
torch>=2.1
rdkit>=2023.9.1
dgl>=1.1
dgllife>=0.3.2
```

Keep those because the existing `pretrained_gin` encoder still uses DGL-LifeSci.

## 4.2 Dependency separation

Use:

```text
PyG       -> AttentiveFP, GINE, GATv2, PNA
Chemprop  -> CheMeleon-pretrained D-MPNN
DGL       -> existing pretrained_gin only
```

Do not rewrite the existing DGL encoder to use PyG.

---

# 4.3 Offline/local-model requirement

This project must be runnable on a machine with **no network access after setup**.

The coding agent must implement:

```text
1. a local CheMeleon checkpoint path in config
2. a one-time download/verification script
3. startup validation that uses the local file
4. no automatic network fallback from training/inference
```

Recommended defaults:

```text
MODEL_DIR=models/pretrained
CHEMELEON_PATH=models/pretrained/chemeleon_mp.pt
```

If `CHEMELEON_PATH` does not exist, the training command should stop with an actionable error such as:

```text
CheMeleon checkpoint not found at ...
Run: python scripts/download_gnn_resources.py --resource chemeleon --output ...
```

Do not embed HTTP-download logic in the normal encoder forward path.

For CI/offline servers, provide an explicit environment/config switch such as `GNN_OFFLINE=1`; when enabled, any attempted network access for a foundation model should be treated as a configuration error rather than automatically retried.

For reproducibility, save the checkpoint provenance into the run config:

```text
chemeleon_checkpoint_path
chemeleon_checkpoint_md5
chemeleon_source_url
chemeleon_downloaded_at (optional)
```

Do not serialize the raw 34.9 MB foundation weights into every compatibility checkpoint unless the existing repository explicitly requires that. The compatibility checkpoint should store the local checkpoint path + verified hash + model architecture config.

# 5. New source files / layout

Add the following modules:

```text
src/
  encoders/
    __init__.py                         # existing: add registrations
    gnn_common.py                       # new: shared graph featurization + batching
    dmpnn_encoder.py                    # new
    attentivefp_encoder.py              # new
    gine_encoder.py                     # new
    gatv2_encoder.py                    # new
    pna_encoder.py                      # new
    pna_utils.py                        # new: degree histogram
```

Add tests:

```text
tests/
  test_gnn_encoders.py                  # new
  test_gnn_model_integration.py         # new
```

Optional utility:

```text
data/
  compute_pna_degree_hist.py            # optional CLI/helper
```

Do not create a completely separate training system for GNNs. Reuse the existing training, loss, evaluation, checkpoint, and metric infrastructure.

---

# 6. Shared graph featurization

## 6.1 RDKit parsing

Every non-empty SMILES must be parsed with:

```python
mol = Chem.MolFromSmiles(smiles)
```

If parsing fails:

```python
raise ValueError(
    f"RDKit cannot parse SMILES for GNN encoder: {smiles!r}"
)
```

Do not silently convert a malformed molecule to an all-zero graph.

For the normal dataset this should never happen because the current split files have no missing SMILES and the descriptor pipeline already uses RDKit parsing, but the GNN encoder must still fail clearly for unexpected malformed input.

## 6.2 39-dimensional atom feature schema

Implement a single reusable function:

```python
def atom_features(atom: Chem.rdchem.Atom) -> list[float]:
    ...
```

The feature layout must be exactly:

```text
1. element symbol one-hot: 16 dims
2. atom degree one-hot:    6 dims
3. formal charge:          1 dim
4. radical electrons:      1 dim
5. hybridization one-hot:  6 dims
6. aromatic flag:           1 dim
7. total H one-hot:         5 dims
8. chirality possible:      1 dim
9. R/S chirality one-hot:   2 dims
--------------------------------------
Total:                     39 dims
```

Use the official PyG AttentiveFP example's categorical lists as the basis:

### Element symbols (16)

```python
[
    "B", "C", "N", "O", "F", "Si", "P", "S",
    "Cl", "As", "Se", "Br", "Te", "I", "At", "other"
]
```

### Degree (6)

```text
0, 1, 2, 3, 4, 5
```

### Hybridization (6)

Use:

```text
SP
SP2
SP3
SP3D
SP3D2
other
```

### Total hydrogens (5)

```text
0, 1, 2, 3, 4
```

### Chirality

Use:

```text
has _ChiralityPossible property -> 1 scalar
CIPCode R/S -> 2-dimensional one-hot
```

### Robustness requirement

The PyG demo assumes the degree and hydrogen values fit their one-hot lists. Production code for this project must be safer:

- values outside the listed categorical ranges go to the final `other` bucket
- never throw an `IndexError` because an unusual molecule has degree > 5 or more than 4 hydrogens
- keep the feature dimension exactly 39

For formal charge and radical electrons, keep the numeric scalar values exactly as RDKit reports them. Do not normalize these separately.

## 6.3 10-dimensional bond/edge features

Implement one reusable function:

```python
def bond_features(bond: Chem.rdchem.Bond) -> list[float]:
    ...
```

Use exactly:

```text
1. single bond one-hot    1
2. double bond one-hot    1
3. triple bond one-hot    1
4. aromatic bond one-hot  1
5. conjugated flag        1
6. ring flag              1
7. stereo one-hot         4
--------------------------------
Total                     10
```

Stereo categories:

```text
STEREONONE
STEREOANY
STEREOZ
STEREOE
```

## 6.4 Directed edges

For each chemical bond `u -- v`, create:

```text
u -> v
v -> u
```

with the same 10-dimensional bond feature vector on both directions.

This is essential because:

- D-MPNN uses directed bond messages
- PyG message passing naturally consumes directed COO edges
- PNA degree normalization must see the same graph convention
- GATv2/GINE/AttentiveFP should receive the same underlying molecular connectivity

Do not create self-loops in the shared raw molecular graph data.

GATv2 may internally add self-loops because that is part of its operator configuration; this does not mean self-loops should be inserted into the shared input `Data` objects.

## 6.5 PyG `Data` object

The shared graph builder should return:

```python
Data(
    x=torch.tensor(atom_features, dtype=torch.float32),
    edge_index=torch.tensor(edge_index, dtype=torch.long),
    edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
)
```

Required invariants:

```text
x.shape[1] == 39
edge_attr.shape[1] == 10
edge_index.shape[0] == 2
```

For a molecule with no bonds:

```text
edge_index.shape == [2, 0]
edge_attr.shape  == [0, 10]
```

Do not add fake bond edges merely to avoid an empty tensor.

## 6.6 Multi-fragment molecules / salts

Do not split a multi-fragment SMILES into separate examples.

For:

```text
CC(=O)[O-].[Na+]
```

create a single graph object containing two disconnected components. PyG's `batch` index still treats the entire molecule as one graph.

The mean pooling should average across all atoms in all components, which is the closest equivalent to the behavior already used in the current pretrained GIN encoder.

## 6.7 Shared graph cache

Create a cache in `gnn_common.py` keyed by exact SMILES string:

```python
self._graph_cache: dict[str, Data]
```

Cache only CPU graph features.

Do NOT cache:

```text
node embeddings
pooled embeddings
attention outputs
cross-attention outputs
```

because the GNN is trainable and those tensors must participate in the current batch's autograd graph.

Because the current dataset has only hundreds of unique API/Excipient SMILES, an in-memory graph cache is practical.

---

# 7. Shared GNN encoder base class

Create a base/helper class in `src/encoders/gnn_common.py`.

Recommended responsibilities:

```text
- is_sequence_capable = True
- output_dim = gnn_hidden_dim
- device handling
- graph cache
- SMILES -> PyG Data conversion
- batch graph construction
- dense node padding
- mean pooling
- missing-SMILES handling
- common validation / finite checks
```

A recommended public method:

```python
def encode(self, smiles_batch: list[str]):
    # 1. build/cached graph objects for non-empty SMILES
    # 2. Batch.from_data_list(...)
    # 3. run model to produce node embeddings [N_total, D]
    # 4. to_dense_batch -> [B, L_max, D], valid_mask [B, L_max]
    # 5. token_mask = ~valid_mask
    # 6. global_mean_pool -> [B, D]
    # 7. return dense node embeddings, pooled vector, token mask
```

Use PyG's `to_dense_batch()` if available:

```python
from torch_geometric.utils import to_dense_batch
```

It is safer than manually writing padding code.

## 7.1 Correct padding semantics

`to_dense_batch()` returns a mask where `True` means a real node.

The project expects the opposite convention:

```python
project_token_mask = ~valid_mask
```

where:

```text
True  = padding / invalid
False = real atom
```

This must be tested explicitly.

## 7.2 Preserve batch ordering

Input:

```python
[smiles_0, smiles_1, smiles_2, ...]
```

must return rows in exactly the same order.

Do not sort graphs by atom count unless an explicit inverse permutation is applied.

## 7.3 Empty SMILES / missing excipient behavior

The existing `CompatibilityModel` explicitly supports unavailable excipients.

For an empty string:

```python
""
```

do NOT call RDKit with the empty string as though it were a normal molecule.

Instead return a dummy sequence of length 1:

```text
token_embeddings: [1, D] = zeros
pooled_embedding: [D] = zeros
token_mask:       [1] = [True]
```

This means the molecular-token branch is completely masked. The existing `CompatibilityModel` will then use its learned missing-excipient global/token fallback in the cross-attention path.

In a mixed batch, valid molecules use ordinary node embeddings and the missing entries are fully masked.

## 7.4 Train/eval behavior

Unlike the existing frozen transformer/GIN encoders, these five encoders are trainable.

Do not put the entire `encode()` method under `torch.no_grad()`.

Only graph construction/cache creation is non-differentiable. The GNN forward must remain differentiable.

---

# 8. Encoder 1 — D-MPNN / Chemprop-style directed bond message passing

## 8.1 Class and key

Use:

```text
config.encoder = "dmpnn_chemprop"
class = DMPNNEncoder
mode = "chemeleon"   # DEFAULT
```

Also support:

```text
config.encoder = "dmpnn_scratch"
mode = "scratch"
```

Set:

```python
is_sequence_capable = True
```

`output_dim` is **dynamic**:

- CheMeleon mode: use the checkpoint's native message-passing output width (do not force to 300)
- scratch mode: 300

The existing downstream `ProjectionHead` must accept this dynamic `output_dim`.

## 8.2 Preferred Tier-1 D-MPNN: CheMeleon-pretrained

Use the official Chemprop foundation-model path rather than recreating the checkpoint architecture manually. Chemprop supports `--from-foundation CHEMELEON`, and the CheMeleon project documents Chemprop 2.2.0+ for fine-tuning. [Chemprop CLI](https://chemprop.readthedocs.io/en/main/cmd.html) [CheMeleon README](https://github.com/JacksonBurns/chemeleon)

Implementation requirements:

1. Load the already-downloaded local `chemeleon_mp.pt` checkpoint.
2. Instantiate the **exact Chemprop message-passing architecture required by the checkpoint**.
3. Use the **native Chemprop molecular graph featurizer** used by CheMeleon.
4. Load the pretrained message-passing weights with strict key/dimension validation.
5. Do **not** freeze the message-passing encoder by default; fine-tune it end-to-end from the compatibility loss.
6. Expose the atom/vertex-level hidden states needed by this project. Chemprop documents vertex-level hidden representations from its message-passing layer. [Chemprop message passing](https://chemprop.readthedocs.io/en/latest/tutorial/python/models/message_passing.html)
7. Pass those atom states to the existing generic encoder adapter.
8. Keep the native hidden width all the way to the existing projection head.

Do not perform a learned 300-dimensional bottleneck before the project's projection head. The purpose of this experiment is to preserve the information in the pretrained representation.

### CheMeleon fine-tuning strategy

Use two optimizer parameter groups:

```text
CheMeleon message-passing parameters : lr = 1.0e-5
new downstream parameters            : lr = 1.5e-4
weight decay                        : 8.0e-4
```

The lower encoder LR protects the pretrained representation from catastrophic forgetting while still allowing task-specific adaptation. The downstream head keeps the project's existing learning-rate scale.

Do not freeze the CheMeleon encoder unless a separate ablation explicitly requests a frozen-foundation run.

For the first experiments use the same effective batch size of 64 as the other GNNs. If GPU memory requires a physical batch of 32, use gradient accumulation of 2 so the effective batch size remains 64.

## 8.3 Scratch D-MPNN diagnostic mode

Retain a local 39/10 Chemprop-style D-MPNN implementation as `dmpnn_scratch` for ablation/debugging.

Recommended:

```text
hidden_dim = 300
depth      = 3
activation = ReLU
dropout    = 0.10
bias       = False
undirected = False
```

The scratch implementation should follow the documented directed-bond message-passing semantics, but it is no longer the primary metric-seeking D-MPNN.

The depth 3 / hidden 300 choice matches the established Chemprop defaults where practical; the 0.10 dropout is the controlled project-level regularization choice for all five new GNNs.

## 8.4 Core computation

Let:

```text
x_v  = 39-dim atom feature for atom v
e_vw = 10-dim bond feature for directed edge v -> w
```

For every directed edge `v -> w` initialize:

```text
h_vw^0 = ReLU(W_i([x_v || e_vw]))
```

where:

```text
W_i: (39 + 10) -> 300
```

For each message-passing iteration:

```text
m_vw = sum_{u in N(v) \ {w}} h_uv
h_vw = ReLU(h_vw^0 + W_h(m_vw))
```

where:

```text
W_h: 300 -> 300
```

Use the reverse-edge index to exclude the message coming directly from `w -> v`.

After the final directed-edge update, form an atom message by summing incoming directed bond states:

```text
m_v = sum_{w in N(v)} h_wv
```

Then compute the final atom embedding:

```text
h_v = ReLU(W_o([x_v || m_v]))
```

with:

```text
W_o: (39 + 300) -> 300
```

Apply dropout in the message-passing path according to the configured 0.10 rate.

## 8.5 No-bond graph

If a graph has one atom and zero real bonds:

```text
m_v = 0
h_v = ReLU(W_o([x_v || 0]))
```

This must work without special fake edges.

## 8.6 Output

Return final atom embeddings:

```text
[N_atoms, 300]
```

then the shared base class creates:

```text
[B, L_max, 300]
[B, 300]
[B, L_max] bool mask
```

## 8.7 CheMeleon checkpoint verification and failure handling

CheMeleon is part of the **main D-MPNN metric experiment**, not a future extension.

At startup, validate:

```text
checkpoint exists / can be downloaded
checkpoint loads successfully
Chemprop architecture matches checkpoint
native featurizer dimensions match checkpoint
message-passing output shape is stable
atom-level embeddings are available
```

If any of these checks fail, do **not** silently fall back to the scratch D-MPNN. Fail loudly and print the exact incompatibility so the environment/version can be corrected.

The official CheMeleon resources are:

```text
GitHub:   https://github.com/JacksonBurns/chemeleon
Zenodo:   https://zenodo.org/records/15460715
Checkpoint: https://zenodo.org/records/15460715/files/chemeleon_mp.pt
```

If the local checkpoint cannot be loaded, verify its MD5 against the published checksum and report the failure. Do not automatically access the internet from the training/inference path; the user should rerun the one-time download/verification utility if the local file is corrupt or missing.

---

# 9. Encoder 2 — AttentiveFP

## 9.1 Class and key

```text
config.encoder = "attentivefp"
class = AttentiveFPEncoder
```

```python
is_sequence_capable = True
output_dim = 300
```

## 9.2 Recommended hyperparameters

Use:

```text
hidden_channels = 300
num_layers      = 2
num_timesteps   = 2
dropout         = 0.10
```

The official PyG AttentiveFP example uses 39 atom features, 10 edge features, 2 GNN layers, and 2 iterative timesteps. The project changes hidden width to 300 so that all five GNN encoders produce the same dimensional output before the existing 128-dimensional projection. [PyG official example](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/attentive_fp.py)

## 9.3 Node embeddings are required

The public `AttentiveFP` model returns a graph-level prediction/output, but this project needs **node-level hidden representations** to support cross-attention.

Do not incorrectly use the final graph prediction as the token sequence.

Implement a small project-local node-returning version based on the official PyG AttentiveFP source, preserving the atom/message-passing part of the model and stopping before the graph-level readout.

Two acceptable implementation strategies:

1. copy the relevant open-source PyG atom/message-passing path into a local `AttentiveFPNodeEncoder`, or
2. subclass/hook the exact pinned PyG implementation to capture the final node representation immediately before molecule-level readout.

Prefer option 1 for robustness because the project must explicitly own the node-embedding contract.

## 9.4 Architectural requirement

The node path must preserve the characteristic AttentiveFP mechanism:

```text
initial node projection
   -> edge-aware gated graph convolution
   -> recurrent node update
   -> additional atom-level attention/message passing
   -> recurrent refinement
   -> final node hidden representation
```

The edge features must be supplied to the edge-aware first stage exactly as expected by the official AttentiveFP implementation.

## 9.5 Output

Return:

```text
[N_atoms, 300]
```

and use the common mean pool for:

```text
[N_atoms, 300] -> [300]
```

Do not use the graph-level scalar/vector readout from the original regression/classification example as the pooled compatibility embedding.

---

# 10. Encoder 3 — GINE

## 10.1 Class and key

```text
config.encoder = "gine"
class = GINEEncoder
```

```python
is_sequence_capable = True
output_dim = 300
```

## 10.2 Recommended hyperparameters

```text
hidden_dim   = 300
num_layers   = 3
edge_dim     = 10
dropout      = 0.10
train_eps    = True
initial_eps  = 0.0
```

Three layers are chosen as a reasonable small-data molecular setting while keeping the computational cost close to the other new models. The downstream dataset is small enough that an unnecessarily deep GINE should be avoided in the first controlled grid.

## 10.3 Per-layer MLP

Each GINE layer should use a two-layer MLP:

```text
Linear(in_dim -> 300)
LayerNorm(300)
ReLU
Linear(300 -> 300)
```

For the first layer:

```text
in_dim = 39
```

For later layers:

```text
in_dim = 300
```

## 10.4 GINE layer

Use PyG:

```python
GINEConv(
    nn=mlp,
    eps=0.0,
    train_eps=True,
    edge_dim=10,
)
```

Then apply the project's chosen post-layer normalization/activation/dropout consistently.

The PyG GINE implementation explicitly incorporates edge features into aggregation, which is the essential property needed here. [GINE API](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html)

## 10.5 Forward

For layer `l`:

```text
x_{l+1} = GINEConv_l(x_l, edge_index, edge_attr)
x_{l+1} = LayerNorm(x_{l+1})
x_{l+1} = ReLU(x_{l+1})
x_{l+1} = Dropout(x_{l+1}, p=0.10)
```

Do not add graph-level pooling inside the GINE module. The shared base class handles mean pooling and padding.

## 10.6 Output

```text
node_embeddings = [N_atoms, D]
pooled          = [D]
```

---

# 11. Encoder 4 — GATv2

## 11.1 Class and key

```text
config.encoder = "gatv2"
class = GATv2Encoder
```

```python
is_sequence_capable = True
output_dim = 300
```

## 11.2 Recommended hyperparameters

```text
hidden_dim     = 300
num_layers     = 3
heads          = 4
out_per_head   = 75
edge_dim       = 10
dropout        = 0.10
negative_slope = 0.2
add_self_loops = True
fill_value     = "mean"
residual       = True
```

Because:

```text
4 heads * 75 channels = 300
```

the output dimensionality stays exactly 300.

The current PyG `GATv2Conv` supports `edge_dim`, multi-head attention, self loops, and a residual option. [GATv2 API](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html)

## 11.3 Layer definition

Use:

```python
GATv2Conv(
    in_channels=in_dim,
    out_channels=75,
    heads=4,
    concat=True,
    dropout=0.10,
    edge_dim=10,
    add_self_loops=True,
    fill_value="mean",
    residual=True,
)
```

Then:

```text
ReLU
LayerNorm(300)
```

for each layer.

For layer 1:

```text
in_dim = 39
```

For later layers:

```text
in_dim = 300
```

## 11.4 Edge features

Always pass:

```python
edge_attr=edge_attr
```

into every GATv2 layer.

Do not accidentally use `edge_dim=None`; that would turn this into a node-only GATv2 and make the comparison inconsistent with the common edge-aware setup.

## 11.5 Output

```text
[N_atoms, 300]
```

then common mean pooling.

---

# 12. Encoder 5 — PNA

## 12.1 Class and key

```text
config.encoder = "pna"
class = PNAEncoder
```

```python
is_sequence_capable = True
output_dim = 300
```

## 12.2 Recommended hyperparameters

Use:

```text
hidden_dim     = 300
num_layers     = 3
edge_dim       = 10
towers         = 5
pre_layers     = 1
post_layers    = 1
divide_input   = False
dropout        = 0.10
activation     = ReLU
```

Aggregators:

```python
["mean", "min", "max", "std"]
```

Scalers:

```python
["identity", "amplification", "attenuation"]
```

These give PNA genuinely different neighborhood aggregation behavior rather than reducing it to a single mean aggregator.

PyG's PNA documentation states that the `deg` argument is a training-set in-degree histogram used by the degree scalers. [PNA API](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.PNAConv.html)

## 12.3 PNA layer

Use approximately:

```python
PNAConv(
    in_channels=in_dim,
    out_channels=300,
    aggregators=["mean", "min", "max", "std"],
    scalers=["identity", "amplification", "attenuation"],
    deg=degree_histogram,
    edge_dim=10,
    towers=5,
    pre_layers=1,
    post_layers=1,
    divide_input=False,
)
```

After each layer:

```text
residual/skip (where dimensions match)
LayerNorm(300)
ReLU
Dropout(0.10)
```

The residual connection is a stabilization aid in the project wrapper; it should not change the defining PNA aggregation itself.

## 12.4 Required degree histogram behavior

This is the one encoder-specific data-derived parameter that must be handled carefully.

### Single train/val/test run

Compute the degree histogram from **training molecules only**.

Do not compute it from:

```text
train + val
train + test
full start_dataset.csv
```

The histogram is not a learned target, but it is still a training-derived normalization statistic and must remain split-local.

### Cross-validation

For each fold:

```text
fold train graphs -> degree histogram -> PNA fold model
```

Never reuse a histogram computed from the full dataset across CV folds.

## 12.5 Recommended implementation

Create:

```text
src/encoders/pna_utils.py
```

with:

```python
def compute_degree_histogram(smiles_iterable) -> torch.Tensor:
    ...
```

Algorithm:

1. parse each training SMILES
2. construct the same shared 39/10 graph
3. count raw in-degrees from `edge_index[1]`
4. concatenate degree counts across training molecules
5. `torch.bincount(...)`
6. ensure dtype is `torch.long`
7. ensure histogram has at least one element
8. move the histogram to CPU when storing it in config/checkpoint

Do not include the synthetic self loops that GATv2 creates internally when computing the PNA histogram.

---

# 13. Important comparison rule: same training conditions

All five GNNs are architectures being compared inside the same downstream task.

Keep these common values fixed for the first experiment matrix:

| Setting | Value |
|---|---:|
| Seed | 42 |
| Batch size | 64 |
| Effective batch size | 64 |
| Learning rate | 1.5e-4 |
| Weight decay | 8e-4 |
| Optimizer | AdamW |
| Gradient clipping | 1.0 |
| Max epochs | 150 |
| Early-stop patience | 6 |
| Early-stop min delta | 0.002 PR-AUC |
| LR scheduler | ReduceLROnPlateau |
| LR factor | 0.5 |
| LR patience | 8 |
| Descriptors | ON |
| Balanced sampler | ON |
| Primary selection metric | validation PR-AUC |
| Decision threshold tuning | validation only |

GNN-specific architecture parameters such as number of layers, number of heads, PNA towers, and AttentiveFP timesteps are model-defining parameters and are allowed to differ as described in their individual sections.

## 13.1 Do not silently lower batch size for one model

If a model causes CUDA out-of-memory at batch size 64, do not quietly run that model at batch size 32 and compare the result as though nothing changed.

Use gradient accumulation to preserve an effective batch size of 64:

```text
micro_batch_size * accumulation_steps = 64
```

and record the micro-batch/accumulation configuration in the run config.

For example:

```text
micro_batch = 32
accumulation = 2
```

The default experiment should still be ordinary batch size 64 whenever memory permits.

## 13.2 Do not freeze the new GNNs

All five new encoders should have:

```python
requires_grad = True
```

for their trainable GNN parameters.

The existing MoLFormer/ChemBERTa/pretrained-GIN freezing behavior should remain unchanged.

---

# 14. Configuration changes — `src/config.py`

Extend the `Config.encoder` Literal from:

```python
Literal[
    "molformer",
    "pretrained_gin",
    "chemberta",
    "fixed_vector",
]
```

to include:

```python
"dmpnn_chemprop",
"attentivefp",
"gine",
"gatv2",
"pna",
```

Add these fields:

```python
# --- New trainable molecular GNN settings ---
gnn_hidden_dim: int = 300
gnn_node_feature_dim: int = 39
gnn_edge_feature_dim: int = 10
gnn_num_layers: int = 3
gnn_dropout: float = 0.10

dmpnn_depth: int = 3
dmpnn_bias: bool = False

dmpnn_undirected: bool = False

attentivefp_num_layers: int = 2
attentivefp_num_timesteps: int = 2

gatv2_heads: int = 4
gatv2_negative_slope: float = 0.2

gine_train_eps: bool = True

gine_eps: float = 0.0

pna_towers: int = 5
pna_pre_layers: int = 1
pna_post_layers: int = 1
pna_divide_input: bool = False
pna_aggregators: tuple[str, ...] = ("mean", "min", "max", "std")
pna_scalers: tuple[str, ...] = ("identity", "amplification", "attenuation")
pna_degree_hist: Optional[list[int]] = None

# Feature schema/version guard
gnn_feature_schema: str = "attentivefp39_edge10_v1"

# CheMeleon foundation model is the primary D-MPNN configuration
dmpnn_use_pretrained: bool = True
dmpnn_pretrained_checkpoint: Optional[str] = "models/pretrained/chemeleon_mp.pt"
```

Use ordinary JSON/pickle-safe Python types in config. Lists are safer than tensors for `pna_degree_hist` because `Config` is serialized into checkpoints.

## 14.1 Encoder output dimension

Do not hard-code 300 inside `src/model.py`.

At encoder construction:

```python
config.encoder_output_dim = encoder.output_dim
```

Expected values:

```text
AttentiveFP/GINE/GATv2/PNA: 300 by default
CheMeleon D-MPNN:           native checkpoint width
```

Never hard-code a single width into the downstream model.

## 14.2 Checkpoint names

The existing `resolve_checkpoint_paths()` should naturally generate distinct model directories.

Examples:

```text
checkpoints/dmpnn_chemprop_cross_attn_global_gated_bce
checkpoints/dmpnn_chemprop_cross_attn_global_gated_weighted_bce
checkpoints/dmpnn_chemprop_cross_attn_global_gated_focal
checkpoints/dmpnn_chemprop_cross_attn_global_gated_asl

checkpoints/attentivefp_concat_bce
checkpoints/gine_cross_attn_global_gated_asl
checkpoints/gatv2_concat_focal
checkpoints/pna_cross_attn_global_gated_weighted_bce
```

No run is allowed to overwrite another encoder/fusion/loss combination.

---

# 15. Encoder registry changes — `src/encoders/__init__.py`

Keep the registry lazy.

Add factories such as:

```python
def _get_dmpnn():
    from src.encoders.dmpnn_encoder import DMPNNEncoder
    return DMPNNEncoder


def _get_attentivefp():
    from src.encoders.attentivefp_encoder import AttentiveFPEncoder
    return AttentiveFPEncoder


def _get_gine():
    from src.encoders.gine_encoder import GINEEncoder
    return GINEEncoder


def _get_gatv2():
    from src.encoders.gatv2_encoder import GATv2Encoder
    return GATv2Encoder


def _get_pna():
    from src.encoders.pna_encoder import PNAEncoder
    return PNAEncoder
```

and add them to `_factories`:

```python
_factories = {
    ...
    "dmpnn_chemprop": _get_dmpnn,
    "attentivefp": _get_attentivefp,
    "gine": _get_gine,
    "gatv2": _get_gatv2,
    "pna": _get_pna,
}
```

Do not eagerly import PyG in the top-level module if that would break existing non-GNN environments.

---

# 16. `main.py` integration

## 16.1 Do not hard-code five independent training paths

Extend the existing `build_encoder(config, device)` function.

Preferred dispatch pattern:

```python
if config.encoder == "molformer":
    ...
elif config.encoder == "pretrained_gin":
    ...
elif config.encoder == "chemberta":
    ...
elif config.encoder == "fixed_vector":
    ...
elif config.encoder == "dmpnn_chemprop":
    ...
elif config.encoder == "attentivefp":
    ...
elif config.encoder == "gine":
    ...
elif config.encoder == "gatv2":
    ...
elif config.encoder == "pna":
    ...
```

It is acceptable to improve this further with a config-aware factory, but the agent must preserve old encoder behavior.

## 16.2 PNA ordering requirement

In a single run, PNA's degree histogram must be available before constructing `PNAEncoder`.

Therefore the safe order is:

```text
parse CLI
-> resolve split paths
-> load train CSV
-> compute positive prior
-> if encoder == pna: compute training degree histogram
-> store pna_degree_hist in config
-> build encoder
-> build CompatibilityModel
-> build dataloaders
-> train
```

Do not build PNA with a placeholder degree histogram and then accidentally train using the wrong one.

## 16.3 All five must remain sequence-capable

Do not trigger the existing non-sequence fallback for these encoders.

All five must expose:

```python
is_sequence_capable = True
```

so both fusion modes remain available.

---

# 17. `src/model.py` integration

## 17.1 Expected result: minimal change

`CompatibilityModel` should not need a GNN-specific branch.

It should continue to do:

```python
enc_dim = encoder.output_dim
self.api_proj = ProjectionHead(enc_dim, ...)
self.exc_proj = ProjectionHead(enc_dim, ...)
```

and then use:

```python
self.encoder.is_sequence_capable
```

The new encoders all satisfy the exact sequence contract already used by MoLFormer and pretrained GIN.

## 17.2 Cross-attention input

For a GNN, with `D = encoder.output_dim`:

```text
api_tokens: [B, L_api, D]
exc_tokens: [B, L_exc, D]
api_pool:   [B, D]
exc_pool:   [B, D]
```

The existing ProjectionHead changes these to:

```text
api_tokens: [B, L_api, 128]
exc_tokens: [B, L_exc, 128]
api_pool:   [B, 128]
exc_pool:   [B, 128]
```

Everything after that remains unchanged.

## 17.3 Concat input

The existing concat path should call:

```python
_, api_pool, _ = encoder.encode(api_smiles)
_, exc_pool, _ = encoder.encode(exc_smiles)
```

then use the existing `ProjectionHead`.

No special GNN concat branch should be created.

---

# 18. Cross-attention + GNN interaction details

This section is important because a GNN is not a text tokenizer.

## 18.1 What counts as a “token” here

For the new GNNs:

```text
one atom = one token
```

The sequence order is the RDKit atom index order.

The GNN embedding for each atom already contains information propagated through its molecular neighborhood. Therefore the downstream cross-attention operates on **graph-aware atom representations**, not raw atom one-hot vectors.

## 18.2 Cross-attention still makes sense

The architecture becomes:

```text
API graph                    Excipient graph
   |                              |
   v                              v
GNN message passing           GNN message passing
   |                              |
   v                              v
atom hidden states            atom hidden states
   |                              |
   +------------+-----------------+
                |
                v
        existing projection
                |
                v
      bidirectional cross-attn
                |
                v
      existing gated pooling
                |
                v
        pair representation
                |
                v
       existing classifier
```

Do not add another graph cross-attention mechanism between the two graphs inside the GNN. The point of this task is to reuse the existing pair-fusion architecture consistently across all encoders.

## 18.3 Global vector

The GNN's pooled mean vector is the existing encoder-level global representation.

The current model then prepends it as a 128-dimensional global token after projection.

Do not create a second “CLS node” inside the GNN.

## 18.4 Gated pooling

The existing global + gated attention pooling must operate on the GNN node tokens exactly like it does for transformer tokens.

The only semantic change is:

```text
transformer token = SMILES token
GNN token         = atom/node hidden state
```

Do not pool the global vector twice.

---

# 19. Concat + GNN behavior

The requested concat experiment is specifically useful because it isolates the effect of the downstream cross-attention block.

For each GNN:

```text
SMILES
-> GNN
-> node embeddings [N,300]
-> mean pool [300]
-> API/Exc projection [128]
-> descriptors
-> elementwise product
-> abs difference
-> classifier
```

Cross-attention version:

```text
SMILES
-> GNN
-> node embeddings [N,300]
-> pooled [300]
-> projection to 128
-> bidirectional cross-attention
-> global + gated token pooling
-> descriptors
-> interaction
-> difference
-> classifier
```

Do not change the classifier head between these two fusion modes.

---

# 20. Loss integration — all four losses must work

No change should be necessary to `src/loss.py` except tests if needed.

The agent must verify all combinations:

```text
for encoder in [dmpnn_chemprop, attentivefp, gine, gatv2, pna]:
    for fusion in [concat, cross_attn]:
        for loss in [bce, weighted_bce, focal, asl]:
            model forward -> loss -> backward -> optimizer step
```

## 20.1 Weighted BCE

The project currently computes:

```text
pos_weight = num_negative / num_positive
```

from the training dataframe.

Keep the existing implementation so that the GNN results remain directly comparable to the existing runs.

The current project also enables the balanced sampler by default. This means weighted BCE effectively combines sampler balancing with `pos_weight`. Do not silently alter this behavior for only the GNNs. If a later experiment wants a pure loss-ablation study without sampler balancing, run that as a separate explicitly named experiment using `--no_balanced_sampler` for every compared loss.

## 20.2 Focal

Use the existing config defaults:

```text
gamma = 2.0
alpha = 0.25
```

## 20.3 ASL

Use the existing config defaults:

```text
gamma_neg = 4.0
gamma_pos = 1.0
clip      = 0.05
```

Do not retune ASL only for one GNN during the initial 40-model matrix.

---

# 21. Training loop requirements for trainable GNNs

The current `src/train.py` is compatible with the new encoders as long as:

- `encoder.encode()` remains differentiable
- the returned tensors are on the correct device
- `model.parameters()` includes the GNN parameters
- no output embedding cache is used

## 21.1 Verify trainable parameter count

At startup, the existing code prints:

```text
Parameters: X trainable / Y total
```

For each GNN run, assert that the GNN contributes a substantial number of trainable parameters.

A suspiciously tiny trainable count indicates an accidental freeze or detached encoder.

## 21.2 Gradient-flow unit test

After a single forward/backward pass:

```python
loss.backward()
```

verify that at least one parameter inside the GNN encoder has:

```python
param.grad is not None
```

and a finite gradient.

This test is mandatory.

---

# 22. Inference/checkpoint integration

## 22.1 Checkpoint must contain the full Config

The existing training code saves:

```python
"config": config
```

inside the checkpoint.

For GNN checkpoints, this must contain enough information to rebuild the exact encoder.

At minimum:

```text
encoder
encoder_output_dim
gnn_hidden_dim
gnn_node_feature_dim
gnn_edge_feature_dim
gnn_dropout
model-specific hyperparameters
pna_degree_hist (for PNA)
gnn_feature_schema
fusion
pooling
loss
use_descriptors
```

## 22.2 PNA inference

Do not recompute PNA degree histograms from validation/test/held-out molecules during inference.

Load the training-derived `pna_degree_hist` stored in the checkpoint config.

## 22.3 Graph cache on checkpoint load

Create a fresh runtime graph cache after constructing the encoder.

Never serialize the in-memory graph cache into the checkpoint.

## 22.4 `inference.py`

Update its encoder factory so all five keys can be reconstructed from the embedded checkpoint config.

Do not rely only on folder names to identify the architecture. The configuration stored in the checkpoint is the source of truth.

---

# 23. Cross-validation changes — `src/cross_validate.py`

## 23.1 New encoder choices

Update all CLI choices and factory branches so the five new keys are accepted.

## 23.2 PNA fold-local histogram

Before constructing the PNA encoder in each fold:

```text
actual_train molecules
    -> shared graph builder
    -> degree histogram
    -> fold_config.pna_degree_hist
    -> PNAEncoder
```

The histogram must not use:

```text
test_fold
actual_val
raw full dataset
```

## 23.3 Fresh trainable encoder per fold

For every fold, instantiate a fresh GNN model.

Do not reuse the previous fold's weights.

Do not reuse its optimizer.

Do not reuse its graph output cache.

## 23.4 Same split protocol

Keep the existing project split behavior:

- cluster split uses Butina clusters + `StratifiedGroupKFold`
- random split uses `StratifiedKFold`

Do not change the grouping definition in the GNN task.

## 23.5 Additional CV rigor recommendation

The current repository uses one existing descriptor normalization-statistics file during CV. For the most rigorous future CV, recompute descriptor means/stds inside each fold from that fold's actual training molecules and pass the fold-specific normalization file to the fold dataset.

This is a methodological cleanup separate from the GNN implementation, but an AI agent implementing this task should not make the existing issue worse. If the agent changes this now, it must be applied consistently across **all** encoder baselines, not just the new GNNs.

---

# 24. Experimental-control and metric-maximization rules

These rules keep the task-level experiment interpretable without deliberately weakening any encoder.

## Keep fixed across the initial matrix

```text
- same dataset split
- same seed
- same loss implementation for the named loss
- same descriptors
- same sampler setting
- same validation/test metrics
- same threshold-selection protocol
- same maximum epoch budget
- same early-stopping rule
- same effective batch-size target
- same downstream CompatibilityModel / classifier
```

## Explicitly allowed to differ because it defines model strength

```text
- architecture-specific hidden size when required by a strong implementation
- number of layers/depth
- number of attention heads
- number of towers
- AttentiveFP timesteps
- PNA aggregators/scalers
- pretrained initialization
- pretrained-encoder learning rate or layer-wise learning-rate policy
- architecture-native normalization / residual behavior
```

## No artificial capacity matching

Do **not** do this merely for symmetry:

```text
- truncate CheMeleon hidden states to 300 before the existing projector
- remove edge features from an edge-aware GNN
- reduce a multi-head GATv2 to one head
- reduce PNA to a single aggregator/scaler
- freeze a trainable GNN without experimental justification
- replace a strong pretrained checkpoint with random initialization in the main metric run
```

The existing downstream projection is the correct place to bring encoder outputs into the common 128-dimensional compatibility space.

## Two-stage experiment policy

### Stage A — initial 40-run matrix

Run all requested combinations:

```text
5 GNN encoders x 2 fusion modes x 4 losses = 40 runs
```

Use the recommended starting configurations in Section 25.

### Stage B — metric-maximization tuning

After Stage A, tune only the strongest/promising GNN+fusion+loss combinations using the training split and validation split. Hyperparameter search may change model-specific parameters and learning-rate policy. Do not tune against the held-out test labels.

Recommended tuning targets:

```text
- hidden dimension / width where architecture permits
- depth / number of layers
- dropout
- attention heads
- AttentiveFP timesteps
- PNA towers / scalers
- GNN learning rate
- downstream-head learning rate
- weight decay
- warmup / scheduler settings
```

Select using validation PR-AUC as the primary metric, while also recording F1, MCC, precision, recall and ROC-AUC/accuracy where already supported by the repository.

Do not change the five model definitions into weaker versions just because another architecture has fewer parameters.

# 25. Recommended initial GNN hyperparameter table

Use this as the agent's source of truth.

| Parameter | D-MPNN | AttentiveFP | GINE | GATv2 | PNA |
|---|---:|---:|---:|---:|---:|
| node input | 39 | 39 | 39 | 39 | 39 |
| edge input | 10 | 10 | 10 | 10 | 10 |
| hidden/output | native CheMeleon width | 300 | 300 | 300 | 300 |
| layers/depth | checkpoint-defined (CheMeleon; do not alter) | 2 | 3 | 3 | 3 |
| dropout | checkpoint-defined / preserve pretrained behavior | 0.10 | 0.10 | 0.10 | 0.10 |
| activation | ReLU | operator-native + ReLU | ReLU | ReLU | ReLU |
| heads | — | operator-native | — | 4 | — |
| head width | — | — | — | 75 | — |
| timesteps | — | 2 | — | — | — |
| `train_eps` | — | — | True | — | — |
| PNA towers | — | — | — | — | 5 |
| PNA aggregators | — | — | — | — | mean/min/max/std |
| PNA scalers | — | — | — | — | identity/amplification/attenuation |
| D-MPNN bias | checkpoint-defined | — | — | — | — |

This table is a strong starting point for a small molecular dataset. It does **not** impose a requirement that all five encoders expose the same internal width; the downstream ProjectionHead provides the task-level dimensional alignment.

---

# 26. Test plan

The tests must be implemented before running the full 40-model grid.

## 26.1 Per-encoder shape test

For every encoder:

```python
smiles = [
    "CCO",
    "c1ccccc1",
    "CC(=O)[O-].[Na+]",
]
```

assert:

```text
token_embeddings.ndim == 3
pooled.ndim == 2
token_mask.ndim == 2
```

and:

```text
token_embeddings.shape[0] == 3
pooled.shape == [3, encoder.output_dim]
token_embeddings.shape[2] == encoder.output_dim
token_mask.shape == token_embeddings.shape[:2]
```

## 26.2 Padding semantics test

Use molecules with different atom counts, for example:

```text
C
CC
c1ccccc1
```

Then verify:

- real atoms have `token_mask=False`
- padded positions have `token_mask=True`
- changing only padded embeddings does not affect the existing gated pooling behavior

## 26.3 Pooled-vector correctness test

For a batch of known graphs, independently calculate:

```python
mean(node_embeddings[:num_atoms], dim=0)
```

and assert it matches the encoder's pooled output to a small tolerance.

## 26.4 Missing excipient test

Input:

```python
["CCO", ""]
```

must produce finite tensors.

For the empty item:

```text
token_mask = all True
pooled = all zeros
```

The full compatibility model must still produce finite logits.

## 26.5 Invalid SMILES test

Input:

```text
"this_is_not_a_valid_smiles"
```

must raise a clear `ValueError` from the GNN graph builder.

Do not silently return zero embeddings.

## 26.6 Multi-fragment test

Use:

```text
CC(=O)[O-].[Na+]
```

and verify:

- graph contains two disconnected components
- node count is correct
- no NaNs
- output is 300-dimensional

## 26.7 Single-atom test

Use:

```text
[Na+]
```

and verify:

- 1 node
- 0 real chemical bonds
- empty edge tensors have correct dimensions
- encoder produces a finite 300-dimensional node vector

## 26.8 Gradient-flow test

For each GNN:

```text
build model
-> forward one batch
-> compute BCE loss
-> backward
```

assert at least one GNN parameter has a finite gradient.

This catches accidental `torch.no_grad()`, output detachment, frozen parameters, or stale embedding caches.

## 26.9 Concat smoke test

For each encoder:

```bash
python main.py \
  --encoder <ENCODER> \
  --fusion concat \
  --loss bce \
  --max_epochs 2 \
  --batch_size 4 \
  --run_name smoke_<ENCODER>_concat_bce
```

The exact smoke-test batch size can be small. This is only a correctness run, not a comparison run.

## 26.10 Cross-attention smoke test

For each encoder:

```bash
python main.py \
  --encoder <ENCODER> \
  --fusion cross_attn \
  --pooling global_gated_attention \
  --loss bce \
  --max_epochs 2 \
  --batch_size 4 \
  --run_name smoke_<ENCODER>_cross_attn_bce
```

Verify:

- no shape mismatch
- no NaN attention output
- no all-masked softmax NaN
- logits have shape `[B]`
- gradients reach GNN parameters

## 26.11 All four loss smoke tests

For at least one encoder, verify:

```text
bce
weighted_bce
focal
asl
```

all complete a forward/backward optimizer step.

Then run the full 40-model matrix.

---

# 27. Optional integration test for the complete 40-model matrix

Create a tiny helper test that loops through:

```python
ENCODERS = [
    "dmpnn_chemprop",
    "attentivefp",
    "gine",
    "gatv2",
    "pna",
]

FUSIONS = ["concat", "cross_attn"]

LOSSES = ["bce", "weighted_bce", "focal", "asl"]
```

For each combination:

1. instantiate config
2. compute PNA degree hist if needed
3. instantiate encoder
4. instantiate `CompatibilityModel`
5. run one forward pass
6. run the appropriate loss
7. run `backward()`
8. check finite logit/loss/gradient
9. delete model and empty CUDA cache if applicable

This test does not train for epochs. Its purpose is to prove that the Cartesian product is wired correctly.

---

# 28. Full experiment commands

After smoke tests pass, the 40 runs can be generated programmatically.

Recommended new script:

```text
scripts/run_gnn_grid.py
```

Pseudo-code:

```python
for encoder in ENCODERS:
    for fusion in FUSIONS:
        for loss in LOSSES:
            run_name = make_run_name(encoder, fusion, loss)
            run_subprocess_or_call_training(
                encoder=encoder,
                fusion=fusion,
                loss=loss,
                pooling="global_gated_attention" if fusion == "cross_attn" else None,
                batch_size=64,
                seed=42,
                run_name=run_name,
            )
```

Do not hide failures. If one run fails, log:

```text
encoder
fusion
loss
exception type
exception message
stack trace
```

and continue only if explicitly configured to do so.

---

# 29. Suggested shell commands for individual runs

## D-MPNN

```bash
python main.py --encoder dmpnn_chemprop --fusion concat --loss asl
python main.py --encoder dmpnn_chemprop --fusion cross_attn --loss asl
```

## AttentiveFP

```bash
python main.py --encoder attentivefp --fusion concat --loss asl
python main.py --encoder attentivefp --fusion cross_attn --loss asl
```

## GINE

```bash
python main.py --encoder gine --fusion concat --loss asl
python main.py --encoder gine --fusion cross_attn --loss asl
```

## GATv2

```bash
python main.py --encoder gatv2 --fusion concat --loss asl
python main.py --encoder gatv2 --fusion cross_attn --loss asl
```

## PNA

```bash
python main.py --encoder pna --fusion concat --loss asl
python main.py --encoder pna --fusion cross_attn --loss asl
```

Repeat with:

```text
--loss bce
--loss weighted_bce
--loss focal
```

for every encoder/fusion pair.

---

# 30. Evaluation and result recording

Use the existing metrics implementation:

```text
PR-AUC
F1
MCC
Precision
Recall
Accuracy
confusion matrix
```

Primary model-selection metric:

```text
validation PR-AUC
```

Do not select a winner based on accuracy alone.

Threshold procedure remains:

```text
train
-> choose best checkpoint by validation PR-AUC
-> tune classification threshold on validation F1
-> freeze threshold
-> evaluate test
```

Never tune the decision threshold on the test set.

## 30.1 Required per-run files

Every run should leave:

```text
checkpoints/<run_name>/best_model.pt
metrics/<run_name>/val_metrics.json
metrics/<run_name>/test_metrics.json
metrics/<run_name>/training_history.json
```

For CV:

```text
metrics/<run_name>/cv_metrics.json
```

and per-fold files as produced by the existing harness.

---

# 31. Recommended result table after the 40 runs

Create a CSV such as:

```text
metrics/gnn_experiment_summary.csv
```

with columns:

```text
encoder
fusion
pooling
loss
seed
batch_size
lr
weight_decay
trainable_params
val_pr_auc
val_f1
val_mcc
val_precision
val_recall
val_accuracy
test_pr_auc
test_f1
test_mcc
test_precision
test_recall
test_accuracy
best_epoch
threshold
```

Do not rank encoders by a single arbitrary metric in the implementation layer. The experiment summary should preserve the full metric set and allow later analysis.

---

# 32. Optional CV protocol after single-split smoke/full runs

After the 40 single split runs are confirmed:

```text
5 encoders
x
2 fusion modes
x
4 losses
x
5 folds
=
200 training runs
```

This is expensive, so the recommended practical workflow is:

### Stage A — all 40 single train/val/test runs

Use these to identify a small shortlist based on validation behavior and resource cost.

### Stage B — CV only for shortlisted combinations

Run 5-fold CV for the combinations that will later be compared/ensembled.

The existing project master specification says that a final “better/worse” claim should ultimately be backed by cross-validation. Follow that standard for the final shortlist.

---

# 33. Important warning about the current pretrained GIN

Do not rewrite the existing `pretrained_gin` implementation as part of this task.

It currently depends on:

```text
dgl
dgllife
```

and contains a DGL/GraphBolt compatibility workaround.

The new five encoders should use PyG instead, so that:

```text
pretrained_gin -> existing DGL path
new GNNs       -> PyG path
```

remain isolated.

This reduces the risk of breaking the existing GIN baseline while adding the requested architectures.

---

# 34. Device-handling rules

## CPU

All five GNNs must work on CPU.

## CUDA

When CUDA is available:

```text
PyG Batch -> .to(device)
node features -> GPU
edge_index -> GPU
edge_attr -> GPU
GNN -> GPU
```

Graph cache remains on CPU.

Do not repeatedly move the same cached graph to different devices permanently; create/move the batched copy used in the current forward pass.

## Mixed precision

Do not introduce AMP only for GNNs in the first controlled grid. Keep the precision policy consistent across the five models.

If AMP is later added, apply it consistently to all comparable experiments.

---

# 35. Performance recommendations

The current project repeatedly calls `encoder.encode()` for API and Excipient SMILES inside each training batch.

For the five GNNs:

1. cache graph objects per SMILES
2. batch all API graphs in one PyG `Batch`
3. batch all Excipient graphs in one PyG `Batch`
4. run one GNN forward for the API side
5. run one GNN forward for the Excipient side
6. dense-pad the two outputs separately

Do not construct one giant graph connecting API and excipient molecules. They are two separate molecular graphs, and pair interaction is handled downstream by the existing fusion architecture.

## 35.1 Do not concatenate API and excipient graphs before GNN

Wrong:

```text
API graph + artificial edges + excipient graph -> one GNN
```

Correct:

```text
API graph -> GNN -> API node states
Exc graph -> GNN -> Exc node states
                     |
                     v
          existing cross-attention
```

The latter preserves the intended separation between molecular encoding and pairwise compatibility modeling.

---

# 36. Reproducibility requirements

At the start of each run:

```python
seed_everything(config.seed)
```

Keep:

```text
torch
numpy
random
cudnn deterministic behavior
```

as the existing utility already does.

The PyG graph builder itself must be deterministic; it should not randomly reorder atoms or edges.

The graph cache must not introduce random tensors.

---

# 37. Checkpoint/config validation

When loading a GNN checkpoint, validate:

```text
checkpoint_config.encoder is one of the five GNN names
checkpoint_config.encoder_output_dim matches the saved encoder configuration
checkpoint_config.gnn_node_feature_dim == 39
checkpoint_config.gnn_edge_feature_dim == 10
checkpoint_config.gnn_feature_schema == "attentivefp39_edge10_v1"
```

For PNA:

```text
pna_degree_hist is present
```

If not, fail loudly with a message explaining that the checkpoint is incomplete.

Do not silently replace a missing PNA histogram with a full-dataset histogram.

---

# 38. Failure modes the agent must guard against

## Failure 1 — GNN outputs are detached

Symptom:

```text
GNN parameters have grad=None
```

Cause:

```python
with torch.no_grad():
```

around trainable encoding, or `.detach()` on node embeddings.

Fix:

Only cache graphs; never cache outputs during trainable runs.

## Failure 2 — Cross-attention dimension mismatch

Symptom:

```text
expected projected embedding dimension 128, got incompatible encoder dimension
```

Fix:

Both node embeddings and pooled vectors must pass through the existing ProjectionHead before cross-attention.

## Failure 3 — token mask inverted

Symptom:

Padded atoms receive attention or real atoms get masked.

Fix:

```python
valid_mask = to_dense_batch(...)[1]
token_mask = ~valid_mask
```

## Failure 4 — empty excipient crashes RDKit

Fix:

Use the required masked dummy output for `""`.

## Failure 5 — GATv2 ignores edges

Symptom:

`edge_dim` is missing or `edge_attr` is not passed.

Fix:

Every layer receives `edge_attr` and has `edge_dim=10`.

## Failure 6 — GINE edge dimension mismatch

Symptom:

GINE expects node and edge dimensions to align internally.

Fix:

Instantiate `GINEConv(..., edge_dim=10)` so PyG performs the required edge projection to the node hidden dimension.

## Failure 7 — PNA construction fails because `deg` is missing

Fix:

Compute the train-only degree histogram before constructing `PNAEncoder`.

## Failure 8 — PNA histogram leaks validation/test structure

Fix:

Compute it from training molecules only; redo it for each CV fold.

## Failure 9 — AttentiveFP returns only graph-level output

Fix:

Use a node-returning implementation of the public AttentiveFP message-passing path. Never treat the final task output as atom tokens.

## Failure 10 — D-MPNN checkpoint loading mismatch

Fix:

Do not load CheMeleon into the common-feature baseline unless exact feature/architecture compatibility is verified.

## Failure 11 — OOM on one encoder changes only its batch size

Fix:

Use gradient accumulation to preserve effective batch size 64 and record the micro-batch/accumulation settings.

## Failure 12 — output cache makes GNN appear frozen

Fix:

Do not cache node/pooled embeddings for trainable encoders.

---

# 39. Suggested implementation order for an AI coding agent

Follow this order exactly.

## Phase 1 — dependency + shared graph layer

1. Add PyG dependency.
2. Add `gnn_common.py`.
3. Implement exact 39/10 feature schema.
4. Implement graph cache.
5. Implement PyG batching.
6. Implement dense padding + inverted padding mask.
7. Implement mean pooling.
8. Implement missing-SMILES behavior.
9. Add unit tests for feature dimensions and graph construction.

Do not touch the downstream model yet.

## Phase 2 — implement the five encoders individually

Implement in this order:

```text
D-MPNN
AttentiveFP
GINE
GATv2
PNA
```

After each one:

```text
shape test
single-atom test
salt test
invalid-SMILES test
gradient test
```

## Phase 3 — registry/config integration

1. Add config fields.
2. Add registry entries.
3. Update `main.py`.
4. Update `cross_validate.py`.
5. Update `inference.py`.
6. Update any ensemble/checkpoint-loading code that enumerates encoder names.

## Phase 4 — model integration

Verify that no new GNN-specific branch is required in `CompatibilityModel`.

Run:

```text
encoder -> concat
encoder -> cross_attn
```

for all five.

## Phase 5 — loss integration

Verify all four losses.

## Phase 6 — full 40-model single-split grid

Run all 40 combinations with fixed training conditions.

## Phase 7 — shortlist + CV

Run the existing CV harness for the final shortlist.

## Phase 8 — optional CheMeleon experiment

Only after the baseline five-way comparison is complete.

---

# 40. Changes that should NOT be made as part of this task

Do not:

- replace MoLFormer
- remove the current pretrained GIN
- remove ChemBERTa
- remove fixed-vector baselines
- change the descriptor list
- change the existing classifier head
- add a separate loss system for GNNs
- add a separate optimizer for GNNs
- add cross-graph GNN edges
- add random SMILES augmentation
- introduce extra training data
- use validation/test molecules to fit graph normalizers
- precompute trainable GNN outputs and treat them as fixed vectors
- silently change the split protocol
- silently change the batch size for only one model
- tune test thresholds
- change the existing ensemble behavior merely to make the GNN experiment pass

The new GNNs are **additional encoder choices**, not a replacement project.

---

# 41. Main D-MPNN foundation-model requirement — CheMeleon

CheMeleon is **not optional** for the primary D-MPNN run in this project. It is the default initialization for `dmpnn_chemprop`.

Required behavior:

1. load the exact local `chemeleon_mp.pt` checkpoint;
2. use the exact Chemprop architecture and native featurization required by that checkpoint;
3. preserve the checkpoint's native hidden width and depth;
4. expose atom/vertex hidden states for the compatibility model;
5. fine-tune the message-passing parameters end-to-end;
6. use a smaller encoder learning rate than the newly initialized downstream head;
7. fail loudly on checkpoint/architecture/featurizer mismatch;
8. never silently fall back to scratch D-MPNN;
9. keep the checkpoint on local disk and do not require internet access after the one-time download.

Official local resource:

```text
models/pretrained/chemeleon_mp.pt
```

Official source:

```text
https://github.com/JacksonBurns/chemeleon
https://zenodo.org/records/15460715
https://zenodo.org/records/15460715/files/chemeleon_mp.pt
```

The published Zenodo record identifies `chemeleon_mp.pt` as a 34.9 MB model file with MD5 `6a80b54fdb7de37ef0374d302f01e8ce`.

The scratch D-MPNN remains available only as an ablation/debugging model under `dmpnn_scratch`; it is not the main metric-seeking configuration.

### Required offline verification

After the one-time download, run a local smoke test that proves:

```text
- the checkpoint can be opened from disk
- the local MD5 matches the published checksum
- Chemprop can construct the exact message-passing block
- a small set of SMILES produces atom-level embeddings
- no HTTP request is made during encoder construction or `encode()`
```

The normal training and inference commands must work with network access disabled once dependencies and the checkpoint have been installed locally.

# 42. Optional future extension: graph-level pretraining

Do not implement generic self-supervised pretraining for the five GNNs in this task.

The requested experiment is:

```text
supervised compatibility training
```

on the existing split.

Any additional pretraining beyond the official CheMeleon foundation initialization must be a separate research experiment with its own data and leakage controls.

---

# 43. Integration checklist for the coding agent

Before declaring the task complete, verify every item below.

## Encoder availability

- [ ] `dmpnn_chemprop` appears in `ENCODER_REGISTRY`
- [ ] `attentivefp` appears in `ENCODER_REGISTRY`
- [ ] `gine` appears in `ENCODER_REGISTRY`
- [ ] `gatv2` appears in `ENCODER_REGISTRY`
- [ ] `pna` appears in `ENCODER_REGISTRY`

## Common contract

- [ ] `is_sequence_capable=True` for all five
- [ ] `output_dim=300` for AttentiveFP/GINE/GATv2/PNA by default
- [ ] `dmpnn_chemprop.output_dim` equals the CheMeleon checkpoint's native message-passing width
- [ ] `encode()` returns exactly three outputs
- [ ] node output is `[B,L,D]` where `D=encoder.output_dim`
- [ ] pooled output is `[B,D]`
- [ ] token mask is `[B,L]`, bool, True=padding

## Graph data

- [ ] 39 atom features
- [ ] 10 bond features
- [ ] bidirectional edges
- [ ] no shared raw-graph self loops
- [ ] multi-fragment molecules supported
- [ ] single-atom molecules supported
- [ ] invalid SMILES fails clearly
- [ ] graph cache is CPU-side and output cache is absent

## Model integration

- [ ] concat works for all five
- [ ] cross-attn works for all five
- [ ] global gated attention works for all five
- [ ] existing descriptor path is unchanged
- [ ] existing classifier path is unchanged
- [ ] missing excipient still works

## Loss integration

- [ ] BCE works
- [ ] weighted BCE works
- [ ] focal works
- [ ] ASL works

## PNA-specific

- [ ] degree histogram is training-only
- [ ] degree histogram is fold-local in CV
- [ ] histogram is stored in checkpoint config
- [ ] inference reconstructs PNA from the saved histogram

## Training

- [ ] GNN parameters are trainable
- [ ] GNN gradients are finite
- [ ] same optimizer settings are used
- [ ] batch size/effective batch size is controlled
- [ ] early stopping remains PR-AUC based

## Reproducibility

- [ ] seed is applied
- [ ] run directories are unique
- [ ] no metrics/checkpoints are overwritten

## Experiment coverage

- [ ] 5 encoders x 2 fusion x 4 losses = 40 combinations can be launched
- [ ] each combination has a unique run name
- [ ] metrics are saved separately

---

# 44. Final expected architecture

For each molecule:

```text
                        RDKit
                          |
                          v
                 Molecular Graph
               x: 39 atom features
             e: 10 bond features
                          |
          +---------------+----------------+
          |               |                |
       D-MPNN        AttentiveFP         GINE
          |               |                |
          +---------------+----------------+
                          |
                 OR GATv2 / PNA
                          |
                          v
              Node embeddings [N,D]
                          |
               +----------+----------+
               |                     |
        mean pooling            node sequence
               |                     |
          [D] global          [N,D] tokens
               |                     |
               +----------+----------+
                          |
                 Existing Model
                    projection
                 [128-dimensional]
                          |
              +-----------+-----------+
              |                       |
         concat fusion          cross-attention
              |                       |
              |                bidirectional 8-head
              |                global+gated pooling
              |                       |
              +-----------+-----------+
                          |
                  21 descriptors
                          |
               API/Excipient structural
                  + interaction
                  + abs difference
                          |
                     classifier
                          |
                        logit
                          |
                     four losses
```

The essential architectural idea is:

```text
GNN = molecule encoder
existing model = pairwise compatibility architecture
```

Do not collapse those two roles into one model.

---

# 44.5 Offline resource acceptance

The implementation is not complete until the following is true:

```text
models/pretrained/chemeleon_mp.pt exists locally
MD5 = 6a80b54fdb7de37ef0374d302f01e8ce
training uses that local path
inference uses that local path
normal runs do not download the checkpoint
missing checkpoint produces an actionable error
```

The four PyG GNN architectures are code-defined and do not need external pretrained weight files for this requested implementation.

# 45. Acceptance criterion

The implementation is accepted only when all of the following are true:

1. All five requested GNNs can be selected using `--encoder`.
2. AttentiveFP/GINE/GATv2/PNA output 300-dimensional atom and pooled embeddings; CheMeleon preserves its native checkpoint width and is aligned by the existing ProjectionHead.
3. All five support both `concat` and `cross_attn`.
4. The existing `global_gated_attention` cross-attention path works with variable atom counts and padding masks.
5. All four existing losses work with every GNN/fusion combination.
6. The GNNs remain trainable end-to-end.
7. PNA uses train-only, fold-local degree histograms.
8. Graph parsing and empty/multi-fragment molecule behavior are tested.
9. Existing MolFormer/pretrained-GIN/ChemBERTa/fixed-vector workflows still work.
10. The 40 requested experiment combinations can be launched without manually editing source code between runs.
11. Checkpoints contain enough config to reconstruct the exact GNN, including CheMeleon foundation provenance.
12. `dmpnn_chemprop` uses the verified pretrained CheMeleon message-passing weights by default.
13. The implementation never silently downgrades CheMeleon to scratch D-MPNN after a checkpoint-load failure.
14. The CheMeleon weight file is stored locally and normal runtime paths do not access the internet.
15. A separate `dmpnn_scratch` ablation is available for isolating the value of pretraining.

---

# 46. Final note to the AI coding agent

**Non-negotiable metric-first rule:** Do not nerf any of the five GNNs. Use CheMeleon-pretrained D-MPNN locally, preserve its native representation, keep all edge-aware mechanisms enabled, and allow architecture-specific tuning in Stage B when it is expected to improve validation metrics.

When implementing this specification, prefer **small, isolated changes** over refactoring the entire repository.

The existing architecture is already designed around a swappable encoder contract. The correct implementation is therefore to make each new GNN satisfy that contract and let the existing pair-fusion/loss/training code do the rest.

The most important correctness properties are:

```text
same input graphs
same feature schema
compatible downstream projection
same training/evaluation protocol where practical
trainable node embeddings
strong pretrained initialization allowed and preferred for D-MPNN
correct padding mask
train-only PNA degree statistics
no hidden runtime download or undocumented initialization
```

Once these are true, the five GNNs become proper additional models in the existing API–Excipient compatibility framework and are directly comparable under the requested concat/cross-attention × four-loss grid.

