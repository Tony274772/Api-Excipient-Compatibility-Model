# Implementation Plan: Explicit Pairwise Token Interaction + Global Representations

## Objective

Replace the current CLS-only / molecule-separated pooling behavior in the **cross-attention** branch with an explicit **token-pair interaction architecture**.

The new architecture must:

1. Keep the existing frozen molecular encoder and 768 -> 256 -> 128 projection pipeline.
2. Keep the existing bidirectional cross-attention.
3. Remove the prepended global position from the molecular-token pair grid.
4. Construct an explicit feature for **every valid API-token × Excipient-token pair**.
5. Mask all pair combinations that contain API or Excipient padding.
6. Pool the valid pair features into one fixed-size pairwise representation.
7. Preserve the cross-attended global API and Excipient vectors separately and add them **at the end of pairwise pooling**.
8. Add the existing descriptor representations and excipient availability flag.
9. Use a new classifier input dimension based on the new representation.
10. Keep the existing training, loss, sampling, thresholding, and evaluation behavior unchanged.

This is a new architecture, not just a pooling replacement.

---

# 1. Existing architecture context

The current repository uses:

```text
Encoder:
MoLFormer-XL by default
output dimension = 768

Projection:
768 -> 256 -> 128

Cross-attention:
embed_dim = 128
num_heads = 8
dropout = config.attn_dropout

Bidirectional:
Excipient queries -> API keys/values
API queries -> Excipient keys/values
```

The projected pooled molecular vector is prepended as position `0`.

Therefore after the current preprocessing:

```text
API sequence:
[B, L_api + 1, 128]

Excipient sequence:
[B, L_exc + 1, 128]
```

The current code then performs cross-attention and obtains:

```text
refined_api:
[B, L_api + 1, 128]

refined_exc:
[B, L_exc + 1, 128]
```

The old code extracts only:

```python
h_api_struct = refined_api[:, 0, :]
h_exc_struct = refined_exc[:, 0, :]
```

The new architecture must NOT do that as the final representation.

---

# 2. Scope of files

## Modify

```text
src/model.py
src/config.py
main.py
src/cross_validate.py
```

## Add

```text
tests/test_pairwise_interaction.py
```

## Do not modify unless a direct compatibility problem is found

```text
src/dataset.py
src/train.py
src/evaluate.py
src/loss.py
src/descriptors.py
src/encoders/molformer_encoder.py
src/encoders/chemberta_encoder.py
src/encoders/gin_encoder.py
```

No new third-party dependency is required.

---

# 3. Target high-level architecture

The final architecture is:

```text
API SMILES
    |
Frozen MoLFormer
    |
768-D token embeddings + 768-D pooled embedding
    |
API projection: 768 -> 256 -> 128
    |
API tokens [B,L_api,128]
    +
global API vector [B,128]
    |
prepend global
    |
API sequence [B,L_api+1,128]


Excipient SMILES
    |
Frozen MoLFormer
    |
768-D token embeddings + 768-D pooled embedding
    |
Excipient projection: 768 -> 256 -> 128
    |
Excipient tokens [B,L_exc,128]
    +
global Excipient vector [B,128]
    |
prepend global
    |
Exc sequence [B,L_exc+1,128]


                 BIDIRECTIONAL
                 CROSS-ATTENTION
                       |
              +--------+--------+
              |                 |
              v                 v
      refined API         refined Excipient
    [B,L_api+1,128]     [B,L_exc+1,128]
              |                 |
              |                 |
       global API          global Exc
        [B,128]             [B,128]
              |                 |
              |                 |
      API molecular       Exc molecular
      tokens [B,L_a,128]  tokens [B,L_e,128]
              |                 |
              +--------+--------+
                       |
          EXPLICIT TOKEN-PAIR GRID
                       |
              every API token i
                       x
              every Exc token j
                       |
                       v
              pair features [512]
                       |
                       v
        pair tensor [B,L_a,L_e,512]
                       |
              pair padding mask
                       |
                       v
            valid pair attention pooling
                       |
                       v
                [B,512]
                       |
        +--------------+--------------+
        |              |              |
        v              v              v
   Pair pooled     Global API     Global Exc
      [512]           [128]          [128]
        |              |              |
        +--------------+--------------+
                       |
                    concat
                  512+128+128
                       |
                    [B,768]
                       |
             API descriptor [24]
                       +
          Excipient descriptor [24]
                       +
          Excipient availability [1]
                       |
                       v
                    [B,817]
                       |
                  classifier
               817 -> 256 -> 64 -> 1
                       |
                    sigmoid
                       |
              P(incompatibility)
```

---

# 4. Important distinction from Global + Gated Attention

This implementation is NOT:

```text
API tokens -> pool API
Exc tokens -> pool Exc
```

Instead:

```text
API token i
      +
Excipient token j
      |
      v
explicit pair feature
```

for every valid `(i,j)`.

The global API and global Excipient representations are retained separately and added **after** pairwise pooling.

---

# 5. Step 1 — Keep existing encoder pipeline

Do not change the encoder.

For MoLFormer:

```text
API:
[B,L_api,768]
[B,768]

Excipient:
[B,L_exc,768]
[B,768]
```

Project with the existing separate projection heads:

```text
API:
768 -> 256 -> 128

Excipient:
768 -> 256 -> 128
```

Therefore:

```text
api_tok_proj:
[B,L_api,128]

api_pool_proj:
[B,128]

exc_tok_proj:
[B,L_exc,128]

exc_pool_proj:
[B,128]
```

---

# 6. Step 2 — Keep the existing global-token construction

The current model prepends the projected pooled embedding.

Keep this:

```python
api_cls = api_pool_proj.unsqueeze(1)
api_seq = torch.cat([api_cls, api_tok_proj], dim=1)
```

Result:

```text
api_seq:
[B,L_api+1,128]
```

Similarly:

```python
exc_cls = exc_pool_proj.unsqueeze(1)
exc_seq = torch.cat([exc_cls, exc_tok_proj], dim=1)
```

Result:

```text
exc_seq:
[B,L_exc+1,128]
```

Keep the current extended masks:

```text
api_mask_ext:
[B,L_api+1]

exc_mask_ext:
[B,L_exc+1]
```

with:

```text
True  = padding / invalid
False = valid
```

---

# 7. Step 3 — Preserve existing missing-excipient handling

The existing repository already supports missing/unavailable excipients using:

```text
self.exc_placeholder
self.exc_global_placeholder
exc_available
```

Do not remove that mechanism.

Keep the existing behavior that:

```text
exc_available = 0
```

causes:

```text
position 0 = valid global placeholder
molecular token positions = masked
```

This is important because the pairwise grid must not treat missing-excipient placeholder tokens as normal molecular tokens.

For an unavailable excipient:

```text
exc_token_mask = exc_mask_ext[:,1:]
```

can become:

```text
[True, True, ..., True]
```

meaning there are zero valid molecular tokens.

The pairwise pooling logic must safely handle this.

---

# 8. Step 4 — Preserve the existing bidirectional cross-attention

Do NOT modify the actual cross-attention architecture.

Keep:

### Excipient queries API

```python
refined_exc, _ = self.cross_attn_exc(
    query=exc_seq,
    key=api_seq,
    value=api_seq,
    key_padding_mask=api_mask_ext,
)
```

followed by existing:

```python
refined_exc = torch.nan_to_num(refined_exc)
refined_exc = self.ln_exc(exc_seq + refined_exc)
```

### API queries Excipient

```python
refined_api, _ = self.cross_attn_api(
    query=api_seq,
    key=exc_seq,
    value=exc_seq,
    key_padding_mask=exc_mask_ext,
)
```

followed by existing:

```python
refined_api = torch.nan_to_num(refined_api)
refined_api = self.ln_api(api_seq + refined_api)
```

Do not change:

```text
embed_dim = 128
num_heads = 8
```

---

# 9. Step 5 — Separate global and molecular token representations

After cross-attention:

```python
api_global = refined_api[:, 0, :]
exc_global = refined_exc[:, 0, :]
```

Dimensions:

```text
api_global:
[B,128]

exc_global:
[B,128]
```

Now explicitly remove position 0 from the token-pair construction:

```python
api_tokens = refined_api[:, 1:, :]
exc_tokens = refined_exc[:, 1:, :]
```

Dimensions:

```text
api_tokens:
[B,L_api,128]

exc_tokens:
[B,L_exc,128]
```

Do not include position 0 in the explicit token-pair grid.

The global vectors are retained separately and used later.

---

# 10. Step 6 — Obtain molecular-token masks

Use the existing extended masks and remove the global position:

```python
api_token_mask = api_mask_ext[:, 1:]
exc_token_mask = exc_mask_ext[:, 1:]
```

Dimensions:

```text
api_token_mask:
[B,L_api]

exc_token_mask:
[B,L_exc]
```

Mask semantics:

```text
True  = invalid / padding
False = valid molecular token
```

Do not rebuild masks from token lengths.

---

# 11. Step 7 — Construct every API-token × Excipient-token pair

This is the key new architecture.

Given:

```text
api_tokens:
[B,L_api,128]

exc_tokens:
[B,L_exc,128]
```

construct:

```text
pair features:
[B,L_api,L_exc,512]
```

For every API token `i` and Excipient token `j`:

```text
api_i = [128]
exc_j = [128]
```

Construct four components:

### Component 1 — API token

```text
api_i
[128]
```

### Component 2 — Excipient token

```text
exc_j
[128]
```

### Component 3 — Element-wise product

```text
api_i * exc_j
[128]
```

### Component 4 — Absolute difference

```text
abs(api_i - exc_j)
[128]
```

Concatenate:

```text
128 + 128 + 128 + 128
= 512
```

So:

```text
pair_ij:
[512]
```

for every `(i,j)`.

---

# 12. Efficient pair construction

Do NOT implement nested Python loops over every molecule/token pair.

Use broadcasting.

For example:

```python
api_expanded = api_tokens.unsqueeze(2)   # [B,L_api,1,128]
exc_expanded = exc_tokens.unsqueeze(1)   # [B,1,L_exc,128]

product = api_expanded * exc_expanded
difference = torch.abs(api_expanded - exc_expanded)

pair_tensor = torch.cat(
    [
        api_expanded.expand(-1, -1, exc_tokens.size(1), -1),
        exc_expanded.expand(-1, api_tokens.size(1), -1, -1),
        product,
        difference,
    ],
    dim=-1,
)
```

Expected:

```text
pair_tensor:
[B,L_api,L_exc,512]
```

The implementation may use another memory-safe vectorized equivalent, but must not use a Python loop over `i,j`.

---

# 13. Pair tensor meaning

For each location:

```text
pair_tensor[:, i, j, :]
```

the 512 dimensions are:

```text
0:128       API token i
128:256     Excipient token j
256:384     API_i * Excipient_j
384:512     abs(API_i - Excipient_j)
```

Maintain this ordering because it makes debugging and reproducibility easier.

---

# 14. Step 8 — Build the pairwise padding mask

A pair is valid ONLY if both component tokens are valid.

Given:

```text
api_token_mask:
[B,L_api]

exc_token_mask:
[B,L_exc]
```

construct:

```python
pair_mask = (
    api_token_mask.unsqueeze(2)
    | exc_token_mask.unsqueeze(1)
)
```

Dimensions:

```text
pair_mask:
[B,L_api,L_exc]
```

Semantics:

```text
True  = invalid pair
False = valid pair
```

Examples:

```text
valid API + valid Excipient
    -> valid pair

valid API + padded Excipient
    -> invalid pair

padded API + valid Excipient
    -> invalid pair

padded API + padded Excipient
    -> invalid pair
```

This exact rule is required.

---

# 15. Step 9 — Handle fully masked pair grids

A fully masked pair grid can happen when:

```text
exc_available = 0
```

because the missing excipient has zero valid molecular tokens.

Do NOT apply normal softmax and assume the result is valid.

The pairwise pooling module must be robust to:

```text
pair_mask[b,:,:] == True
```

for every pair.

For such a sample:

```text
pair attention weights = all zero
pair pooled vector = zero vector
```

This must be finite.

Then later the model can rely on the retained global placeholder representation:

```text
exc_global
```

and the existing availability flag.

The zero pair-pooled vector is intentional for the missing-excipient case.

---

# 16. Step 10 — Flatten valid pair locations

The pair tensor:

```text
[B,L_api,L_exc,512]
```

has variable numbers of valid pairs.

Flatten the pair dimensions:

```python
pair_flat = pair_tensor.view(
    B,
    L_api * L_exc,
    512,
)
```

Result:

```text
[B,L_api*L_exc,512]
```

Flatten the pair mask the same way:

```python
pair_mask_flat = pair_mask.view(
    B,
    L_api * L_exc,
)
```

Result:

```text
[B,L_api*L_exc]
```

The order must match between:

```text
pair_flat
pair_mask_flat
```

---

# 17. Step 11 — Pairwise attention pooling

Create a reusable pairwise attention pooling module.

It should take:

```text
pair_flat:
[B,N_pairs,512]

pair_mask_flat:
[B,N_pairs]
```

and return:

```text
pair_pooled:
[B,512]

pair_weights:
[B,N_pairs]
```

where:

```text
N_pairs = L_api * L_exc
```

Use masked attention:

```text
score = learn(pair_feature)

invalid pair -> score masked
softmax over valid pairs

pair_pooled =
Σ weight_ij * pair_feature_ij
```

A simple learned attention scorer is sufficient.

Recommended:

```python
class PairwiseAttentionPooling(nn.Module):

    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()

        self.score_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, x, padding_mask=None):
        scores = self.score_net(x).squeeze(-1)

        if padding_mask is None:
            valid_mask = torch.ones(
                x.shape[:2],
                dtype=torch.bool,
                device=x.device,
            )
        else:
            valid_mask = ~padding_mask

        scores = scores.masked_fill(~valid_mask, -1e9)

        weights = torch.softmax(scores, dim=1)

        # Explicitly zero invalid locations.
        weights = weights * valid_mask.to(weights.dtype)

        # Re-normalize valid weights.
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        weights = weights / denom

        pooled = torch.sum(
            weights.unsqueeze(-1) * x,
            dim=1,
        )

        return pooled, weights
```

The same fully-masked safety requirement applies here.

---

# 18. Pairwise pooling output

For normal examples:

```text
pair_flat:
[B,N_pairs,512]

pair_weights:
[B,N_pairs]

pair_pooled:
[B,512]
```

For missing-excipient examples:

```text
pair_weights:
all zeros

pair_pooled:
[B,512] zero vector
```

The final representation remains finite because the global placeholder is retained separately.

---

# 19. Step 12 — Preserve global API and Excipient representations

Do NOT pool the global vectors into the pairwise grid.

Keep them separately:

```text
api_global:
[B,128]

exc_global:
[B,128]
```

The global vectors must be added **after** pairwise pooling.

The representation at this point is:

```text
pair_pooled:
[B,512]

api_global:
[B,128]

exc_global:
[B,128]
```

Concatenate:

```python
pair_core = torch.cat(
    [
        pair_pooled,
        api_global,
        exc_global,
    ],
    dim=1,
)
```

Dimension:

```text
512 + 128 + 128
= 768
```

Therefore:

```text
pair_core:
[B,768]
```

This is the meaning of "global at pairwise end":

```text
explicit token-pair information
+
global API information
+
global Excipient information
```

---

# 20. Step 13 — Add descriptor representations

Keep the current descriptor pipeline unchanged.

API descriptor:

```text
[B,21]
    ↓
21 -> 32 -> 24
    ↓
[B,24]
```

Excipient descriptor:

```text
[B,21]
    ↓
21 -> 32 -> 24
    ↓
[B,24]
```

Then concatenate:

```python
pair_vector = torch.cat(
    [
        pair_core,       # [B,768]
        d_api,           # [B,24]
        d_exc,           # [B,24]
        exc_available.unsqueeze(1),  # [B,1]
    ],
    dim=1,
)
```

Dimension:

```text
768 + 24 + 24 + 1
= 817
```

Therefore:

```text
pair_vector:
[B,817]
```

---

# 21. Important change: the classifier input is no longer 609

The old architecture had:

```text
609 -> 128 -> 64 -> 1
```

That is NOT correct for this new explicit-pairwise architecture unless an additional projection is deliberately introduced.

The recommended new classifier is:

```text
817 -> 256 -> 64 -> 1
```

Keep:

```text
GELU
Dropout
GELU
Dropout
```

using the existing configured dropout values.

Recommended implementation:

```python
self.classifier = nn.Sequential(
    nn.Linear(817, 256),
    nn.GELU(),
    nn.Dropout(config.clf_dropout_1),
    nn.Linear(256, 64),
    nn.GELU(),
    nn.Dropout(config.clf_dropout_2),
    nn.Linear(64, 1),
)
```

Do not keep the old 609-input classifier.

---

# 22. Alternative if the existing classifier dimensions must be preserved

Do NOT automatically do this.

Only use this alternative if there is a specific backward-compatibility requirement:

```text
817 -> 609 -> 128 -> 64 -> 1
```

However, this adds another arbitrary projection.

The preferred architecture for this new experiment is:

```text
817 -> 256 -> 64 -> 1
```

because 817 is the actual representation dimension.

---

# 23. Final dimensionality chain

The complete new dimensionality chain is:

```text
API MoLFormer tokens:
[B,L_api,768]

Excipient MoLFormer tokens:
[B,L_exc,768]

Projection:
API      [B,L_api,128]
Excipient[B,L_exc,128]

Global prepend:
API      [B,L_api+1,128]
Excipient[B,L_exc+1,128]

Cross-attention:
refined API      [B,L_api+1,128]
refined Excipient[B,L_exc+1,128]

Split:
API global       [B,128]
Exc global       [B,128]

API tokens       [B,L_api,128]
Exc tokens       [B,L_exc,128]

Explicit pair construction:
[B,L_api,L_exc,512]

Flatten:
[B,L_api*L_exc,512]

Pairwise attention pooling:
[B,512]

Add global API:
[B,128]

Add global Exc:
[B,128]

Concat:
512 + 128 + 128
= 768

API descriptors:
[B,24]

Exc descriptors:
[B,24]

Availability:
[B,1]

Final pair vector:
768 + 24 + 24 + 1
= 817

Classifier:
817 -> 256 -> 64 -> 1

Output:
[B,1] internally
[B] after squeeze
```

---

# 24. Complete architecture diagram

```text
                              API SMILES
                                   |
                             Frozen MoLFormer
                                   |
                         +---------+---------+
                         |                   |
                    tokens 768-D        pooled 768-D
                         |                   |
                    768→256→128         768→256→128
                         |                   |
                    API tokens          API global
                  [B,L_a,128]            [B,128]
                         |                   |
                         +---------+---------+
                                   |
                              prepend global
                                   |
                           [B,L_a+1,128]


                           EXCIPIENT SMILES
                                   |
                             Frozen MoLFormer
                                   |
                         +---------+---------+
                         |                   |
                    tokens 768-D        pooled 768-D
                         |                   |
                    768→256→128         768→256→128
                         |                   |
                   Exc tokens          Exc global
                  [B,L_e,128]            [B,128]
                         |                   |
                         +---------+---------+
                                   |
                              prepend global
                                   |
                           [B,L_e+1,128]


                    +-----------------------------+
                    |  BIDIRECTIONAL              |
                    |  CROSS-ATTENTION             |
                    |                             |
                    | Exc queries -> API K/V      |
                    | API queries -> Exc K/V      |
                    +--------------+--------------+
                                   |
                    +--------------+--------------+
                    |                             |
                    v                             v
             refined API                  refined Excipient
            [B,L_a+1,128]                [B,L_e+1,128]
                    |                             |
             +------+-----+                 +-----+------+
             |            |                 |            |
             v            v                 v            v
         global        tokens           global        tokens
          [128]       [L_a,128]          [128]       [L_e,128]
             |            |                 |            |
             |            +--------+--------+            |
             |                     |                     |
             |                     v                     |
             |             EXPLICIT TOKEN PAIRS          |
             |                     |                     |
             |          every API token i ×             |
             |          every Excipient token j         |
             |                     |                     |
             |                     v                     |
             |               [B,L_a,L_e,512]             |
             |                     |                     |
             |              pair padding mask            |
             |                     |                     |
             |                     v                     |
             |           Pairwise Attention Pooling      |
             |                     |                     |
             |                  [B,512]                  |
             |                     |                     |
             +---------------------+---------------------+
                                   |
                     pair [512] + API global [128]
                               + Exc global [128]
                                   |
                               CONCAT
                                   |
                                [B,768]
                                   |
                         +---------+---------+
                         |                   |
                   API descriptors      Exc descriptors
                      [B,21]               [B,21]
                         |                   |
                       21→32→24            21→32→24
                         |                   |
                       [B,24]              [B,24]
                         |                   |
                         +---------+---------+
                                   |
                         + availability [B,1]
                                   |
                                   v
                             [B,817]
                                   |
                            classifier MLP
                             817→256→64→1
                                   |
                                  logit
                                   |
                                sigmoid
                                   |
                         P(INCOMPATIBILITY)
```

---

# 25. Pair feature layout

Every token pair `(i,j)` must contain:

```text
pair_ij =
[
    API_i,                    # 128
    EXC_j,                    # 128
    API_i * EXC_j,            # 128
    abs(API_i - EXC_j),       # 128
]
```

Therefore:

```text
pair_ij = 512-D
```

The exact order must remain:

```text
API
EXC
PRODUCT
ABS_DIFF
```

---

# 26. Why the global vectors are kept separately

The pairwise token grid represents **local interaction information**.

The global vectors represent **overall molecule-level information after cross-attention**.

The final pair representation combines:

```text
local token-pair interaction
+
global API context
+
global Excipient context
```

Do not replace the globals with pairwise pooling.

Do not include global position `0` inside the token-pair grid.

---

# 27. Padding behavior

The pair grid must respect BOTH masks.

For pair `(i,j)`:

```text
valid_pair(i,j)
=
valid_api(i)
AND
valid_excipient(j)
```

Equivalent invalid-mask formula:

```python
pair_mask = api_token_mask.unsqueeze(2) | exc_token_mask.unsqueeze(1)
```

where:

```text
True = invalid
False = valid
```

The pairwise attention pooling must use:

```text
pair_mask_flat
```

before softmax.

Padding combinations must have zero attention.

---

# 28. Missing excipient behavior

For:

```text
exc_available = 0
```

the repository's current masking produces no valid molecular excipient tokens.

Therefore:

```text
pair_mask = all True
```

for that sample.

The pairwise pooling module must return:

```text
pair_weights = all zero
pair_pooled  = zero [512]
```

The preserved global representation remains:

```text
exc_global = refined_exc[:,0,:]
```

which is based on the existing learned missing-excipient global placeholder mechanism.

The final representation then contains:

```text
pair_pooled    [512] = zero
api_global     [128]
exc_global     [128] = placeholder-based
api_desc        [24]
exc_desc        [24]
availability    [1] = 0
```

No NaNs are permitted.

---

# 29. Pairwise attention weights for inspection

Keep:

```python
pair_weights
```

available for optional inspection.

For a flattened pair grid:

```text
pair_weights:
[B,L_api*L_exc]
```

If stored on the model:

```python
self.pair_token_weights = pair_weights.detach()
```

Do not retain the computation graph.

The weights can later be reshaped to:

```text
[B,L_api,L_exc]
```

for visualization.

This can show:

```text
which API token ↔ which Excipient token
```

pair the pooling layer emphasized.

Do not call attention weights causal explanations.

---

# 30. Memory requirements

The explicit pairwise architecture is more expensive than Global + Gated Pooling.

The pair tensor is:

```text
[B,L_api,L_exc,512]
```

Therefore memory grows approximately with:

```text
L_api × L_exc
```

Do NOT create pair features using Python nested loops.

Use vectorized broadcasting.

If the repository encounters very long sequences, the implementation may use a memory-conscious equivalent, but it must preserve exactly the same pair-feature definition and masking semantics.

Do not silently truncate tokens.

---

# 31. Configuration changes

In `src/config.py`, add:

```python
pair_pool_hidden_dim: int = 256
pair_pooling: Literal["pairwise_attention"] = "pairwise_attention"
```

There is no need to expose multiple pair-pooling modes unless desired.

Do not confuse this with the previous `pooling` configuration for Global + Gated Attention.

Recommended configuration:

```text
fusion = "cross_attn"
pair_pooling = "pairwise_attention"
```

This explicit-pairwise architecture is intended for the cross-attention path.

---

# 32. Experiment path naming

Update `resolve_paths()` so this architecture receives a distinct experiment directory.

Recommended:

```text
molformer_cross_attn_pairwise_asl
```

For example:

```text
checkpoints/molformer_cross_attn_pairwise_asl
metrics/molformer_cross_attn_pairwise_asl
```

Do not overwrite results from:

```text
molformer_cross_attn_cls_asl
molformer_cross_attn_global_gated_attention_asl
```

Use a distinct directory.

---

# 33. CLI changes

In `main.py`, add:

```python
parser.add_argument(
    "--pairwise",
    action="store_true",
)
```

However, because this is replacing the structural cross-attention representation, a cleaner configuration is preferable.

Recommended approach:

```python
parser.add_argument(
    "--pairwise",
    action="store_true",
    help="Use explicit API-token × Excipient-token pairwise pooling",
)
```

Then:

```python
if args.pairwise:
    config.pairwise = True
```

If the repository already has a general `fusion/pooling` configuration mechanism, use that instead of inventing redundant flags.

The final implementation must have one unambiguous way to select:

```text
CLS baseline
Global + Gated Attention
Explicit Pairwise
```

Do not allow conflicting options to silently override one another.

---

# 34. Recommended configuration abstraction

Prefer a single field such as:

```python
pooling: Literal[
    "cls",
    "global_gated_attention",
    "explicit_pairwise",
] = "explicit_pairwise"
```

Then the model branch is:

```python
if config.pooling == "cls":
    ...

elif config.pooling == "global_gated_attention":
    ...

elif config.pooling == "explicit_pairwise":
    ...

else:
    raise ValueError(...)
```

This is cleaner than having:

```text
pooling
+
pairwise boolean
```

because those two options could otherwise conflict.

If `pooling` already exists because another implementation has been added, extend that same field rather than creating a second unrelated selection mechanism.

---

# 35. Configuration semantics

The three options should mean:

### `cls`

```text
cross-attention
    ↓
position 0
    ↓
API [128], Exc [128]
```

This reproduces the old baseline.

### `global_gated_attention`

```text
cross-attention
    ↓
global + token gated pooling
    ↓
API [128], Exc [128]
```

This is the previous proposed architecture.

### `explicit_pairwise`

```text
cross-attention
    ↓
all API-token × Exc-token pairs
    ↓
pairwise pooling [512]
    +
API global [128]
    +
Exc global [128]
    ↓
768
    +
descriptors
    ↓
817
```

This is the architecture in this specification.

---

# 36. Training and evaluation must remain unchanged

Do not change:

```text
ASL
balanced sampler
AdamW
learning rate
weight decay
gradient clipping
ReduceLROnPlateau
early stopping
PR-AUC monitoring
threshold tuning
test evaluation
```

The only changed model component is the representation/fusion path and classifier input size required by that new representation.

---

# 37. Checkpoint behavior

The explicit-pairwise architecture is not checkpoint-compatible with:

```text
CLS architecture
Global + Gated architecture
```

because it adds:

```text
pairwise attention pooling
```

and changes:

```text
classifier input:
609 -> 817
```

Train from scratch.

Do not silently load incompatible old checkpoints.

---

# 38. Unit tests

Create:

```text
tests/test_pairwise_interaction.py
```

Use standard `unittest`.

Do not download MoLFormer for unit tests.

---

# 39. Test 1 — Pairwise feature construction

Use:

```text
B = 2
L_api = 3
L_exc = 4
D = 128
```

Create:

```python
api = torch.randn(2, 3, 128)
exc = torch.randn(2, 4, 128)
```

Construct pair features.

Verify:

```text
pair_tensor.shape == (2,3,4,512)
```

Also verify a known pair manually:

```python
pair[0, i, j, :128] == api[0, i]
pair[0, i, j, 128:256] == exc[0, j]
pair[0, i, j, 256:384] == api[0,i] * exc[0,j]
pair[0, i, j, 384:512] == abs(api[0,i] - exc[0,j])
```

Use `torch.testing.assert_close`.

---

# 40. Test 2 — Pair mask construction

Example:

```text
API mask:
[F,F,T]

Exc mask:
[F,F,F,T]
```

Expected invalid pair mask:

```text
API valid × Exc valid = valid
API padding × anything = invalid
anything × Exc padding = invalid
```

Verify the exact result.

---

# 41. Test 3 — Pairwise attention padding

Create:

```text
pair_flat:
[B,N,512]
mask:
[B,N]
```

Verify:

```text
padded pair weights < 1e-5
```

Verify valid weights sum to approximately 1.

---

# 42. Test 4 — Padding invariance

Create two identical pair tensors except for invalid positions.

Change invalid pair features drastically.

Verify:

```text
pooled_original ≈ pooled_modified
```

This confirms padding truly has no influence.

---

# 43. Test 5 — All-pairs-masked behavior

Use:

```python
mask = torch.ones(B, N, dtype=torch.bool)
```

Verify:

```text
weights = all zeros
pooled = all zeros
no NaNs
```

This is required for missing-excipient support.

---

# 44. Test 6 — Full model dimensions

Use a mock sequence-capable encoder instead of real MoLFormer.

Verify:

```text
api_global:
[B,128]

exc_global:
[B,128]

pair_pooled:
[B,512]

pair_core:
[B,768]

pair_vector:
[B,817]

classifier input:
[B,817]

logits:
[B]
```

Use a forward hook on `model.classifier` to verify classifier input shape instead of changing the production return signature.

---

# 45. Test 7 — Missing excipient

Use a mixed batch:

```text
example 0:
exc_available = 1

example 1:
exc_available = 0
```

Verify:

```text
no NaNs
pair_pooled for missing example is zero
exc_global is still finite
pair_vector is [B,817]
logits are [B]
```

---

# 46. Test 8 — CLS baseline remains reproducible

Set:

```text
config.pooling = "cls"
```

and verify the old structural extraction still works.

The goal is to preserve a fair baseline.

---

# 47. Test 9 — Global + Gated mode remains isolated

If the repository already includes the Global + Gated implementation, verify that:

```text
config.pooling = "global_gated_attention"
```

still uses that architecture and does not accidentally enter the explicit-pairwise path.

The three pooling modes must be mutually exclusive.

---

# 48. Real MoLFormer smoke test

Only after unit tests pass:

1. Load actual MoLFormer.
2. Build one small real dataloader batch.
3. Run one forward pass.
4. Verify:
   - logits finite
   - logits shape `[B]`
   - classifier input `[B,817]`

Do not perform full training just to validate tensor shapes.

---

# 49. Performance warning

The explicit pairwise representation can become large.

Example:

```text
L_api = 50
L_exc = 50
```

gives:

```text
50 × 50 = 2,500
```

pair vectors per sample.

Each pair is:

```text
512 dimensions
```

so:

```text
2,500 × 512
```

features per sample before pooling.

The implementation must be vectorized and should use the actual sequence lengths/masks already provided by the encoder.

Do not introduce arbitrary token truncation as a hidden workaround.

---

# 50. Do not change downstream semantic features

Keep the existing descriptors:

```text
MolWt
MolLogP
TPSA
NumHDonors
NumHAcceptors
...
```

and their:

```text
21 -> 32 -> 24
```

projection.

Keep:

```text
exc_available
```

as an explicit final scalar.

---

# 51. Final classifier architecture

Recommended:

```text
pairwise pooled                 512
API global                     128
Excipient global               128
                               ----
                                768

API descriptors                 24
Exc descriptors                 24
availability                     1
                               ----
                                817
                                 |
                            Linear 817→256
                                 |
                               GELU
                                 |
                            Dropout .5
                                 |
                            Linear 256→64
                                 |
                               GELU
                                 |
                            Dropout .4
                                 |
                            Linear 64→1
                                 |
                               logit
                                 |
                              sigmoid
                                 |
                        P(incompatibility)
```

The production `forward()` should return:

```python
return logits.squeeze(1)
```

giving:

```text
[B]
```

---

# 52. Parameter sharing

Keep:

```text
MoLFormer = shared frozen encoder
```

Use separate:

```text
API projection
Excipient projection
```

The pairwise feature construction itself is parameter-free:

```text
concatenation
elementwise product
absolute difference
```

The pairwise attention pooling has its own trainable parameters.

Do not share pairwise pooling with API/Excipient pooling because this architecture does not separately pool the two molecules.

---

# 53. Important: pairwise interactions happen AFTER cross-attention

The pair feature must use:

```text
refined_api[:,1:,:]
refined_exc[:,1:,:]
```

not the original pre-cross-attention projected tokens.

Correct:

```text
MoLFormer
 -> projection
 -> cross-attention
 -> refined tokens
 -> explicit pair construction
```

Incorrect:

```text
MoLFormer
 -> projection
 -> explicit pair construction
 -> cross-attention
```

The requested architecture is based on **cross-attended token representations**.

---

# 54. Important: do not use cross-attention weights as the pair feature

The explicit pairwise feature is NOT:

```text
cross-attention score matrix
```

Do not simply take:

```python
attention_weights
```

and call that the pair representation.

Instead construct:

```text
[API_i,
 EXC_j,
 API_i*EXC_j,
 abs(API_i-EXC_j)]
```

for every valid pair.

The cross-attention weights are an internal part of the transformer interaction.

---

# 55. Important: pairwise pooling is separate from cross-attention

Cross-attention answers:

```text
Which information from the other molecule should each token receive?
```

Pairwise construction answers:

```text
What is the explicit relationship between API token i and Excipient token j?
```

Pairwise attention pooling answers:

```text
Which token-pair relationships are most important?
```

These are three separate operations.

---

# 56. Final success criteria

The implementation is correct only if all of the following hold:

### Architecture

- Frozen MoLFormer remains unchanged.
- Existing 768 -> 256 -> 128 projections remain.
- Bidirectional cross-attention remains unchanged.
- Position 0 is kept as a global API/Excipient representation.
- Position 0 is excluded from explicit molecular token-pair construction.
- Every valid API token is paired with every valid Excipient token.
- Each pair is exactly 512-dimensional.
- Pairwise pooling produces 512 dimensions.
- API global and Excipient global are retained and added after pairwise pooling.
- Pair core is exactly 768 dimensions.
- Descriptor fusion creates exactly 817 final dimensions.
- Classifier is exactly 817 -> 256 -> 64 -> 1.

### Padding

- Existing mask convention is preserved.
- Invalid API/Excipient combinations produce invalid pair masks.
- Padding pairs receive approximately zero attention.
- Changing invalid pair feature values does not change the pooled representation.
- Fully masked pair grids produce finite zero pair-pooled vectors.

### Missing excipient

- Existing placeholder logic remains unchanged.
- `exc_available == 0` produces zero valid molecular pair locations.
- Pairwise pooling handles this safely.
- Global Excipient placeholder representation remains available.
- No NaNs occur.

### Compatibility

- `forward()` still returns `[B]`.
- Training code continues to work.
- Evaluation code continues to work.
- ASL remains unchanged.
- Threshold tuning remains unchanged.
- Old models are not silently loaded.
- New experiment gets a separate checkpoint/metrics directory.

---

# 57. Final conceptual summary

The model should now reason at three levels:

```text
LEVEL 1 — Molecular encoding
SMILES
  ↓
MoLFormer
  ↓
768-D
  ↓
128-D


LEVEL 2 — Contextual token interaction
API tokens <------> Excipient tokens
        cross-attention
                ↓
      interaction-aware tokens


LEVEL 3 — Explicit pair reasoning
API token i × Excipient token j
                ↓
          512-D pair feature
                ↓
       attention over all pairs
                ↓
             512-D


GLOBAL INFORMATION
cross-attended API global      128-D
cross-attended Exc global      128-D

                ↓

512 pairwise
+ 128 API global
+ 128 Exc global
= 768

                ↓

+ 24 API descriptors
+ 24 Exc descriptors
+ 1 availability
= 817

                ↓

817 → 256 → 64 → 1

                ↓

P(INCOMPATIBILITY)
```

This is the required **Explicit Pairwise Token Interaction + Global-at-the-End** architecture.
