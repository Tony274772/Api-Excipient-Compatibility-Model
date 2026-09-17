# Global + Gated Attention Pooling — Implementation Specification

## Objective

Replace the current CLS-only structural representation extraction in the MoLFormer + bidirectional cross-attention path.

Current:

```text
Cross-attention
    -> refined sequence [B, L+1, 128]
    -> refined[:, 0, :]
    -> structural vector [B, 128]
```

New:

```text
Cross-attention
    -> refined sequence [B, L+1, 128]
    -> global token refined[:, 0, :]
    -> molecular tokens refined[:, 1:, :]
    -> gated attention pooling over VALID molecular tokens
    -> concatenate global + pooled = [B, 256]
    -> 256 -> 128 fusion
    -> structural vector [B, 128]
```

The goal is to retain distributed token-level information instead of discarding every token except position 0.

---

## 1. Files to modify

### Required

```text
src/model.py
```

### Optional

```text
src/config.py
```

Only add configuration for selecting the pooling mode if the existing project has a centralized config.

Do not modify the dataset, encoder, descriptor, loss, training, evaluation, split, or threshold logic unless a direct compatibility problem is discovered.

---

## 2. Existing architecture that must remain unchanged

Keep:

```text
MoLFormer
768-D output
    |
API projection: 768 -> 256 -> 128
Exc projection: 768 -> 256 -> 128
    |
prepend projected global representation
    |
bidirectional cross-attention
    |
residual + LayerNorm
```

Keep the existing cross-attention settings:

```text
embed_dim = 128
num_heads = 8
```

Do not alter the directionality:

```text
API queries -> Excipient keys/values
Excipient queries -> API keys/values
```

Only change what happens AFTER the cross-attention outputs have been produced.

---

## 3. New module: GatedAttentionPooling

Add this class in `src/model.py`, before the main compatibility model class.

```python
class GatedAttentionPooling(nn.Module):
    """
    Gated attention pooling over a variable-length sequence.

    x:
        [B, L, D]

    padding_mask:
        [B, L]
        True  = padding / invalid token
        False = valid token

    Returns:
        pooled:
            [B, D]

        weights:
            [B, L]
    """

    def __init__(self, input_dim=128, hidden_dim=128):
        super().__init__()

        self.tanh_proj = nn.Linear(input_dim, hidden_dim)
        self.sigmoid_proj = nn.Linear(input_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, padding_mask=None):
        tanh_part = torch.tanh(self.tanh_proj(x))
        sigmoid_part = torch.sigmoid(self.sigmoid_proj(x))

        gated = tanh_part * sigmoid_part

        scores = self.score(gated).squeeze(-1)  # [B, L]

        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, -1e9)

        weights = torch.softmax(scores, dim=1)

        pooled = torch.sum(
            weights.unsqueeze(-1) * x,
            dim=1,
        )

        return pooled, weights
```

Do not use unmasked softmax.

---

## 4. Padding-mask semantics

Use the same convention as PyTorch `MultiheadAttention`:

```text
True  = padding / ignore
False = valid
```

For example:

```text
tokens:
[A, B, C, D, PAD, PAD]

mask:
[F, F, F, F,  T,   T]
```

The attention scores must be masked before softmax:

```python
scores = scores.masked_fill(padding_mask, -1e9)
```

Therefore padded positions receive effectively zero attention.

---

## 5. Never pool the global token twice

The cross-attention sequence has:

```text
position 0 = projected global representation
positions 1...L = molecular token representations
```

Therefore:

```python
global_api = refined_api[:, 0, :]
api_tokens = refined_api[:, 1:, :]

global_exc = refined_exc[:, 0, :]
exc_tokens = refined_exc[:, 1:, :]
```

The gated pooling module MUST receive only:

```python
refined_api[:, 1:, :]
refined_exc[:, 1:, :]
```

Do not pass the entire `[L+1, 128]` sequence to the pooling module.

The desired architecture is:

```text
global token ----------------------+
                                    |
molecular tokens -> gated pooling -+-> concat -> 256 -> 128
```

---

## 6. Add modules in CompatibilityModel.__init__

Inside the existing cross-attention initialization section, add:

```python
self.api_pool = GatedAttentionPooling(
    input_dim=proj_dim,
    hidden_dim=proj_dim,
)

self.exc_pool = GatedAttentionPooling(
    input_dim=proj_dim,
    hidden_dim=proj_dim,
)
```

With the current default:

```text
proj_dim = 128
```

both pooling modules operate at 128 dimensions.

Add two separate fusion heads:

```python
self.api_pool_fusion = nn.Sequential(
    nn.Linear(proj_dim * 2, proj_dim),
    nn.LayerNorm(proj_dim),
    nn.GELU(),
    nn.Dropout(config.proj_dropout),
)

self.exc_pool_fusion = nn.Sequential(
    nn.Linear(proj_dim * 2, proj_dim),
    nn.LayerNorm(proj_dim),
    nn.GELU(),
    nn.Dropout(config.proj_dropout),
)
```

Do not share the API and excipient fusion weights.

---

## 7. Replace current CLS extraction

Find the current logic equivalent to:

```python
h_api_struct = refined_api[:, 0, :]
h_exc_struct = refined_exc[:, 0, :]
```

Remove that as the only structural extraction mechanism.

Replace with:

```python
# Global + Gated Attention Pooling

api_global = refined_api[:, 0, :]       # [B, 128]
exc_global = refined_exc[:, 0, :]       # [B, 128]

api_tokens = refined_api[:, 1:, :]      # [B, L_api, 128]
exc_tokens = refined_exc[:, 1:, :]      # [B, L_exc, 128]

api_token_mask = api_mask_ext[:, 1:]    # [B, L_api]
exc_token_mask = exc_mask_ext[:, 1:]    # [B, L_exc]

api_token_pool, api_token_weights = self.api_pool(
    api_tokens,
    api_token_mask,
)

exc_token_pool, exc_token_weights = self.exc_pool(
    exc_tokens,
    exc_token_mask,
)

# Missing-excipient fallback
missing_exc = ~exc_available.bool()

exc_placeholder_pool = self.exc_global_placeholder.expand(
    exc_tokens.size(0),
    -1,
)

exc_token_pool = torch.where(
    missing_exc.unsqueeze(1),
    exc_placeholder_pool,
    exc_token_pool,
)

# Global + token-level fusion
api_combined = torch.cat(
    [api_global, api_token_pool],
    dim=-1,
)  # [B, 256]

exc_combined = torch.cat(
    [exc_global, exc_token_pool],
    dim=-1,
)  # [B, 256]

h_api_struct = self.api_pool_fusion(
    api_combined
)  # [B, 128]

h_exc_struct = self.exc_pool_fusion(
    exc_combined
)  # [B, 128]
```

Adapt variable names only when necessary to match the existing code. Do not duplicate variables that already exist.

---

## 8. Missing excipient behavior

The repository already supports unavailable/missing excipients.

Preserve the existing behavior.

When:

```text
exc_available = 0
```

there may be no valid excipient molecular tokens.

Do not perform a normal masked softmax over a sequence where every token is masked and assume the result is meaningful.

Instead:

```text
available excipient:
    valid tokens
       |
       v
    gated pooling
       |
    [B,128]

missing excipient:
    learned placeholder
       |
    [B,128]
```

Use the existing:

```python
self.exc_global_placeholder
```

as the fallback if it already exists in the model.

Do not remove the existing placeholder mechanism.

---

## 9. Missing-excipient ordering

For an unavailable excipient:

```text
exc_global = refined_exc[:, 0, :]
```

is still preserved because position 0 is the explicit global/placeholder representation used by the existing architecture.

The pooled token branch falls back to:

```python
self.exc_global_placeholder
```

Then:

```text
exc_global [128]
+
exc_placeholder_pool [128]
=
256
```

Then:

```text
256 -> 128
```

---

## 10. Resulting dimensionality

API:

```text
MoLFormer                  [B, L_api, 768]
projection                 [B, L_api, 128]
global-token prepend       [B, L_api+1, 128]
cross-attention            [B, L_api+1, 128]
global                     [B, 128]
tokens                     [B, L_api, 128]
gated pooling              [B, 128]
global + pooled            [B, 256]
fusion 256 -> 128          [B, 128]
```

Therefore:

```text
h_api_struct = [B, 128]
```

Excipient:

```text
MoLFormer                  [B, L_exc, 768]
projection                 [B, L_exc, 128]
global-token prepend       [B, L_exc+1, 128]
cross-attention            [B, L_exc+1, 128]
global                     [B, 128]
tokens                     [B, L_exc, 128]
gated pooling              [B, 128]
global + pooled            [B, 256]
fusion 256 -> 128          [B, 128]
```

Therefore:

```text
h_exc_struct = [B, 128]
```

---

## 11. Downstream architecture MUST remain unchanged

Descriptors:

```text
API [B,21] -> 21 -> 32 -> 24 -> [B,24]
Exc [B,21] -> 21 -> 32 -> 24 -> [B,24]
```

Then:

```text
API:
128 + 24 = 152

Excipient:
128 + 24 + availability(1) = 153
```

Existing pair interactions:

```text
elementwise product       [152]
absolute difference       [152]
```

Final pair vector:

```text
152 + 153 + 152 + 152 = 609
```

Classifier remains:

```text
609 -> 128 -> 64 -> 1
```

Do not change these dimensions.

---

## 12. Full target flow

```text
API SMILES
    |
MoLFormer
    |
768-D
    |
768 -> 256 -> 128
    |
API sequence [L_api,128]
    |
prepend global
    |
API [L_api+1,128]
    |
    |<------ bidirectional cross-attention ------>|
    |
refined API [L_api+1,128]
    |
    +---- position 0 ------> global API [128]
    |
    +---- positions 1..L -> gated attention
                                  |
                               [128]
                                  |
                         concat with global
                                  |
                               [256]
                                  |
                              256 -> 128
                                  |
                            API struct [128]


Excipient follows the same structure:

Excipient SMILES
    |
MoLFormer
    |
768-D
    |
768 -> 256 -> 128
    |
Exc sequence [L_exc,128]
    |
prepend global
    |
Exc [L_exc+1,128]
    |
bidirectional cross-attention
    |
refined Exc [L_exc+1,128]
    |
    +---- global [128]
    |
    +---- valid tokens
             |
             v
      gated attention pooling
             |
           [128]
             |
          concat
             |
           [256]
             |
        256 -> 128
             |
      Exc struct [128]


Then:

API struct [128] + API descriptors [24]
    -> API [152]

Exc struct [128] + Exc descriptors [24] + availability [1]
    -> Exc [153]

API [152] + Exc core [152]
    -> elementwise product [152]
    -> absolute difference [152]

[152 | 153 | 152 | 152]
          |
         609
          |
    609 -> 128 -> 64 -> 1
          |
        logit
          |
       sigmoid
          |
P(incompatibility)
```

---

## 13. Optional configurable pooling

If practical, preserve the old architecture as an ablation.

Add:

```python
pooling = "global_gated_attention"
pool_hidden_dim = 128
```

Support:

```text
pooling = "cls"
pooling = "global_gated_attention"
```

For `cls`:

```python
h_api_struct = refined_api[:, 0, :]
h_exc_struct = refined_exc[:, 0, :]
```

For `global_gated_attention`, use the new implementation.

Do not change any other component between the two modes.

---

## 14. Attention weights

Keep these internally:

```python
api_token_weights
exc_token_weights
```

The primary model output must remain compatible with the existing training/evaluation code.

The weights can later support token-level visualization.

Do not describe attention weights as causal explanations.

---

## 15. Numerical safety

Preserve existing `torch.nan_to_num(...)` handling around cross-attention if present.

The new pooling must not introduce NaNs.

The all-masked-token case must use the missing-excipient placeholder fallback.

---

## 16. Testing requirements

Before full training, run a small forward pass and verify:

```text
api_global       [B,128]
api_token_pool   [B,128]
h_api_struct     [B,128]

exc_global       [B,128]
exc_token_pool   [B,128]
h_exc_struct     [B,128]

pair vector      [B,609]

classifier       existing output shape
```

Temporary assertions:

```python
assert h_api_struct.shape[-1] == 128
assert h_exc_struct.shape[-1] == 128
assert pair_vec.shape[-1] == 609

assert torch.isfinite(h_api_struct).all()
assert torch.isfinite(h_exc_struct).all()
```

Adapt variable names to the actual code.

---

## 17. Padding correctness test

Verify padded positions receive approximately zero attention.

For example:

```python
padded_weights = api_token_weights.masked_select(api_token_mask)
```

and verify they are near zero.

Repeat for excipient.

Also verify that changing padded embedding values does not materially change the pooled representation.

---

## 18. Missing-excipient test

Run a batch containing:

```text
exc_available = 0
```

Verify:

```text
no NaNs
placeholder fallback is used
h_exc_struct = [B,128]
pair vector = [B,609]
```

---

## 19. Checkpoint behavior

The new architecture adds:

```text
api_pool
exc_pool
api_pool_fusion
exc_pool_fusion
```

Old CLS checkpoints should not be silently loaded into the new architecture.

Train the new architecture from scratch.

Recommended experiment names:

```text
molformer_cross_attn_cls_asl
molformer_cross_attn_global_gated_asl
```

---

## 20. Do NOT change

This task is ONLY a pooling modification.

Do not:

- remove MoLFormer
- unfreeze MoLFormer
- change MoLFormer output dimension
- remove bidirectional cross-attention
- change number of attention heads
- change RDKit descriptors
- change descriptor projection dimensions
- change interaction features
- change the 609-dimensional pair representation
- change classifier architecture
- change ASL
- add SMOTE
- change data splitting
- change threshold tuning
- change evaluation metrics
- remove missing-excipient handling

---

## 21. Implementation success criteria

The implementation is correct only if:

1. Cross-attention remains bidirectional.
2. Position 0 remains available as the global cross-attended representation.
3. Positions 1...L are pooled using learned gated attention.
4. Padding positions are masked before softmax.
5. Position 0 is NOT included again inside token pooling.
6. Available excipients use real token pooling.
7. Missing excipients use the existing learned placeholder fallback.
8. Global [128] + pooled [128] creates [256].
9. Fusion maps [256] -> [128].
10. API structural representation remains [B,128].
11. Excipient structural representation remains [B,128].
12. Final pair vector remains [B,609].
13. Classifier remains 609 -> 128 -> 64 -> 1.
14. No NaNs occur.
15. Existing training/evaluation continues to run.
16. Old CLS results remain available as the baseline.

---

## 22. Final required change

Old:

```text
Cross Attention
      |
      v
[L+1,128]
      |
take position 0
      |
128-D
```

New:

```text
Cross Attention
      |
      v
[L+1,128]
      |
      +----------------------+
      |                      |
      v                      v
position 0              positions 1..L
global [128]             tokens [L,128]
                              |
                              v
                     Gated Attention Pooling
                              |
                            [128]
                              |
      +-----------------------+
      |
      v
CONCAT [256]
      |
256 -> 128
      |
structural representation [128]
```

The sole architectural objective is to retain valid token-level information after cross-attention while preserving the existing downstream architecture.
