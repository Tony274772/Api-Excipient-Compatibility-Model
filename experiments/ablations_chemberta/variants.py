"""Ablation variants of `chemberta__x_pubchemfp` (encoder grid, fp_secondary block).

The model is the grid's PGBCompatibilityModel: frozen ChemBERTa -> projection -> cross-attention
(API <-> excipient) -> CLS pooling as the structural vector, PubChem 881 bits as the secondary
description, the shared tower features (12 descriptors, 18 chemistry flags, 8 salt-context values)
and the unchanged PGB head (5-D excipient prior, gate, rank-16 bilinear term, prior trust w,
5 SMARTS mechanism flags).

How a variant is made (same rule as final_project/apiexc/ablations/variants.py):
every variant first builds the FULL model with the run's seed, exactly as run_grid.py does, and
then cuts its part out of the existing weight matrices. No new layers and no extra random draws, so
every layer a variant keeps starts from the same weights as `full` at the same seed, and the
comparison with `full` is paired seed by seed.

Three variants cannot be made by cutting and change the build instead (they are marked in
DESCRIPTIONS and in run_info.json as `weight_paired: false` or note how they differ):

    pool_gated      adds the gated-attention pooling layers. They are created LAST in
                    PGBCompatibilityModel.__init__, after the towers, head, projections and
                    cross-attention, so every layer `full` has still gets the same initial weights.
    mech_v2         13 mechanism inputs instead of 5 widens the head's gate and MLP, so the head
                    draws different random numbers (towers and encoder side are unchanged).
    shuf_pubchemfp  same model and weights as `full`; only the data changes (PubChem bits deranged).

Ablations combine with "+", applied left to right: "prior1+no_w", "no_flags_metal+no_salt".
"""

import torch
import torch.nn as nn

from experiments.encoder_comparison.towers import ComparisonTower
from src.model import PGBCompatibilityModel, PGBHead

PRIOR_NAMES = ["p_exact", "p_delta", "p_family", "log_n", "unseen"]
MECH_NAMES = ["maillard", "ester_amide_hydrolysis", "metal_ion_oxidation",
              "acid_base_salt_displacement", "schiff_base"]
# the 18 chemistry flags of src/features.py, in order
FLAG_NAMES = ["no_carbons", "oxidizer", "strong_acid", "strong_base", "has_metal", "has_Mg", "has_Ca",
              "has_Al", "has_Si", "has_Ti_Fe", "has_Mn_Cr", "is_salt", "very_short", "many_OH",
              "has_amine", "carboxylic_acid", "ester", "phenol"]
METAL_FLAGS = ["no_carbons", "has_metal", "has_Mg", "has_Ca", "has_Al", "has_Si", "has_Ti_Fe",
               "has_Mn_Cr", "is_salt", "very_short"]
NON_METAL_FLAGS = [f for f in FLAG_NAMES if f not in METAL_FLAGS]


@torch.no_grad()
def keep_input_columns(linear: nn.Linear, keep):
    """Cut a Linear layer down to the input columns in `keep`, in place (no new random draws)."""
    linear.weight = nn.Parameter(linear.weight[:, keep].clone())
    linear.in_features = len(keep)


# =========================================================================== head
class AblationPGBHead(PGBHead):
    """src.model.PGBHead.forward with seven switches (same as the ARES ablation head):

        prior_cols       which of the 5 prior values the head sees
        use_prior_trust  keep the w * logit(p_exact) term
        use_gate         gate the bilinear term
        use_bilinear     put the (gated) bilinear term into the MLP input
        mech_cols        which mechanism flags the head sees (5 in v1, 13 in v2)
        api_flag_cols    which API chemistry flags reach the gate
        exc_flag_cols    which excipient chemistry flags reach the gate

    With every switch at its default this is exactly PGBHead.forward.
    """

    def forward(self, h_api, h_exc, prior_vec, api_flags, exc_flags, mech_flags):
        prior_vec = prior_vec[:, self.prior_cols]
        mech_flags = mech_flags[:, self.mech_cols]
        api_flags = api_flags[:, self.api_flag_cols]
        exc_flags = exc_flags[:, self.exc_flag_cols]

        pair = [h_api, h_exc, h_api * h_exc, torch.abs(h_api - h_exc)]
        if self.use_bilinear:
            bilinear = self.U(h_api) * self.V(h_exc)
            if self.use_gate:
                gate = self.gate(torch.cat([prior_vec, api_flags, exc_flags, mech_flags], dim=-1))
                pair.append(bilinear * gate)
            else:
                pair.append(bilinear)

        residual = self.mlp(torch.cat(pair + [prior_vec, mech_flags], dim=-1)).squeeze(-1)

        logit = residual + self.prior_bias
        if self.use_prior_trust:
            w = self.prior_trust_mlp(prior_vec).squeeze(-1)
            p = prior_vec[:, 0].clamp(1e-6, 1 - 1e-6)
            logit = logit + w * torch.log(p / (1 - p))
        return logit


def ablation_head(model):
    """The model's head as an AblationPGBHead (same weights; switches keep earlier ablations)."""
    head = model.head
    if not isinstance(head, AblationPGBHead):
        head.__class__ = AblationPGBHead
        head.tower_dim = head.U.in_features
        head.bilinear_dim = head.U.out_features
        if getattr(model.config, "mech_set", "v1") == "v2":
            from experiments.ablations_chemberta.mech_v2 import MECH_V2_NAMES
            head.mech_names = list(MECH_V2_NAMES)
        else:
            head.mech_names = list(MECH_NAMES)
        head.prior_cols = list(range(len(PRIOR_NAMES)))
        head.use_prior_trust = True
        head.use_gate = True
        head.use_bilinear = True
        head.mech_cols = list(range(len(head.mech_names)))
        head.api_flag_cols = list(range(len(FLAG_NAMES)))
        head.exc_flag_cols = list(range(len(FLAG_NAMES)))
    return head


# =========================================================================== towers
class AblationComparisonTower(ComparisonTower):
    """ComparisonTower.forward with the mix-layer input tracked as named segments, so any segment
    can be cut. `segments` is an ordered list of [name, width]:

        struct (128, ChemBERTa after cross-attention), [primary 64, unused here], secondary (32,
        PubChem), desc (24), flags (16), salt (8)

    Extra switches: `flag_cols` (which of the 18 flags enter the flag block) and `use_desc`.
    has_struct / has_secondary / use_salt_context are the parent's own switches.
    """

    def forward(self, struct_vec, primary, primary_avail, secondary, pgb_desc, salt, flags):
        parts = []
        if self.has_struct:
            if struct_vec is None:
                raise RuntimeError("tower expects a structural vector but the model passed none")
            parts.append(struct_vec)
        if self.has_primary:
            parts.append(self.primary_block(primary))
        if self.has_secondary:
            if secondary is None:
                raise RuntimeError("tower has a secondary block but the batch carries no secondary")
            parts.append(self.secondary_block(secondary))
        if self.use_desc:
            parts.append(self.desc_block(pgb_desc))
        if self.flag_cols:
            parts.append(self.flag_block(flags[:, self.flag_cols]))
        if self.use_salt_context:
            parts.append(salt)
        x = torch.cat(parts, dim=-1)
        if x.shape[-1] != self.mix[0].in_features:
            raise RuntimeError(f"mix input {x.shape[-1]} != mix layer {self.mix[0].in_features}; "
                               f"segments {self.segments}")
        return self.mix(x)


def ablation_tower(tower):
    """A ComparisonTower as an AblationComparisonTower (same weights)."""
    if not isinstance(tower, AblationComparisonTower):
        known = ((tower.PRIMARY_OUT if tower.has_primary else 0)
                 + (tower.SECONDARY_OUT if tower.has_secondary else 0)
                 + 24 + 16 + (8 if tower.use_salt_context else 0))
        struct_w = tower.mix[0].in_features - known
        segs = []
        if tower.has_struct:
            segs.append(["struct", struct_w])
        if tower.has_primary:
            segs.append(["primary", tower.PRIMARY_OUT])
        if tower.has_secondary:
            segs.append(["secondary", tower.SECONDARY_OUT])
        segs += [["desc", 24], ["flags", 16]]
        if tower.use_salt_context:
            segs.append(["salt", 8])
        tower.__class__ = AblationComparisonTower
        tower.segments = segs
        tower.flag_cols = list(range(len(FLAG_NAMES)))
        tower.use_desc = True
    return tower


def cut_segment(tower, name):
    """Remove one named segment from the tower's mix-layer input."""
    tower = ablation_tower(tower)
    start = 0
    for seg, width in tower.segments:
        if seg == name:
            break
        start += width
    else:
        return tower                          # already gone
    mix0 = tower.mix[0]
    keep_input_columns(mix0, list(range(start)) + list(range(start + width, mix0.in_features)))
    tower.segments = [s for s in tower.segments if s[0] != name]
    return tower


def both_towers(model):
    return [ablation_tower(model.api_tower), ablation_tower(model.exc_tower)]


# =========================================================================== encoder side
class AblationPGBModel(PGBCompatibilityModel):
    """PGBCompatibilityModel with one switch, `use_attention_layers`.

    False: the structural vector is the projected mean-pooled ChemBERTa embedding of the molecule
    itself (`api_proj(pooled)`), with no cross-attention to the partner and no LayerNorm on top.
    That is exactly the input the CLS position of the cross-attention starts from, so the cut
    removes the attention and nothing else. A missing excipient still gets the learned placeholder.
    """

    def _encode_struct(self, batch, device):
        if not self.use_cross_attn or getattr(self, "use_attention_layers", True):
            return super()._encode_struct(batch, device)
        _, api_pool, _ = self.encoder.encode(batch["api_smiles"])
        _, exc_pool, _ = self.encoder.encode(batch["exc_smiles"])
        api_struct = self.api_proj(api_pool.to(device))
        exc_struct = self.exc_proj(exc_pool.to(device))
        avail = batch["exc_available"].to(device).unsqueeze(1)
        B = exc_struct.shape[0]
        exc_struct = avail * exc_struct + (1 - avail) * self.exc_global_placeholder.expand(B, -1)
        return api_struct, exc_struct


def ablation_model(model):
    if not isinstance(model, AblationPGBModel):
        model.__class__ = AblationPGBModel
        model.use_attention_layers = True
    return model


# =========================================================================== A. head
def no_w(model):
    """Remove the adaptive prior trust w: z = r + b. The prior still enters the gate and the MLP."""
    head = ablation_head(model)
    if hasattr(head, "prior_trust_mlp"):
        del head.prior_trust_mlp
    head.use_prior_trust = False
    return model


def keep_prior(head, names):
    """Cut the head down to the prior values in `names`: gate input, MLP input, prior trust input."""
    old = head.prior_cols
    keep = [i for i, c in enumerate(old) if PRIOR_NAMES[c] in names]
    k = len(old)
    d, r = head.tower_dim, head.bilinear_dim
    if hasattr(head, "gate"):
        g0 = head.gate[0]                     # [prior (k), api flags, exc flags, mech]
        keep_input_columns(g0, keep + list(range(k, g0.in_features)))
    m0 = head.mlp[0]                          # [4d, gated (r), prior (k), mech]
    start = 4 * d + r
    keep_input_columns(m0, list(range(start)) + [start + i for i in keep]
                       + list(range(start + k, m0.in_features)))
    if hasattr(head, "prior_trust_mlp"):
        keep_input_columns(head.prior_trust_mlp[0], keep)
    head.prior_cols = [old[i] for i in keep]


def prior1(model):
    """Keep only p_exact of the 5 prior values (gate 46 -> 42, MLP 410 -> 406, prior trust 5 -> 1)."""
    keep_prior(ablation_head(model), ["p_exact"])
    return model


def no_gate(model):
    """Remove the gate on the bilinear term: gated = bilinear (MLP input still 410)."""
    head = ablation_head(model)
    if hasattr(head, "gate"):
        del head.gate
    head.use_gate = False
    return model


def no_bilinear(model):
    """Remove U, V and the gate; cut the 16 gated columns from the MLP input (410 -> 394)."""
    head = ablation_head(model)
    if not head.use_bilinear:
        return model
    d, r = head.tower_dim, head.bilinear_dim
    del head.U, head.V
    if hasattr(head, "gate"):
        del head.gate
    m0 = head.mlp[0]
    keep_input_columns(m0, list(range(4 * d)) + list(range(4 * d + r, m0.in_features)))
    head.use_bilinear = False
    head.use_gate = False
    head.bilinear_dim = 0
    return model


def no_prior(model):
    """Remove the excipient prior everywhere (gate, MLP, w term): z = r + b."""
    head = ablation_head(model)
    no_w(model)
    keep_prior(head, [])
    return model


def keep_mech(head, names):
    """Keep only the mechanism flags in `names`; they are the last m inputs of gate and MLP."""
    old = head.mech_cols
    keep = [i for i, c in enumerate(old) if head.mech_names[c] in names]
    m = len(old)
    for layer in ([head.gate[0]] if hasattr(head, "gate") else []) + [head.mlp[0]]:
        start = layer.in_features - m
        keep_input_columns(layer, list(range(start)) + [start + i for i in keep])
    head.mech_cols = [old[i] for i in keep]


def no_mech(model):
    """Remove all mechanism flags from the head (v1: gate 46 -> 41, MLP 410 -> 405)."""
    keep_mech(ablation_head(model), [])
    return model


def mech_v2(model):
    """13 literature mechanism rules in place of the 5 SMARTS flags. Changes the build (see
    BUILD_SETTINGS): the head is created with 13 mechanism inputs and the dataset swaps the flags."""
    return model


# =========================================================================== B. tower shared features
def no_salt(model):
    """Remove the 8-value salt context from both towers (mix -8)."""
    for t in both_towers(model):
        if t.use_salt_context:
            cut_segment(t, "salt")
            t.use_salt_context = False
    return model


def no_desc(model):
    """Remove the 12 RDKit descriptors (desc block, mix -24) from both towers."""
    for t in both_towers(model):
        if t.use_desc:
            cut_segment(t, "desc")
            del t.desc_block
            t.use_desc = False
    return model


def keep_flags(model, side, names):
    """Keep only the chemistry flags in `names` for one side, in the tower's flag block and in the
    gate. With none left the flag block is deleted and its 16 mix inputs are cut."""
    tower = ablation_tower(model.api_tower if side == "api" else model.exc_tower)
    old = tower.flag_cols
    keep = [i for i, c in enumerate(old) if FLAG_NAMES[c] in names]
    if old and not keep:
        cut_segment(tower, "flags")
        del tower.flag_block
    elif keep != list(range(len(old))):
        keep_input_columns(tower.flag_block[0], keep)
    tower.flag_cols = [old[i] for i in keep]

    head = ablation_head(model)
    attr = f"{side}_flag_cols"
    old = getattr(head, attr)
    keep = [i for i, c in enumerate(old) if FLAG_NAMES[c] in names]
    if hasattr(head, "gate"):
        start = len(head.prior_cols) + (0 if side == "api" else len(head.api_flag_cols))
        g0 = head.gate[0]
        keep_input_columns(g0, list(range(start)) + [start + i for i in keep]
                           + list(range(start + len(old), g0.in_features)))
    setattr(head, attr, [old[i] for i in keep])


def no_flags(model):
    """Remove all 18 chemistry flags from both towers and the gate (46 -> 10)."""
    keep_flags(model, "api", [])
    keep_flags(model, "exc", [])
    return model


def no_flags_api(model):
    keep_flags(model, "api", [])
    return model


def no_flags_exc(model):
    keep_flags(model, "exc", [])
    return model


def no_flags_metal(model):
    """Remove the 10 metal / inorganic flags on both sides (flag blocks 18 -> 8, gate 46 -> 26)."""
    keep_flags(model, "api", NON_METAL_FLAGS)
    keep_flags(model, "exc", NON_METAL_FLAGS)
    return model


# =========================================================================== C. data
def no_family(model):
    """Data only: prior lookup exact -> nearest neighbour -> global (no family step)."""
    return model


# =========================================================================== D. the description
def no_secondary(model):
    """Remove the PubChem branch (secondary block, mix -32) from both towers."""
    for t in both_towers(model):
        if t.has_secondary:
            cut_segment(t, "secondary")
            del t.secondary_block
            t.has_secondary = False
    return model


def no_chemberta(model):
    """Remove ChemBERTa's structural vector: projections, cross-attention, LayerNorms, placeholders
    and the 128 struct columns of both mix layers. PubChem, the shared features and the head stay.
    The frozen encoder is still loaded but never called."""
    for t in both_towers(model):
        if t.has_struct:
            cut_segment(t, "struct")
            t.has_struct = False
    for name in ("api_proj", "exc_proj", "exc_placeholder", "exc_global_placeholder",
                 "cross_attn_exc", "cross_attn_api", "ln_exc", "ln_api",
                 "api_pool", "exc_pool", "api_pool_fusion", "exc_pool_fusion"):
        if hasattr(model, name):
            delattr(model, name)
    model.use_cross_attn = False
    return model


def no_cross_attn(model):
    """Remove the two cross-attention layers and their LayerNorms (and the token placeholder):
    the structural vector is the molecule's own projected ChemBERTa embedding. PubChem stays."""
    model = ablation_model(model)
    if getattr(model, "pooling", "cls") != "cls":
        raise ValueError("no_cross_attn is defined for CLS pooling only")
    for name in ("cross_attn_exc", "cross_attn_api", "ln_exc", "ln_api", "exc_placeholder"):
        if hasattr(model, name):
            delattr(model, name)
    model.use_attention_layers = False
    return model


def pool_gated(model):
    """Global gated-attention pooling instead of CLS. Changes the build (BUILD_SETTINGS)."""
    return model


def shuf_pubchemfp(model):
    """Data only: every molecule gets another molecule's PubChem bits (fixed derangement,
    seed 20251110, the grid's SHUFFLE_SEED). Same model, same parameters, zero information."""
    return model


def chemberta_parent(model):
    """Data only: ChemBERTa reads the parent molecule (largest fragment, neutralised, no
    stereochemistry). Everything else, including salt context and flags, reads the original SMILES."""
    return model


def full(model):
    """The unchanged model, trained through this harness."""
    return model


# =========================================================================== registry
ABLATIONS = {
    "full": full,
    # A. head
    "no_w": no_w, "prior1": prior1, "no_prior": no_prior, "no_gate": no_gate,
    "no_bilinear": no_bilinear, "no_mech": no_mech, "mech_v2": mech_v2,
    # B. tower shared features
    "no_salt": no_salt, "no_flags": no_flags, "no_flags_api": no_flags_api,
    "no_flags_exc": no_flags_exc, "no_flags_metal": no_flags_metal, "no_desc": no_desc,
    # C. data
    "no_family": no_family,
    # D. the molecular description
    "no_secondary": no_secondary, "no_chemberta": no_chemberta, "no_cross_attn": no_cross_attn,
    "pool_gated": pool_gated, "shuf_pubchemfp": shuf_pubchemfp,
    "chemberta_parent": chemberta_parent,
}

DESCRIPTIONS = {
    "full": "full model (harness baseline)",
    "no_w": "A. prior trust w removed, z = r + b",
    "prior1": "A. prior 5 -> 1 (p_exact only)",
    "no_prior": "A. excipient prior removed from gate, MLP and w term",
    "no_gate": "A. gate on the bilinear term removed",
    "no_bilinear": "A. bilinear term U,V and its gate removed",
    "no_mech": "A. 5 SMARTS mechanism flags removed from the head",
    "mech_v2": "A. 13 literature mechanism rules instead of 5 (head re-drawn: not weight-paired)",
    "no_salt": "B. 8-value salt context removed from both towers",
    "no_flags": "B. all 18 chemistry flags removed (towers and gate)",
    "no_flags_api": "B. API chemistry flags removed",
    "no_flags_exc": "B. excipient chemistry flags removed",
    "no_flags_metal": "B. 10 metal / inorganic flags removed on both sides",
    "no_desc": "B. 12 RDKit descriptors removed from both towers",
    "no_family": "C. prior lookup without the family step (data only)",
    "no_secondary": "D. PubChem branch removed",
    "no_chemberta": "D. ChemBERTa struct branch removed (projection, cross-attention)",
    "no_cross_attn": "D. cross-attention removed; struct = own projected ChemBERTa embedding",
    "pool_gated": "D. gated-attention pooling instead of CLS (adds layers, drawn last)",
    "shuf_pubchemfp": "D. PubChem bits deranged across molecules (control, data only)",
    "chemberta_parent": "D. ChemBERTa reads the parent molecule (no counterions / charges / stereo)",
}

# Ablations that change how the run is BUILT rather than (or as well as) the model afterwards.
# Applied to the run before any data or model is made. Keys:
#   prior_lookup   'no_family'  -> prior_lookup.NoFamilyPriorTable
#   mech_set       'v2'         -> mech_v2 dataset, head sized for 13 mechanism inputs
#   pooling        'global_gated_attention'
#   secondary      representation name of the secondary slot
#   parent_smiles  True         -> ChemBERTa reads the parent molecule
BUILD_SETTINGS = {
    "no_family": {"prior_lookup": "no_family"},
    "mech_v2": {"mech_set": "v2"},
    "pool_gated": {"pooling": "global_gated_attention"},
    "shuf_pubchemfp": {"secondary": "shuf_pubchemfp"},
    "chemberta_parent": {"parent_smiles": True},
}

# Variants whose kept layers do NOT start from exactly the weights of `full` at the same seed.
NOT_WEIGHT_PAIRED = {"mech_v2"}

# The plan's run list (paper/chemberta_pubchemfp_ablation_plan.md, section 3). Combinations are
# written with "+".
PLAN = [
    "full",
    # A
    "no_w", "prior1", "prior1+no_w", "no_prior", "no_gate", "no_bilinear", "no_mech", "mech_v2",
    # B
    "no_salt", "no_flags", "no_flags_api", "no_flags_exc", "no_flags_metal", "no_flags_metal+no_salt",
    # C
    "prior1+no_family",
    # D
    "no_secondary", "no_chemberta", "no_cross_attn", "pool_gated", "no_desc", "shuf_pubchemfp",
    "chemberta_parent",
]

# Which run each ablation is compared with in collect.py (default: full).
REFERENCE = {
    "prior1+no_family": "prior1",       # identical weights, only the prior lookup differs
}

# Candidate designs, filled in after the ablation results (plan section 5, rule 4).
DESIGNS = {}


def parse_ablation(name):
    """'prior1+no_w' -> ['prior1', 'no_w'] with designs expanded; raises on unknown parts."""
    parts = [p for part in name.split("+") for p in DESIGNS.get(part, [part])]
    unknown = [p for p in parts if p not in ABLATIONS]
    if unknown:
        raise ValueError(f"unknown ablation(s) {unknown}; available: {sorted(ABLATIONS)}")
    if len(parts) > 1 and "full" in parts:
        raise ValueError("'full' cannot be combined with other ablations")
    return parts


def build_settings(name):
    settings = {}
    for part in parse_ablation(name):
        settings.update(BUILD_SETTINGS.get(part, {}))
    return settings


def describe(name):
    return " + ".join(DESCRIPTIONS[p] for p in parse_ablation(name))


def weight_paired(name):
    return not any(p in NOT_WEIGHT_PAIRED for p in parse_ablation(name))


def apply_ablation(model, name):
    """Apply every part of `name` to an already-built full model, in order."""
    for part in parse_ablation(name):
        model = ABLATIONS[part](model)
    return model
