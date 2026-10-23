"""
Ablation variants of the Morgan-PGB model.

Every variant starts from the full model built by pgb_model_from_config() and then removes or
changes one part. Building the full model first keeps the random draws identical to the baseline,
so every layer the ablation keeps starts from exactly the same weights as in the full model.
Removing inputs cuts columns out of the existing weight matrices in place (no new layers, so no
extra random draws).

Ablations combine with "+", applied left to right: "prior1+no_w" = 1-D prior and no prior trust.

Add a new ablation by writing a function (model) -> model and registering it in ABLATIONS and
DESCRIPTIONS. Ablations of the head switch it to AblationPGBHead and set its switches.
"""

import torch
import torch.nn as nn

from src.model import MoleculeTower, PGBHead, pgb_model_from_config

PRIOR_NAMES = ["p_exact", "p_delta", "p_family", "log_n", "unseen"]
MECH_NAMES = ["maillard", "ester_amide_hydrolysis", "metal_ion_oxidation", "acid_base_salt_displacement", "schiff_base"]
# the 18 chemistry flags of src/features.py, in order
FLAG_NAMES = ["no_carbons", "oxidizer", "strong_acid", "strong_base", "has_metal", "has_Mg", "has_Ca", "has_Al",
              "has_Si", "has_Ti_Fe", "has_Mn_Cr", "is_salt", "very_short", "many_OH", "has_amine",
              "carboxylic_acid", "ester", "phenol"]
METAL_FLAGS = ["no_carbons", "has_metal", "has_Mg", "has_Ca", "has_Al", "has_Si", "has_Ti_Fe", "has_Mn_Cr",
               "is_salt", "very_short"]


class AblationPGBHead(PGBHead):
    """PGBHead.forward with seven switches:

        prior_cols       which of the 5 prior values the head sees (default all, [0..4])
        use_prior_trust  keep the w * logit(p_exact) term (default True)
        use_gate         gate the bilinear term with g = sigmoid(MLP(gate context)) (default True)
        use_bilinear     put the (gated) bilinear term into the MLP input (default True)
        mech_cols        which mechanism flags the head sees (default all; 5 in v1, 13 in v2)
        api_flag_cols    which of the 18 API chemistry flags reach the gate (default all)
        exc_flag_cols    which of the 18 excipient chemistry flags reach the gate (default all)

        full model       : z = r + w * logit(p_exact) + b
        use_prior_trust=0: z = r + b
    """

    prior_cols = list(range(len(PRIOR_NAMES)))
    use_prior_trust = True
    use_gate = True
    use_bilinear = True
    mech_cols = list(range(len(MECH_NAMES)))
    api_flag_cols = list(range(len(FLAG_NAMES)))
    exc_flag_cols = list(range(len(FLAG_NAMES)))

    def forward(self, h_api, h_exc, prior_vec, api_flags, exc_flags, mech_flags):
        prior_vec  = prior_vec[:, self.prior_cols]   # [B, k], p_exact always column 0
        mech_flags = mech_flags[:, self.mech_cols]   # [B, m]
        api_flags  = api_flags[:, self.api_flag_cols]   # [B, fa]
        exc_flags  = exc_flags[:, self.exc_flag_cols]   # [B, fe]

        product = h_api * h_exc
        diff    = torch.abs(h_api - h_exc)

        pair = [h_api, h_exc, product, diff]
        if self.use_bilinear:
            bilinear = self.U(h_api) * self.V(h_exc)              # [B, r]

            if self.use_gate:
                gate_input = torch.cat([prior_vec, api_flags, exc_flags, mech_flags], dim=-1)  # [B, k + fa + fe + m]
                gate       = self.gate(gate_input)                     # [B, r]
                gated      = bilinear * gate                           # [B, r]
            else:
                gated      = bilinear                                  # [B, r]
            pair.append(gated)

        concat = torch.cat(pair + [prior_vec, mech_flags], dim=-1)  # [B, 4d + r + k + m], r = 0 without bilinear

        residual = self.mlp(concat).squeeze(-1)  # [B]

        logit = residual + self.prior_bias
        if self.use_prior_trust:
            prior_weight = self.prior_trust_mlp(prior_vec).squeeze(-1)
            p = prior_vec[:, 0].clamp(1e-6, 1 - 1e-6)   # p_exact
            logit = logit + prior_weight * torch.log(p / (1 - p))
        return logit   # [B]


class AblationMoleculeTower(MoleculeTower):
    """MoleculeTower.forward with one extra switch:

        flag_cols   which of the 18 chemistry flags go into the tower's flag block (default all);
                    with none left the flag block is gone and the mix layer loses its 16 flag inputs
    """

    flag_cols = list(range(len(FLAG_NAMES)))

    def forward(self, struct_vec, primary, primary_avail, secondary, pgb_desc, salt, flags):
        parts = []
        if self.has_struct and struct_vec is not None:
            parts.append(struct_vec)

        prim_proj = self.primary_block(primary)
        if self.use_missing_embedding and primary_avail is not None:
            missing_mask = (primary_avail == 0.0).unsqueeze(1)
            prim_proj = torch.where(missing_mask, self.missing_primary, prim_proj)
            parts.append(prim_proj)
            parts.append(primary_avail.unsqueeze(1))
        else:
            parts.append(prim_proj)

        if self.has_secondary and secondary is not None:
            parts.append(self.secondary_block(secondary))

        parts.append(self.desc_block(pgb_desc))
        if self.flag_cols:
            parts.append(self.flag_block(flags[:, self.flag_cols]))

        if self.use_salt_context and salt is not None:
            parts.append(salt)

        return self.mix(torch.cat(parts, dim=-1))


def ablation_tower(tower):
    """A MoleculeTower as an AblationMoleculeTower (same weights; switches keep earlier ablations)."""
    if not isinstance(tower, AblationMoleculeTower):
        tower.__class__ = AblationMoleculeTower
        tower.flag_cols = list(range(len(FLAG_NAMES)))
    return tower


def ablation_head(model):
    """The model's head as an AblationPGBHead (same weights; switches keep earlier ablations)."""
    head = model.head
    if not isinstance(head, AblationPGBHead):
        head.__class__ = AblationPGBHead
        head.tower_dim = head.U.in_features      # d, kept here because no_bilinear deletes U
        head.bilinear_dim = head.U.out_features  # r, width of the gated slot in the MLP input (0 once removed)
        if getattr(model.config, "mech_set", "v1") == "v2":    # v2 mechanism flags (ablations/mech_v2.py)
            from ablations.mech_v2 import MECH_V2_NAMES
            head.mech_names = list(MECH_V2_NAMES)
        else:
            head.mech_names = list(MECH_NAMES)
        head.mech_cols = list(range(len(head.mech_names)))
    return head


@torch.no_grad()
def keep_input_columns(linear: nn.Linear, keep):
    """Cut a Linear layer down to the input columns in `keep`, in place (no new random draws)."""
    linear.weight = nn.Parameter(linear.weight[:, keep].clone())
    linear.in_features = len(keep)


# --------------------------------------------------------------------------- A. no prior trust (w)
def no_w(model):
    """Remove the adaptive prior trust w and the w * logit(p_exact) term: z = r + b.
    The prior still enters the gate and the MLP; b keeps its init and stays trainable."""
    head = ablation_head(model)
    if hasattr(head, "prior_trust_mlp"):
        del head.prior_trust_mlp             # no longer in the state dict (already gone after no_prior)
    head.use_prior_trust = False
    return model


# --------------------------------------------------------------------------- B. 1-D prior (p_exact only)
def keep_prior(head, names):
    """Cut the head down to the prior values in `names` (of those it still sees), everywhere it reads
    the prior: gate input, MLP input and prior trust MLP input. The dataset still builds all 5."""
    old = head.prior_cols                    # prior columns the head currently sees
    keep = [i for i, c in enumerate(old) if PRIOR_NAMES[c] in names]
    k = len(old)
    d, r = head.tower_dim, head.bilinear_dim

    # gate input: [prior(k), api_flags, exc_flags, mech_flags]
    if hasattr(head, "gate"):
        gate0 = head.gate[0]
        keep_input_columns(gate0, keep + list(range(k, gate0.in_features)))
    # MLP input: [h_api, h_exc, product, diff (4d), gated (r), prior (k), mech_flags]
    mlp0 = head.mlp[0]
    start = 4 * d + r
    keep_input_columns(mlp0, list(range(start)) + [start + i for i in keep] + list(range(start + k, mlp0.in_features)))
    # prior trust MLP input: prior (k)
    if hasattr(head, "prior_trust_mlp"):
        keep_input_columns(head.prior_trust_mlp[0], keep)

    head.prior_cols = [old[i] for i in keep]


def prior1(model):
    """Keep only p_exact of the 5 prior values, everywhere the head reads the prior:
    gate 46 -> 42, MLP concat 410 -> 406, prior trust MLP 5 -> 8 -> 1 becomes 1 -> 8 -> 1.
    The dataset and prior table are unchanged; the head drops the other 4 columns."""
    keep_prior(ablation_head(model), ["p_exact"])
    return model


# --------------------------------------------------------------------------- C. no gate on the bilinear term
def no_gate(model):
    """Remove the prior-informed gate: gated = bilinear instead of bilinear * g.
    The 16 bilinear values keep their slot in the MLP input (still 410); the flags still reach the
    model through the towers, and the prior and mechanism flags through the MLP input."""
    head = ablation_head(model)
    del head.gate                            # 2,032 parameters, no longer in the state dict
    head.use_gate = False
    return model


# --------------------------------------------------------------------------- D. no bilinear term
def no_bilinear(model):
    """Remove the low-rank bilinear term U*h_api (.) V*h_exc and the gate that scales it:
    U, V and the gate are deleted and the 16 gated columns are cut from the MLP input (410 -> 394).
    h_api, h_exc, their product and |difference|, the prior, mechanism flags and w are unchanged."""
    head = ablation_head(model)
    d, r = head.tower_dim, head.bilinear_dim
    del head.U, head.V                       # 2 x 96 x 16 = 3,072 parameters
    if hasattr(head, "gate"):
        del head.gate                        # 2,032 parameters (already gone after no_gate)
    # MLP input: [h_api, h_exc, product, diff (4d), gated (r), prior, mech_flags]
    mlp0 = head.mlp[0]
    keep_input_columns(mlp0, list(range(4 * d)) + list(range(4 * d + r, mlp0.in_features)))
    head.use_bilinear = False
    head.use_gate = False
    head.bilinear_dim = 0
    return model


# --------------------------------------------------------------------------- E. no excipient prior
def no_prior(model):
    """Remove the excipient prior completely: all 5 values out of the gate (46 -> 41) and the MLP input
    (410 -> 405), and the w * logit(p_exact) term (w and p_exact both come from the prior): z = r + b.
    The dataset still builds the prior and the prior table is still saved; the model never reads them."""
    head = ablation_head(model)
    no_w(model)                              # prior trust MLP and its term
    keep_prior(head, [])
    return model


# --------------------------------------------------------------------------- F. no salt context
SALT_DIM = 8


def drop_salt_context(tower):
    """Remove the 8 raw salt-context values from one MoleculeTower. They are the last 8 inputs of the
    mix layer (64 + 1 + 32 + 24 + 16 + 8 = 145 -> 137); MoleculeTower.forward already skips the salt
    input when use_salt_context is False."""
    if tower.use_salt_context:
        mix0 = tower.mix[0]
        keep_input_columns(mix0, list(range(mix0.in_features - SALT_DIM)))
        tower.use_salt_context = False


def no_salt(model):
    """Remove the 8-value salt context (fragments, metals, inorganic fragments, charges, organic and
    inorganic heavy atoms, counterion) from both towers. Fingerprints, descriptors, chemistry flags
    (incl. is_salt / has_metal), the prior and the head are unchanged."""
    drop_salt_context(model.api_tower)
    drop_salt_context(model.exc_tower)
    return model


# --------------------------------------------------------------------------- G. no availability flag
def drop_availability(tower):
    """Remove the primary-fingerprint availability flag from one MoleculeTower: the flag input of the
    mix layer (column 64, right after the 64-d Morgan projection; 145 -> 144) and the learned
    missing-fingerprint embedding that replaces the projection when the flag is 0.
    MoleculeTower.forward already skips both when use_missing_embedding is False."""
    if tower.use_missing_embedding:
        assert not tower.has_struct, "Morgan towers have no structural branch before the flag"
        avail_col = tower.primary_block[0].out_features   # mix input: [Morgan proj (64), flag, ...]
        mix0 = tower.mix[0]
        keep_input_columns(mix0, [i for i in range(mix0.in_features) if i != avail_col])
        del tower.missing_primary
        tower.use_missing_embedding = False


def no_avail(model):
    """Remove the availability flag (1 for every pair in train, val and held-out, since every
    molecule has SMILES) and the never-used missing-fingerprint embedding from both towers.
    The flag's mix weights are not folded into the bias: the constant input is simply gone."""
    drop_availability(model.api_tower)
    drop_availability(model.exc_tower)
    return model


# --------------------------------------------------------------------------- H. no mechanism flags
def keep_mech(head, names):
    """Cut the head down to the mechanism flags in `names` (of those it still sees). The flags are the
    last m inputs of both the gate ([prior, api_flags, exc_flags, mech]) and the MLP
    ([pair features, prior, mech]). The dataset still computes all 5."""
    old = head.mech_cols                     # mechanism columns the head currently sees
    keep = [i for i, c in enumerate(old) if head.mech_names[c] in names]
    m = len(old)
    for layer in ([head.gate[0]] if hasattr(head, "gate") else []) + [head.mlp[0]]:
        start = layer.in_features - m
        keep_input_columns(layer, list(range(start)) + [start + i for i in keep])
    head.mech_cols = [old[i] for i in keep]


def no_mech(model):
    """Remove all SMARTS mechanism flags from the head (v1: gate 46 -> 41, MLP input 410 -> 405).
    Chemistry flags, prior, w, gate and bilinear term are unchanged."""
    keep_mech(ablation_head(model), [])
    return model


# --------------------------------------------------------------------------- I. chemistry flags
def keep_flags(model, side, names):
    """Keep only the chemistry flags in `names` (of those still used) for one side ("api" or "exc"),
    in both places they are read: the side's tower (flag block input; with none left the flag block is
    deleted and its 16 mix inputs cut) and the gate (that side's flag inputs). The dataset still
    computes all 18."""
    tower = ablation_tower(model.api_tower if side == "api" else model.exc_tower)
    old = tower.flag_cols
    keep = [i for i, c in enumerate(old) if FLAG_NAMES[c] in names]
    if old and not keep:
        assert not tower.has_struct, "Morgan towers have no structural branch"
        # mix input: [Morgan proj, (flag), (MACCS proj), descriptor proj, flag proj (16), (salt)]
        start = (tower.primary_block[0].out_features + int(tower.use_missing_embedding)
                 + (tower.secondary_block[0].out_features if tower.has_secondary else 0)
                 + tower.desc_block[0].out_features)
        width = tower.flag_block[0].out_features
        mix0 = tower.mix[0]
        keep_input_columns(mix0, list(range(start)) + list(range(start + width, mix0.in_features)))
        del tower.flag_block
    elif keep != list(range(len(old))):
        keep_input_columns(tower.flag_block[0], keep)
    tower.flag_cols = [old[i] for i in keep]

    head = ablation_head(model)
    attr = f"{side}_flag_cols"
    old = getattr(head, attr)
    keep = [i for i, c in enumerate(old) if FLAG_NAMES[c] in names]
    if hasattr(head, "gate"):
        # gate input: [prior (k), api flags (fa), exc flags (fe), mech flags]
        start = len(head.prior_cols) + (0 if side == "api" else len(head.api_flag_cols))
        gate0 = head.gate[0]
        keep_input_columns(gate0, list(range(start)) + [start + i for i in keep]
                           + list(range(start + len(old), gate0.in_features)))
    setattr(head, attr, [old[i] for i in keep])


NON_METAL_FLAGS = [f for f in FLAG_NAMES if f not in METAL_FLAGS]


def no_flags(model):
    """Remove all 18 chemistry flags from both towers (flag block, mix 145 -> 129) and the gate (46 -> 10)."""
    keep_flags(model, "api", [])
    keep_flags(model, "exc", [])
    return model


def no_flags_api(model):
    """Remove the 18 API chemistry flags: API tower flag block (mix 145 -> 129) and gate (46 -> 28)."""
    keep_flags(model, "api", [])
    return model


def no_flags_exc(model):
    """Remove the 18 excipient chemistry flags: excipient tower flag block (mix 145 -> 129) and gate (46 -> 28)."""
    keep_flags(model, "exc", [])
    return model


def no_flags_metal(model):
    """Remove the 10 metal / inorganic flags (no_carbons, has_metal, has_Mg, has_Ca, has_Al, has_Si,
    has_Ti_Fe, has_Mn_Cr, is_salt, very_short) on both sides: flag blocks 18 -> 8, gate 46 -> 26."""
    keep_flags(model, "api", NON_METAL_FLAGS)
    keep_flags(model, "exc", NON_METAL_FLAGS)
    return model


# --------------------------------------------------------------------------- registry
def full(model):
    """The unchanged model (baseline trained through the same harness)."""
    return model


ABLATIONS = {
    "full": full,
    "no_w": no_w,
    "prior1": prior1,
    "no_gate": no_gate,
    "no_bilinear": no_bilinear,
    "no_prior": no_prior,
    "no_salt": no_salt,
    "no_avail": no_avail,
    "no_mech": no_mech,
    "no_flags": no_flags,
    "no_flags_api": no_flags_api,
    "no_flags_exc": no_flags_exc,
    "no_flags_metal": no_flags_metal,
}

DESCRIPTIONS = {
    "full": "full model, z = r + w * logit(p_exact) + b",
    "no_w": "A. adaptive prior trust w removed, z = r + b (prior still in gate and MLP)",
    "prior1": "B. prior 5 -> 1 (p_exact only) in gate, MLP and prior trust; w kept",
    "no_gate": "C. gate on the bilinear term removed, gated = bilinear (MLP input still 410)",
    "no_bilinear": "D. bilinear term and its gate removed (U, V, gate), MLP input 410 -> 394",
    "no_prior": "E. excipient prior removed (gate 46 -> 41, MLP 410 -> 405, no w term), z = r + b",
    "no_salt": "F. 8-value salt context removed from both towers (mix 145 -> 137)",
    "no_avail": "G. availability flag and missing-fingerprint embedding removed from both towers (mix 145 -> 144)",
    "no_mech": "H. 5 SMARTS mechanism flags removed from the head (gate 46 -> 41, MLP 410 -> 405)",
    "no_flags": "I. all 18 chemistry flags removed from both towers (mix 145 -> 129) and the gate (46 -> 10)",
    "no_flags_api": "I-a. API chemistry flags removed (API tower mix 145 -> 129, gate 46 -> 28)",
    "no_flags_exc": "I-b. excipient chemistry flags removed (excipient tower mix 145 -> 129, gate 46 -> 28)",
    "no_flags_metal": "I-c. 10 metal / inorganic chemistry flags removed on both sides (flag blocks 18 -> 8, gate 46 -> 26)",
}


def parse_ablation(name):
    """'prior1+no_w' -> ['prior1', 'no_w']; raises ValueError for unknown parts."""
    parts = name.split("+")
    unknown = [p for p in parts if p not in ABLATIONS]
    if unknown:
        raise ValueError(f"unknown ablation(s) {unknown}; available: {sorted(ABLATIONS)}")
    if len(parts) > 1 and "full" in parts:
        raise ValueError("'full' cannot be combined with other ablations")
    return parts


def describe(name):
    return " + ".join(DESCRIPTIONS[p] for p in parse_ablation(name))


def build_model(config, encoder):
    """Full model from the config, then the ablation(s) named in config.ablation applied in order."""
    model = pgb_model_from_config(config, encoder)
    for part in parse_ablation(config.ablation):
        model = ABLATIONS[part](model)
    return model
