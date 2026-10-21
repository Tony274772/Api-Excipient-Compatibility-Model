"""Simple top-to-bottom architecture diagram of the Morgan-PGB model, with the tensor size at
every layer. Layer shapes checked against checkpoints/morgan_pgb_8020_seed42/best_model.pt
(api_tower.primary_block 64x1024, secondary_block 32x167, desc_block 24x12, flag_block 16x18,
mix 96x145, head.U/V 16x96, gate 32x46 -> 16x32, mlp 128x410 -> 48x128 -> 1x48,
prior_trust_mlp 8x5 -> 1x8, prior_bias -0.421).

Run:  python figures/make_architecture_diagram.py
Writes figures/model_architecture.png and figures/model_architecture.svg
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
W, H = 100, 156

# (fill, edge) per component type
COLORS = {
    "input":  ("#f4f4f5", "#3f3f46"),
    "api":    ("#dbeafe", "#1d4ed8"),
    "exc":    ("#ffedd5", "#c2410c"),
    "pair":   ("#ede9fe", "#6d28d9"),
    "head":   ("#dcfce7", "#15803d"),
    "out":    ("#fee2e2", "#b91c1c"),
}
INK = "#111827"
GREY = "#4b5563"

fig, ax = plt.subplots(figsize=(16, 16 * H / W))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("white")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)


def box(x0, x1, yc, h, text, kind, fs=11, bold=False, lw=1.6):
    fill, edge = COLORS[kind]
    ax.add_patch(FancyBboxPatch((x0, yc - h / 2), x1 - x0, h,
                                boxstyle="round,pad=0,rounding_size=0.8",
                                fc=fill, ec=edge, lw=lw, zorder=2))
    ax.text((x0 + x1) / 2, yc, text, ha="center", va="center", fontsize=fs * 1.15,
            fontweight="bold" if bold else "normal", color=INK, zorder=3, linespacing=1.35)


def panel(x0, x1, y0, y1, title, kind):
    edge = COLORS[kind][1]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=1.5",
                                fc="white", ec=edge, lw=2, ls=(0, (5, 3)), zorder=1))
    ax.text((x0 + x1) / 2, y1 - 2.4, title, ha="center", va="center", fontsize=14,
            fontweight="bold", color=edge, zorder=3)


def arrow(x0, y0, x1, y1, color=GREY, lw=1.8):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=18,
                                shrinkA=0, shrinkB=0))


def line(xs, ys, color=GREY, lw=1.8):
    ax.plot(xs, ys, color=color, lw=lw, zorder=1.5, solid_capstyle="round")


def label(x, y, text, color=GREY, fs=10, ha="center", bold=False):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color, zorder=5,
            fontweight="bold" if bold else "normal",
            bbox=dict(fc="white", ec="none", pad=1.2))


# ---------------------------------------------------------------- title
ax.text(W / 2, 153, "Morgan-PGB: API–Excipient Compatibility Model", ha="center", va="center",
        fontsize=20, fontweight="bold", color=INK)
ax.text(W / 2, 149.6, "Prior-Gated Bilinear head  ·  236,219 trainable parameters  ·  "
        "[n] = vector size for one API–excipient pair", ha="center", va="center",
        fontsize=12, color=GREY)

# ---------------------------------------------------------------- step 1: towers
TOWER_TOP, TOWER_BOT = 131, 68
ROWS_Y = [122.5, 115, 107.5, 100, 92.5, 85]
ROW_H = 5.8
ROWS = [
    ("Morgan fingerprint\n(radius 2)  [1024]", "Linear 1024 → 64\nLayerNorm · GELU · Dropout 0.15", "64"),
    ("MACCS keys  [167]", "Linear 167 → 32\nLayerNorm · GELU · Dropout 0.075", "32"),
    ("12 descriptors, z-scored\n(MolWt, LogP, TPSA …)  [12]", "Linear 12 → 24\nLayerNorm · GELU", "24"),
    ("18 chemistry flags\n(Mg, Ca, amine, ester …)  [18]", "Linear 18 → 16\nGELU", "16"),
    ("Salt context\n(charges, fragments …)  [8]", "passed through\nunchanged", "8"),
    ("SMILES-available flag  [1]", "passed through\nunchanged", "1"),
]


def tower(x0, kind, who, h_name):
    edge = COLORS[kind][1]
    # input SMILES
    box(x0 + 7, x0 + 37, 140, 5, f"{who} SMILES", "input", fs=13, bold=True)
    arrow(x0 + 22, 137.5, x0 + 22, TOWER_TOP)
    label(x0 + 22, 134.2, "RDKit computes the features below", fs=10)

    panel(x0, x0 + 44, TOWER_BOT, TOWER_TOP, f"① {who} tower", kind)
    bus_x = x0 + 41.5
    for y, (feat, layer, out) in zip(ROWS_Y, ROWS):
        box(x0 + 1.5, x0 + 18, y, ROW_H, feat, "input", fs=9.5)
        arrow(x0 + 18, y, x0 + 20.5, y)
        box(x0 + 20.5, x0 + 38.5, y, ROW_H, layer, kind, fs=9.5)
        line([x0 + 38.5, bus_x], [y, y], edge)
        label(x0 + 40, y + 1.5, out, color=edge, fs=10, bold=True)
    line([bus_x, bus_x], [ROWS_Y[0], ROWS_Y[-1]], edge)
    arrow(bus_x, ROWS_Y[-1], bus_x, 79.25, color=edge)

    box(x0 + 1.5, x0 + 42.5, 77, 4.5, "Concatenate:  64 + 32 + 24 + 16 + 8 + 1  =  145", kind, fs=10.5, bold=True)
    arrow(x0 + 22, 74.75, x0 + 22, 73.25, color=edge)
    box(x0 + 1.5, x0 + 42.5, 71, 4.5, "Mix:  Linear 145 → 96 · LayerNorm · GELU", kind, fs=10.5, bold=True)

    arrow(x0 + 22, 68.75, x0 + 22, 63.25, color=edge)
    box(x0 + 14, x0 + 30, 61, 4.5, f"{h_name}  [96]", kind, fs=13, bold=True, lw=2.2)


tower(3, "api", "API", "h_API")
tower(53, "exc", "Excipient", "h_EXC")
ax.text(W / 2, 140, "same design,\nseparate weights", ha="center", va="center", fontsize=10.5,
        color=GREY, style="italic")

# pair-level inputs between the two h boxes
box(34.2, 49.6, 61, 7.5, "Excipient prior  [5]\np_exact, p_delta, p_family,\nlog n, unseen\n(train-only lookup table)",
    "pair", fs=9)
box(50.4, 65.8, 61, 7.5, "Mechanism flags  [5]\nSMARTS rules on API + exc.\nMaillard, hydrolysis,\nsalt displacement …",
    "pair", fs=9)

# ---------------------------------------------------------------- step 3: PGB head
HEAD_TOP, HEAD_BOT = 55, 18
panel(3, 97, HEAD_BOT, HEAD_TOP, "② PGB interaction head", "head")
for x in (25, 42.5, 57.5, 75):
    arrow(x, 58.75 if x in (25, 75) else 57.25, x, HEAD_TOP)

# row A: interaction terms + gate
YA, HA = 46, 7.5
box(5, 20, YA, HA, "Element-wise product\nh_API ⊙ h_EXC\n[96]", "head", fs=10)
box(22, 37, YA, HA, "Absolute difference\n|h_API − h_EXC|\n[96]", "head", fs=10)
box(39, 61, YA, HA, "Low-rank bilinear\nU·h_API (96 → 16)  ⊙\nV·h_EXC (96 → 16)   [16]", "head", fs=10)
box(63, 95, YA, HA, "Gate network\ninput: prior 5 + API flags 18 + exc. flags 18\n+ mech 5 = 46\n"
    "Linear 46 → 32 · GELU · Linear 32 → 16 · Sigmoid   [16]", "head", fs=9.2)

# row B: gated bilinear
YB = 37.5
box(50, 80, YB, 4.2, "Gated bilinear = bilinear ⊙ gate   [16]", "head", fs=10.5, bold=True)
arrow(50, YA - HA / 2, 58, YB + 2.1)
arrow(79, YA - HA / 2, 72, YB + 2.1)

# row C: concat
YC = 31
box(5, 95, YC, 4.6, "Concatenate:  h_API 96 + h_EXC 96 + product 96 + |difference| 96 + gated 16 + prior 5 + mech 5  =  410",
    "head", fs=10.5, bold=True)
arrow(12.5, YA - HA / 2, 12.5, YC + 2.3)
arrow(29.5, YA - HA / 2, 29.5, YC + 2.3)
arrow(65, YB - 2.1, 65, YC + 2.3)

# row D: MLP + prior trust
YD = 23.5
box(5, 23, YD, 5.4, "Linear 410 → 128\nGELU · Dropout 0.30", "head", fs=10)
arrow(23, YD, 26, YD)
box(26, 44, YD, 5.4, "Linear 128 → 48\nGELU · Dropout 0.20", "head", fs=10)
arrow(44, YD, 47, YD)
box(47, 61, YD, 5.4, "Linear 48 → 1\nresidual r   [1]", "head", fs=10, bold=True)
arrow(14, YC - 2.3, 14, YD + 2.7)
box(66, 95, YD, 5.4, "Adaptive prior trust\nprior 5 → Linear 5 → 8 · GELU\n→ Linear 8 → 1 · Sigmoid  =  w  [1]", "head", fs=9.5)
label(80.5, YD + 3.9, "input: excipient prior [5]", fs=9)

# ---------------------------------------------------------------- step 3: output
YE = 12
box(14, 86, YE, 5.5, "③ Output logit   z  =  r  +  w · logit(p_exact)  +  b\n"
    "p_exact = excipient's incompatibility rate in train  ·  b = learned bias (−0.421)", "out", fs=11.5)
arrow(54, YD - 2.7, 54, YE + 2.75)
arrow(80.5, YD - 2.7, 80.5, YE + 2.75)

arrow(W / 2, YE - 2.75, W / 2, 6.3, color=COLORS["out"][1])
box(10, 90, 4, 4.6, "P(incompatible) = sigmoid(z)      P ≥ 0.671  →  INCOMPATIBLE      P < 0.671  →  COMPATIBLE",
    "out", fs=13, bold=True, lw=2.2)

for ext, dpi in (("png", 200), ("svg", None)):
    fig.savefig(os.path.join(OUT_DIR, f"model_architecture.{ext}"), dpi=dpi, facecolor="white")
print("saved", os.path.join(OUT_DIR, "model_architecture.png"))
