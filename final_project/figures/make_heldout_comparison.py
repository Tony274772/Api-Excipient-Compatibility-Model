"""Held-out comparison table: FormulationDE, DeInteract, Towards Safe and ARES on the 22 external pairs.

All sources hold the original 24 pairs; rows 10 and 11 (Hydrocortisone + magnesium stearate /
beta cyclodextrin, also in train.csv) are dropped before plotting, as in data/held_out_test_set.csv.

Sources
  DeInteract    : extra work/metrics_4_models/heldout/Results_De_interact_heldout.csv
                  (the CSV reports the probability of the predicted class; P(incompatible) is
                  1 - p when the call is "Possibly Compatible")
  Towards Safe  : paper_model_og/held_out/held_out_predictions.csv, p_incompat_final, threshold 0.5
  FormulationDE, ARES : transcribed from the previous version of this figure
                  (ARES threshold 0.671; the 0.67 in rows 2 and 21 are rounded values on either side of it)

Run:  python figures/make_heldout_comparison.py
Writes figures/heldout_predictions_comparison.png
"""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\pardhu\Desktop"
DE_CSV = os.path.join(ROOT, "extra work", "metrics_4_models", "heldout", "Results_De_interact_heldout.csv")
TS_CSV = os.path.join(ROOT, "versions_of_api-exc", "paper_model_og", "held_out", "held_out_predictions.csv")
OUT = os.path.join(HERE, "heldout_predictions_comparison.png")

PAIRS = [
    ("Warfarin", "Magnesium aluminum silicate"), ("Phenytoin", "Magnesium oxide"),
    ("Trichlormethiazide", "Magnesium oxide"), ("Tetracycline", "Calcium phosphate"),
    ("Fluvoxamine", "Lactose"), ("Acetazolamide", "Lactose"), ("Acetazolamide", "Mannitol"),
    ("Phenytoin sodium", "Lactose"), ("Phenytoin sodium", "Calcium sulfate"),
    ("Hydrocortisone", "Magnesium stearate"), ("Hydrocortisone", "Beta cyclodextrin"),
    ("Quercetin", "Magnesium carbonate"),
    ("Metformin hydrochloride", "Magnesium stearate"), ("Lamotrigine", "Sodium lauryl sulfate"),
    ("Gabapentin", "Mannitol"), ("Omeprazole", "Silicon dioxide"), ("Carbamazepine", "Propylene glycol"),
    ("Hydrochlorothiazide", "Stearic acid"), ("Celecoxib", "Titanium dioxide"),
    ("Topiramate", "Triacetin"), ("Levetiracetam", "Talc"), ("Fenofibrate", "Sucrose"),
    ("Risperidone", "Lactose monohydrate"), ("Olanzapine", "Magnesium stearate"),
]
LABELS = [1] * 12 + [0] * 12

# (call, P(incompatible)) as shown in the previous figure
FORMDE = [(0, .28), (1, .86), (1, .92), (0, .07), (0, .09), (1, .98), (1, .98), (1, .68), (1, .84), (0, .25),
          (0, .06), (1, .61), (0, .35), (1, .99), (1, .58), (1, .87), (1, .83), (1, 1.0), (1, 1.0), (1, .93),
          (0, .02), (1, .96), (1, .61), (1, .86)]
ARES = [(0, .56), (1, .67), (1, .82), (1, .71), (0, .57), (1, .72), (0, .44), (0, .51), (1, .80), (1, .68),
        (1, .76), (1, .71), (0, .53), (0, .28), (0, .23), (0, .14), (0, .27), (0, .44), (0, .05), (0, .17),
        (0, .67), (0, .35), (0, .21), (0, .36)]


def load_deinteract():
    df = pd.read_csv(DE_CSV, usecols=[2, 3])
    df = df.dropna()
    out = []
    for gt, txt in zip(df.iloc[:, 0], df.iloc[:, 1]):
        p = float(re.search(r"Probability: ([0-9.]+)", txt).group(1))
        inc = "Possibly Incompatible" in txt
        out.append((int(inc), p if inc else 1.0 - p))
    assert len(out) == 24 and list(df.iloc[:, 0].astype(int)) == LABELS
    return out


def load_towards_safe():
    df = pd.read_csv(TS_CSV)
    assert list(df["ground_truth"]) == LABELS
    return [(int(p >= 0.5), float(p)) for p in df["p_incompat_final"]]


MODELS = [
    ("FormulationDE", "prediction · P(incompatible)", FORMDE, False),
    ("DeInteract", "prediction · P(incompatible)", load_deinteract(), False),
    ("Towards Safe", "prediction · P(incompatible), t = 0.5", load_towards_safe(), False),
    ("ARES", "prediction · P(incompatible), t = 0.671", ARES, True),
]

# drop the 2 pairs that are also in train.csv (removed from data/held_out_test_set.csv on 2026-11-04)
DROP = {9, 10}
KEEP = [i for i in range(len(PAIRS)) if i not in DROP]
PAIRS = [PAIRS[i] for i in KEEP]
LABELS = [LABELS[i] for i in KEEP]
MODELS = [(name, sub, [preds[i] for i in KEEP], ours) for name, sub, preds, ours in MODELS]
N, N_INC = len(PAIRS), sum(LABELS)

# ----- palette -----
INK, MUTED, RULE = "#1b1f27", "#5b6270", "#e3e6eb"
GREEN, RED = "#2a7f5f", "#c4412b"
PINK_BG, PINK_FG = "#fde7e2", "#c4412b"
MINT_BG, MINT_FG = "#dcf2ea", "#2a7f5f"
ARES_BG = "#eef1f7"

# ----- layout (pixels) -----
W = 2660
X_NUM, X_NAME, X_LABEL = 100, 150, 900
X_MODEL0, COL_W = 1090, 390
ROW_H, SEC_H, HEAD_H = 56, 46, 240
n_sec = 2
BODY = N * ROW_H + n_sec * SEC_H
FOOT_H = 260
H = HEAD_H + BODY + FOOT_H

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")
fig.patch.set_facecolor("white")

FONT = "DejaVu Sans"


def T(x, y, s, size=13, color=INK, weight="normal", ha="left", style="normal"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va="center",
            family=FONT, style=style)


T(60, 58, f"Held-out test set: predictions on all {N} API–excipient pairs", 25, weight="bold")
T(60, 112, f"{N_INC} incompatible + {N - N_INC} compatible external pairs. Each cell shows the model's call and its predicted "
           "probability of incompatibility; ✓ correct, ✕ wrong.", 13.5, MUTED)

# ARES column highlight
ares_x = X_MODEL0 + 3 * COL_W
ax.add_patch(FancyBboxPatch((ares_x + 6, 165), COL_W - 12, HEAD_H - 165 + BODY + FOOT_H - 40,
                            boxstyle="round,pad=0,rounding_size=14", fc=ARES_BG, ec="none", zorder=0))

# header
T(X_NUM, 205, "#", 13, MUTED)
T(X_NAME, 205, "API + excipient", 13, MUTED)
T(X_LABEL, 205, "True label", 13, MUTED)
for i, (name, sub, _, ours) in enumerate(MODELS):
    cx = X_MODEL0 + i * COL_W + COL_W / 2
    T(cx, 182, name, 17, weight="bold", ha="center")
    T(cx, 214, sub, 10.5, MUTED, ha="center")
ax.add_line(plt.Line2D([60, W - 60], [HEAD_H - 6, HEAD_H - 6], color=INK, lw=2.2))


def pill(x, y, label):
    inc = label == 1
    txt = "Incompatible" if inc else "Compatible"
    w = 150 if inc else 130
    ax.add_patch(FancyBboxPatch((x, y - 15), w, 30, boxstyle="round,pad=0,rounding_size=15",
                                fc=PINK_BG if inc else MINT_BG, ec="none"))
    T(x + w / 2, y, txt, 11.5, PINK_FG if inc else MINT_FG, ha="center")


def mark(x, y, ok):
    r = 15
    if ok:
        ax.add_patch(Circle((x, y), r, fc="#e1f2ea", ec="none"))
        ax.plot([x - 6, x - 2, x + 7], [y, y + 5, y - 6], color=GREEN, lw=2.4, solid_capstyle="round")
    else:
        ax.add_patch(Circle((x, y), r, fc=RED, ec="none"))
        ax.plot([x - 5, x + 5], [y - 5, y + 5], color="white", lw=2.4, solid_capstyle="round")
        ax.plot([x - 5, x + 5], [y + 5, y - 5], color="white", lw=2.4, solid_capstyle="round")


y = HEAD_H
idx = 0
stats = [dict(tp=0, tn=0, fp=0, fn=0) for _ in MODELS]
for sec_title, rng in (("INCOMPATIBLE PAIRS (LABEL 1)", range(0, N_INC)), ("COMPATIBLE PAIRS (LABEL 0)", range(N_INC, N))):
    T(X_NUM, y + SEC_H / 2, sec_title, 11.5, MUTED, weight="bold")
    y += SEC_H
    for r in rng:
        yc = y + ROW_H / 2
        api, exc = PAIRS[r]
        T(X_NUM + 5, yc, str(r + 1), 13, MUTED, ha="center")
        t_api = ax.text(X_NAME, yc, api, fontsize=14, fontweight="bold", color=INK, ha="left", va="center", family=FONT)
        x_end = t_api.get_window_extent(renderer=fig.canvas.get_renderer()).x1
        T(x_end + 16, yc, "+", 12, "#9aa1ad")
        T(x_end + 40, yc, exc, 14)
        pill(X_LABEL, yc, LABELS[r])
        for i, (_, _, preds, _) in enumerate(MODELS):
            call, p = preds[r]
            ok = call == LABELS[r]
            s = stats[i]
            s["tp" if (call == 1 and LABELS[r] == 1) else "tn" if (call == 0 and LABELS[r] == 0)
              else "fp" if call == 1 else "fn"] += 1
            x0 = X_MODEL0 + i * COL_W
            mark(x0 + 36, yc, ok)
            T(x0 + 70, yc, "Incompatible" if call else "Compatible", 14, weight="normal")
            T(x0 + 285, yc, f"{p:.2f}", 13, MUTED)
        ax.add_line(plt.Line2D([60, ares_x - 6], [y + ROW_H, y + ROW_H], color=RULE, lw=1))
        ax.add_line(plt.Line2D([ares_x + COL_W + 6, W - 60], [y + ROW_H, y + ROW_H], color=RULE, lw=1)) if False else None
        y += ROW_H
    if rng.start == 0:
        ax.add_line(plt.Line2D([60, W - 60], [y, y], color=RULE, lw=1.2))

ax.add_line(plt.Line2D([60, W - 60], [y, y], color=INK, lw=2.2))

# footer summary
T(X_NAME, y + 55, "Correct predictions", 18, weight="bold")
T(X_NAME, y + 90, f"out of {N} held-out pairs", 12.5, MUTED)
for i, (name, _, _, ours) in enumerate(MODELS):
    s = stats[i]
    n_ok = s["tp"] + s["tn"]
    cx = X_MODEL0 + i * COL_W + COL_W / 2
    T(cx - 40, y + 55, str(n_ok), 40, weight="bold", ha="center")
    T(cx + 5, y + 66, f"/{N}", 17, MUTED, ha="left")
    bx, bw = cx - 140, 280
    ax.add_patch(FancyBboxPatch((bx, y + 108), bw, 12, boxstyle="round,pad=0,rounding_size=6", fc="#e3e6eb", ec="none"))
    ax.add_patch(FancyBboxPatch((bx, y + 108), bw * n_ok / N, 12, boxstyle="round,pad=0,rounding_size=6",
                                fc=INK if ours else "#8b95a7", ec="none"))
    T(cx, y + 145, f"{n_ok / N:.1%} correct", 13.5, weight="bold", ha="center")
    T(cx, y + 172, f"TP {s['tp']} · TN {s['tn']} · FP {s['fp']} · FN {s['fn']}", 11.5, MUTED, ha="center")

T(60, H - 38,
  "Two of the original 24 pairs (Hydrocortisone + magnesium stearate / beta cyclodextrin) are in the ARES training set and are left out.   "
  "DeInteract reports the probability of its predicted class; P(incompatible) is 1 − p for its Compatible calls.",
  11.5, MUTED)

fig.savefig(OUT, dpi=100, facecolor="white")
for (name, *_), s in zip(MODELS, stats):
    print(f"{name:14s} {s['tp'] + s['tn']}/{N}  TP {s['tp']} TN {s['tn']} FP {s['fp']} FN {s['fn']}")
print("wrote", OUT)
