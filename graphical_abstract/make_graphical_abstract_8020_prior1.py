"""Graphical abstract, architecture variant: Morgan-PGB with a 1-D excipient prior (p_exact only)
and no adaptive prior trust branch. Copy of make_graphical_abstract_8020.py with these changes:
  - prior vector 5 -> 1, so gate context 46 -> 42 and concat 410 -> 406
  - adaptive prior trust network removed; prediction is z = r + b
Panel D numbers are unchanged from the 80/20 model (5-D prior + adaptive prior trust).

Original notes for the 80/20 model (final_project/):

Every number is taken from final_project/:
  data/split_log.txt                    3,544 pairs; train 2,770 / val 774; held-out 24 (12 / 12)
  reference_results/val_metrics.json    PR-AUC 0.686, MCC 0.639, 734/774, threshold 0.671
  reference_results/heldout_metrics.json PR-AUC 0.959, MCC 0.707, 20/24
  reference_results/heldout_unseen_metrics.json  18/22 on held-out pairs absent from train
  ROC-AUC (val 0.924, held-out 0.958) recomputed from reference_results/predictions_*.csv
Architecture verified from checkpoints/morgan_pgb_8020_seed42/best_model.pt weight shapes.
Example pair: data/train.csv row with API_CID 4908 (primaquine) + Excipient_CID 6134 (lactose), label 1.
"""

import math
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_NAME = "morgan_pgb_8020_graphical_abstract_prior1"
W, H = 1600, 900

NAVY = "#1F2A44"
SLATE = "#5B6B7F"
SLATE_L = "#8A97A8"
LINE = "#CDD4DD"
PANEL = "#F8F9FB"
TEAL = "#2A8C82"
TEAL_L = "#E2F1EE"
TEAL_M = "#9ACFC7"
CORAL = "#C9614A"
CORAL_L = "#FBE8E2"
AMBER = "#B9822A"
AMBER_D = "#7A5413"
AMBER_L = "#FBF0DA"
TRAIN = "#3E4F6A"
VAL = "#8394AB"
FONT = "Arial, Helvetica, sans-serif"
MONO = "Consolas, 'Courier New', monospace"

parts = []


def add(s):
    parts.append(s)


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=12, fill=NAVY, weight="normal", anchor="start", style="normal",
         raw=False, spacing=None, family=FONT):
    body = s if raw else esc(s)
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" font-style="{style}" text-anchor="{anchor}"{ls}>{body}</text>')


def rect(x, y, w, h, fill="none", stroke=LINE, sw=1, rx=6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def line(pts, color=SLATE, sw=1.5, arrow=True, dash=None):
    d = "M " + " L ".join(f"{x} {y}" for x, y in pts)
    mk = f' marker-end="url(#ah-{color[1:]})"' if arrow else ""
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" '
        f'stroke-linejoin="round" stroke-linecap="round"{ds}{mk}/>')


def circle(cx, cy, r, fill, stroke=None, sw=1):
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"{st}/>')


def hline(x1, x2, y, color=LINE, sw=1):
    add(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="{sw}"/>')


def panel(x, y, w, h, letter, label):
    rect(x, y, w, h, fill=PANEL, stroke=LINE, sw=1, rx=10)
    rect(x + 12, y + 11, 20, 20, fill=NAVY, stroke="none", rx=4)
    text(x + 22, y + 26, letter, 13, "#FFFFFF", "bold", "middle")
    text(x + 40, y + 26, label, 15, NAVY, "bold")


def module_heading(x, y, label):
    text(x, y, label.upper(), 10.5, SLATE, "bold", spacing="0.8")


def pill(x, y, w, label, fill, size=12.5, h=24):
    rect(x, y, w, h, fill=fill, stroke="none", rx=h / 2)
    text(x + w / 2, y + h / 2 + size * 0.36, label, size, "#FFFFFF", "bold", "middle")


# ════════════════════════════════════════════════════════════════════════════
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add("<defs>")
for c in (SLATE, NAVY, TEAL, AMBER, CORAL, SLATE_L):
    add(f'<marker id="ah-{c[1:]}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0 0.8 L9 5 L0 9.2 z" fill="{c}"/></marker>')
add("</defs>")
add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

PT, PB = 24, 876

# ════════════════════════════════════════════════════════════════════════════
# PANEL A — datasets, then the SMILES input pair
# ════════════════════════════════════════════════════════════════════════════
panel(24, PT, 290, PB - PT, "A", "Data and model input")

# Datasets box
rect(38, 62, 262, 410, fill="#FFFFFF", stroke=LINE, rx=8)
text(52, 94, "Datasets", 20, NAVY, "bold")
text(52, 134, "3,544", 26, NAVY, "bold")
text(133, 134, "API–excipient pairs", 12.5, NAVY)
cw = round(234 * 3200 / 3544, 1)
rect(52, 146, cw, 14, fill=TEAL, stroke="none", rx=2)
rect(52 + cw, 146, 234 - cw, 14, fill=CORAL, stroke="none", rx=2)
text(52, 180, "3,200 compatible (90.3%)", 12, TEAL, "bold")
text(52, 197, "344 incompatible (9.7%)", 12, CORAL, "bold")
hline(52, 286, 214)

text(52, 238, "80/20 API-cluster split", 13.5, NAVY, "bold")
for i, (lab, n, pct, col) in enumerate([("Train", 2770, "78.2%", TRAIN), ("Val", 774, "21.8%", VAL)]):
    ry = 252 + i * 28
    text(52, ry + 14, lab, 12, NAVY, "bold")
    rect(94, ry, round(92 * n / 2770, 1), 18, fill=col, stroke="none", rx=3)
    text(286, ry + 14, f"{n:,} ({pct})", 12.5, NAVY, "bold", "end")
text(52, 324, "Butina API clusters (Morgan, 2,048-bit);", 11, SLATE)
text(52, 339, "no API cluster in both sets", 11, SLATE)
hline(52, 286, 354)

text(52, 376, "External held-out test", 13.5, NAVY, "bold")
for i in range(24):
    rect(round(52 + i * 9.75, 2), 388, 8.25, 12, fill=CORAL if i < 12 else TEAL, stroke="none", rx=1.5)
text(52, 420, "24 pairs: 12 incompatible, 12 compatible", 11.5, NAVY)
text(52, 437, "not used for training, early stopping", 11, SLATE)
text(52, 452, "or threshold selection", 11, SLATE)

# SMILES input pair
IN_Y = 540
module_heading(40, IN_Y - 14, "Model input · SMILES pair")


def smiles_card(y, badge, badge_w, badge_col, name, smiles_lines):
    h = 50 + 18 * len(smiles_lines) + 12
    rect(38, y, 262, h, fill="#FFFFFF", stroke=badge_col, sw=1.3, rx=8)
    pill(50, y + 12, badge_w, badge, badge_col, 12, 22)
    text(60 + badge_w, y + 29, name, 18, badge_col if badge_col != SLATE else NAVY, "bold")
    rect(50, y + 44, 238, 18 * len(smiles_lines) + 8, fill="#F1F3F6", stroke="none", rx=4)
    for i, s in enumerate(smiles_lines):
        text(58, y + 61 + i * 18, s, 12, NAVY, family=MONO)
    return h


h_api = smiles_card(IN_Y, "API", 46, NAVY, "Primaquine", ["CC(CCCN)NC1=C2C(=CC(=C1)OC)C=CC=N2"])
circle(169, IN_Y + h_api + 24, 12, "#FFFFFF", NAVY, 1.4)
text(169, IN_Y + h_api + 29, "+", 16, NAVY, "bold", "middle")
exc_y = IN_Y + h_api + 48
h_exc = smiles_card(exc_y, "Excipient", 82, SLATE, "Lactose",
                    ["C(C1C(C(C(C(O1)OC2C(OC(", "C(C2O)O)O)CO)O)O)O)O"])
text(169, exc_y + h_exc + 20, "training pair · label 1 (incompatible)", 11, SLATE_L, anchor="middle")

# ════════════════════════════════════════════════════════════════════════════
# PANEL B — two encoding towers
# ════════════════════════════════════════════════════════════════════════════
panel(330, PT, 400, PB - PT, "B", "Molecular feature encoding")

FEATS = [
    ("Morgan FP", "r = 2 · 1,024 bits", "64", "morgan"),
    ("MACCS keys", "167 bits", "32", "maccs"),
    ("Descriptors", "12 physicochemical", "24", "desc"),
    ("Chem. flags", "18 binary", "16", "flags"),
    ("Salt context", "8 counts / charges", "8", "salt"),
    ("Availability", "1 flag", "1", "avail"),
]


def glyph(kind, x, y):
    if kind in ("morgan", "maccs"):
        on = {"morgan": {2, 9, 13, 21, 26}, "maccs": {1, 4, 8, 11, 15, 19, 22, 25}}[kind]
        col = TEAL if kind == "morgan" else SLATE
        for i in range(28):
            add(f'<rect x="{x + i * 1.75:.2f}" y="{y}" width="1.75" height="12" '
                f'fill="{col if i in on else "#FFFFFF"}" stroke="{LINE}" stroke-width="0.3"/>')
    elif kind == "desc":
        for i, hh in enumerate([6, 10, 4, 8, 12, 5, 9, 7, 11, 3, 8, 6]):
            add(f'<rect x="{x + i * 4}" y="{y + 12 - hh}" width="3" height="{hh}" fill="{SLATE_L}"/>')
    elif kind == "flags":
        on = {1, 4, 9, 13, 16}
        for i in range(18):
            r_, c_ = divmod(i, 9)
            circle(x + 2.5 + c_ * 5.3, y + 3 + r_ * 6, 2.0, TEAL if i in on else "#FFFFFF", SLATE_L, 0.6)
    elif kind == "salt":
        circle(x + 10, y + 6, 5.5, "#FFFFFF", SLATE, 1)
        text(x + 10, y + 9.5, "+", 9, SLATE, "bold", "middle")
        circle(x + 26, y + 6, 4, "#FFFFFF", SLATE, 1)
        text(x + 26, y + 9.3, "−", 9, SLATE, "bold", "middle")
    elif kind == "avail":
        add(f'<rect x="{x + 4}" y="{y}" width="12" height="12" fill="{TEAL}" rx="1.5"/>')
        text(x + 10, y + 10, "1", 9, "#FFFFFF", "bold", "middle")


def tower(top, title, accent, emb_word):
    pill(346, top, 138 if title.startswith("Exc") else 104, title, accent, 13.5, 26)
    chip_x, chip_w, chip_h, step = 360, 186, 48, 58
    y0 = top + 38
    centers = []
    for i, (t, s, od, kind) in enumerate(FEATS):
        cy = y0 + i * step
        primary = kind == "morgan"
        rect(chip_x, cy, chip_w, chip_h, fill="#FFFFFF",
             stroke=TEAL if primary else LINE, sw=1.6 if primary else 1, rx=5)
        text(chip_x + 9, cy + 20, t, 12.5, TEAL if primary else NAVY, "bold")
        text(chip_x + 9, cy + 37, s, 11, SLATE)
        glyph(kind, chip_x + chip_w - 52, cy + 12 if kind != "desc" else cy + 10)
        centers.append(cy + chip_h / 2)
    bx_ = 350
    line([(bx_, y0 + 6), (bx_, y0 + 5 * step + chip_h - 6)], color=accent, sw=1.5, arrow=False)
    for c in centers:
        line([(bx_, c), (chip_x - 2, c)], color=accent, sw=1.1, arrow=False)
    px, pw = 562, 34
    for c, (t, s, od, kind) in zip(centers, FEATS):
        direct = kind in ("salt", "avail")
        line([(chip_x + chip_w + 1, c), (px - 2, c)], color=SLATE_L, sw=1.2)
        rect(px, c - 12, pw, 24, fill=TEAL_L if not direct else "#FFFFFF",
             stroke=TEAL if not direct else SLATE_L, sw=1, rx=4, dash=None if not direct else "3 2")
        text(px + pw / 2, c + 4.5, od, 12.5, TEAL if not direct else SLATE, "bold", "middle")
    cbx = 606
    line([(px + pw + 1, centers[0]), (cbx, centers[0]), (cbx, centers[-1]), (px + pw + 1, centers[-1])],
         color=SLATE, sw=1.3, arrow=False)
    for c in centers[1:-1]:
        line([(px + pw + 1, c), (cbx, c)], color=SLATE, sw=1.3, arrow=False)
    mid = (centers[0] + centers[-1]) / 2
    mx, mw, mh = 620, 46, 76
    line([(cbx, mid), (mx - 2, mid)], color=SLATE)
    rect(mx, mid - mh / 2, mw, mh, fill=NAVY, stroke="none", rx=6)
    text(mx + mw / 2, mid - 6, "Mix", 12.5, "#FFFFFF", "bold", "middle")
    text(mx + mw / 2, mid + 11, "145→96", 10.5, "#FFFFFF", anchor="middle")
    # single vector symbol for the 96-D embedding
    vx, vh = 686, 120
    line([(mx + mw + 1, mid), (vx - 3, mid)], color=SLATE)
    fill = TEAL_L if accent == NAVY else "#E8EDF2"
    rect(vx, mid - vh / 2, 20, vh, fill=fill, stroke=accent, sw=1.6, rx=4)
    for k in range(1, 4):
        hline(vx + 3, vx + 17, mid - vh / 2 + k * vh / 4, color="#FFFFFF", sw=1.5)
    text(vx + 10, mid - vh / 2 - 24, emb_word, 12, accent, "bold", "middle")
    text(vx + 10, mid - vh / 2 - 10, "embedding", 12, accent, "bold", "middle")
    text(vx + 10, mid + vh / 2 + 17, "96-D", 12, accent, "bold", "middle")
    return vx + 20, mid


api_out = tower(62, "API tower", NAVY, "API")
text(530, 455, "same architecture · separate (untied) weights", 10.5, SLATE_L, anchor="middle", style="italic")
line([(350, 464), (712, 464)], color=LINE, sw=1, arrow=False, dash="4 4")
exc_out = tower(478, "Excipient tower", SLATE, "Excipient")

# A → B: each SMILES feeds its own tower
api_arrow_y = IN_Y + 29
line([(300, api_arrow_y), (322, api_arrow_y), (322, 420), (348, 420)], color=NAVY, sw=1.6)
line([(300, exc_y + 29), (348, exc_y + 29)], color=SLATE, sw=1.6)

# ════════════════════════════════════════════════════════════════════════════
# PANEL C — prior-gated bilinear head
# ════════════════════════════════════════════════════════════════════════════
panel(746, PT, 560, PB - PT, "C", "Prior-gated bilinear (PGB) interaction head")

# both embeddings are routed to every interaction (no operation at the junction)
JY, TX = 470, 786
line([api_out, (764, api_out[1]), (764, JY), (TX, JY)], color=NAVY, sw=1.6, arrow=False)
line([exc_out, (764, exc_out[1]), (764, JY)], color=SLATE, sw=1.6, arrow=False)

module_heading(800, 84, "① Pairwise interactions")
OPS_X, OPS_W = 800, 196
cards = [(96, 46), (154, 52), (218, 52), (284, 174)]
gy = 368
mids = [119, 180, 244, gy]
line([(TX, JY), (TX, mids[0])], color=SLATE, sw=1.4, arrow=False)
for m in mids:
    line([(TX, m), (OPS_X - 2, m)], color=SLATE, sw=1.4)

y, h = cards[0]
rect(OPS_X, y, OPS_W, h, fill="#FFFFFF", stroke=LINE)
text(OPS_X + 10, y + 20, "API ‖ Excipient", 13.5, NAVY, "bold")
text(OPS_X + 10, y + 36, "both embeddings", 11, SLATE)
text(OPS_X + OPS_W - 10, y + 29, "192", 13.5, NAVY, "bold", "end")
for (y, h), eq, lab in [(cards[1], "API ⊙ Excipient", "element-wise product"),
                        (cards[2], "|API − Excipient|", "absolute difference")]:
    rect(OPS_X, y, OPS_W, h, fill="#FFFFFF", stroke=LINE)
    text(OPS_X + 10, y + 21, eq, 13.5, NAVY, "bold")
    text(OPS_X + 10, y + 39, lab, 11, SLATE)
    text(OPS_X + OPS_W - 10, y + 31, "96", 13.5, NAVY, "bold", "end")

y, h = cards[3]
rect(OPS_X, y, OPS_W, h, fill="#FFFFFF", stroke=NAVY, sw=1.5)
text(OPS_X + 10, y + 21, "Low-rank bilinear", 13, NAVY, "bold")
sx = OPS_X + 14
for lab, yy, col in [("API", gy - 44, NAVY), ("Exc.", gy + 18, SLATE)]:
    for k in range(6):
        add(f'<rect x="{sx}" y="{yy + k * 4}" width="10" height="4" fill="{TEAL_M if k % 2 else TEAL}" '
            f'stroke="#FFFFFF" stroke-width="0.6"/>')
    text(sx + 5, yy - 5, lab, 9.5, col, "bold", "middle")
    px0 = sx + 20
    add(f'<path d="M {px0} {yy - 2} L {px0 + 34} {yy + 6} L {px0 + 34} {yy + 18} L {px0} {yy + 26} Z" '
        f'fill="#E8ECF2" stroke="{col}" stroke-width="1"/>')
    text(px0 + 15, yy + 16, "U" if lab == "API" else "V", 11, col, "bold", "middle")
    rx_ = px0 + 42
    for k in range(4):
        add(f'<rect x="{rx_}" y="{yy + 4 + k * 4}" width="9" height="4" fill="#B8C2D0" '
            f'stroke="#FFFFFF" stroke-width="0.6"/>')
    line([(px0 + 35, yy + 12), (rx_ - 1, yy + 12)], color=SLATE_L, sw=1, arrow=False)
pnx = sx + 94
line([(sx + 65, gy - 32), (pnx - 6, gy - 5)], color=SLATE, sw=1.2, arrow=False)
line([(sx + 65, gy + 30), (pnx - 6, gy + 5)], color=SLATE, sw=1.2, arrow=False)
circle(pnx, gy, 8, "#FFFFFF", NAVY, 1.3)
text(pnx, gy + 4.5, "⊙", 13, NAVY, "bold", "middle")
bvx = pnx + 20
line([(pnx + 8, gy), (bvx - 2, gy)], color=SLATE, sw=1.2)
for k in range(8):
    add(f'<rect x="{bvx}" y="{gy - 16 + k * 4}" width="9" height="4" '
        f'fill="{NAVY if k % 3 else SLATE_L}" stroke="#FFFFFF" stroke-width="0.6"/>')
text(bvx + 22, gy - 4, "rank", 10.5, SLATE)
text(bvx + 22, gy + 12, "16", 13.5, NAVY, "bold")
text(OPS_X + 10, y + h - 26, "U·API ⊙ V·Excipient", 11.5, NAVY)
text(OPS_X + 10, y + h - 10, "U, V: 96 → 16", 11, SLATE)

# gating
gx = 1026
line([(OPS_X + OPS_W + 1, gy), (gx - 10, gy)], color=NAVY, sw=1.5)
circle(gx, gy, 10, AMBER_L, AMBER, 1.6)
text(gx, gy + 5, "⊙", 15, AMBER, "bold", "middle")

module_heading(800, 488, "② Prior-informed gating")
GB = (966, 500, 120, 58)
rect(*GB, fill=AMBER_L, stroke=AMBER, sw=1.5, rx=7)
text(GB[0] + GB[2] / 2, GB[1] + 23, "Learned gate g", 12.5, AMBER_D, "bold", "middle")
text(GB[0] + GB[2] / 2, GB[1] + 42, "42 → 32 → 16 · σ", 11.5, AMBER_D, anchor="middle")
line([(gx, GB[1] - 1), (gx, gy + 11)], color=AMBER, sw=1.8)
text(gx + 7, 470, "g ∈ (0, 1)¹⁶", 11, AMBER_D)

KX, KY, KW, KH = 800, 588, 262, 230
rect(KX, KY, KW, KH, fill="#FFFFFF", stroke=AMBER, sw=1.2)
text(KX + 12, KY + 22, "Gate context", 13, NAVY, "bold")
text(KX + KW - 12, KY + 22, "42", 14, AMBER, "bold", "end")
rows = [
    ("1", "Excipient prior", AMBER, AMBER_L),
    ("18", "API chemistry flags", TEAL, TEAL_L),
    ("18", "Excipient chemistry flags", TEAL, TEAL_L),
    ("5", "Mechanism flags (SMARTS)", SLATE, "#EDF0F4"),
]
for i, (n_, lab, col, bg) in enumerate(rows):
    ry = KY + 36 + i * 27
    rect(KX + 12, ry, 30, 19, fill=bg, stroke=col, sw=1, rx=3)
    text(KX + 27, ry + 14, n_, 11.5, col, "bold", "middle")
    text(KX + 52, ry + 14, lab, 12, NAVY)
text(KX + 12, KY + 165, "prior = p_exact: the excipient's", 10.5, SLATE)
text(KX + 12, KY + 180, "incompatibility rate in training labels;", 10.5, SLATE)
text(KX + 12, KY + 195, "training rows leave their own", 10.5, SLATE)
text(KX + 12, KY + 210, "API cluster out", 10.5, SLATE)
line([(gx, KY - 1), (gx, GB[1] + GB[3] + 1)], color=AMBER, sw=1.8)

# concatenation
CBX, CBW, CBT, CBB = 1100, 26, 96, 652
bounds = [CBT, 148, 212, 276, 470, CBB]
seg_lab = ["192", "96", "96", "16", "6"]
seg_fill = [TEAL_L, "#EEF1F5", "#EEF1F5", AMBER_L, AMBER_L]
for i in range(5):
    add(f'<rect x="{CBX}" y="{bounds[i]}" width="{CBW}" height="{bounds[i + 1] - bounds[i]}" fill="{seg_fill[i]}"/>')
    if i:
        hline(CBX, CBX + CBW, bounds[i], color=NAVY, sw=0.8)
    text(CBX + CBW / 2, (bounds[i] + bounds[i + 1]) / 2 + 4, seg_lab[i], 11, NAVY, "bold", "middle")
rect(CBX, CBT, CBW, CBB - CBT, fill="none", stroke=NAVY, sw=1.3, rx=3)
text(CBX + CBW / 2, CBT - 7, "concat", 10.5, SLATE, anchor="middle")
text(CBX + CBW / 2, CBB + 17, "406-D", 12.5, NAVY, "bold", "middle")
for m in mids[:3]:
    line([(OPS_X + OPS_W + 1, m), (CBX - 2, m)], color=SLATE, sw=1.4)
line([(gx + 10, gy), (CBX - 2, gy)], color=AMBER, sw=1.6)
text(gx + 18, gy - 8, "gated 16", 11, AMBER_D, "bold")
line([(KX + KW + 1, 612), (CBX - 2, 612)], color=AMBER, sw=1.4)

# prediction module
module_heading(1140, 84, "③ Prediction")
rect(1138, 96, 162, 68, fill="#FFFFFF", stroke=NAVY, sw=1.2, rx=6)
text(1219, 122, "z = r + b", 12.5, NAVY, "bold", "middle")
text(1219, 141, "r: MLP output · b: learned bias", 10.5, SLATE, anchor="middle")
text(1219, 156, "prior acts via gate and concat", 10.5, SLATE, anchor="middle")

text(1200, 290, "MLP", 13, NAVY, "bold", "middle")
text(1200, 307, "406 → 128 → 48 → 1", 11.5, NAVY, anchor="middle")
line([(CBX + CBW + 1, gy), (1142, gy)], color=SLATE, sw=1.5)
layers = [(1144, 128 * 0.62), (1186, 48 * 0.62), (1224, 10)]
for i, (lx, lh) in enumerate(layers):
    rect(lx, gy - lh / 2, 18, lh, fill=NAVY if i < 2 else TEAL, stroke="none", rx=4)
    if i < 2:
        line([(lx + 19, gy), (layers[i + 1][0] - 2, gy)], color=SLATE, sw=1.4)
text(1153, gy + 128 * 0.31 + 16, "128", 11.5, SLATE, "bold", "middle")
text(1195, gy + 48 * 0.31 + 16, "48", 11.5, SLATE, "bold", "middle")
text(1233, gy + 23, "r", 13, TEAL, "bold", "middle", style="italic")

sxn = 1276
line([(1243, gy), (sxn - 11, gy)], color=SLATE, sw=1.5)
circle(sxn, gy, 10, "#FFFFFF", NAVY, 1.6)
text(sxn, gy + 5, "+", 15, NAVY, "bold", "middle")
text(sxn, gy - 17, "z", 14, NAVY, "bold", "middle", style="italic")
text(sxn, gy + 40, "b", 13, SLATE, "bold", "middle", style="italic")
line([(sxn, gy + 28), (sxn, gy + 12)], color=SLATE, sw=1.2)

# ════════════════════════════════════════════════════════════════════════════
# PANEL D — prediction and evaluation
# ════════════════════════════════════════════════════════════════════════════
panel(1322, PT, 254, PB - PT, "D", "Prediction and evaluation")

rect(1336, 62, 226, 88, fill=CORAL_L, stroke=CORAL, sw=1.3, rx=8)
circle(1352, 84, 6, CORAL)
text(1364, 89, "Potential incompatibility", 13, CORAL, "bold")
text(1348, 110, "elevated predicted risk →", 11.5, NAVY)
text(1348, 128, "prioritise for formulation", 11.5, NAVY, "bold")
text(1348, 143, "experiments", 11.5, NAVY, "bold")

THR = 0.671
PX0, PX1, PY0, PY1 = 1368, 1556, 184, 440
t_y = PY1 - THR * (PY1 - PY0)
rect(PX0, PY0, PX1 - PX0, t_y - PY0, fill=CORAL_L, stroke="none", rx=0)
rect(PX0, t_y, PX1 - PX0, PY1 - t_y, fill=TEAL_L, stroke="none", rx=0)
pts = []
for i in range(81):
    z = -6 + 12 * i / 80
    pts.append((PX0 + (PX1 - PX0) * i / 80, PY1 - (1 / (1 + math.exp(-z))) * (PY1 - PY0)))
add('<path d="M ' + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts) +
    f'" fill="none" stroke="{NAVY}" stroke-width="2.2"/>')
line([(PX0, t_y), (PX1, t_y)], color=NAVY, sw=1, arrow=False, dash="4 3")
add(f'<line x1="{PX0}" y1="{PY0}" x2="{PX0}" y2="{PY1}" stroke="{SLATE}" stroke-width="1"/>')
add(f'<line x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}" stroke="{SLATE}" stroke-width="1"/>')
text(PX0 - 5, PY0 + 4, "1", 10.5, SLATE, anchor="end")
text(PX0 - 5, PY1 + 4, "0", 10.5, SLATE, anchor="end")
text(PX0 + 8, PY0 + 18, "P(incompatible) = σ(z)", 11.5, NAVY, "bold")
text(PX1 - 4, t_y - 6, "t = 0.671", 11.5, NAVY, "bold", "end")
text(PX1, PY1 + 15, "z", 12, SLATE, anchor="end", style="italic")
line([(sxn + 11, gy), (PX0 - 2, gy)], color=NAVY, sw=1.6)
line([(1462, PY0 - 2), (1462, 152)], color=CORAL, sw=1.6)
line([(1462, PY1 + 3), (1462, 470)], color=TEAL, sw=1.6)

rect(1336, 472, 226, 54, fill=TEAL_L, stroke=TEAL, sw=1.3, rx=8)
circle(1352, 492, 6, TEAL)
text(1364, 497, "Compatible", 13, TEAL, "bold")
text(1348, 516, "lower predicted risk", 11.5, NAVY)
text(1449, 544, "Predicted risk, not experimental confirmation", 10, SLATE, anchor="middle", style="italic")

# evaluation table
rect(1336, 560, 226, 302, fill="#FFFFFF", stroke=NAVY, sw=1.4, rx=8)
text(1350, 586, "Evaluation", 14.5, NAVY, "bold")
C1, C2 = 1454, 1528
text(C1, 612, "Validation", 11, SLATE, "bold", "middle")
text(C1, 627, "774 pairs", 10.5, SLATE, anchor="middle")
text(C2, 612, "Held-out", 11, NAVY, "bold", "middle")
text(C2, 627, "24 pairs", 10.5, NAVY, anchor="middle")
rows = [("PR-AUC", "0.686", "0.959"), ("ROC-AUC", "0.924", "0.958"),
        ("MCC", "0.639", "0.707"), ("Correct", "734/774", "20/24")]
for i, (lab, v1, v2) in enumerate(rows):
    ry = 666 + i * 38
    hline(1350, 1550, ry - 24)
    text(1350, ry, lab, 12, SLATE, "bold")
    text(C1, ry, v1, 14, SLATE, "bold", "middle")
    text(C2, ry + 1, v2, 19, NAVY, "bold", "middle")
hline(1350, 1550, 666 + 4 * 38 - 24)
text(1350, 816, "Validation drives early stopping and", 10, SLATE)
text(1350, 830, "threshold (TNR ≥ 0.97). 18/22 correct on", 10, SLATE)
text(1350, 844, "held-out pairs absent from training.", 10, SLATE)

add("</svg>")

svg = "\n".join(parts)
with open(os.path.join(OUT_DIR, OUT_NAME + ".svg"), "w", encoding="utf-8") as f:
    f.write(svg)
print("wrote", OUT_NAME + ".svg", len(svg))
