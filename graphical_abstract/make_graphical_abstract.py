"""Builds the ARES (Morgan-PGB) graphical abstract as a self-contained, editable SVG.

Molecules are rendered by RDKit from SMILES of a real training-set pair
(data/train.csv: API_CID 4908 primaquine + Excipient_CID 6134 lactose, Outcome1 = 1).
Architecture dimensions follow src/model.py (MoleculeTower, PGBHead) and
src/dataset.py (morgan family: primary Morgan r=2 1,024 bits, secondary MACCS 167).
Metrics: experiments/results/metrics_pgb/group_split/morgan/test_metrics.json.
"""

import math
import os
import random
import re

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
W, H = 1600, 900

# ── palette (one role per hue) ─────────────────────────────────────────────
NAVY = "#1F2A44"      # structure, model operations
SLATE = "#5B6B7F"     # secondary text, connectors
SLATE_L = "#8A97A8"
LINE = "#CDD4DD"
PANEL = "#F8F9FB"
TEAL = "#2A8C82"      # molecular features, compatible outcome
TEAL_L = "#E2F1EE"
TEAL_M = "#9ACFC7"
CORAL = "#C9614A"     # potential incompatibility
CORAL_L = "#FBE8E2"
AMBER = "#B9822A"     # prior information and gating
AMBER_D = "#7A5413"
AMBER_L = "#FBF0DA"
TRAIN = "#3E4F6A"
VAL = "#8394AB"
TEST = "#C3CCD8"
FONT = "Arial, Helvetica, sans-serif"

parts = []


def add(s):
    parts.append(s)


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=12, fill=NAVY, weight="normal", anchor="start", style="normal",
         raw=False, spacing=None):
    body = s if raw else esc(s)
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    add(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" font-style="{style}" text-anchor="{anchor}"{ls}>{body}</text>')


def sub(base, s, size=None):
    """Inline base_subscript as tspans (dy-based, renderer independent)."""
    small = f' font-size="{size}"' if size else ' font-size="0.72em"'
    return f'{esc(base)}<tspan dy="3"{small}>{esc(s)}</tspan><tspan dy="-3"> </tspan>'


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


def panel(x, y, w, h, letter, label):
    rect(x, y, w, h, fill=PANEL, stroke=LINE, sw=1, rx=10)
    rect(x + 12, y + 11, 20, 20, fill=NAVY, stroke="none", rx=4)
    text(x + 22, y + 26, letter, 13, "#FFFFFF", "bold", "middle")
    text(x + 40, y + 26, label, 14.5, NAVY, "bold")


def module_heading(x, y, label):
    text(x, y, label.upper(), 10, SLATE, "bold", spacing="0.8")


# ── molecule rendering ─────────────────────────────────────────────────────
def mol_svg(smiles, w, h):
    mol = Chem.MolFromSmiles(smiles)
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    o = d.drawOptions()
    o.clearBackground = False
    o.bondLineWidth = 1.6
    o.fixedFontSize = 13
    o.padding = 0.06
    o.updateAtomPalette({6: (0.12, 0.16, 0.27), 7: (0.20, 0.36, 0.62),
                         8: (0.74, 0.30, 0.25), 1: (0.12, 0.16, 0.27)})
    d.DrawMolecule(mol)
    d.FinishDrawing()
    s = d.GetDrawingText()
    s = re.sub(r"<\?xml[^>]*\?>", "", s)
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    return re.search(r"<svg[^>]*>(.*)</svg>", s, flags=re.S).group(1)


def place_mol(smiles, x, y, w, h):
    add(f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewBox="0 0 {w} {h}" overflow="visible">')
    add(mol_svg(smiles, w, h))
    add("</svg>")


PRIMAQUINE = "CC(CCCN)NC1=C2C(=CC(=C1)OC)C=CC=N2"          # API_CID 4908
LACTOSE = "C(C1C(C(C(C(O1)OC2C(OC(C(C2O)O)O)CO)O)O)O)O"   # Excipient_CID 6134

# ════════════════════════════════════════════════════════════════════════════
# canvas + defs
# ════════════════════════════════════════════════════════════════════════════
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add("<defs>")
for c in (SLATE, NAVY, TEAL, AMBER, CORAL, SLATE_L):
    add(f'<marker id="ah-{c[1:]}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0 0.8 L9 5 L0 9.2 z" fill="{c}"/></marker>')
add("</defs>")
add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

PT, PB = 24, 742          # top / bottom of the four main panels

# ════════════════════════════════════════════════════════════════════════════
# PANEL A — data
# ════════════════════════════════════════════════════════════════════════════
panel(24, PT, 270, PB - PT, "A", "Data: API–excipient pairs")

rect(38, 62, 242, 146, fill="#FFFFFF", stroke=LINE)
rect(46, 70, 40, 18, fill=NAVY, stroke="none", rx=9)
text(66, 83, "API", 11, "#FFFFFF", "bold", "middle")
text(272, 83, "primaquine", 11, SLATE, anchor="end", style="italic")
place_mol(PRIMAQUINE, 48, 90, 222, 112)

circle(159, 232, 12, "#FFFFFF", NAVY, 1.4)
text(159, 237, "+", 16, NAVY, "bold", "middle")
rect(50, 256, 104, 24, fill=TEAL_L, stroke=TEAL, rx=12)
text(102, 272, "compatible (0)", 11.5, TEAL, "bold", "middle")
rect(160, 256, 114, 24, fill=CORAL_L, stroke=CORAL, rx=12)
text(217, 272, "incompatible (1)", 11.5, CORAL, "bold", "middle")

rect(38, 296, 242, 146, fill="#FFFFFF", stroke=LINE)
rect(46, 304, 74, 18, fill=SLATE, stroke="none", rx=9)
text(83, 317, "Excipient", 11, "#FFFFFF", "bold", "middle")
text(272, 317, "lactose", 11, SLATE, anchor="end", style="italic")
place_mol(LACTOSE, 48, 324, 222, 112)
text(159, 458, "training-set example · label 1", 10.5, SLATE_L, anchor="middle")

# class balance
text(40, 512, "3,544", 22, NAVY, "bold")
text(106, 512, "labelled pairs", 12.5, NAVY)
bx, bw = 40, 238
cw = round(bw * 3200 / 3544, 1)
rect(bx, 524, cw, 14, fill=TEAL, stroke="none", rx=2)
rect(bx + cw, 524, bw - cw, 14, fill=CORAL, stroke="none", rx=2)
text(40, 557, "3,200 compatible (90.3%)", 11.5, TEAL, "bold")
text(40, 573, "344 incompatible (9.7%)", 11.5, CORAL, "bold")

# split sizes (procedure is described once, in panel E)
text(40, 614, "API-cluster split", 12.5, NAVY, "bold")
segs = [("Train", 2030, TRAIN), ("Val", 842, VAL), ("Test", 672, TEST)]
x0, gap = bx, 4
avail = bw - 2 * gap
seg_pos = []
for name, n, col in segs:
    w_ = avail * n / 3544
    rect(round(x0, 1), 624, round(w_, 1), 26, fill=col, stroke="none", rx=3)
    seg_pos.append((x0, w_, name, n, col))
    x0 += w_ + gap
random.seed(7)
for (sx, sw_, name, n, col) in seg_pos:
    k = max(2, int(sw_ // 20))
    for i in range(k):
        cx = sx + (i + 0.5) * sw_ / k
        for _ in range(3):
            circle(round(cx + random.uniform(-4, 4), 1), round(637 + random.uniform(-5, 5), 1), 1.9,
                   "#FFFFFF" if col != TEST else NAVY)
for (sx, sw_, name, n, col) in seg_pos:
    text(round(sx + sw_ / 2, 1), 668, name, 10.5, SLATE, "bold", "middle")
    text(round(sx + sw_ / 2, 1), 684, f"{n:,}", 12, NAVY, "bold", "middle")

# ════════════════════════════════════════════════════════════════════════════
# PANEL B — two Morgan-based towers
# ════════════════════════════════════════════════════════════════════════════
panel(310, PT, 410, PB - PT, "B", "Morgan-based molecular towers")

FEATS = [  # title, sub, projected dim, kind
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


def tower(top, who, accent, h_label):
    text(326, top + 14, "API tower" if who == "API" else "Excipient tower", 12.5, accent, "bold")
    chip_x, chip_w, chip_h, step = 334, 178, 40, 46
    y0 = top + 26
    centers = []
    for i, (t, s, od, kind) in enumerate(FEATS):
        cy = y0 + i * step
        primary = kind == "morgan"
        rect(chip_x, cy, chip_w, chip_h, fill="#FFFFFF",
             stroke=TEAL if primary else LINE, sw=1.6 if primary else 1, rx=5)
        text(chip_x + 8, cy + 16, t, 12, TEAL if primary else NAVY, "bold")
        text(chip_x + 8, cy + 31, s, 10.5, SLATE)
        glyph(kind, chip_x + chip_w - 52, cy + 8 if kind != "desc" else cy + 6)
        centers.append(cy + chip_h / 2)
    bx_ = 324
    line([(bx_, y0 + 6), (bx_, y0 + 5 * step + chip_h - 6)], color=accent, sw=1.4, arrow=False)
    for c in centers:
        line([(bx_, c), (chip_x - 2, c)], color=accent, sw=1.1, arrow=False)
    px, pw = 526, 34
    for c, (t, s, od, kind) in zip(centers, FEATS):
        direct = kind in ("salt", "avail")
        line([(chip_x + chip_w + 1, c), (px - 2, c)], color=SLATE_L, sw=1.2)
        rect(px, c - 11, pw, 22, fill=TEAL_L if not direct else "#FFFFFF",
             stroke=TEAL if not direct else SLATE_L, sw=1, rx=4, dash=None if not direct else "3 2")
        text(px + pw / 2, c + 4.5, od, 12, TEAL if not direct else SLATE, "bold", "middle")
    cbx = 572
    line([(px + pw + 1, centers[0]), (cbx, centers[0]), (cbx, centers[-1]), (px + pw + 1, centers[-1])],
         color=SLATE, sw=1.3, arrow=False)
    for c in centers[1:-1]:
        line([(px + pw + 1, c), (cbx, c)], color=SLATE, sw=1.3, arrow=False)
    mid = (centers[0] + centers[-1]) / 2
    mx, mw, mh = 596, 48, 70
    line([(cbx, mid), (mx - 2, mid)], color=SLATE)
    rect(mx, mid - mh / 2, mw, mh, fill=NAVY, stroke="none", rx=6)
    text(mx + mw / 2, mid - 6, "Mix", 12, "#FFFFFF", "bold", "middle")
    text(mx + mw / 2, mid + 10, "145→96", 10.5, "#FFFFFF", anchor="middle")
    vx, cell, n = 662, 12, 12
    vy0 = mid - n * cell / 2
    line([(mx + mw + 1, mid), (vx - 3, mid)], color=SLATE)
    shades = ["#2A8C82", "#5FAFA5", "#9ACFC7", "#CFE7E3"]
    for i in range(n):
        fill = shades[(i * 7 + (3 if who == "EXC" else 0)) % 4]
        add(f'<rect x="{vx}" y="{vy0 + i * cell}" width="16" height="{cell}" fill="{fill}" '
            f'stroke="#FFFFFF" stroke-width="1"/>')
    rect(vx, vy0, 16, n * cell, fill="none", stroke=accent, sw=1.2, rx=1)
    text(vx + 8, vy0 - 8, h_label, 13, accent, "bold", "middle", raw=True)
    text(vx + 8, vy0 + n * cell + 16, "96-D", 11.5, accent, "bold", "middle")
    return vx + 16, mid


api_out = tower(64, "API", NAVY, sub("h", "API"))
text(517, 379, "identical architecture · untied weights", 10.5, SLATE_L, anchor="middle", style="italic")
line([(330, 388), (704, 388)], color=LINE, sw=1, arrow=False, dash="4 4")
exc_out = tower(398, "EXC", SLATE, sub("h", "EXC"))

line([(280, 135), (322, 135)], color=NAVY, sw=1.6)     # API molecule → API tower
line([(280, 440), (322, 440)], color=SLATE, sw=1.6)    # excipient → excipient tower

# ════════════════════════════════════════════════════════════════════════════
# PANEL C — ARES prior-gated bilinear head, three modules
# ════════════════════════════════════════════════════════════════════════════
panel(736, PT, 560, PB - PT, "C", "ARES interaction head (prior-gated bilinear)")

# h_API, h_EXC converge into the pair node, then fan out
nx, ny = 756, 400
line([api_out, (nx, api_out[1]), (nx, ny - 9)], color=NAVY, sw=1.6)
line([exc_out, (nx, exc_out[1]), (nx, ny + 9)], color=SLATE, sw=1.6)
circle(nx, ny, 7, "#FFFFFF", NAVY, 1.6)
circle(nx, ny, 2.5, NAVY)

# ── module 1: pairwise interactions ──
module_heading(794, 74, "① Pairwise interactions")
OPS_X, OPS_W = 794, 186
cards = [(84, 40), (134, 44), (188, 44), (244, 140)]
gy = 314                                   # bilinear output / gated path / MLP axis
mids = [104, 156, 210, gy]
tx = 778
line([(nx + 7, ny), (tx, ny), (tx, mids[0])], color=SLATE, sw=1.4, arrow=False)
for m in mids:
    line([(tx, m), (OPS_X - 2, m)], color=SLATE, sw=1.4)

y, h = cards[0]
rect(OPS_X, y, OPS_W, h, fill="#FFFFFF", stroke=LINE)
text(OPS_X + 10, y + 25, sub("h", "API") + " ‖ " + sub("h", "EXC"), 13, NAVY, "bold", raw=True)
text(OPS_X + OPS_W - 10, y + 25, "192", 13, NAVY, "bold", "end")

for (y, h), eq, lab in [(cards[1], sub("h", "API") + " ⊙ " + sub("h", "EXC"), "element-wise product"),
                        (cards[2], "|" + sub("h", "API") + "− " + sub("h", "EXC") + "|", "absolute difference")]:
    rect(OPS_X, y, OPS_W, h, fill="#FFFFFF", stroke=LINE)
    text(OPS_X + 10, y + 19, eq, 13, NAVY, "bold", raw=True)
    text(OPS_X + 10, y + 35, lab, 10.5, SLATE)
    text(OPS_X + OPS_W - 10, y + 27, "96", 13, NAVY, "bold", "end")

# low-rank bilinear card (emphasised)
y, h = cards[3]
rect(OPS_X, y, OPS_W, h, fill="#FFFFFF", stroke=NAVY, sw=1.5)
text(OPS_X + 10, y + 19, "Low-rank bilinear", 12.5, NAVY, "bold")
sx = OPS_X + 12
for lab, yy, col in [("API", y + 36, NAVY), ("EXC", y + 78, SLATE)]:
    for k in range(6):
        add(f'<rect x="{sx}" y="{yy + k * 4}" width="10" height="4" fill="{TEAL_M if k % 2 else TEAL}" '
            f'stroke="#FFFFFF" stroke-width="0.6"/>')
    px0 = sx + 18
    add(f'<path d="M {px0} {yy - 2} L {px0 + 34} {yy + 6} L {px0 + 34} {yy + 18} L {px0} {yy + 26} Z" '
        f'fill="#E8ECF2" stroke="{col}" stroke-width="1"/>')
    text(px0 + 15, yy + 16, "U" if lab == "API" else "V", 11, col, "bold", "middle")
    rx_ = px0 + 42
    for k in range(4):
        add(f'<rect x="{rx_}" y="{yy + 4 + k * 4}" width="9" height="4" fill="#B8C2D0" '
            f'stroke="#FFFFFF" stroke-width="0.6"/>')
    line([(px0 + 35, yy + 12), (rx_ - 1, yy + 12)], color=SLATE_L, sw=1, arrow=False)
pnx, pny = sx + 92, gy
line([(sx + 63, y + 48), (pnx - 6, pny - 5)], color=SLATE, sw=1.2, arrow=False)
line([(sx + 63, y + 90), (pnx - 6, pny + 5)], color=SLATE, sw=1.2, arrow=False)
circle(pnx, pny, 8, "#FFFFFF", NAVY, 1.3)
text(pnx, pny + 4.5, "⊙", 13, NAVY, "bold", "middle")
bvx = pnx + 20
line([(pnx + 8, pny), (bvx - 2, pny)], color=SLATE, sw=1.2)
for k in range(8):
    add(f'<rect x="{bvx}" y="{pny - 16 + k * 4}" width="9" height="4" '
        f'fill="{NAVY if k % 3 else SLATE_L}" stroke="#FFFFFF" stroke-width="0.6"/>')
text(bvx + 22, pny - 4, "rank", 10.5, SLATE)
text(bvx + 22, pny + 11, "16", 13, NAVY, "bold")
text(OPS_X + 10, y + h - 12, "U " + sub("h", "API") + "⊙ V " + sub("h", "EXC") + "· U, V: 96 → 16",
     10.5, SLATE, raw=True)

# ── module 2: prior-informed gating ──
gx = 1010
line([(OPS_X + OPS_W + 1, gy), (gx - 10, gy)], color=NAVY, sw=1.5)
circle(gx, gy, 10, AMBER_L, AMBER, 1.6)
text(gx, gy + 5, "⊙", 15, AMBER, "bold", "middle")

module_heading(794, 414, "② Prior-informed gating")
GB = (950, 426, 120, 56)
rect(*GB, fill=AMBER_L, stroke=AMBER, sw=1.5, rx=7)
text(GB[0] + GB[2] / 2, GB[1] + 22, "Learned gate g", 12.5, AMBER_D, "bold", "middle")
text(GB[0] + GB[2] / 2, GB[1] + 41, "46 → 32 → 16 · σ", 11.5, AMBER_D, anchor="middle")
line([(gx, GB[1] - 1), (gx, gy + 11)], color=AMBER, sw=1.8)
text(gx + 7, 398, "g ∈ (0, 1)¹⁶", 10.5, AMBER_D)

KX, KY, KW, KH = 794, 508, 250, 178
rect(KX, KY, KW, KH, fill="#FFFFFF", stroke=AMBER, sw=1.2)
text(KX + 10, KY + 20, "Gate context", 12.5, NAVY, "bold")
text(KX + KW - 10, KY + 20, "46", 13, AMBER, "bold", "end")
rows = [
    ("5", "Excipient prior", AMBER, AMBER_L),
    ("18", "API chemistry flags", TEAL, TEAL_L),
    ("18", "Excipient chemistry flags", TEAL, TEAL_L),
    ("5", "Mechanism flags (SMARTS)", SLATE, "#EDF0F4"),
]
for i, (n_, lab, col, bg) in enumerate(rows):
    ry = KY + 32 + i * 24
    rect(KX + 10, ry, 28, 18, fill=bg, stroke=col, sw=1, rx=3)
    text(KX + 24, ry + 13, n_, 11, col, "bold", "middle")
    text(KX + 46, ry + 13, lab, 11.5, NAVY)
text(KX + 10, KY + 148, "prior = [p_exact, Δ global, p_family,", 10, SLATE)
text(KX + 10, KY + 162, "log(1+n), unseen]", 10, SLATE)
line([(gx, KY - 1), (gx, GB[1] + GB[3] + 1)], color=AMBER, sw=1.8)

# ── module 3: prediction head ──
CBX, CBW, CBT, CBB = 1090, 26, 84, 636
bounds = [CBT, 130, 183, 240, 420, CBB]
seg_lab = ["192", "96", "96", "16", "10"]
seg_fill = [TEAL_L, "#EEF1F5", "#EEF1F5", AMBER_L, AMBER_L]
for i in range(5):
    add(f'<rect x="{CBX}" y="{bounds[i]}" width="{CBW}" height="{bounds[i + 1] - bounds[i]}" fill="{seg_fill[i]}"/>')
    if i:
        add(f'<line x1="{CBX}" y1="{bounds[i]}" x2="{CBX + CBW}" y2="{bounds[i]}" stroke="{NAVY}" stroke-width="0.8"/>')
    text(CBX + CBW / 2, (bounds[i] + bounds[i + 1]) / 2 + 4, seg_lab[i], 10.5, NAVY, "bold", "middle")
rect(CBX, CBT, CBW, CBB - CBT, fill="none", stroke=NAVY, sw=1.3, rx=3)
text(CBX + CBW / 2, CBT - 7, "concat", 10, SLATE, anchor="middle")
text(CBX + CBW / 2, CBB + 16, "410-D", 12, NAVY, "bold", "middle")
for m in mids[:3]:
    line([(OPS_X + OPS_W + 1, m), (CBX - 2, m)], color=SLATE, sw=1.4)
line([(gx + 10, gy), (CBX - 2, gy)], color=AMBER, sw=1.6)
text(gx + 18, gy - 7, "gated 16", 10.5, AMBER_D, "bold")
line([(KX + KW + 1, 560), (CBX - 2, 560)], color=AMBER, sw=1.4)

module_heading(1132, 74, "③ Prediction")
# logit equation — the single place where z is defined
rect(1130, 92, 158, 62, fill="#FFFFFF", stroke=NAVY, sw=1.2, rx=6)
text(1209, 116, "z = r + w·logit(p_exact) + b", 11.5, NAVY, "bold", "middle")
text(1209, 134, "r: MLP residual · b: bias", 10, SLATE, anchor="middle")
text(1209, 147, "w: adaptive prior trust", 10, SLATE, anchor="middle")

text(1190, 238, "MLP", 12.5, NAVY, "bold", "middle")
text(1190, 254, "410 → 128 → 48 → 1", 11.5, NAVY, anchor="middle")
line([(CBX + CBW + 1, gy), (1132, gy)], color=SLATE, sw=1.5)
layers = [(1134, 128 * 0.62), (1176, 48 * 0.62), (1214, 10)]
for i, (lx, lh) in enumerate(layers):
    rect(lx, gy - lh / 2, 18, lh, fill=NAVY if i < 2 else TEAL, stroke="none", rx=4)
    if i < 2:
        line([(lx + 19, gy), (layers[i + 1][0] - 2, gy)], color=SLATE, sw=1.4)
text(1143, gy + 128 * 0.31 + 15, "128", 11, SLATE, "bold", "middle")
text(1185, gy + 48 * 0.31 + 15, "48", 11, SLATE, "bold", "middle")
text(1223, gy + 22, "r", 12.5, TEAL, "bold", "middle", style="italic")

sxn = 1264
line([(1233, gy), (sxn - 11, gy)], color=SLATE, sw=1.5)
circle(sxn, gy, 10, "#FFFFFF", NAVY, 1.6)
text(sxn, gy + 5, "+", 15, NAVY, "bold", "middle")
text(sxn, gy - 17, "z", 13, NAVY, "bold", "middle", style="italic")

TB = (1134, 604, 152, 76)
rect(*TB, fill=AMBER_L, stroke=AMBER, sw=1.3, rx=7)
text(TB[0] + 10, TB[1] + 21, "Adaptive prior trust", 12, AMBER_D, "bold")
text(TB[0] + 10, TB[1] + 40, "w = σ(MLP(prior))", 11, AMBER_D)
text(TB[0] + 10, TB[1] + 58, "× logit(p_exact)", 11, AMBER_D)
line([(KX + KW + 1, 666), (TB[0] - 2, 666)], color=AMBER, sw=1.4)
line([(sxn, TB[1] - 1), (sxn, gy + 11)], color=AMBER, sw=1.6)

# ════════════════════════════════════════════════════════════════════════════
# PANEL D — prediction, use and evidence
# ════════════════════════════════════════════════════════════════════════════
panel(1312, PT, 264, PB - PT, "D", "Prediction & test evidence")

rect(1326, 62, 236, 80, fill=CORAL_L, stroke=CORAL, sw=1.3, rx=8)
circle(1343, 83, 6, CORAL)
text(1355, 88, "Potential incompatibility", 12.5, CORAL, "bold")
text(1338, 108, "elevated predicted risk →", 11, NAVY)
text(1338, 125, "prioritise for formulation experiments", 11, NAVY, "bold")

PX0, PX1, PY0, PY1 = 1356, 1548, 170, 410
t_y = PY1 - 0.727 * (PY1 - PY0)
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
text(PX1 - 4, t_y - 6, "t = 0.727", 11, NAVY, "bold", "end")
text(PX1, PY1 + 15, "z", 11.5, SLATE, anchor="end", style="italic")
line([(sxn + 11, gy), (PX0 - 2, gy)], color=NAVY, sw=1.6)
line([(1454, PY0 - 2), (1454, 144)], color=CORAL, sw=1.6)
line([(1454, PY1 + 3), (1454, 432)], color=TEAL, sw=1.6)

rect(1326, 434, 236, 50, fill=TEAL_L, stroke=TEAL, sw=1.3, rx=8)
circle(1343, 453, 6, TEAL)
text(1355, 458, "Compatible", 12.5, TEAL, "bold")
text(1338, 476, "lower predicted risk", 11, NAVY)
text(1444, 503, "Predicted risk, not experimental confirmation", 10, SLATE, anchor="middle", style="italic")

# metrics — principal quantitative evidence
rect(1326, 522, 236, 186, fill="#FFFFFF", stroke=NAVY, sw=1.4, rx=8)
text(1340, 546, "Held-out test set", 13.5, NAVY, "bold")
text(1340, 564, "672 pairs · unseen API clusters", 11, SLATE)
for i, (lab, val) in enumerate([("PR-AUC", "0.725"), ("ROC-AUC", "0.912"), ("MCC", "0.584")]):
    ry = 604 + i * 42
    add(f'<line x1="1340" y1="{ry - 26}" x2="1548" y2="{ry - 26}" stroke="{LINE}" stroke-width="1"/>')
    text(1340, ry, lab, 13, SLATE, "bold")
    text(1548, ry + 2, val, 25, NAVY, "bold", "end")

# ════════════════════════════════════════════════════════════════════════════
# STRIP E — leakage-aware evaluation (procedure only; counts live in panel A)
# ════════════════════════════════════════════════════════════════════════════
SY = 758
rect(24, SY, 1552, 118, fill="#FFFFFF", stroke=LINE, sw=1, rx=10)
rect(40, SY + 36, 20, 20, fill=SLATE, stroke="none", rx=4)
text(50, SY + 51, "E", 13, "#FFFFFF", "bold", "middle")
text(70, SY + 46, "Leakage-aware", 14.5, NAVY, "bold")
text(70, SY + 64, "evaluation", 14.5, NAVY, "bold")

steps = [
    ("API-cluster group split", ["Butina clusters (Morgan 2,048-bit; split only)",
                                 "StratifiedGroupKFold 3 : 1 : 1 · no shared cluster"], NAVY, "#EEF1F5", LINE),
    ("Training", ["Excipient priors from training labels only;",
                  "training rows: leave-own-API-cluster-out"], TRAIN, AMBER_L, AMBER),
    ("Validation", ["Early stopping on PR-AUC",
                    "Threshold t = 0.727 (TNR ≥ 0.97)"], VAL, "#EEF1F5", LINE),
    ("Held-out test", ["Unseen API clusters",
                       "Single final evaluation"], NAVY, "#EEF1F5", LINE),
]
sx0, sx1, sgap = 214, 1562, 28
sw_ = (sx1 - sx0 - sgap * (len(steps) - 1)) / len(steps)
for i, (t, lines_, col, bg, st) in enumerate(steps):
    x = round(sx0 + i * (sw_ + sgap), 1)
    rect(x, SY + 14, round(sw_, 1), 90, fill=bg, stroke=st, sw=1.1, rx=7)
    rect(x, SY + 14, 5, 90, fill=col, stroke="none", rx=2)
    text(round(x + 16, 1), SY + 38, t, 13, NAVY, "bold")
    for j, l_ in enumerate(lines_):
        text(round(x + 16, 1), SY + 60 + j * 17, l_, 11, SLATE)
    if i < len(steps) - 1:
        ax_ = x + sw_
        line([(round(ax_ + 4, 1), SY + 59), (round(ax_ + sgap - 4, 1), SY + 59)], color=SLATE, sw=1.6)

add("</svg>")

svg = "\n".join(parts)
with open(os.path.join(OUT_DIR, "morgan_pgb_graphical_abstract.svg"), "w", encoding="utf-8") as f:
    f.write(svg)
print("wrote SVG", len(svg))
