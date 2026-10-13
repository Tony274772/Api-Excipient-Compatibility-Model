"""Builds the Morgan-PGB graphical abstract as a self-contained, editable SVG.

Molecules are rendered by RDKit from SMILES of a real training-set pair
(data/train.csv: API_CID 4908 primaquine + Excipient_CID 6134 lactose, Outcome1 = 1).
Architecture dimensions follow src/model.py (MoleculeTower, PGBHead) and
src/dataset.py (morgan family: primary Morgan r=2 1,024 bits, secondary MACCS 167).
"""

import os
import re

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
W, H = 1600, 900

# ── palette ────────────────────────────────────────────────────────────────
NAVY = "#1F2A44"
SLATE = "#5B6B7F"
SLATE_L = "#8A97A8"
LINE = "#C9D1DB"
PANEL = "#F8F9FB"
TEAL = "#2A8C82"
TEAL_L = "#E2F1EE"
TEAL_M = "#9ACFC7"
CORAL = "#D2694F"
CORAL_L = "#FBE8E2"
AMBER = "#C08A2B"
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


def text(x, y, s, size=12, fill=NAVY, weight="normal", anchor="start", style="normal", raw=False):
    body = s if raw else esc(s)
    add(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" font-style="{style}" text-anchor="{anchor}">{body}</text>')


def sub(base, s, size=None):
    """Inline base_subscript as tspans (dy-based, renderer independent)."""
    small = f' font-size="{size}"' if size else ' font-size="0.72em"'
    return f'{esc(base)}<tspan dy="3"{small}>{esc(s)}</tspan><tspan dy="-3"> </tspan>'


def rect(x, y, w, h, fill="none", stroke=LINE, sw=1, rx=6, dash=None, extra=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d} {extra}/>')


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
    text(x + 40, y + 26, label, 14, NAVY, "bold")


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
    inner = re.search(r"<svg[^>]*>(.*)</svg>", s, flags=re.S).group(1)
    return inner


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

# ════════════════════════════════════════════════════════════════════════════
# PANEL A — data
# ════════════════════════════════════════════════════════════════════════════
AX, AY, AW, AH = 24, 24, 270, 696
panel(AX, AY, AW, AH, "A", "API–excipient pairs")

# API molecule card
rect(38, 62, 242, 150, fill="#FFFFFF", stroke=LINE)
rect(46, 70, 40, 18, fill=NAVY, stroke="none", rx=9)
text(66, 83, "API", 11, "#FFFFFF", "bold", "middle")
text(272, 83, "primaquine", 11, SLATE, anchor="end", style="italic")
place_mol(PRIMAQUINE, 48, 90, 222, 116)

# pairing row
circle(159, 241, 13, "#FFFFFF", NAVY, 1.4)
text(159, 246, "+", 17, NAVY, "bold", "middle")
text(159, 276, "Classify each pair as", 11.5, SLATE, anchor="middle")
rect(52, 286, 100, 24, fill=TEAL_L, stroke=TEAL, rx=12)
text(102, 302, "compatible", 11.5, TEAL, "bold", "middle")
rect(160, 286, 112, 24, fill=CORAL_L, stroke=CORAL, rx=12)
text(216, 302, "incompatible", 11.5, CORAL, "bold", "middle")
text(159, 326, "Outcome1: 0 = compatible, 1 = incompatible", 10.5, SLATE_L, anchor="middle")

# Excipient molecule card
rect(38, 340, 242, 150, fill="#FFFFFF", stroke=LINE)
rect(46, 348, 74, 18, fill=SLATE, stroke="none", rx=9)
text(83, 361, "Excipient", 11, "#FFFFFF", "bold", "middle")
text(272, 361, "lactose", 11, SLATE, anchor="end", style="italic")
place_mol(LACTOSE, 48, 368, 222, 116)
text(159, 506, "Example pair from the training set (label 1)", 10.5, SLATE_L, anchor="middle")

# dataset statistics
text(40, 538, "3,544", 20, NAVY, "bold")
text(100, 538, "curated pairs", 12.5, NAVY)
bx, bw = 40, 238
cw = round(bw * 3200 / 3544, 1)
rect(bx, 548, cw, 14, fill=TEAL, stroke="none", rx=2)
rect(bx + cw, 548, bw - cw, 14, fill=CORAL, stroke="none", rx=2)
text(40, 578, "Compatible 3,200 (90.3%)", 11, TEAL, "bold")
text(278, 578, "344 (9.7%)", 11, CORAL, "bold", "end")
text(278, 591, "incompatible", 10.5, CORAL, anchor="end")

# split bar with disjoint API clusters
text(40, 618, "API-cluster group split", 12, NAVY, "bold")
segs = [("Train", 2030, TRAIN), ("Val", 842, VAL), ("Test", 672, TEST)]
x0 = bx
gap = 4
avail = bw - 2 * gap
seg_pos = []
for name, n, col in segs:
    w_ = avail * n / 3544
    rect(round(x0, 1), 628, round(w_, 1), 26, fill=col, stroke="none", rx=3)
    seg_pos.append((x0, w_, name, n, col))
    x0 += w_ + gap
# cluster glyphs: each split owns different clusters (dot groups)
import random
random.seed(7)
for (sx, sw_, name, n, col) in seg_pos:
    k = max(2, int(sw_ // 20))
    for i in range(k):
        cx = sx + (i + 0.5) * sw_ / k
        for j in range(3):
            dx = random.uniform(-4, 4)
            dy = random.uniform(-5, 5)
            circle(round(cx + dx, 1), round(641 + dy, 1), 1.9,
                   "#FFFFFF" if col != TEST else NAVY)
for (sx, sw_, name, n, col) in seg_pos:
    text(round(sx + sw_ / 2, 1), 670, name, 10.5, SLATE, "bold", "middle")
    text(round(sx + sw_ / 2, 1), 683, f"{n:,}", 11, NAVY, "bold", "middle")
text(40, 703, "Clusters of similar APIs never cross splits", 10.5, SLATE)

# ════════════════════════════════════════════════════════════════════════════
# PANEL B — two Morgan-based towers
# ════════════════════════════════════════════════════════════════════════════
BX, BY, BW, BH = 310, 24, 410, 696
panel(BX, BY, BW, BH, "B", "Morgan-based molecular towers")

FEATS = [
    # title, sub, out-dim, kind
    ("Morgan FP", "r = 2 · 1,024 bits", "64", "morgan"),
    ("MACCS keys", "167 bits", "32", "maccs"),
    ("Descriptors (12)", "MW, logP, TPSA, QED…", "24", "desc"),
    ("Chem. flags (18)", "acid, base, metal, ester…", "16", "flags"),
    ("Salt context (8)", "fragments, charges", "8", "salt"),
    ("Availability (1)", "primary FP present", "1", "avail"),
]


def glyph(kind, x, y):
    """Small icon on the right side of a feature chip (x,y = top-left, 46x14)."""
    if kind in ("morgan", "maccs"):
        on = {"morgan": {2, 9, 13, 21, 26}, "maccs": {1, 4, 8, 11, 15, 19, 22, 25}}[kind]
        col = TEAL if kind == "morgan" else SLATE
        for i in range(28):
            fill = col if i in on else "#FFFFFF"
            add(f'<rect x="{x + i * 1.75:.2f}" y="{y}" width="1.75" height="12" fill="{fill}" '
                f'stroke="{LINE}" stroke-width="0.3"/>')
    elif kind == "desc":
        hs = [6, 10, 4, 8, 12, 5, 9, 7, 11, 3, 8, 6]
        for i, hh in enumerate(hs):
            add(f'<rect x="{x + i * 4}" y="{y + 12 - hh}" width="3" height="{hh}" fill="{SLATE_L}"/>')
    elif kind == "flags":
        on = {1, 4, 9, 13, 16}
        for i in range(18):
            r_, c_ = divmod(i, 9)
            circle(x + 2.5 + c_ * 5.3, y + 3 + r_ * 6, 2.0,
                   TEAL if i in on else "#FFFFFF", SLATE_L, 0.6)
    elif kind == "salt":
        circle(x + 10, y + 6, 5.5, "#FFFFFF", SLATE, 1)
        text(x + 10, y + 9.5, "+", 9, SLATE, "bold", "middle")
        circle(x + 26, y + 6, 4, "#FFFFFF", SLATE, 1)
        text(x + 26, y + 9.3, "−", 9, SLATE, "bold", "middle")
    elif kind == "avail":
        add(f'<rect x="{x + 4}" y="{y}" width="12" height="12" fill="{TEAL}" rx="1.5"/>')
        text(x + 10, y + 10, "1", 9, "#FFFFFF", "bold", "middle")


def tower(top, who, accent, h_label, mol_y):
    """Draws one tower lane; returns (x_out, y_center) of its 96-D output."""
    lane_h = 316
    title = "API tower" if who == "API" else "Excipient tower"
    text(326, top + 14, title, 12.5, accent, "bold")
    text(704, top + 14, "θ" if False else "", 11, SLATE)
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
    # input bracket from panel A
    bx_ = 324
    line([(bx_, y0 + 6), (bx_, y0 + 5 * step + chip_h - 6)], color=accent, sw=1.4, arrow=False)
    for c in centers:
        line([(bx_, c), (chip_x - 2, c)], color=accent, sw=1.1, arrow=False)
    # projection blocks
    px, pw = 524, 34
    for c, (t, s, od, kind) in zip(centers, FEATS):
        direct = kind in ("salt", "avail")
        line([(chip_x + chip_w + 1, c), (px - 2, c)], color=SLATE_L, sw=1.2)
        rect(px, c - 11, pw, 22,
             fill=(TEAL_L if not direct else "#FFFFFF"),
             stroke=(TEAL if not direct else SLATE_L), sw=1, rx=4,
             dash=None if not direct else "3 2")
        text(px + pw / 2, c + 4.5, od, 12, TEAL if not direct else SLATE, "bold", "middle")
    # concat bracket
    cbx = 570
    line([(px + pw + 1, centers[0]), (cbx, centers[0]), (cbx, centers[-1]), (px + pw + 1, centers[-1])],
         color=SLATE, sw=1.3, arrow=False)
    for c in centers[1:-1]:
        line([(px + pw + 1, c), (cbx, c)], color=SLATE, sw=1.3, arrow=False)
    mid = (centers[0] + centers[-1]) / 2
    # mix block
    mx, mw, mh = 596, 46, 70
    line([(cbx, mid), (mx - 2, mid)], color=SLATE)
    rect(mx, mid - mh / 2, mw, mh, fill=NAVY, stroke="none", rx=6)
    text(mx + mw / 2, mid - 6, "Mix", 12, "#FFFFFF", "bold", "middle")
    text(mx + mw / 2, mid + 10, "145→96", 10.5, "#FFFFFF", anchor="middle")
    # 96-D vector
    vx, cell, n = 660, 12, 12
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


api_out = tower(64, "API", NAVY, sub("h", "API"), 137)
# separator between the two lanes
line([(330, 382), (704, 382)], color=LINE, sw=1, arrow=False, dash="4 4")
text(517, 377, "identical architecture · separate (untied) weights", 10.5, SLATE_L, anchor="middle", style="italic")
exc_out = tower(394, "EXC", SLATE, sub("h", "EXC"), 415)

# arrows A → B (molecule → its own tower)
line([(280, 137), (322, 137)], color=NAVY, sw=1.6)
line([(280, 446), (322, 446)], color=SLATE, sw=1.6)

# ════════════════════════════════════════════════════════════════════════════
# PANEL C — prior-gated bilinear head
# ════════════════════════════════════════════════════════════════════════════
CX, CY, CW, CH = 736, 24, 560, 696
panel(CX, CY, CW, CH, "C", "Prior-gated bilinear (PGB) interaction head")

# merge node
nx, ny = 756, 380
line([api_out, (nx, api_out[1]), (nx, ny - 9)], color=NAVY, sw=1.6)
line([exc_out, (nx, exc_out[1]), (nx, ny + 9)], color=SLATE, sw=1.6)
circle(nx, ny, 7, "#FFFFFF", NAVY, 1.6)
circle(nx, ny, 2.5, NAVY)

# fan-out trunk to interaction cards
tx = 778
OPS_X, OPS_W = 794, 186
cards = [  # y_top, height
    (66, 46),
    (122, 58),
    (190, 58),
    (258, 150),
]
mids = [y + h / 2 for y, h in cards]
mids[3] = 330  # bilinear card: arrow enters at schematic midline
line([(nx + 7, ny), (tx, ny), (tx, mids[0])], color=SLATE, sw=1.4, arrow=False)
for m in mids:
    line([(tx, m), (OPS_X - 2, m)], color=SLATE, sw=1.4)

# card 1: pass-through representations
y, h = cards[0]
rect(OPS_X, y, OPS_W, h, fill="#FFFFFF", stroke=LINE)
text(OPS_X + 10, y + 19, sub("h", "API") + " ‖ " + sub("h", "EXC"), 13, NAVY, "bold", raw=True)
text(OPS_X + 10, y + 36, "both representations", 10.5, SLATE)
text(OPS_X + OPS_W - 10, y + 28, "192", 13, NAVY, "bold", "end")

# card 2: element-wise product
y, h = cards[1]
rect(OPS_X, y, OPS_W, h, fill="#FFFFFF", stroke=LINE)
text(OPS_X + 10, y + 20, sub("h", "API") + " ⊙ " + sub("h", "EXC"), 13, NAVY, "bold", raw=True)
text(OPS_X + 10, y + 36, "Element-wise product", 11, NAVY)
text(OPS_X + 10, y + 50, "aligned feature interactions", 10.5, SLATE)
text(OPS_X + OPS_W - 10, y + 32, "96", 13, NAVY, "bold", "end")

# card 3: absolute difference
y, h = cards[2]
rect(OPS_X, y, OPS_W, h, fill="#FFFFFF", stroke=LINE)
text(OPS_X + 10, y + 20, "| " + sub("h", "API") + "− " + sub("h", "EXC") + "|", 13, NAVY, "bold", raw=True)
text(OPS_X + 10, y + 36, "Absolute difference", 11, NAVY)
text(OPS_X + 10, y + 50, "representation contrast", 10.5, SLATE)
text(OPS_X + OPS_W - 10, y + 32, "96", 13, NAVY, "bold", "end")

# card 4: low-rank bilinear (emphasised)
y, h = cards[3]
rect(OPS_X, y, OPS_W, h, fill="#FFFFFF", stroke=NAVY, sw=1.5)
text(OPS_X + 10, y + 19, "Low-rank bilinear interaction", 12, NAVY, "bold")
# schematic: two vectors -> U, V -> rank-16 vectors -> product
sx = OPS_X + 12
# API vector (short)
for i, (lab, yy, col) in enumerate([("API", y + 40, NAVY), ("EXC", y + 82, SLATE)]):
    for k in range(6):
        add(f'<rect x="{sx}" y="{yy + k * 4}" width="10" height="4" fill="{TEAL_M if k % 2 else TEAL}" stroke="#FFFFFF" stroke-width="0.6"/>')
    text(sx + 5, yy - 4, "h", 10, col, "bold", "middle")
    # projection trapezoid
    px0 = sx + 18
    add(f'<path d="M {px0} {yy - 2} L {px0 + 34} {yy + 6} L {px0 + 34} {yy + 18} L {px0} {yy + 26} Z" '
        f'fill="{"#E8ECF2"}" stroke="{col}" stroke-width="1"/>')
    text(px0 + 15, yy + 16, "U" if lab == "API" else "V", 11, col, "bold", "middle")
    # rank-16 vectors
    rx_ = px0 + 42
    for k in range(4):
        add(f'<rect x="{rx_}" y="{yy + 4 + k * 4}" width="9" height="4" fill="{AMBER_L if False else "#B8C2D0"}" stroke="#FFFFFF" stroke-width="0.6"/>')
    line([(px0 + 35, yy + 12), (rx_ - 1, yy + 12)], color=SLATE_L, sw=1, arrow=False)
# product node
pnx, pny = sx + 92, y + 72
line([(sx + 63, y + 52), (pnx - 6, pny - 5)], color=SLATE, sw=1.2, arrow=False)
line([(sx + 63, y + 94), (pnx - 6, pny + 5)], color=SLATE, sw=1.2, arrow=False)
circle(pnx, pny, 8, "#FFFFFF", NAVY, 1.3)
text(pnx, pny + 4.5, "⊙", 13, NAVY, "bold", "middle")
# bilinear output vector
bvx = pnx + 20
line([(pnx + 8, pny), (bvx - 2, pny)], color=SLATE, sw=1.2)
for k in range(8):
    add(f'<rect x="{bvx}" y="{pny - 16 + k * 4}" width="9" height="4" fill="{NAVY if k % 3 else SLATE_L}" stroke="#FFFFFF" stroke-width="0.6"/>')
text(bvx + 24, pny - 4, "rank", 10.5, SLATE)
text(bvx + 24, pny + 10, "16", 13, NAVY, "bold")
text(OPS_X + 10, y + h - 12, "U " + sub("h", "API") + "⊙ V " + sub("h", "EXC") + "· U, V: 96 → 16", 10.5, SLATE, raw=True)

# gating
gx, gy = 1010, pny  # gate multiplication node
line([(OPS_X + OPS_W + 1, gy), (gx - 10, gy)], color=NAVY, sw=1.5)
circle(gx, gy, 10, AMBER_L, AMBER, 1.6)
text(gx, gy + 5, "⊙", 15, AMBER, "bold", "middle")

GB = (950, 432, 120, 70)  # gate block
rect(*GB, fill=AMBER_L, stroke=AMBER, sw=1.5, rx=7)
text(GB[0] + GB[2] / 2, GB[1] + 19, "Learned gate g", 12, "#7A5413", "bold", "middle")
text(GB[0] + GB[2] / 2, GB[1] + 36, "46 → 32 → 16", 11.5, "#7A5413", anchor="middle")
text(GB[0] + GB[2] / 2, GB[1] + 53, "sigmoid · g ∈ (0, 1)¹⁶", 10.5, "#7A5413", anchor="middle")
line([(gx, GB[1] - 1), (gx, gy + 11)], color=AMBER, sw=1.8)
text(gx + 8, gy + 34, "modulates each", 10, "#7A5413")
text(gx + 8, gy + 46, "bilinear channel", 10, "#7A5413")

# context card (gate inputs)
KX, KY, KW, KH = 794, 538, 262, 164
rect(KX, KY, KW, KH, fill="#FFFFFF", stroke=AMBER, sw=1.2)
text(KX + 10, KY + 19, "Gate context", 12, NAVY, "bold")
text(KX + KW - 10, KY + 19, "46 features", 11.5, AMBER, "bold", "end")
rows = [
    ("5", "Excipient prior (training labels only)", AMBER, AMBER_L),
    ("18", "API chemistry flags", TEAL, TEAL_L),
    ("18", "Excipient chemistry flags", TEAL, TEAL_L),
    ("5", "Pairwise mechanism flags (SMARTS)", SLATE, "#EDF0F4"),
]
for i, (n_, lab, col, bg) in enumerate(rows):
    ry = KY + 30 + i * 22
    rect(KX + 10, ry, 28, 17, fill=bg, stroke=col, sw=1, rx=3)
    text(KX + 24, ry + 13, n_, 11, col, "bold", "middle")
    text(KX + 46, ry + 13, lab, 11, NAVY)
text(KX + 10, KY + 136, "prior = [p_exact, Δ global, p_family,", 10, SLATE)
text(KX + 10, KY + 149, "            log(1+n), unseen]", 10, SLATE)
line([(gx, KY - 1), (gx, GB[1] + GB[3] + 1)], color=AMBER, sw=1.8)

# concatenation bar (segments labelled with their dimensions)
CBX, CBW, CBT, CBB = 1092, 28, 66, 640
bounds = [CBT, 120, 185, 270, 420, CBB]
seg_lab = ["192", "96", "96", "16", "10"]
seg_fill = [TEAL_L, "#EEF1F5", "#EEF1F5", AMBER_L, AMBER_L]
for i in range(5):
    add(f'<rect x="{CBX}" y="{bounds[i]}" width="{CBW}" height="{bounds[i + 1] - bounds[i]}" fill="{seg_fill[i]}"/>')
    if i:
        add(f'<line x1="{CBX}" y1="{bounds[i]}" x2="{CBX + CBW}" y2="{bounds[i]}" stroke="{NAVY}" stroke-width="0.8"/>')
    text(CBX + CBW / 2, (bounds[i] + bounds[i + 1]) / 2 + 4, seg_lab[i], 10.5, NAVY, "bold", "middle")
rect(CBX, CBT, CBW, CBB - CBT, fill="none", stroke=NAVY, sw=1.3, rx=3)
for m in mids[:3]:
    line([(OPS_X + OPS_W + 1, m), (CBX - 2, m)], color=SLATE, sw=1.4)
line([(gx + 10, gy), (CBX - 2, gy)], color=AMBER, sw=1.6)
text(gx + 20, gy - 7, "gated · 16", 10.5, "#7A5413", "bold")
line([(KX + KW + 1, 600), (CBX - 2, 600)], color=AMBER, sw=1.4)
text(CBX + CBW / 2, CBB + 16, "410-D", 12, NAVY, "bold", "middle")
text(CBX + CBW / 2, CBT - 6, "concat", 10.5, SLATE, anchor="middle")

# MLP
my = 330
line([(CBX + CBW + 1, my), (1132, my)], color=SLATE, sw=1.5)
text(1192, 214, "MLP", 12.5, NAVY, "bold", "middle")
text(1192, 230, "410 → 128 → 48 → 1", 11.5, NAVY, anchor="middle")
layers = [(1134, 128 * 0.62), (1176, 48 * 0.62), (1214, 10)]
for i, (lx, lh) in enumerate(layers):
    rect(lx, my - lh / 2, 18, lh, fill=NAVY if i < 2 else TEAL, stroke="none", rx=4)
    if i < 2:
        nxt = layers[i + 1][0]
        line([(lx + 19, my), (nxt - 2, my)], color=SLATE, sw=1.4)
text(1143, my + 128 * 0.31 + 15, "128", 11, SLATE, "bold", "middle")
text(1185, my + 48 * 0.31 + 15, "48", 11, SLATE, "bold", "middle")
text(1223, my + 21, "r", 12, TEAL, "bold", "middle", style="italic")
text(1223, my + 34, "residual", 10, SLATE, anchor="middle")

# logit composition
sxn, syn = 1264, my
line([(1233, my), (sxn - 11, my)], color=SLATE, sw=1.5)
circle(sxn, syn, 10, "#FFFFFF", NAVY, 1.6)
text(sxn, syn + 5, "+", 15, NAVY, "bold", "middle")
text(sxn, syn - 18, "logit z", 11.5, NAVY, "bold", "middle")

# adaptive prior trust
TB = (1134, 540, 150, 88)
rect(*TB, fill=AMBER_L, stroke=AMBER, sw=1.3, rx=7)
text(TB[0] + 10, TB[1] + 19, "Adaptive prior trust", 12, "#7A5413", "bold")
text(TB[0] + 10, TB[1] + 37, "w = σ(MLP(prior))", 11, "#7A5413")
text(TB[0] + 10, TB[1] + 54, "w · logit(p_exact) + b", 11, "#7A5413")
text(TB[0] + 10, TB[1] + 71, "learned bias b", 10, "#7A5413")
line([(KX + KW + 1, 680), (1124, 680), (1124, TB[1] + 44), (TB[0] - 2, TB[1] + 44)], color=AMBER, sw=1.4)
line([(sxn, TB[1] - 1), (sxn, syn + 11)], color=AMBER, sw=1.6)
text(sxn - 6, 470, "prior", 10, "#7A5413", anchor="end")
text(sxn - 6, 482, "log-odds", 10, "#7A5413", anchor="end")

# ════════════════════════════════════════════════════════════════════════════
# PANEL D — prediction
# ════════════════════════════════════════════════════════════════════════════
DX, DY, DW, DH = 1312, 24, 264, 696
panel(DX, DY, DW, DH, "D", "Predicted incompatibility risk")

# coral outcome card (top)
rect(1326, 62, 236, 86, fill=CORAL_L, stroke=CORAL, sw=1.3, rx=8)
circle(1344, 84, 6, CORAL)
text(1356, 89, "Potential incompatibility", 12.5, CORAL, "bold")
text(1340, 108, "elevated predicted risk", 11, NAVY)
text(1340, 124, "→ prioritise for experimental", 11, NAVY)
text(1340, 139, "    compatibility testing", 11, NAVY)

# sigmoid plot
PX0, PX1, PY0, PY1 = 1356, 1548, 176, 470   # plot box
t_y = PY1 - 0.727 * (PY1 - PY0)
rect(PX0, PY0, PX1 - PX0, t_y - PY0, fill=CORAL_L, stroke="none", rx=0)
rect(PX0, t_y, PX1 - PX0, PY1 - t_y, fill=TEAL_L, stroke="none", rx=0)
import math
pts = []
for i in range(81):
    z = -6 + 12 * i / 80
    p = 1 / (1 + math.exp(-z))
    pts.append((PX0 + (PX1 - PX0) * i / 80, PY1 - p * (PY1 - PY0)))
add('<path d="M ' + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts) +
    f'" fill="none" stroke="{NAVY}" stroke-width="2.2"/>')
line([(PX0, t_y), (PX1, t_y)], color=NAVY, sw=1, arrow=False, dash="4 3")
add(f'<line x1="{PX0}" y1="{PY0}" x2="{PX0}" y2="{PY1}" stroke="{SLATE}" stroke-width="1"/>')
add(f'<line x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}" stroke="{SLATE}" stroke-width="1"/>')
text(PX0 - 5, PY0 + 4, "1", 10.5, SLATE, anchor="end")
text(PX0 - 5, PY1 + 4, "0", 10.5, SLATE, anchor="end")
text(PX1 - 4, t_y - 6, "t = 0.727", 11, NAVY, "bold", "end")
text(PX1 - 4, t_y + 14, "set on validation", 10, SLATE, anchor="end")
text(PX1, PY1 + 15, "logit z", 11, SLATE, anchor="end")
text(PX0 + 8, PY0 + 18, "P(incompatible) = σ(z)", 11.5, NAVY, "bold")
# logit arrow into plot
line([(sxn + 11, syn), (PX0 - 2, syn)], color=NAVY, sw=1.6)
# arrows to outcome cards
line([(1454, PY0 - 2), (1454, 150)], color=CORAL, sw=1.6)
line([(1454, PY1 + 3), (1454, 492)], color=TEAL, sw=1.6)

# teal outcome card (below plot)
rect(1326, 494, 236, 58, fill=TEAL_L, stroke=TEAL, sw=1.3, rx=8)
circle(1344, 514, 6, TEAL)
text(1356, 519, "Compatible", 12.5, TEAL, "bold")
text(1340, 539, "lower predicted risk", 11, NAVY)
text(1444, 570, "Model-predicted risk categories,", 10, SLATE, anchor="middle", style="italic")
text(1444, 583, "not experimental confirmation", 10, SLATE, anchor="middle", style="italic")

# metrics inset
rect(1326, 602, 236, 104, fill="#FFFFFF", stroke=NAVY, sw=1.2, rx=8)
text(1340, 622, "Group-split test set", 12, NAVY, "bold")
text(1548, 622, "n = 672", 11, SLATE, anchor="end")
for i, (lab, val) in enumerate([("PR-AUC", "0.725"), ("ROC-AUC", "0.912"), ("MCC", "0.584")]):
    cx = 1365 + i * 79
    text(cx, 660, val, 20, NAVY, "bold", "middle")
    text(cx, 679, lab, 11, SLATE, "bold", "middle")
    if i:
        add(f'<line x1="{cx - 40}" y1="{640}" x2="{cx - 40}" y2="{686}" stroke="{LINE}" stroke-width="1"/>')
text(1444, 698, "Morgan-PGB · API-cluster held-out", 10, SLATE_L, anchor="middle")

# ════════════════════════════════════════════════════════════════════════════
# STRIP — leakage-aware protocol
# ════════════════════════════════════════════════════════════════════════════
SY, SH = 736, 140
rect(24, SY, 1552, SH, fill="#FFFFFF", stroke=LINE, sw=1, rx=10)
rect(36, SY + 11, 20, 20, fill=SLATE, stroke="none", rx=4)
text(46, SY + 26, "E", 13, "#FFFFFF", "bold", "middle")
text(64, SY + 26, "Leakage-aware evaluation protocol", 14, NAVY, "bold")

steps = [
    ("Curated pairs", ["3,544 API–excipient pairs", "binary compatibility label"], NAVY, "#EEF1F5"),
    ("API-cluster group split", ["Butina clustering of APIs (Morgan,", "2,048-bit, Tanimoto) — used for splitting only", "StratifiedGroupKFold: 3 : 1 : 1 folds"], NAVY, "#EEF1F5"),
    ("Training · 2,030", ["Excipient priors from training labels only;", "training rows use leave-own-API-", "cluster-out priors"], TRAIN, AMBER_L),
    ("Validation · 842", ["Early stopping on validation PR-AUC", "Threshold t = 0.727 (TNR ≥ 0.97)", "Priors: full training table"], VAL, "#EEF1F5"),
    ("Held-out test · 672", ["Unseen API clusters", "Single final evaluation", "Priors: full training table"], NAVY, "#EEF1F5"),
]
sx0, sgap = 40, 22
sw_ = (1560 - sx0 - sgap * (len(steps) - 1)) / len(steps)
for i, (t, lines_, col, bg) in enumerate(steps):
    x = sx0 + i * (sw_ + sgap)
    rect(round(x, 1), SY + 40, round(sw_, 1), 88, fill=bg, stroke=AMBER if bg == AMBER_L else LINE,
         sw=1.2 if bg == AMBER_L else 1, rx=7)
    rect(round(x, 1), SY + 40, 5, 88, fill=col, stroke="none", rx=2)
    text(round(x + 14, 1), SY + 60, t, 12.5, NAVY, "bold")
    for j, l_ in enumerate(lines_):
        text(round(x + 14, 1), SY + 79 + j * 15, l_, 10.5, SLATE)
    if i < len(steps) - 1:
        ax_ = x + sw_
        line([(round(ax_ + 3, 1), SY + 84), (round(ax_ + sgap - 3, 1), SY + 84)], color=SLATE, sw=1.6)

add("</svg>")

svg = "\n".join(parts)
with open(os.path.join(OUT_DIR, "morgan_pgb_graphical_abstract.svg"), "w", encoding="utf-8") as f:
    f.write(svg)
print("wrote SVG", len(svg))
