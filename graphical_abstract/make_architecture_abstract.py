"""Architecture-style graphical abstract for the Morgan-PGB 80/20 model (final_project/).

Pastel, panel-based layout: dashed input region, molecule tower, PGB head,
prior-calibrated decision column and predicted-outcome panel.

Every number comes from final_project/:
  src/features.py        Morgan r=2 1,024 bits, MACCS 167, 12 PGB descriptors, 18 SMARTS flags,
                         8 salt-context values; fragments kept and OR-ed (salt-aware parsing)
  src/lookup.py          5 mechanism SMARTS rules
  MORGAN_PGB_8020_MODEL_SUMMARY.txt  layer shapes (145 -> 96, U/V 96 -> 16, gate 46 -> 32 -> 16,
                         MLP 410 -> 128 -> 48 -> 1, prior trust 5 -> 8 -> 1), threshold 0.671
  reference_results/heldout_metrics.json  22 pairs: 18 correct, PR-AUC 0.942, ROC-AUC 0.950,
                         MCC 0.671, FP 0
  Example probabilities: Trichlormethiazide + MgO 0.8185 (label 1), Celecoxib + TiO2 0.0539 (label 0)

Run:  python graphical_abstract/make_architecture_abstract.py
Writes .svg and a 2x .png (rendered with headless Chrome/Edge) next to this script.
"""

import os
import shutil
import subprocess

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_NAME = "morgan_pgb_architecture_abstract"
W, H = 1600, 1000

INK = "#2B2F3A"
SUB = "#5E6575"
MUTE = "#8A90A0"
FONT = "'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "Consolas, 'Courier New', monospace"

# reference-style pastels
ORANGE, ORANGE_D = "#F8B48C", "#D9773F"
BLUE_BG, BLUE_LN = "#EEF3FC", "#B8C9EC"
BLUE_BOX, BLUE_BOXD = "#9FC0EE", "#5C86C9"
LAV_BG, LAV_LN = "#F3EEFB", "#CDBCEB"
PINK_BG, PINK_LN = "#FDEFEE", "#EBB3AE"
RED_BOX, RED_LN = "#F6A097", "#D65E55"
GREEN_BOX, GREEN_LN = "#BFE6CB", "#5DAE76"
AMBER_BG, AMBER_LN = "#FCEFD6", "#D6A04A"

C_MORGAN = "#77C3A3"
C_MACCS = "#8DB2EA"
C_DESC = "#F09A9A"
C_FLAG = "#B79BE8"
C_SALT = "#F6B98A"
C_GRAY = "#C9CDD6"
C_PRIOR = "#E7B556"

parts = []


def add(s):
    parts.append(s)


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=12, fill=INK, weight="normal", anchor="start", style="normal",
         family=FONT, raw=False):
    body = s if raw else esc(s)
    add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" font-style="{style}" text-anchor="{anchor}">{body}</text>')


def rect(x, y, w, h, fill="none", stroke=None, sw=1.2, rx=8, dash=None, opacity=None):
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    op = f' opacity="{opacity}"' if opacity else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"{st}{d}{op}/>')


def circle(cx, cy, r, fill, stroke=None, sw=1.2):
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"{st}/>')


def path(pts, color="#5E6575", sw=1.6, arrow=True, dash=None):
    d = "M " + " L ".join(f"{x} {y}" for x, y in pts)
    mk = f' marker-end="url(#ah-{color[1:]})"' if arrow else ""
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" '
        f'stroke-linejoin="round" stroke-linecap="round"{ds}{mk}/>')


def ribbon(x, y, cols, size=12, gap=3, dashed_from=None):
    """Row of small rounded squares; cols is a list of fill colours."""
    for i, c in enumerate(cols):
        dash = "2 2" if dashed_from is not None and i >= dashed_from else None
        rect(x + i * (size + gap), y, size, size, fill=c, stroke="#FFFFFF" if not dash else MUTE,
             sw=0.8, rx=2.5, dash=dash)
    return x + len(cols) * (size + gap) - gap


def oplus(cx, cy, r=11):
    circle(cx, cy, r, "#FFFFFF", INK, 1.5)
    path([(cx - r + 4, cy), (cx + r - 4, cy)], INK, 1.5, arrow=False)
    path([(cx, cy - r + 4), (cx, cy + r - 4)], INK, 1.5, arrow=False)


def odot(cx, cy, r=9, color=INK):
    circle(cx, cy, r, "#FFFFFF", color, 1.5)
    circle(cx, cy, 2.6, color)


# ════════════════════════════════════════════════════════════════════════════
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add("<defs>")
for c in ("#5E6575", INK, MUTE, RED_LN, AMBER_LN, "#7E6BB5", ORANGE_D):
    add(f'<marker id="ah-{c[1:]}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0 0.8 L9 5 L0 9.2 z" fill="{c}"/></marker>')
add('<filter id="sh" x="-10%" y="-10%" width="120%" height="130%">'
    '<feDropShadow dx="0" dy="1.5" stdDeviation="1.6" flood-color="#000" flood-opacity="0.12"/></filter>')
add("</defs>")
add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

# ── dashed input region ────────────────────────────────────────────────────
rect(18, 18, 1166, 646, fill="none", stroke="#A7ACB8", sw=1.4, rx=16, dash="7 5")
text(38, 50, "Input", 19, INK, "bold")

# ── input pills ────────────────────────────────────────────────────────────
def input_pill(x, y, w, h, label):
    add(f'<g filter="url(#sh)">')
    rect(x, y, w, h, fill=ORANGE, stroke=ORANGE_D, sw=1.4, rx=h / 2)
    add("</g>")
    text(x + w / 2, y + h / 2 + 6, label, 16, INK, "bold", "middle")


input_pill(34, 168, 160, 46, "API SMILES")
text(36, 236, "CN(C)C(=N)N=C(N)N.Cl", 10.5, MUTE, family=MONO)
text(36, 251, "metformin hydrochloride", 10.5, MUTE, style="italic")

input_pill(34, 470, 160, 46, "Excipient SMILES")
text(36, 538, "lactose · Mg stearate · TiO\u2082 …", 10.5, MUTE)

path([(194, 191), (226, 191)], "#5E6575")
# excipient also passes through the same parser
path([(194, 493), (210, 493), (210, 262), (226, 262)], "#5E6575")
path([(210, 493), (226, 493)], "#5E6575")

# ── salt-aware parsing panel ───────────────────────────────────────────────
SX, SY, SW, SH = 230, 78, 248, 262
text(SX + SW / 2, SY - 10, "Salt-aware parsing", 15, INK, "normal", "middle", "italic")
rect(SX, SY, SW, SH, fill="#FFFFFF", stroke="#E7C2A6", sw=1.3, rx=12)
rect(SX + 14, SY + 18, 170, 32, fill="#F7F8FA", stroke="#D5D9E0", sw=1, rx=6)
text(SX + 99, SY + 39, "CN(C)C(=N)N=C(N)N.Cl", 11.5, INK, "normal", "middle", family=MONO)
circle(SX + 214, SY + 34, 16, "#F5B7B0", RED_LN, 1.2)
text(SX + 214, SY + 39, "Cl\u207b", 12.5, INK, "bold", "middle")
text(SX + 124, SY + 70, "split on '.'", 10.5, MUTE, anchor="middle", style="italic")
path([(SX + 70, SY + 52), (SX + 70, SY + 86)], "#5E6575", 1.4)
path([(SX + 214, SY + 52), (SX + 214, SY + 86)], "#5E6575", 1.4)
rect(SX + 14, SY + 90, 140, 36, fill=BLUE_BOX, stroke=BLUE_BOXD, sw=1.2, rx=7)
text(SX + 84, SY + 113, "Parent molecule", 13, INK, "bold", "middle")
rect(SX + 166, SY + 90, 68, 36, fill="#F9D2CD", stroke=RED_LN, sw=1.2, rx=7)
text(SX + 200, SY + 113, "Counter-ion", 11, INK, "bold", "middle")
rect(SX + 14, SY + 140, 220, 50, fill="#F7F8FA", stroke=MUTE, sw=1, rx=7, dash="4 3")
text(SX + 124, SY + 160, "all fragments kept (not stripped)", 11.5, INK, "bold", "middle")
text(SX + 124, SY + 178, "bits OR-ed · flags fire on any fragment", 10.5, SUB, anchor="middle")
text(SX + 124, SY + 214, "+ salt context: fragments, metals,", 10.5, SUB, anchor="middle")
text(SX + 124, SY + 229, "charges, inorganic heavy atoms", 10.5, SUB, anchor="middle")
text(SX + 124, SY + 250, "applied to API and excipient", 10.5, MUTE, anchor="middle", style="italic")

# ── excipient prior + mechanism panel ──────────────────────────────────────
PX, PY, PW, PH = 230, 386, 248, 232
text(PX + PW / 2, PY - 10, "Excipient prior · reaction rules", 15, INK, "normal", "middle", "italic")
rect(PX, PY, PW, PH, fill="#FFFFFF", stroke=AMBER_LN, sw=1.3, rx=12)
prior_rows = [("p_exact", "smoothed incompatibility rate"),
              ("p_\u0394", "p_exact \u2212 global rate 0.094"),
              ("p_family", "rate after removing water"),
              ("log(1+n)", "training rows with excipient"),
              ("unseen", "1 if not in training")]
for i, (k, d) in enumerate(prior_rows):
    yy = PY + 24 + i * 19
    rect(PX + 14, yy - 10, 11, 11, fill=C_PRIOR, stroke="#FFFFFF", sw=0.8, rx=2.5)
    text(PX + 32, yy, k, 11, INK, "bold")
    text(PX + 96, yy, d, 10, SUB)
text(PX + 14, PY + 128, "train labels only · leave-own-cluster-out", 9.8, MUTE, style="italic")
path([(PX + 14, PY + 140), (PX + PW - 14, PY + 140)], "#E3E6EB", 1, arrow=False)
text(PX + 14, PY + 160, "Mechanism flags · 5 SMARTS rules", 11.5, INK, "bold")
for i in range(5):
    circle(PX + 22 + i * 22, PY + 178, 7.5, ORANGE if i == 0 else "#7FC6A4",
           ORANGE_D if i == 0 else "#4E9E79", 1)
text(PX + 138, PY + 182, "API \u00d7 excipient", 10.5, SUB)
text(PX + 14, PY + 202, "Maillard · ester/amide hydrolysis · metal-", 10, SUB)
text(PX + 14, PY + 216, "ion oxidation · salt displacement · Schiff base", 10, SUB)

# ── molecule tower panel ───────────────────────────────────────────────────
RX, RY, RW, RH = 510, 34, 656, 528
rect(RX, RY, RW, RH, fill=BLUE_BG, stroke=BLUE_LN, sw=1.3, rx=14)
text(RX + RW / 2, RY + 26, "Molecule tower", 16, INK, "normal", "middle", "italic")
text(RX + RW / 2, RY + 44, "one per molecule · API and excipient towers share the design, separate weights",
     10.5, SUB, anchor="middle")

path([(478, 191), (520, 191)], "#5E6575")          # from parser into the bus
BUS = 522
CARD_X, CARD_W, CARD_H = 538, 262, 58
cards = [
    ("Morgan FP", "r = 2 · 1,024 bits", "morgan", 64, 8, C_MORGAN, False),
    ("MACCS keys", "167 structural keys", "maccs", 32, 6, C_MACCS, False),
    ("RDKit descriptors", "12 physicochemical values", "desc", 24, 5, C_DESC, False),
    ("Chemistry flags", "18 SMARTS: metals, acids, amines…", "flag", 16, 4, C_FLAG, False),
    ("Salt context + availability", "8 counts / charges + 1 flag", "salt", 9, 3, C_SALT, True),
]
card_y0, card_gap = 92, 10
RIB_X = 836
COLLECT = 1132
path([(BUS, 101), (BUS, card_y0 + 4 * (CARD_H + card_gap) + CARD_H / 2)], "#5E6575", 1.4, arrow=False)
concat_cols = []
for i, (name, sub, icon, dim, nsq, col, passthru) in enumerate(cards):
    cy = card_y0 + i * (CARD_H + card_gap)
    mid = cy + CARD_H / 2
    path([(BUS, mid), (CARD_X - 2, mid)], "#5E6575", 1.4)
    add('<g filter="url(#sh)">')
    rect(CARD_X, cy, CARD_W, CARD_H, fill="#FFFFFF", stroke="#C9D3E6", sw=1.1, rx=10)
    add("</g>")
    ix, iy = CARD_X + 12, cy + 12   # icon box 48 x 38
    if icon == "morgan":
        pal = ["#77C3A3", "#B5E0CD", "#E6F4EE", "#8DB2EA", "#F09A9A", "#F6B98A", "#B79BE8"]
        k = 0
        for r_ in range(3):
            for c_ in range(5):
                rect(ix + c_ * 9.5, iy + r_ * 12.5, 8, 10.5, fill=pal[(k * 5 + r_) % len(pal)], rx=1.5)
                k += 1
    elif icon == "maccs":
        bits = "1101001110100101011101"
        for j in range(16):
            h_ = 34 if bits[j] == "1" else 16
            add(f'<line x1="{ix + 2 + j * 2.9}" y1="{iy + 37}" x2="{ix + 2 + j * 2.9}" y2="{iy + 37 - h_}" '
                f'stroke="{C_MACCS if bits[j] == "1" else "#D3DCEB"}" stroke-width="1.8"/>')
    elif icon == "desc":
        hs = [18, 30, 12, 26, 34, 20, 9, 24, 15]
        pal = [C_DESC, C_MACCS, C_MORGAN, C_SALT, C_FLAG]
        for j, h_ in enumerate(hs):
            rect(ix + j * 5.4, iy + 38 - h_, 4.2, h_, fill=pal[j % 5], rx=1)
        path([(ix - 1, iy + 38.5), (ix + 49, iy + 38.5)], MUTE, 1, arrow=False)
    elif icon == "flag":
        on = "101100110010100110"
        for j in range(18):
            r_, c_ = divmod(j, 6)
            circle(ix + 4 + c_ * 8, iy + 7 + r_ * 12, 3.2, C_FLAG if on[j] == "1" else "#FFFFFF",
                   "#9B80D6", 0.9)
    elif icon == "salt":
        circle(ix + 13, iy + 19, 11, "#FBE3D2", ORANGE_D, 1.1)
        text(ix + 13, iy + 25, "+", 16, ORANGE_D, "bold", "middle")
        circle(ix + 36, iy + 19, 11, "#E3ECFA", BLUE_BOXD, 1.1)
        text(ix + 36, iy + 24, "\u2212", 16, BLUE_BOXD, "bold", "middle")
    text(CARD_X + 72, cy + 27, name, 14, INK, "bold")
    text(CARD_X + 72, cy + 45, sub, 10.5, SUB)
    # projection to ribbon
    path([(CARD_X + CARD_W, mid), (RIB_X - 6, mid)], "#5E6575", 1.4)
    if not passthru:
        text((CARD_X + CARD_W + RIB_X) / 2, mid + 14, "Linear", 9.5, MUTE, anchor="middle")
    else:
        text((CARD_X + CARD_W + RIB_X) / 2, mid + 14, "direct", 9.5, MUTE, anchor="middle")
    end = ribbon(RIB_X, mid - 6, [col] * nsq, dashed_from=None)
    text(RIB_X, mid - 12, f"{dim} values", 10.5, SUB)
    path([(end + 4, mid), (COLLECT, mid)], "#5E6575", 1.4, arrow=False)
    concat_cols += [col] * nsq

path([(COLLECT, card_y0 + CARD_H / 2), (COLLECT, 470), (1016, 470)], "#5E6575", 1.6)
# concat ribbon (145 values)
text(560, 474, "Concat", 14, INK, "bold")
cend = ribbon(624, 464, concat_cols, size=11, gap=3)
text(624, 454, "molecule features · 64 + 32 + 24 + 16 + 9 = 145 values", 11, SUB)
# Mix box
path([(cend + 2, 470), (1016, 470)], "#5E6575", 1.4, arrow=False)
MIXX, MIXY = 772, 510
path([(838, 482), (838, MIXY - 2)], "#5E6575", 1.4)
add('<g filter="url(#sh)">')
rect(MIXX, MIXY, 132, 40, fill=GREEN_BOX, stroke=GREEN_LN, sw=1.3, rx=9)
add("</g>")
text(MIXX + 66, MIXY + 18, "Mix", 13.5, INK, "bold", "middle")
text(MIXX + 66, MIXY + 33, "Linear + LayerNorm", 10, SUB, anchor="middle")
text(MIXX + 146, MIXY + 18, "145 \u2192 96", 12, INK, "bold")
text(MIXX + 146, MIXY + 34, "embedding h \u00d7 2 (API, excipient)", 10.5, SUB)

# ── pair junction ──────────────────────────────────────────────────────────
JX, JY = 838, 606
path([(838, MIXY + 40), (838, JY - 13)], "#5E6575", 1.6)
path([(PX + PW, 594), (720, 594), (720, JY), (JX - 13, JY)], "#5E6575", 1.6)
text(492, 588, "prior 5 + mechanism 5", 10.5, SUB)
oplus(JX, JY)
ribbon(870, 600, [C_MORGAN] * 4 + [C_MACCS] * 4 + [C_PRIOR] * 2 + [ORANGE] * 2, size=12, gap=3)
text(870, 632, "pair input: h_API 96 · h_exc 96 · prior 5 · mechanisms 5", 10.5, SUB)
path([(JX, JY + 11), (JX, 700)], "#5E6575", 1.6)

# ── PGB interaction head ──────────────────────────────────────────────────
GX, GY, GW, GH = 498, 702, 686, 250
rect(GX, GY, GW, GH, fill=LAV_BG, stroke=LAV_LN, sw=1.3, rx=14)
text(GX + GW / 2, GY + 26, "Prior-gated bilinear (PGB) interaction head", 16, INK, "normal", "middle", "italic")
hcards = [
    ("Concat", "API \u2225 Exc", "192", 192, C_MORGAN),
    ("Product", "API \u2299 Exc", "96", 96, C_MACCS),
    ("Difference", "|API \u2212 Exc|", "96", 96, C_DESC),
    ("Bilinear", "U·h \u2299 V·h", "rank 16", 16, C_FLAG),
    ("Prior gate g", "46 \u2192 32 \u2192 16 · \u03c3", "16 gates", None, C_PRIOR),
    ("MLP", "410 \u2192 128 \u2192 48 \u2192 1", "residual r", None, "#7E6BB5"),
]
HC_W, HC_H, HC_G = 100, 150, 11
hx0 = GX + 16
for i, (nm, sub, lab, dim, col) in enumerate(hcards):
    x = hx0 + i * (HC_W + HC_G)
    y = GY + 44
    is_gate = nm.startswith("Prior")
    add('<g filter="url(#sh)">')
    rect(x, y, HC_W, HC_H, fill=AMBER_BG if is_gate else "#FFFFFF",
         stroke=AMBER_LN if is_gate else "#D3C8EC", sw=1.1, rx=10)
    add("</g>")
    cx = x + HC_W / 2
    iy = y + 14
    # icons
    if nm == "Concat":
        ribbon(cx - 39, iy + 10, [C_MORGAN] * 3, size=10, gap=2)
        ribbon(cx + 3, iy + 10, [C_MACCS] * 3, size=10, gap=2)
    elif nm == "Product":
        ribbon(cx - 39, iy + 10, [C_MORGAN] * 3, size=8, gap=2)
        odot(cx, iy + 14, 7)
        ribbon(cx + 11, iy + 10, [C_MACCS] * 3, size=8, gap=2)
    elif nm == "Difference":
        ribbon(cx - 39, iy + 10, [C_MORGAN] * 3, size=8, gap=2)
        text(cx, iy + 19, "\u2212", 16, INK, "bold", "middle")
        ribbon(cx + 11, iy + 10, [C_MACCS] * 3, size=8, gap=2)
    elif nm == "Bilinear":
        add(f'<path d="M {cx-36} {iy} L {cx-14} {iy+7} L {cx-14} {iy+15} L {cx-36} {iy+22} Z" '
            f'fill="#E4DBF6" stroke="#7E6BB5" stroke-width="1.1"/>')
        text(cx - 26, iy + 15, "U", 9.5, INK, "bold", "middle")
        add(f'<path d="M {cx+36} {iy} L {cx+14} {iy+7} L {cx+14} {iy+15} L {cx+36} {iy+22} Z" '
            f'fill="#E4DBF6" stroke="#7E6BB5" stroke-width="1.1"/>')
        text(cx + 26, iy + 15, "V", 9.5, INK, "bold", "middle")
        odot(cx, iy + 11, 7)
    elif is_gate:
        add(f'<path d="M {cx-30} {iy+24} C {cx-12} {iy+24} {cx-8} {iy} {cx+30} {iy}" fill="none" '
            f'stroke="{AMBER_LN}" stroke-width="2"/>')
        path([(cx - 32, iy + 26), (cx + 32, iy + 26)], MUTE, 0.8, arrow=False)
    else:  # MLP
        for j, (hh, ww) in enumerate([(26, 9), (18, 9), (10, 9), (5, 9)]):
            rect(cx - 30 + j * 17, iy + 13 - hh / 2, ww, hh, fill="#7E6BB5", rx=2.5)
    text(cx, y + 62, nm, 13, INK, "bold", "middle")
    text(cx, y + 80, sub, 10.5, SUB, anchor="middle")
    # bottom bar: share of the 410-D MLP input
    if dim:
        rect(x + 12, y + 104, HC_W - 24, 8, fill="#ECE8F4", rx=4)
        rect(x + 12, y + 104, max(4, (HC_W - 24) * dim / 192), 8, fill=col, rx=4)
    text(cx, y + 134, lab, 12, INK if dim else (AMBER_LN if is_gate else "#7E6BB5"), "bold", "middle")

# gate multiplies the bilinear term
g_x = hx0 + 4 * (HC_W + HC_G)
b_x = hx0 + 3 * (HC_W + HC_G)
add(f'<path d="M {g_x + 30} {GY + 196} C {g_x + 10} {GY + 222}, {b_x + 90} {GY + 222}, {b_x + 70} {GY + 198}" '
    f'fill="none" stroke="{AMBER_LN}" stroke-width="1.6" marker-end="url(#ah-{AMBER_LN[1:]})"/>')
text(g_x + 6, GY + 232, "gated \u2299", 10.5, AMBER_LN, "bold", "middle")
text(GX + 18, GY + 236, "192 + 96 + 96 + gated 16 + prior 5 + mechanisms 5 = 410 \u2192 MLP", 10.5, SUB)
text(GX + GW - 16, GY + 236, "236,219 parameters", 10.5, MUTE, anchor="end", style="italic")

# r out of the head → decision column
path([(GX + GW, GY + 120), (1197, GY + 120), (1197, 139), (1243, 139)], "#5E6575", 1.6)
text(1203, GY + 112, "r", 13, "#7E6BB5", "bold", style="italic")

# ── prior-calibrated decision column ──────────────────────────────────────
DX, DY, DW, DH = 1212, 34, 350, 520
rect(DX, DY, DW, DH, fill=PINK_BG, stroke=PINK_LN, sw=1.3, rx=14)
text(DX + DW / 2, DY + 32, "Prior-calibrated decision", 17, INK, "bold", "middle")
boxes = [
    ("Residual logit r", "pair-interaction score from the PGB head"),
    ("Adaptive prior trust", "w = \u03c3(MLP(prior)) \u2208 (0, 1)  · 5 \u2192 8 \u2192 1"),
    ("Logit fusion", "z = r + w·logit(p_exact) + b"),
    ("Threshold calibration", "P(incompatible) = \u03c3(z) \u2265 0.671"),
]
BX, BW, BH = 1246, 282, 66
for i, (t1, t2) in enumerate(boxes):
    by = 106 + i * 108
    add('<g filter="url(#sh)">')
    rect(BX, by, BW, BH, fill=RED_BOX, stroke=RED_LN, sw=1.4, rx=14)
    add("</g>")
    text(BX + BW / 2, by + 28, t1, 15, INK, "bold", "middle")
    text(BX + BW / 2, by + 48, t2, 11.5, "#3E3540", anchor="middle")
    if i < 3:
        path([(BX + BW / 2, by + BH + 2), (BX + BW / 2, by + 106)], INK, 1.6)
text(DX + DW / 2, 456 + 66 + 22, "t: best validation MCC with TNR \u2265 0.97", 10.5, SUB, anchor="middle",
     style="italic")

# ── output panel ──────────────────────────────────────────────────────────
OX, OY, OW, OH = 1212, 594, 350, 358
path([(DX + DW / 2, DY + DH), (DX + DW / 2, OY - 2)], INK, 1.6)
rect(OX, OY, OW, OH, fill=LAV_BG, stroke=LAV_LN, sw=1.3, rx=14)


def chip(x, y, w, label, fill, stroke, size=13, h=30, color=INK):
    rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.1, rx=h / 2)
    text(x + w / 2, y + h / 2 + size * 0.36, label, size, color, "bold", "middle")


chip(1232, 616, 150, "Incompatible", "#F7B5AE", RED_LN, 14, 34)
chip(1394, 616, 148, "Compatible", "#BFE6CB", GREEN_LN, 14, 34)

# probability gauge
GX0, GX1, GYY = 1240, 1534, 700
t_x = GX0 + (GX1 - GX0) * 0.671
rect(GX0, GYY, t_x - GX0, 10, fill="#BFE6CB", rx=5)
rect(t_x, GYY, GX1 - t_x, 10, fill="#F7B5AE", rx=5)
path([(t_x, GYY - 8), (t_x, GYY + 18)], INK, 1.4, arrow=False, dash="3 2")
text(GX0, GYY - 8, "P(incompatible)", 10.5, SUB)
text(t_x, GYY + 30, "t = 0.671", 10.5, INK, "bold", "middle")
text(GX0, GYY + 30, "0", 10, MUTE)
text(GX1, GYY + 30, "1", 10, MUTE, anchor="end")
for p, lbl, col in ((0.8185, "Trichlormethiazide + MgO  0.82", RED_LN),
                    (0.0539, "Celecoxib + TiO\u2082  0.05", GREEN_LN)):
    xx = GX0 + (GX1 - GX0) * p
    circle(xx, GYY + 5, 6.5, "#FFFFFF", col, 2)
y_ex = GYY + 54
circle(1246, y_ex - 4, 5, "#FFFFFF", RED_LN, 2)
text(1258, y_ex, "Trichlormethiazide + MgO  P = 0.82 \u2192 incompatible \u2713", 10.5, INK)
circle(1246, y_ex + 14, 5, "#FFFFFF", GREEN_LN, 2)
text(1258, y_ex + 18, "Celecoxib + TiO\u2082  P = 0.05 \u2192 compatible \u2713", 10.5, INK)

# held-out metrics
text(1232, 806, "External held-out test · 22 pairs", 12, INK, "bold")
chip(1232, 818, 106, "18 / 22", "#D7CBF2", "#9B86D3", 13)
chip(1346, 818, 108, "PR-AUC 0.94", "#BBD3F2", BLUE_BOXD, 12)
chip(1462, 818, 80, "MCC 0.67", "#F9D3B6", ORANGE_D, 12)
chip(1232, 856, 118, "ROC-AUC 0.95", "#BFE6CB", GREEN_LN, 12)
chip(1358, 856, 132, "0 false positives", "#F7B5AE", RED_LN, 12)
chip(1498, 856, 44, "\u2026", "#ECEAF2", "#C3BFD0", 13)
text(OX + OW / 2, 935, "Predicted compatibility", 16, INK, "bold", "middle", "italic")

# ── prior feedback (dashed) ───────────────────────────────────────────────
DASH = "6 5"
path([(PX + PW - 30, PY + PH), (PX + PW - 30, 966), (1582, 966), (1582, 239), (BX + BW + 4, 239)],
     MUTE, 1.5, dash=DASH)
path([(1582, 347), (BX + BW + 4, 347)], MUTE, 1.5, dash=DASH)
text(820, 986, "the excipient prior also gates the bilinear term and sets how far the final logit "
     "trusts p_exact", 12, SUB, anchor="middle", style="italic")

# ── decorative ball-and-stick pair (bottom left) ──────────────────────────
def molecule(ox, oy, atoms, bonds, s=1.0):
    for a, b in bonds:
        (x1, y1, _), (x2, y2, _) = atoms[a], atoms[b]
        add(f'<line x1="{ox + x1 * s}" y1="{oy + y1 * s}" x2="{ox + x2 * s}" y2="{oy + y2 * s}" '
            f'stroke="#C3C8D2" stroke-width="5" stroke-linecap="round"/>')
    for x, y, c in atoms:
        add(f'<circle cx="{ox + x * s}" cy="{oy + y * s}" r="{9 * s}" fill="{c}" opacity="0.92"/>')
        add(f'<circle cx="{ox + x * s - 3 * s}" cy="{oy + y * s - 3 * s}" r="{3 * s}" fill="#FFFFFF" '
            f'opacity="0.55"/>')


Cc, Nc, Oc, Sc = "#F2C46D", "#8DB2EA", "#F09A9A", "#B79BE8"
import math
hexa = [(math.cos(math.radians(30 + 60 * k)) * 30, math.sin(math.radians(30 + 60 * k)) * 30) for k in range(6)]
api_atoms = [(x, y, Cc) for x, y in hexa] + [(56, -15, Nc), (82, 0, Cc), (-56, -15, Oc), (0, -60, Sc)]
api_atoms[2] = (*hexa[2], Nc)
api_bonds = [(k, (k + 1) % 6) for k in range(6)] + [(0, 6), (6, 7), (3, 8), (4, 9)]
molecule(110, 830, api_atoms, api_bonds, 1.05)
text(110, 905, "API", 12, MUTE, "bold", "middle")

sugar = [(math.cos(math.radians(60 * k)) * 28, math.sin(math.radians(60 * k)) * 28) for k in range(6)]
exc_atoms = [(x, y, Cc) for x, y in sugar] + [(52, -30, Oc), (-52, 30, Oc), (0, 58, Oc)]
exc_atoms[1] = (*sugar[1], Oc)
exc_bonds = [(k, (k + 1) % 6) for k in range(6)] + [(5, 6), (3, 7), (2, 8)]
molecule(370, 800, exc_atoms, exc_bonds, 1.0)
text(370, 905, "excipient", 12, MUTE, "bold", "middle")
path([(215, 828), (300, 816)], MUTE, 1.4, dash="4 4", arrow=True)
path([(300, 816), (215, 828)], MUTE, 1.4, dash="4 4", arrow=True)
text(258, 808, "?", 18, ORANGE_D, "bold", "middle")

add("</svg>")

svg = "\n".join(parts)
svg_path = os.path.join(OUT_DIR, OUT_NAME + ".svg")
with open(svg_path, "w", encoding="utf-8") as f:
    f.write(svg)
print("wrote", svg_path)

# ── rasterise with headless Chrome / Edge ──────────────────────────────────
browser = next((p for p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                            shutil.which("chrome"), shutil.which("msedge")) if p and os.path.exists(p)), None)
if browser:
    html_path = os.path.join(OUT_DIR, OUT_NAME + "_render.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f'<!doctype html><html><body style="margin:0">{svg}</body></html>')
    png_path = os.path.join(OUT_DIR, OUT_NAME + ".png")
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    "--force-device-scale-factor=2", f"--window-size={W},{H}",
                    f"--screenshot={png_path}", "file:///" + html_path.replace("\\", "/")],
                   check=True, capture_output=True)
    os.remove(html_path)
    print("wrote", png_path)
else:
    print("no Chrome/Edge found; open the SVG in a browser to export PNG")
