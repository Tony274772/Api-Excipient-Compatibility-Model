"""Table of the 5 mechanism flags (REACTION_TABLE in src/lookup.py) in the style of the salt-context table.

Each flag is 1 only when BOTH the API matches one of its API SMARTS AND the excipient matches one of its
excipient SMARTS (get_mechanism_matches). The examples are real incompatible pairs from data/start_dataset.csv
(matched atoms highlighted); "fires on" is counted over the whole file.

Run (needs rdkit + matplotlib, e.g. the anaconda python):  python figures/make_mechanism_flags_table.py
Writes figures/mechanism_flags_table.png
"""

import io
import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from src.lookup import REACTION_TABLE, get_mechanism_matches  # noqa: E402

EXAMPLES = "--no-examples" not in sys.argv
OUT = os.path.join(HERE, "mechanism_flags_table.png" if EXAMPLES else "mechanism_flags_table_no_examples.png")
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# ---------------------------------------------------------------- content
# name, what it measures, chemical meaning, example (api name, api smiles, exc name, exc smiles-for-drawing,
# api_cid, exc_cid), row colour
ROWS = [
    dict(
        name="Maillard reaction",
        measures="API has a free primary amine and the excipient has an aldehyde (reducing-sugar-type) group",
        meaning="Amine + carbonyl condense to a coloured adduct (browning). Classic API–lactose / glucose problem: "
                "discolouration and loss of API.",
        ex=("Carbidopa", "CC(CC1=CC(=C(C=C1)O)O)(C(=O)O)NN", "Butyraldehyde", "CCCC=O", 34359, 261),
        color="#fdf3df",
    ),
    dict(
        name="Ester / amide hydrolysis",
        measures="API has an ester or amide bond and the excipient supplies hydroxide ions",
        meaning="Alkaline excipients (NaOH, KOH, Mg(OH)$_2$) cleave ester and amide bonds, splitting the API into "
                "acid + alcohol or amine.",
        ex=("Benzyl benzoate", "C1=CC=C(C=C1)COC(=O)C2=CC=CC=C2", "Sodium hydroxide", "[OH-].[Na+]", 2345, 14798),
        color="#e8f2e4",
    ),
    dict(
        name="Metal-ion catalysed oxidation",
        measures="API has an oxidisable phenol (–OH on aromatic ring) or thiol (–SH) and the excipient contains "
                 "Fe, Cu or Mn",
        meaning="Transition-metal ions catalyse radical oxidation of phenols and thiols (e.g. antioxidants and "
                "captopril-type drugs), so the API degrades.",
        ex=("α-Tocopherol", "CC1=C(C2=C(CCC(O2)(C)CCCC(C)CCCC(C)CCCC(C)C)C(=C1O)C)C", "Copper(II) ion", "[Cu+2]",
            14985, 27099),
        color="#fbe3e8",
    ),
    dict(
        name="Acid–base salt displacement",
        measures="API has a carboxylic acid and the excipient contains Na, K, Ca or Mg",
        meaning="An acidic API reacts with the basic metal salt and is converted into a different salt form, "
                "changing solubility and dissolution.",
        ex=("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O", "Calcium phosphate (dibasic)", "OP(=O)([O-])[O-].[Ca+2]",
            2244, 104805),
        color="#e1ecf7",
    ),
    dict(
        name="Schiff base formation",
        measures="API has a primary or secondary amine and the excipient has an aldehyde or ketone",
        meaning="Amine + carbonyl give an imine (C=N), a covalent API–excipient adduct. Overlaps with Maillard "
                "chemistry, so both can fire together.",
        ex=("Vigabatrin", "C=CC(CCC(=O)O)N", "5-Hydroxymethylfurfural", "C1=C(OC(=C1)C=O)CO", 5665, 237332),
        color="#ebe6f4",
    ),
]

# ---------------------------------------------------------------- data checks / stats
df = pd.read_csv(os.path.join(ROOT, "data", "start_dataset.csv"))
M = np.array([get_mechanism_matches(r.API_Smiles, str(r.Excipient_Smiles)) if isinstance(r.Excipient_Smiles, str)
              else [0] * 5 for r in df.itertuples()])
base = df.Outcome1.mean()
for j, row in enumerate(ROWS):
    sel = df[M[:, j] == 1]
    row["n"], row["pct"], row["inc"] = len(sel), len(sel) / len(df), sel.Outcome1.mean()
    entry = REACTION_TABLE[j]
    row["api_smarts"] = entry.get("api_smarts_list", [entry.get("api_smarts")])
    row["exc_smarts"] = entry["exc_smarts_list"]
    # the example must come from the data, be labelled incompatible, and fire exactly this flag
    _, _, _, _, a, e = row["ex"]
    pair = df[(df.API_CID == a) & (df.Excipient_CID == e)].iloc[0]
    assert pair.Outcome1 == 1 and get_mechanism_matches(pair.API_Smiles, pair.Excipient_Smiles)[j] == 1, row["name"]


# ---------------------------------------------------------------- molecule drawings
def draw(smiles, smarts_list, color, w=520, h=340):
    mol = Chem.MolFromSmiles(smiles)
    atoms = set()
    for s in smarts_list:
        patt = Chem.MolFromSmarts(s)
        for match in mol.GetSubstructMatches(patt):
            atoms.update(match)
    bonds = [b.GetIdx() for b in mol.GetBonds()
             if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms]
    d = rdMolDraw2D.MolDraw2DCairo(w, h)
    o = d.drawOptions()
    o.clearBackground = False
    o.bondLineWidth = 2
    o.baseFontSize = 0.55
    o.highlightRadius = 0.32
    d.DrawMolecule(mol, highlightAtoms=sorted(atoms), highlightBonds=bonds,
                   highlightAtomColors={i: color for i in atoms}, highlightBondColors={b: color for b in bonds})
    d.FinishDrawing()
    return mpimg.imread(io.BytesIO(d.GetDrawingText()), format="png")


API_RGBA, EXC_RGBA = (0.96, 0.55, 0.20, 0.55), (0.22, 0.55, 0.90, 0.55)
for row in ROWS if EXAMPLES else []:
    _, asmi, _, esmi, _, _ = row["ex"]
    row["img_api"] = draw(asmi, row["api_smarts"], API_RGBA)
    row["img_exc"] = draw(esmi, row["exc_smarts"], EXC_RGBA)

# ---------------------------------------------------------------- layout
W = 2900 if EXAMPLES else 2100
COLS = ([20, 100, 410, 940, 1250, 1690, 1890, 2880] if EXAMPLES
        else [20, 100, 420, 1000, 1380, 1860, 2080])    # #, flag, rule, measures, meaning, fires-on, example, end
HEAD_Y, HEAD_H = 90, 70
ROW_H = 250 if EXAMPLES else 215
TITLE_H = 90
FOOT_H = 120
H = TITLE_H + HEAD_H + ROW_H * len(ROWS) + FOOT_H + 10

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")

NAVY, INK, MUTED = "#1f4e79", "#1b1f27", "#667085"
SERIF, MONO = "DejaVu Serif", "DejaVu Sans Mono"
API_C, EXC_C = "#c2611b", "#1f66b8"


def T(x, y, s, size=13.5, color=INK, weight="normal", ha="left", va="center", family=SERIF, style="normal"):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, family=family, style=style)


def wrapped(x, y, s, chars, size=13.5, color=INK, line=27, family=SERIF):
    lines = textwrap.wrap(s, chars)
    y0 = y - (len(lines) - 1) * line / 2
    for i, ln in enumerate(lines):
        T(x, y0 + i * line, ln, size, color, family=family)


T(20, 40, "The 5 mechanism flags (one value per API–excipient pair)", 22, weight="bold", family="DejaVu Sans")
T(20, 72, "A flag is 1 only if the API matches one of its API patterns AND the excipient matches one of its "
          "excipient patterns; otherwise 0.", 13, MUTED, family="DejaVu Sans")

# header
y0 = TITLE_H
ax.add_patch(Rectangle((COLS[0], y0), COLS[-1] - COLS[0], HEAD_H, fc=NAVY, ec="none"))
heads = (["#", "Mechanism flag", "Rule (SMARTS patterns)", "What it measures", "Chemical meaning / purpose",
         f"Fires on\n({len(df):,} pairs)", "Example (real pair from the data)"] if EXAMPLES else
         ["#", "Mechanism flag", "Rule (SMARTS patterns)", "What it measures", "Chemical meaning / purpose",
          f"Fires on\n({len(df):,} pairs)"])
for i, hd in enumerate(heads):
    cx = (COLS[i] + COLS[i + 1]) / 2
    T(cx, y0 + HEAD_H / 2, hd, 15 if "\n" not in hd else 13, "white", "bold", "center", family="DejaVu Serif")

# rows
for r, row in enumerate(ROWS):
    top = TITLE_H + HEAD_H + r * ROW_H
    mid = top + ROW_H / 2
    ax.add_patch(Rectangle((COLS[0], top), COLS[-1] - COLS[0], ROW_H, fc=row["color"], ec="#d4d7dd", lw=1))
    for c in COLS[1:-1]:
        ax.plot([c, c], [top, top + ROW_H], color="#d4d7dd", lw=1)

    T((COLS[0] + COLS[1]) / 2, mid, str(r + 1), 20, weight="bold", ha="center")
    lines = textwrap.wrap(row["name"], 17)
    yy = mid - (len(lines) - 1) * 30 / 2
    for i, ln in enumerate(lines):
        T(COLS[1] + 20, yy + i * 30, ln, 16, weight="bold")
    T(COLS[1] + 20, yy + len(lines) * 30 + 4, f"flag index {r}", 11.5, MUTED, family="DejaVu Sans")

    # rule block
    ys = []
    for label, lst, colr in (("API", row["api_smarts"], API_C), ("EXC", row["exc_smarts"], EXC_C)):
        ys.append((label, lst, colr))
    n_lines = sum(len(lst) for _, lst, _ in ys) + 1
    ry = mid - (n_lines - 1) * 30 / 2 - 4
    for label, lst, colr in ys:
        T(COLS[2] + 20, ry, label, 13.5, colr, "bold", family=MONO)
        for k, patt in enumerate(lst):
            T(COLS[2] + 85, ry, ("" if k == 0 else "or ") + patt, 13, INK, family=MONO)
            ry += 30
        ry += 4
    T(COLS[2] + 20, ry + 2, "flag = API match  AND  EXC match", 11.5, MUTED, family="DejaVu Sans", style="italic")

    wrapped(COLS[3] + 20, mid, row["measures"], 27 if EXAMPLES else 34)
    wrapped(COLS[4] + 20, mid, row["meaning"], 42 if EXAMPLES else 44)

    # fires on
    cx = (COLS[5] + COLS[6]) / 2
    T(cx, mid - 42, f"{row['n']}", 26, weight="bold", ha="center")
    T(cx, mid - 6, f"pairs ({row['pct']:.1%})", 12.5, MUTED, ha="center", family="DejaVu Sans")
    T(cx, mid + 28, f"{row['inc']:.1%}", 18, weight="bold", ha="center",
      color="#b3361d" if row["inc"] > base else "#2a7f5f")
    T(cx, mid + 54, f"incompatible (all data: {base:.1%})", 10.5, MUTED, ha="center", family="DejaVu Sans")

    if not EXAMPLES:
        continue
    # example
    an, _, en, _, _, _ = row["ex"]
    x0 = COLS[6]
    ax.imshow(row["img_api"], extent=(x0 + 8, x0 + 258, top + ROW_H - 38, top + 38), aspect="auto", zorder=3)
    T(x0 + 274, mid - 10, "+", 22, "#7a8190", ha="center", family="DejaVu Sans")
    ax.imshow(row["img_exc"], extent=(x0 + 290, x0 + 540, top + ROW_H - 38, top + 38), aspect="auto", zorder=3)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    T(x0 + 22, top + ROW_H - 16, an, 12, API_C, "bold", family="DejaVu Sans")
    T(x0 + 300, top + ROW_H - 16, en, 12, EXC_C, "bold", family="DejaVu Sans")
    tx = x0 + 560
    T(tx, mid - 55, f"{an}", 15, weight="bold")
    T(tx, mid - 28, f"+ {en}", 15, weight="bold")
    flags = list(M[(df.API_CID == row["ex"][4]).values & (df.Excipient_CID == row["ex"][5]).values][0])
    T(tx, mid + 6, f"mech_flags = [{', '.join(str(int(f)) for f in flags)}]", 12.5, INK, family=MONO)
    T(tx, mid + 36, "labelled incompatible", 12.5, "#b3361d", family="DejaVu Sans")
    T(tx, mid + 60, "highlight: matched atoms", 11, MUTED, family="DejaVu Sans")

ytab = TITLE_H + HEAD_H + ROW_H * len(ROWS)
ax.plot([COLS[0], COLS[-1]], [ytab, ytab], color=NAVY, lw=2)
foot = ("Mechanism flags: 5 values per API–excipient pair (not per molecule), computed with RDKit from the SMILES "
        "(src/lookup.py, get_mechanism_matches) and fed to the PGB head, both into the gate network and into the "
        "final MLP. Examples are real pairs from start_dataset.csv with the matched atoms highlighted "
        "(orange = API pattern, blue = excipient pattern). Notes: the Schiff-base rule is broad (an amide N–H or "
        "an acetate C(=O)–C also matches), so it fires on about one pair in five and is only weakly "
        "associated with incompatibility; the Maillard, hydrolysis and metal-oxidation flags are rare (0.7–1.3% of pairs) and the acid–base flag fires on 6%.")
for i, ln in enumerate(textwrap.wrap(foot if EXAMPLES else foot.replace(' Examples are real pairs from start_dataset.csv with the matched atoms highlighted (orange = API pattern, blue = excipient pattern).', ''), 285 if EXAMPLES else 200)):
    T(COLS[0], ytab + 28 + i * 25, ln, 12, MUTED, family="DejaVu Serif")

fig.savefig(OUT, dpi=100, facecolor="white")
print("wrote", OUT)
for row in ROWS:
    print(f"{row['name']:32s} n={row['n']:4d} ({row['pct']:.1%})  incompatible={row['inc']:.1%}")
