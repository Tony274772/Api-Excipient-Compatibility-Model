"""Table figure of the 8 salt-context features (src/features.py: salt_context), in the style of the
descriptor table. Example values are computed with the model's own salt_context() on molecules
from the dataset.

Run:  python graphical_abstract/make_salt_context_table.py
Out:  graphical_abstract/salt_context_features.{html,png}
"""

import os
import subprocess
import sys

from PIL import Image, ImageChops
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "final_project"))
from src.features import salt_context  # noqa: E402

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUT = os.path.join(HERE, "salt_context_features")

ROWS = [
    # name, definition (html), measures, meaning, example name, smiles, feature index, value note
    ("Number of Fragments<br>(n_frag)",
     "n_frag = number of “.”-separated<br>fragments in the SMILES",
     "How many separate species the<br>molecule is written as",
     "Separates salts, mixtures and hydrates<br>from single neutral molecules.",
     "Magnesium stearate", "CCCCCCCCCCCCCCCCCC(=O)[O-].CCCCCCCCCCCCCCCCCC(=O)[O-].[Mg+2]", 0,
     "2 stearate + Mg²⁺"),
    ("Metal Atom Count<br>(n_metal)",
     "Σ atoms with element in {Li, Na, Mg,<br>Al, K, Ca, Ti, V, Cr, Mn, Fe, Co,<br>Ni, Cu, Zn}",
     "Number of metal atoms",
     "Metal ions can catalyse oxidation,<br>chelate APIs and form insoluble salts.",
     "Mg Al silicate", "[O-][Si]([O-])([O-])[O-].[Mg+2].[Al+3]", 1, "Mg + Al"),
    ("Inorganic Fragment<br>Count (n_inorg_frag)",
     "Count of fragments that contain<br>no carbon atom",
     "Number of inorganic pieces",
     "Flags inorganic counterions and<br>mineral excipients (phosphates,<br>sulfates, oxides).",
     "Calcium sulfate", "[O-]S(=O)(=O)[O-].[Ca+2]", 2, "SO₄²⁻ + Ca²⁺"),
    ("Total Formal Charge<br>(charge_total)",
     "Σ formal charge of all atoms<br>(over all fragments)",
     "Net charge of the whole<br>species",
     "0 for a balanced salt; non-zero for<br>charged species or unbalanced<br>records.",
     "Sodium benzoate", "C1=CC=C(C=C1)C(=O)[O-].[Na+]", 3, "(−1) + (+1)"),
    ("Maximum |Formal<br>Charge| (max_abs_charge)",
     "max |formal charge| over all atoms",
     "Highest charge on any<br>single atom",
     "Separates monovalent (Na⁺, K⁺)<br>from di- and trivalent ions<br>(Mg²⁺, Ca²⁺, Al³⁺).",
     "Mg Al silicate", "[O-][Si]([O-])([O-])[O-].[Mg+2].[Al+3]", 4, "Al³⁺"),
    ("Organic Heavy Atoms<br>(org_heavy)",
     "Σ heavy atoms in fragments that<br>contain carbon",
     "Size of the organic part",
     "Size of the drug or organic<br>excipient, without its counterion.",
     "Sodium lauryl sulfate", "CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]", 5, "lauryl sulfate ion"),
    ("Inorganic Heavy Atoms<br>(inorg_heavy)",
     "Σ heavy atoms in fragments<br>without carbon",
     "Size of the inorganic part",
     "Amount of inorganic material<br>(counterions, minerals, metal<br>oxides).",
     "Tricalcium phosphate", "[O-]P(=O)([O-])[O-].[O-]P(=O)([O-])[O-].[Ca+2].[Ca+2].[Ca+2]", 6,
     "2 × PO₄ (10) + 3 Ca"),
    ("Has Counterion<br>(has_counterion)",
     "1 if n_frag &gt; 1, else 0",
     "Whether the molecule is<br>a salt / multi-component",
     "Salt form vs free form: phenytoin<br>sodium = 1, phenytoin = 0.",
     "Phenytoin sodium", "C1=CC=C(C=C1)C2(C(=O)[N-]C(=O)N2)C3=CC=CC=C3.[Na+]", 7, "vs phenytoin = 0"),
]
FILLS = ["#FCF3DC", "#E6F2E1", "#FBE3E8", "#E1ECF7", "#ECE6F5", "#FCEBDB", "#E6F2E1", "#FBE3E8"]
SHORT = ["n_frag", "n_metal", "n_inorg_frag", "charge_total", "max_abs_charge", "org_heavy",
         "inorg_heavy", "has_counterion"]


def mol_svg(smiles, w=190, h=92):
    mol = Chem.MolFromSmiles(smiles)
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    o = d.drawOptions()
    o.clearBackground = False
    o.bondLineWidth = 1.6
    o.minFontSize = 11
    o.padding = 0.06
    d.DrawMolecule(mol)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    return svg[svg.index("<svg"):]


def fmt(v):
    v = int(v)
    return f"−{abs(v)}" if v < 0 else str(v)


rows_html = []
for i, (name, defin, meas, mean, ex_name, smi, idx, note) in enumerate(ROWS):
    val = salt_context(smi)[idx]
    rows_html.append(f"""
<tr style="background:{FILLS[i]}">
  <td class="num">{i + 1}</td>
  <td class="name">{name}</td>
  <td class="def">{defin.replace('<br>', ' ')}</td>
  <td class="meas">{meas.replace('<br>', ' ')}</td>
  <td class="mean">{mean.replace('<br>', ' ')}</td>
  <td class="ex"><div class="exwrap"><div class="mol">{mol_svg(smi)}</div>
      <div class="exlab">{ex_name}<br><b>{SHORT[idx]} = {fmt(val)}</b><br><span class="note">{note}</span></div></div></td>
</tr>""")

html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
body {{ margin:0; background:#fff; font-family: Cambria, Georgia, 'Times New Roman', serif; color:#1b1b1b; }}
.wrap {{ padding:6px; width:1560px; }}
table {{ border-collapse:separate; border-spacing:0; width:100%; border:1.5px solid #c9cfd6;
         border-radius:8px; overflow:hidden; font-size:18.5px; line-height:1.28; }}
th {{ background:#1f4f72; color:#fff; font-weight:bold; font-size:20px; padding:10px 8px;
      border-right:1px solid #5f86a3; }}
th:last-child, td:last-child {{ border-right:none; }}
td {{ padding:9px 12px; border-right:1px solid #c9cfd6; border-top:1px solid #c9cfd6; vertical-align:middle; }}
td.num {{ text-align:center; font-weight:bold; font-size:20px; width:34px; }}
td.name {{ font-weight:bold; font-size:19.5px; width:215px; }}
td.def {{ width:290px; }}
td.meas {{ width:230px; }}
td.mean {{ width:330px; }}
td.ex {{ width:370px; padding:4px 8px; }}
.exwrap {{ display:flex; align-items:center; gap:6px; }}
.mol svg {{ display:block; }}
.exlab {{ font-size:17px; line-height:1.3; }}
.note {{ color:#555; font-size:15px; }}
.foot {{ font-size:14.5px; color:#4a5563; padding:8px 4px 2px; line-height:1.4; }}
</style></head><body><div class="wrap">
<table>
<tr><th>#</th><th>Feature Name</th><th>Definition / Formula</th><th>What it Measures</th>
<th>Chemical Meaning / Purpose</th><th>Example / Illustration</th></tr>
{''.join(rows_html)}
</table>
<div class="foot">Salt context: 8 values per molecule (API and excipient separately), computed with RDKit from the SMILES
(src/features.py, <i>salt_context</i>) and fed into each molecule tower. Example values are the model's own output for
these SMILES. Notes: water in a hydrate counts as a fragment (lactose monohydrate → has_counterion = 1); HCl salts written
as “.Cl” are neutral fragments (metformin HCl → charge_total = 0).</div>
</div></body></html>"""

with open(OUT + ".html", "w", encoding="utf-8") as f:
    f.write(html)

png = OUT + ".png"
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                "--window-size=1572,1400", f"--screenshot={png}", "file:///" + (OUT + ".html").replace("\\", "/")],
               check=True, capture_output=True)
im = Image.open(png).convert("RGB")
bbox = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
im = im.crop((0, 0, im.size[0], min(im.size[1], bbox[3] + 16)))
im.save(png, dpi=(300, 300))
print("saved", png, im.size)
