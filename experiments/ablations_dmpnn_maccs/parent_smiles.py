"""`dmpnn_parent`: D-MPNN reads the desalted parent molecule for the graph structure.

CheMeleon pretraining was conducted on single-fragment organic graphs, leaving degree-0 atoms
(isolated ions such as [Na+], [Ca+2], [Mg+2]) unbonded.
The parent molecule extraction strips disconnected solvent/water and isolated counterions:
    1. split into fragments, drop water
    2. keep the largest ORGANIC fragment (most heavy atoms among fragments with carbon);
       a purely inorganic molecule (MgO, talc, CaHPO4) keeps its dataset string unchanged
    3. neutralise charges where a neutral form exists (carboxylates -> acid, ammonium -> amine)
    4. return standard SMILES

Only the D-MPNN GRAPH ENCODER sees the parent structure. The salt context, chemistry flags,
descriptors, MACCS keys, excipient prior and mechanism flags still read the original SMILES,
so no salt information is lost from the model.
"""

from functools import lru_cache

import torch.nn as nn
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def needs_parent(smiles: str) -> bool:
    """Check if SMILES contains multi-fragment dot separator."""
    return "." in str(smiles)


def _is_water(frag) -> bool:
    return frag.GetNumAtoms() == 1 and frag.GetAtomWithIdx(0).GetSymbol() == "O"


@lru_cache(maxsize=None)
def parent_smiles(smiles: str) -> str:
    """Parent molecule of `smiles` as a clean SMILES string."""
    s = str(smiles).strip()
    if not s or not needs_parent(s):
        return s
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return s
    frags = [f for f in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True) if not _is_water(f)]
    if not frags:
        return s
    organic = [f for f in frags if any(a.GetAtomicNum() == 6 for a in f.GetAtoms())]
    if not organic:
        return s
    frag = max(organic, key=lambda f: (f.GetNumHeavyAtoms(), Chem.MolToSmiles(f)))
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        frag = rdMolStandardize.Uncharger().uncharge(frag)
    except Exception:
        pass
    out = Chem.MolToSmiles(frag, isomericSmiles=True)
    return out or s


class ParentSmilesEncoder(nn.Module):
    """Wrapper around a graph encoder (e.g. D-MPNN) that translates each SMILES through
    parent_smiles before encoding."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        for attr in ("d_model", "dim", "hidden_dim", "output_dim"):
            if hasattr(encoder, attr):
                setattr(self, attr, getattr(encoder, attr))

    def forward(self, smiles_list):
        translated = [parent_smiles(s) for s in smiles_list]
        return self.encoder(translated)


def parent_change_report(all_smiles) -> dict:
    """Count how many molecules change under parent_smiles."""
    smiles_set = set(all_smiles)
    changed = {s: parent_smiles(s) for s in smiles_set if parent_smiles(s) != s}
    return {
        "molecules": len(smiles_set),
        "changed": len(changed),
        "examples": list(changed.items())[:5],
    }

