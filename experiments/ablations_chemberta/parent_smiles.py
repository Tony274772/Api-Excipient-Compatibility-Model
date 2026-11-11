"""`chemberta_parent`: ChemBERTa reads the parent molecule instead of the dataset SMILES.

ChemBERTa-77M-MTR was pretrained without salts, charges or stereochemistry, so the tokens for
'.', bracket atoms ([Na+], [O-], [C@@H]) and '/', '\\' were never trained (audit_pretrained.py).
The parent molecule removes as many of those as the chemistry allows:

    1. split into fragments, drop water
    2. keep the largest ORGANIC fragment (most heavy atoms among fragments with carbon);
       a purely inorganic molecule (MgO, talc, CaHPO4) keeps its dataset string unchanged
    3. neutralise charges where a neutral form exists (RDKit Uncharger: carboxylate -> acid,
       ammonium -> amine); permanent charges such as quaternary N+ stay
    4. write it without stereochemistry (isomericSmiles=False), so @, @@, / and \\ disappear

Cl and Br inside the parent stay: they are part of the molecule. So this fixes the salt / charge /
stereo part of the "45% unreadable" problem, not the halogen part (token_subsets.py reports both).

Only the ENCODER sees the parent. The salt context, chemistry flags, descriptors, PubChem bits,
excipient prior and mechanism flags still read the original SMILES, so the salt information is not
lost from the model, only from ChemBERTa.
"""

from functools import lru_cache

import torch.nn as nn
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


# characters whose ChemBERTa tokens were never trained and that the parent can remove
FIXABLE_CHARS = (".", "[", "@", "/", "\\")
# untrained tokens the parent cannot remove: halogens are part of the molecule
HALOGENS = ("Cl", "Br")


def needs_parent(smiles: str) -> bool:
    return any(c in smiles for c in FIXABLE_CHARS)


def _is_water(frag) -> bool:
    return frag.GetNumAtoms() == 1 and frag.GetAtomWithIdx(0).GetSymbol() == "O"


@lru_cache(maxsize=None)
def parent_smiles(smiles: str) -> str:
    """Parent molecule of `smiles` as a SMILES string (see the module docstring)."""
    s = str(smiles).strip()
    if not s or not needs_parent(s):
        # nothing ChemBERTa cannot read except halogens: keep the dataset string exactly, so a
        # re-canonicalised SMILES does not change the input of molecules this ablation is not about
        return s
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return s
    frags = [f for f in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True) if not _is_water(f)]
    if not frags:
        return s
    organic = [f for f in frags if any(a.GetAtomicNum() == 6 for a in f.GetAtoms())]
    if not organic:
        return s          # purely inorganic: a fragment of it would only lose the counter-ion
    frag = max(organic, key=lambda f: (f.GetNumHeavyAtoms(), Chem.MolToSmiles(f)))
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        frag = rdMolStandardize.Uncharger().uncharge(frag)
    except Exception:
        pass
    out = Chem.MolToSmiles(frag, isomericSmiles=False)
    return out or s


class ParentSmilesEncoder(nn.Module):
    """Wraps a sequence encoder so it encodes parent_smiles(s) for every SMILES it is given.
    Same output_dim and is_sequence_capable as the wrapped encoder; still frozen."""

    def __init__(self, encoder):
        super().__init__()
        self.inner = encoder
        self.output_dim = encoder.output_dim
        self.is_sequence_capable = encoder.is_sequence_capable

    def encode(self, smiles_batch):
        return self.inner.encode([parent_smiles(s) for s in smiles_batch])

    def train(self, mode: bool = True):
        super().train(mode)
        self.inner.train(mode)        # ChemBERTaEncoder.train keeps its transformer in eval mode
        return self

    def to(self, device, *args, **kwargs):
        self.inner.to(device, *args, **kwargs)
        return super().to(device, *args, **kwargs)


def parent_change_report(molecules) -> dict:
    """How many of `molecules` ChemBERTa now reads differently, for run_info.json."""
    mols = sorted(set(str(m).strip() for m in molecules if str(m).strip()))
    changed = [m for m in mols if parent_smiles(m) != m]
    return {"molecules": len(mols), "changed": len(changed),
            "examples": [[m, parent_smiles(m)] for m in changed[:5]]}
