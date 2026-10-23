"""
Mechanism flags v2: rule-based API-excipient reaction flags.

Replaces the 5 flags of src/lookup.py (v1) with 13 pair-level rules. Each rule fires when the API has
a reactive group AND the excipient has the matching partner (or the reverse, where stated). The rules
are written from the standard API-excipient incompatibility classes in the pharmaceutical literature,
e.g. Narang, Desai & Badawy (2012) "Impact of excipient interactions on drug bioavailability ...",
Bharate, Bharate & Bajaj (2010) "Interactions and incompatibilities of pharmaceutical excipients with
active pharmaceutical ingredients", Wasylaschuk et al. (2007) on peroxides in excipients, and the
Handbook of Pharmaceutical Excipients. They were checked on train / val only (ablations/check_mech_v2.py).

What changed against v1:
  - Maillard: v1 looked for an open-chain aldehyde or a gem-diol, so it missed every reducing sugar written
    as a ring (lactose, glucose, maltose). v2 finds the free anomeric OH (hemiacetal) and the open-chain
    aldose / ketose, and ignores non-reducing sugars (sucrose, trehalose) and sugar alcohols (mannitol).
  - Schiff base: v1 matched any carbonyl (stearates, esters, triacetin) and fired on 20 % of all pairs.
    v2 only uses aldehydes that are not sugars (formaldehyde, vanillin, HMF, glyceraldehyde ...).
  - Hydrolysis: v1 required a free hydroxide ion. v2 uses alkaline excipients (oxides, hydroxides,
    carbonates, tribasic phosphates, metal carboxylates such as Mg stearate, amine bases).
  - Acid-base: v1 only matched carboxylic acids with Na / K / Ca / Mg anywhere in the excipient, including
    neutral salts. v2 needs an alkaline excipient, covers more acidic groups and adds weak NH acids.
  - Metal-ion oxidation: unchanged idea, more oxidisable groups and more redox metals.
  - Metal-ion oxidation (v1) is not used: it never fired on an incompatible train / val pair (see RULES_NOT_USED).
  - New: salt disproportionation, acid-catalysed degradation, metal chelation, metal-catalysed beta-lactam
    hydrolysis, strong oxidisers, sulfite addition, ion-pair precipitation with cationic surfactants,
    aluminosilicate (clay) surface interactions, weak NH-acid deprotonation, carbonyl-amine condensation.
  - Considered and left out (no support on train / val): metal-catalysed oxidation, peroxide oxidation,
    Michael addition, amine acylation.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

from src.dataset import PGBDataset

RDLogger.DisableLog("rdApp.*")


# --------------------------------------------------------------------------- SMARTS
def _p(smarts):
    pat = Chem.MolFromSmarts(smarts)
    assert pat is not None, smarts
    return pat


_NOT_ACYL = "!$(N[#6]=[#7,#8,#16]);!$(N[!#6;!#1]);!$(N[CX3]=[CX3])"   # not amide / urea / amidine /
                                                                          # sulfonamide / N-hetero / enamine
SMARTS = {
    # nucleophilic nitrogen and sulfur
    "prim_amine":      [_p(f"[NX3;H2;{_NOT_ACYL};!$(Nc:n)]"), _p("[NX4+;H3][#6]")],
    "sec_amine":       [_p(f"[NX3;H1;{_NOT_ACYL};!$(N[a])]([#6])[#6]"), _p("[NX4+;H2]([CX4])[CX4]")],
    "tert_amine":      [_p(f"[NX3;H0;{_NOT_ACYL};!$(N[a])]([CX4])([CX4])[CX4]"), _p("[NX4+;H1]([CX4])([CX4])[CX4]")],
    "hydrazine":       [_p("[NX3;H2][NX3]")],
    "amidine":         [_p("[NX3;!$(N[!#6;!#1])][CX3;!a]=[NX2;!a]"), _p("[NX3][CX3;!a]=[NX3+]")],
    "quat":            [_p("[NX4+;H0;!$([N+][O-]);!$([N+]~[!#6])]([#6])([#6])([#6])[#6]")],
    "thiol":           [_p("[SX2H1]")],
    "thioether":       [_p("[#6][S;X2;!a][#6]")],
    "phenothiazine":   [_p("c1ccc2c(c1)Sc1ccccc1N2")],
    # carbonyl and acyl groups
    "aldehyde":        [_p("[CX3H1](=O)[#6]"), _p("[CX3H2]=O")],
    "michael_acceptor": [_p("[CX3;!a]=[CX3;!a][CX3;!a]=O")],
    "acrylic_fumaric": [_p("[CH2]=[CX3][CX3](=O)[O,N]"), _p("[CX3H1]([CX3](=O)[O,N])=[CX3H1][CX3](=O)[O,N]")],
    "ester":           [_p("[#6][CX3](=O)[OX2][#6]")],
    "acetate_ester":   [_p("[CH3][CX3](=O)[OX2][#6]")],
    "anhydride":       [_p("[CX3](=O)[OX2][CX3](=O)")],
    "beta_lactam":     [_p("[#6;r4](=O)[#7;r4]")],
    # acids
    "carboxylic_acid": [_p("[CX3](=O)[OX2H1]")],
    "carboxylate":     [_p("[CX3](=O)[OX1-]")],
    "sulfonic_acid":   [_p("[SX4](=O)(=O)[OX2H1]")],
    "sulfonate":       [_p("[SX4](=O)(=O)[OX1-]")],
    "phosphonic_acid": [_p("[PX4](=O)[OX2H1]")],
    "tetrazole_nh":    [_p("c1nn[nH]n1"), _p("c1nnn[nH]1")],
    "acyl_sulfonamide": [_p("[CX3](=O)[NX3H1][SX4](=O)=O")],
    "sulfonylurea":    [_p("[SX4](=O)(=O)[NX3H1][CX3](=O)[NX3]")],
    "vinylogous_acid": [_p("[OX2H1][CX3]=[CX3][CX3]=O"), _p("[OX2H1]c:c:c=O")],
    "imide":           [_p("[#6](=O)[NX3H1][#6](=O)"), _p("[#6](=O)[nH][#6](=O)")],
    "sulfonamide_nh":  [_p("[SX4](=O)(=O)[NX3;H2,H1;!$(N[CX3]=O)]")],
    "acidic_anion":    [_p("[#7-;!$([N-][N+]);!$([N-]=*)]")],
    # acid-labile groups
    "acetal":          [_p("[CX4]([OX2][CX4])[OX2][#6]")],
    "epoxide":         [_p("[CX4;r3]1[OX2;r3][CX4;r3]1")],
    "enol_ether":      [_p("[CX3;!a]=[CX3;!a][OX2][#6;!$(C=O)]")],
    "ppi_sulfinyl":    [_p("[SX3](=O)([CX4])c1nc2ccccc2[n;H1,-1]1")],
    "imine":           [_p("[CX3;!a;!$(C[#7,#8,#16])]=[NX2;!a]")],
    # oxidisable groups
    "catechol":        [_p("c([OX2H1])c[OX2H1]")],
    "hydroquinone":    [_p("[OX2H1]c1ccc([OX2H1])cc1")],
    "aminophenol":     [_p("[OX2H1]c1ccc([NX3;H2,H1])cc1"), _p("[OX2H1]c:c[NX3;H2,H1]")],
    "enediol":         [_p("[OX2H1][CX3]=[CX3][OX2H1]")],
    "polyene":         [_p("[CX3;!a]=[CX3;!a][CX3;!a]=[CX3;!a][CX3;!a]=[CX3;!a]")],
    "phenol":          [_p("c[OX2H1]")],
    "alcohol":         [_p("[CX4][OX2H1]")],
    "alkene":          [_p("[CX3;!a]=[CX3;!a]")],
    # metal chelators
    "beta_diketone":   [_p("[#6][CX3](=O)[CX4;H1,H2][CX3](=O)[#6]")],
    "keto_acid":       [_p("[#6;R](=O)~[#6;R]-[CX3](=O)[O;H1,-1]")],
    "o_hydroxy_carbonyl": [_p("[OX2H1]c:c-,:[#6]=O")],
    "alpha_hydroxy_ring_carbonyl": [_p("[OX2H1][#6;R]=,:[#6;R]=O"), _p("[OX2H1][#6;R]=,:[#6;R][#6;R]=O")],
    "hydroxamic_acid": [_p("[CX3](=O)[NX3][OX2H1]")],
    "bisphosphonate":  [_p("P(=O)~[CX4]~P(=O)")],
    "hydroxyquinoline": [_p("[OX2H1]c1cccc2cccnc12")],
    "aminopolycarboxylate": [_p("[NX3]([CH2][CX3](=O)[O;H1,-1])[CH2][CX3](=O)[O;H1,-1]")],
    # excipient-side inorganic chemistry
    "hydroxide_ion":   [_p("[OX1H1-]")],
    "oxide_ion":       [_p("[O-2]")],
    "basic_metal_oxide": [_p("[#8;!$([#8][#6]);!$([#8][#1])]~[Mg,Ca,Zn,Ba,Sr,Na,K,Li]"),
                          _p("[OX2H1][Mg,Ca,Zn,Ba,Sr,Na,K,Li]")],
    "carbonate":       [_p("[CX3;!$(C~[#6])](=O)([OX1-,OX2H1])[OX1-]")],
    "tribasic_phosphate": [_p("[PX4](=O)([OX1-])([OX1-])[OX1-]")],
    "phosphate_oh":    [_p("[PX4](=O)[OX2H1]")],
    "sulfite":         [_p("[SX3](=O)([OX2H1,OX1-])[OX2H1,OX1-]"), _p("[SX4](=O)(=O)[SX3](=O)")],
    "nitric":          [_p("[NX3+](=O)([OX1-])[OX1-,OX2H1]")],
    "oxygen_oxygen":   [_p("[#8]~[#8]")],
    "permanganate_chromate": [_p("[Mn,Cr](=O)(=O)=O"), _p("[Mn,Cr](=O)(=O)[OX1-]")],
    "halogen_oxyanion": [_p("[Cl,Br,I](=O)=O"), _p("[Cl,Br,I][OX1-]"), _p("[Cl,Br,I][OX2H1]")],
    "n_vinyl_lactam":  [_p("O=C1CCCN1[#6]")],
    "polyether_unit":  [_p("[OX2]([#6])[CH2][CH2][OX2]")],
}

ALKALI_ALKALINE_EARTH = {"Li", "Na", "K", "Mg", "Ca", "Ba", "Sr", "Zn"}
POLYVALENT_METALS = {"Mg", "Ca", "Al", "Zn", "Fe", "Cu", "Mn", "Bi", "Ba", "Sr", "Cr", "Co", "Ni"}
REDOX_METALS = {"Fe", "Cu", "Mn", "Cr", "Co", "Ni", "V", "Ce"}


# --------------------------------------------------------------------------- molecule properties
def _mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:                                   # odd inorganic notation: try without sanitising
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is not None:
            try:
                mol.UpdatePropertyCache(strict=False)
                Chem.FastFindRings(mol)
            except Exception:
                return None
    return mol


def _has(mol, name):
    return any(mol.HasSubstructMatch(p) for p in SMARTS[name])


def _count(mol, name):
    return sum(len(mol.GetSubstructMatches(p)) for p in SMARTS[name])


def _frags(mol):
    """Fragments of a molecule, with ring info set so ring SMARTS work on unsanitised molecules too."""
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    for f in frags:
        f.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(f)
    return frags


def _is_water(frag):
    return frag.GetNumAtoms() == 1 and frag.GetAtomWithIdx(0).GetSymbol() == "O" and frag.GetAtomWithIdx(0).GetFormalCharge() == 0


def _reducing_sugar(mol):
    """Free hemiacetal (anomeric C with ring O and OH) or open-chain aldose / ketose, in a molecule that is
    only the sugar (water of hydration allowed). Polymer surrogates written as sugar + other monomers
    (e.g. sugar + acetic acid + Na for carboxymethylcellulose) do not count; neither do sucrose,
    trehalose (no free anomeric OH) or sugar alcohols (no anomeric carbon)."""
    frags = [f for f in _frags(mol) if not _is_water(f)]
    if len(frags) != 1:
        return False
    f = frags[0]
    hemiacetal = f.HasSubstructMatch(_p("[OX2H1][CX4;R][OX2;R]"))
    aldose = f.HasSubstructMatch(_p("[CX3H1](=O)[CX4][OX2H1]"))
    ketose = f.HasSubstructMatch(_p("[OX2H1][CH2][CX3](=O)[CX4][OX2H1]"))
    polyol = len(f.GetSubstructMatches(_p("[CX4][OX2H1]"))) >= 2
    return polyol and (hemiacetal or aldose or ketose)


def _reactive_aldehyde(mol):
    """Aldehyde in a fragment that is not a sugar (fewer than 3 OH groups)."""
    for f in _frags(mol):
        if _has(f, "aldehyde") and len(f.GetSubstructMatches(_p("[CX4][OX2H1]"))) < 3:
            return True
    return False


def _small_organic_acid(mol, elements):
    """A free carboxylic (or vinylogous) acid fragment of at most 15 heavy atoms, with no metal in the
    excipient and no basic amine in the fragment (citric, tartaric, malic, succinic, fumaric, ascorbic,
    acrylic, benzoic acid ...; long-chain fatty acids such as stearic acid are left out)."""
    if elements & (ALKALI_ALKALINE_EARTH | POLYVALENT_METALS):
        return False
    for f in _frags(mol):
        if f.GetNumHeavyAtoms() <= 15 and (_has(f, "carboxylic_acid") or _has(f, "vinylogous_acid")) \
                and not (_has(f, "prim_amine") or _has(f, "sec_amine") or _has(f, "tert_amine")):
            return True
    return False


def _counter_anion(mol):
    """A small acid / halide fragment next to the main molecule (HCl, HBr, mesylate, maleate, sulfate ...)."""
    frags = _frags(mol)
    if len(frags) < 2:
        return False
    for f in frags:
        if f.GetNumHeavyAtoms() == 1 and f.GetAtomWithIdx(0).GetSymbol() in ("Cl", "Br", "I"):
            return True
        if f.GetNumHeavyAtoms() <= 15 and (_has(f, "carboxylic_acid") or _has(f, "sulfonic_acid")
                                           or _has(f, "sulfonate") or _has(f, "phosphate_oh")):
            return True
    return False


@lru_cache(maxsize=20000)
def properties(smiles):
    """Set of reactive-group / class names for one molecule (API or excipient)."""
    if not isinstance(smiles, str) or not smiles:
        return frozenset()
    mol = _mol(smiles)
    if mol is None:
        return frozenset()
    elements = {a.GetSymbol() for a in mol.GetAtoms()}
    props = {name for name in SMARTS if _has(mol, name)}
    organic = "C" in elements

    # --- composite classes
    if props & {"prim_amine", "sec_amine", "tert_amine", "amidine", "quat"} or mol.HasSubstructMatch(
            _p("[NX4+;!$([N+][O-]);H1,H2,H3]")):
        props.add("basic_n")
    if props & {"carboxylic_acid", "sulfonic_acid", "phosphonic_acid", "tetrazole_nh", "acyl_sulfonamide",
                "sulfonylurea", "vinylogous_acid"}:
        props.add("acid_group")
    if props & {"imide", "sulfonamide_nh"}:
        props.add("weak_nh_acid")
    if props & {"acid_group", "carboxylate", "sulfonate", "acidic_anion"}:
        props.add("anionic")
    if props & {"acetal", "epoxide", "enol_ether", "ppi_sulfinyl", "imine", "beta_lactam"}:
        props.add("acid_labile")
    if props & {"catechol", "hydroquinone", "aminophenol", "thiol", "enediol", "phenothiazine", "aldehyde",
                "hydrazine", "polyene"}:
        props.add("easily_oxidised")
    if props & {"tert_amine", "sec_amine", "thioether", "thiol", "phenothiazine", "hydrazine", "aldehyde"}:
        props.add("peroxide_target")
    if organic and props & {"prim_amine", "sec_amine", "tert_amine", "alcohol", "phenol", "thioether", "thiol",
                            "aldehyde", "alkene", "hydrazine", "phenothiazine"}:
        props.add("oxidisable")
    if props & {"catechol", "beta_diketone", "vinylogous_acid", "keto_acid", "o_hydroxy_carbonyl",
                "alpha_hydroxy_ring_carbonyl", "hydroxamic_acid", "bisphosphonate", "hydroxyquinoline",
                "aminopolycarboxylate"}:
        props.add("chelator")
    if props & {"anhydride", "acetate_ester"}:
        props.add("acyl_donor")
    if elements & POLYVALENT_METALS:
        props.add("polyvalent_metal")
        # metal ions available to a drug: not locked in a silicate lattice, not an iron-oxide pigment
        iron_pigment = elements & POLYVALENT_METALS == {"Fe"} and not elements & {"C", "S", "P"}
        if "Si" not in elements and not iron_pigment:
            props.add("available_metal_ion")
    if elements & REDOX_METALS:
        props.add("redox_metal")
    if {"Al", "Si"} <= elements:
        props.add("aluminosilicate")
    if _reducing_sugar(mol):
        props.add("reducing_sugar")
    if _reactive_aldehyde(mol):
        props.add("reactive_aldehyde")
    if "quat" in props and not props & {"carboxylate", "sulfonate", "phosphate_oh"} and not mol.HasSubstructMatch(
            _p("[PX4][OX1-]")):
        props.add("cationic_surfactant")          # quaternary ammonium, not a zwitterion such as lecithin

    # --- alkaline / acidic excipient classes
    metal_base = elements & ALKALI_ALKALINE_EARTH
    ammonium = mol.HasSubstructMatch(_p("[NX4+;H4]"))
    # alkalising amines used as excipients are amino alcohols (meglumine, tromethamine, triethanolamine)
    amine_base = (props & {"prim_amine", "sec_amine", "tert_amine"} and "alcohol" in props
                  and not mol.GetAromaticAtoms() and not props & {"acid_group", "anionic"}
                  and not _counter_anion(mol))
    strong_base = (("hydroxide_ion" in props and (metal_base or ammonium)) or "carbonate" in props
                   or "tribasic_phosphate" in props
                   or (metal_base and ("oxide_ion" in props or "basic_metal_oxide" in props)) or bool(amine_base))
    # metal soaps / carboxylate salts (Mg / Ca / Zn stearate, sodium acetate ...): every organic fragment is a
    # carboxylate. Polymer surrogates such as sugar + acetate + Na (croscarmellose, pH 5-7) are left out.
    metal_carboxylate = metal_base and "carboxylate" in props and all(
        f.HasSubstructMatch(SMARTS["carboxylate"][0]) for f in _frags(mol)
        if any(a.GetSymbol() == "C" for a in f.GetAtoms()))
    if strong_base:
        props.add("strong_alkaline")
    if strong_base or metal_carboxylate:
        props.add("alkaline")
    n_poh = _count(mol, "phosphate_oh")
    if ("sulfonic_acid" in props or ("nitric" in props and mol.HasSubstructMatch(_p("[NX3+](=O)[OX2H1]")))
            or n_poh >= 2 or (n_poh == 1 and not metal_base)
            or ("sulfite" in props and mol.HasSubstructMatch(_p("[SX3](=O)[OX2H1]")))
            or _small_organic_acid(mol, elements)
            or any(f.GetNumHeavyAtoms() == 1 and f.GetAtomWithIdx(0).GetSymbol() in ("Cl", "Br")
                   and f.GetAtomWithIdx(0).GetFormalCharge() == 0 and not metal_base
                   for f in _frags(mol))):
        props.add("acidic")

    # --- salts of the API
    if "basic_n" in props and (_counter_anion(mol) or mol.HasSubstructMatch(_p("[NX4+;H1,H2,H3]"))):
        props.add("base_salt")                    # e.g. amine hydrochloride, mesylate, maleate
    if organic and metal_base and (props & {"carboxylate", "acidic_anion"}
                                   or mol.HasSubstructMatch(_p("[#6][OX1-,NX2-]"))):
        props.add("weak_acid_salt")               # e.g. phenytoin sodium, ibuprofen sodium

    # --- oxidisers and peroxide sources
    if props & {"oxygen_oxygen", "permanganate_chromate", "halogen_oxyanion"} or (
            "nitric" in props and not organic):
        props.add("strong_oxidiser")
    if "n_vinyl_lactam" in props or _count(mol, "polyether_unit") >= 2 or "oxygen_oxygen" in props:
        props.add("peroxide_source")              # PVP / crospovidone / copovidone, PEG / polysorbate, peroxides
    return frozenset(props)


# --------------------------------------------------------------------------- rules
def _any(props, *names):
    return any(n in props for n in names)


RULES = [
    ("maillard",
     "primary / secondary amine or hydrazine API + reducing sugar excipient",
     lambda A, E: _any(A, "prim_amine", "sec_amine", "hydrazine") and "reducing_sugar" in E),
    ("carbonyl_amine_condensation",
     "primary amine / hydrazine + non-sugar aldehyde (either side): imine / hydrazone formation",
     lambda A, E: (_any(A, "prim_amine", "hydrazine") and "reactive_aldehyde" in E)
     or ("reactive_aldehyde" in A and _any(E, "prim_amine", "hydrazine"))),
    ("base_catalysed_hydrolysis",
     "ester / lactone / beta-lactam / anhydride API + alkaline excipient",
     lambda A, E: _any(A, "ester", "beta_lactam", "anhydride") and "alkaline" in E),
    ("acid_base_neutralisation",
     "acidic API (carboxylic, sulfonic, phosphonic, tetrazole, acyl-sulfonamide, sulfonylurea, vinylogous acid) "
     "+ alkaline excipient",
     lambda A, E: "acid_group" in A and "alkaline" in E),
    ("weak_acid_deprotonation",
     "weak NH acid API (imide / hydantoin / barbiturate, sulfonamide) + strongly alkaline excipient",
     lambda A, E: "weak_nh_acid" in A and "strong_alkaline" in E),
    ("salt_disproportionation",
     "salt of a basic API + strongly alkaline excipient, or alkali salt of an acidic API + acidic excipient",
     lambda A, E: ("base_salt" in A and "strong_alkaline" in E) or ("weak_acid_salt" in A and "acidic" in E)),
    ("acid_catalysed_degradation",
     "acid-labile API (acetal / glycoside, epoxide, enol ether, imine, beta-lactam, PPI sulfinyl-benzimidazole) "
     "+ acidic excipient",
     lambda A, E: "acid_labile" in A and "acidic" in E),
    ("metal_chelation",
     "chelating API (catechol, beta-diketone / enol, 4-oxo-3-carboxylic acid, o-hydroxy aryl carbonyl, "
     "hydroxamic acid, bisphosphonate, 8-hydroxyquinoline) + di- / trivalent metal excipient, or EDTA-type "
     "excipient + metal-containing API",
     lambda A, E: ("chelator" in A and "available_metal_ion" in E)
     or ("aminopolycarboxylate" in E and "polyvalent_metal" in A)),
    ("metal_catalysed_beta_lactam_hydrolysis",
     "beta-lactam API + di- / trivalent metal ion excipient",
     lambda A, E: "beta_lactam" in A and "available_metal_ion" in E),
    ("strong_oxidiser",
     "oxidisable API + strong oxidiser (peroxide, ozone, O2, permanganate, chromate, chlorate, hypochlorite, "
     "nitrate)",
     lambda A, E: "oxidisable" in A and "strong_oxidiser" in E),
    ("sulfite_addition",
     "aldehyde / Michael acceptor / catechol API + sulfite, bisulfite or metabisulfite excipient",
     lambda A, E: _any(A, "aldehyde", "michael_acceptor", "catechol") and "sulfite" in E),
    ("ion_pair_precipitation",
     "anionic API + cationic surfactant (quaternary ammonium, e.g. benzalkonium, cetrimide)",
     lambda A, E: "anionic" in A and "cationic_surfactant" in E),
    ("aluminosilicate_surface",
     "basic-nitrogen or beta-lactam API + aluminosilicate (clays, Mg / Na Al silicates): adsorption and "
     "surface-catalysed degradation",
     lambda A, E: _any(A, "basic_n", "beta_lactam") and "aluminosilicate" in E),
]

# Written from the literature but left out of the flags: on train + val they fire only on compatible pairs
# or no more often on incompatible pairs than the base rate (check_mech_v2.py --unused), so they would add
# false alarms. Iron-oxide pigments and povidone / PEG behave as inert in this dataset.
RULES_NOT_USED = [
    ("metal_catalysed_oxidation",
     "easily oxidised API (catechol, hydroquinone, aminophenol, thiol, enediol, phenothiazine, aldehyde, "
     "hydrazine, polyene) + redox-active metal excipient (Fe, Cu, Mn, Cr, Co, Ni)",
     lambda A, E: "easily_oxidised" in A and "redox_metal" in E),
    ("peroxide_oxidation",
     "API with tertiary / secondary amine, thioether, thiol, phenothiazine, hydrazine or aldehyde "
     "+ peroxide-carrying excipient (povidone / crospovidone, PEG / polysorbate, peroxides)",
     lambda A, E: "peroxide_target" in A and "peroxide_source" in E),
    ("michael_addition",
     "primary / secondary amine or thiol API + acrylate, methacrylate, fumarate or maleate excipient",
     lambda A, E: _any(A, "prim_amine", "sec_amine", "thiol") and "acrylic_fumaric" in E),
    ("amine_acylation",
     "primary / secondary amine API + acyl donor excipient (anhydride, acetate ester such as triacetin)",
     lambda A, E: _any(A, "prim_amine", "sec_amine") and "acyl_donor" in E),
]

MECH_V2_NAMES = [name for name, _, _ in RULES]
NUM_MECH_V2 = len(RULES)


@lru_cache(maxsize=50000)
def _flags(api_smiles, exc_smiles):
    A, E = properties(api_smiles), properties(exc_smiles)
    return tuple(int(bool(rule(A, E))) for _, _, rule in RULES)


def mechanism_flags_v2(api_smiles, exc_smiles):
    """Binary vector [NUM_MECH_V2] of mechanism flags for one API-excipient pair."""
    if not exc_smiles:
        return np.zeros(NUM_MECH_V2, dtype=np.float32)
    return np.array(_flags(api_smiles, exc_smiles), dtype=np.float32)


class MechV2PGBDataset(PGBDataset):
    """PGBDataset with the v2 mechanism flags in place of the 5 v1 flags; everything else is unchanged."""

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        row = self.df.iloc[idx]
        exc = str(row["Excipient_Smiles"]) if pd.notna(row.get("Excipient_Smiles")) else ""
        item["mech_flags"] = torch.tensor(mechanism_flags_v2(str(row["API_Smiles"]), exc), dtype=torch.float32)
        return item
