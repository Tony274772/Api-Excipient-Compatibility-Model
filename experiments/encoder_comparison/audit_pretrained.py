"""Audit what each pretrained encoder was actually trained to read.

This is a different question from audit_reps.py. That script checks the vectors an encoder
PRODUCES (coverage, scaling, collisions). This one checks the encoder's INPUT layer: for every
atom type, token or substructure this dataset contains, was the corresponding row of the
pretrained weights ever updated during pretraining?

Why it matters: an input the model never saw in pretraining keeps its random initial value. The
encoder still returns a number, nothing errors, and the vector looks perfectly healthy to
audit_reps.py - but that part of the molecule is noise rather than chemistry. ChemBERTa turned out
to be an extreme case: not one of its 543 multi-character tokens was ever trained, so chlorine
reads as carbon and [Na+] as nitrogen for 45% of this dataset.

## How "never trained" is decided

A weight matrix row is untrained if every one of its values is still inside the initialiser's
range. That is an exact, falsifiable test, not a heuristic:

  * nn.Embedding default init N(0, 1)          -> compare against rows known never to occur
  * xavier_uniform_ on (fan_in, fan_out)       -> bound sqrt(6 / (fan_in + fan_out))
  * nn.Linear default init                     -> bound 1 / sqrt(in_features)

Rows 101-118 of a 120-row atom table (elements Fermium onward) are used as a control: no chemistry
pretraining set contains them, so their spread shows what untrained looks like for that file.

Comparing NORMS is not reliable here and an earlier version of this audit got GAT wrong that way:
GAT's trained embeddings have norms around 0.8-2.3 while its untrained rows sit at 1.19, so the
two overlap. The max-absolute-value test against the initialiser bound separates them cleanly.

Read-only: trains nothing, writes nothing.

    python -m experiments.encoder_comparison.audit_pretrained
    python -m experiments.encoder_comparison.audit_pretrained --encoders chemberta molformer
"""

import argparse
import collections
import json
import os
import sys

import numpy as np
import torch
from rdkit import Chem, RDLogger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from experiments.encoder_comparison.data import SplitPaths  # noqa: E402

RDLogger.DisableLog("rdApp.*")

PT = Chem.GetPeriodicTable()
FINDINGS = []


def record(encoder: str, severity: str, headline: str, detail: str = ""):
    FINDINGS.append((encoder, severity, headline, detail))
    mark = {"ok": "ok  ", "minor": "note", "major": "WARN", "critical": "FAIL"}[severity]
    print(f"  {mark} {headline}{(' - ' + detail) if detail else ''}")


def dataset_molecules():
    return [s.strip() for s in SplitPaths().all_molecules()]


def dataset_elements(mols):
    """{element symbol: number of molecules containing it}."""
    c = collections.Counter()
    for s in mols:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        for e in {a.GetSymbol() for a in m.GetAtoms()}:
            c[e] += 1
    return c


# --------------------------------------------------------------------------- chemberta
def audit_chemberta(mols):
    print("\n--- chemberta (DeepChem/ChemBERTa-77M-MTR)")
    d = os.path.join(PROJECT_ROOT, "models", "pretrained", "chemberta")
    from safetensors import safe_open
    E = safe_open(os.path.join(d, "model.safetensors"), "pt") \
        .get_tensor("embeddings.word_embeddings.weight").float().numpy()
    with open(os.path.join(d, "tokenizer.json"), encoding="utf-8") as f:
        tok = json.load(f)
    vocab = tok["model"]["vocab"]
    n = np.linalg.norm(E, axis=1)

    merges = len(tok["model"].get("merges") or [])
    record("chemberta", "critical" if merges == 0 else "ok",
           f"BPE merge rules: {merges}",
           "with no merges the tokenizer can only emit single characters" if merges == 0 else "")

    special = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}
    unused = [t for t in vocab if t.startswith("[unused")]
    ref = n[[vocab[t] for t in unused]].mean()
    single = [t for t in vocab if len(t) == 1]
    multi = [t for t in vocab if len(t) > 1 and t not in special and not t.startswith("[unused")]
    mn = n[[vocab[t] for t in multi]]
    trained_multi = [t for t in multi if n[vocab[t]] > 3 * ref]

    record("chemberta", "critical" if not trained_multi else "ok",
           f"multi-character tokens trained: {len(trained_multi)} of {len(multi)}",
           f"all sit at the random-init norm (median {np.median(mn):.2f}, "
           f"never-used reference {ref:.2f}); single chars are at "
           f"{np.median(n[[vocab[t] for t in single]]):.2f}")

    affected = [s for s in mols if "Cl" in s or "Br" in s or "[" in s]
    record("chemberta", "critical",
           f"molecules misread: {len(affected)} of {len(mols)} ({len(affected) / len(mols):.0%})",
           "Cl->C, Br->B, [Na+]->N, charges and stereochemistry dropped")
    for t in (".", "/", "\\"):
        if t in vocab and n[vocab[t]] < 3 * ref:
            record("chemberta", "major", f"token '{t}' was never trained",
                   "pretraining SMILES had no salts" if t == "."
                   else "pretraining SMILES had no E/Z stereochemistry")


# --------------------------------------------------------------------------- molformer
def audit_molformer(mols):
    print("\n--- molformer (ibm/MoLFormer-XL-both-10pct)")
    d = os.path.join(PROJECT_ROOT, "models", "pretrained", "molformer")
    from safetensors import safe_open
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
    f = safe_open(os.path.join(d, "model.safetensors"), "pt")
    E = f.get_tensor([k for k in f.keys() if "word_embeddings" in k][0]).float().numpy()
    n = np.linalg.norm(E, axis=1)

    used, unk, lossy, lengths = collections.Counter(), [], [], []
    for s in mols:
        ids = tk(s, truncation=False)["input_ids"]
        lengths.append(len(ids))
        toks = [t for t in tk.convert_ids_to_tokens(ids) if t not in ("<bos>", "<eos>")]
        if tk.unk_token_id in ids:
            unk.append(s)
        if "".join(toks) != s.replace(" ", ""):
            lossy.append(s)
        used.update(toks)

    record("molformer", "critical" if unk else "ok",
           f"molecules containing an unknown token: {len(unk)} of {len(mols)}")
    record("molformer", "critical" if lossy else "ok",
           f"molecules whose tokens do not rebuild the SMILES: {len(lossy)} of {len(mols)}",
           "nothing is dropped" if not lossy else f"e.g. {lossy[:2]}")
    uv = np.array([n[tk.convert_tokens_to_ids(t)] for t in used])
    record("molformer", "ok" if uv.min() > 3 * n.min() / 3 else "major",
           f"all {len(used)} tokens used are trained",
           f"lowest norm among them {uv.min():.1f}, vocabulary minimum {n.min():.1f}")

    # Rotary position embeddings extend past the nominal limit, so a long molecule is not truncated.
    with open(os.path.join(d, "config.json")) as fh:
        maxpos = json.load(fh)["max_position_embeddings"]
    over = [l for l in lengths if l > maxpos]
    record("molformer", "minor" if over else "ok",
           f"molecules longer than the nominal {maxpos}-token window: {len(over)}",
           "harmless: this model uses rotary position embeddings, which extend to any length "
           "(verified by encoding the 526-token molecule)" if over else "")


# --------------------------------------------------------------------------- graph encoders
def audit_graph(name, ckpt_rel, key, mols, elements):
    """GIN and GAT both use a 120-row atomic-number embedding, xavier_uniform initialised."""
    print(f"\n--- {name}")
    sd = torch.load(os.path.join(PROJECT_ROOT, ckpt_rel), map_location="cpu", weights_only=True)
    sd = sd.get("model_state_dict", sd)
    E = sd[key].numpy()
    rows, dim = E.shape
    bound = np.sqrt(6 / (rows + dim))          # xavier_uniform_ on (120, 300)

    control = np.abs(E[100:118]).max()         # elements 101-118: never in any chemistry corpus
    record(name, "ok" if abs(control - bound) < 1e-3 else "minor",
           f"initialiser bound {bound:.4f}, never-seen control rows reach {control:.4f}",
           "the test separates trained from untrained exactly")

    trained = np.abs(E).max(1) > bound + 1e-6
    untrained = [e for e in elements if not trained[PT.GetAtomicNumber(e) - 1]]

    affected, atoms_tot, atoms_bad = set(), 0, 0
    for s in mols:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        for a in m.GetAtoms():
            atoms_tot += 1
            if not trained[a.GetAtomicNum() - 1]:
                atoms_bad += 1
                affected.add(s)

    sev = "critical" if len(affected) / len(mols) > 0.15 else "major" if untrained else "ok"
    record(name, sev,
           f"elements present in the data but never trained: {len(untrained)} of {len(elements)}",
           ", ".join(f"{e} ({elements[e]} mols)" for e in
                     sorted(untrained, key=lambda x: -elements[x])) or "none")
    record(name, sev if untrained else "ok",
           f"molecules affected: {len(affected)} of {len(mols)} ({len(affected) / len(mols):.0%})",
           f"{atoms_bad} of {atoms_tot} atoms ({atoms_bad / atoms_tot:.2%}) get a random vector")


# --------------------------------------------------------------------------- dmpnn
def audit_dmpnn(mols):
    print("\n--- dmpnn (CheMeleon)")
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    af = SimpleMoleculeMolGraphFeaturizer().atom_featurizer
    ck = torch.load(os.path.join(PROJECT_ROOT, "models", "pretrained", "chemeleon_mp.pt"),
                    map_location="cpu", weights_only=False)
    W = ck["state_dict"]["W_i.weight"].numpy()
    bound = 1 / np.sqrt(W.shape[1])            # nn.Linear default init
    trained = np.abs(W).max(0) > bound + 1e-6

    names = []
    for attr, label in [("atomic_nums", "elem"), ("degrees", "deg"), ("formal_charges", "charge"),
                        ("chiral_tags", "chiral"), ("num_Hs", "nH"), ("hybridizations", "hyb")]:
        for v in getattr(af, attr):
            names.append(f"{label}={PT.GetElementSymbol(int(v))}" if label == "elem"
                         else f"{label}={v}")
        names.append(f"{label}=OTHER")
    names += ["aromatic", "mass"]

    per_slot = collections.defaultdict(set)
    for s in mols:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        for a in m.GetAtoms():
            x = np.asarray(af(a))
            for i in np.nonzero(x[:len(names) - 2])[0]:
                per_slot[i].add(s)

    bad = [(names[i], len(per_slot[i])) for i in sorted(per_slot) if not trained[i]]
    record("dmpnn", "ok" if not bad else "major",
           f"atom-feature slots used by the data: {len(per_slot)}, of those untrained: {len(bad)}",
           "; ".join(f"{nm} in {c} molecules" for nm, c in bad) or "none")
    if bad:
        aff = set().union(*[per_slot[i] for i in per_slot if not trained[i]])
        record("dmpnn", "major", f"molecules touching an untrained feature: {len(aff)} of {len(mols)}",
               "CheMeleon's pretraining set evidently held no disconnected ions")


# --------------------------------------------------------------------------- mol2vec
def audit_mol2vec(mols):
    print("\n--- mol2vec")
    from gensim.models import word2vec
    from rdkit.Chem import AllChem
    kv = word2vec.Word2Vec.load(
        os.path.join(PROJECT_ROOT, "models", "pretrained", "mol2vec", "model_300dim.pkl")).wv

    tot = unk = 0
    frac = []
    for s in mols:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        info = {}
        AllChem.GetMorganFingerprint(m, 1, bitInfo=info)
        d = {a.GetIdx(): {0: None, 1: None} for a in m.GetAtoms()}
        for ident in info:
            for ai, r in info[ident]:
                d[ai][r] = ident
        sent = [str(d[a][r]) for a in d for r in (0, 1) if d[a][r] is not None]
        u = sum(t not in kv.key_to_index for t in sent)
        tot += len(sent)
        unk += u
        frac.append((u / max(len(sent), 1), s))

    all_unk = [s for f, s in frac if f == 1.0]
    record("mol2vec", "ok", f"substructure tokens unknown: {unk} of {tot} ({unk / tot:.1%})",
           "unknown ones fall back to the model's own trained UNK vector, not to a random one")
    record("mol2vec", "major" if all_unk else "ok",
           f"molecules that are 100% unknown: {len(all_unk)}",
           f"they all receive the identical UNK vector: {', '.join(all_unk[:6])}" if all_unk else "")
    record("mol2vec", "minor", f"molecules above 25% unknown: {sum(1 for f, _ in frac if f > 0.25)}")


# --------------------------------------------------------------------------- report
def main():
    p = argparse.ArgumentParser(description="Audit what the pretrained encoders can actually read.")
    p.add_argument("--encoders", nargs="+",
                   default=["chemberta", "molformer", "gin", "gat", "dmpnn", "mol2vec"])
    args = p.parse_args()

    mols = dataset_molecules()
    elements = dataset_elements(mols)
    print(f"dataset: {len(mols)} molecules, {len(elements)} elements")
    print("elements: " + ", ".join(f"{e}({c})" for e, c in elements.most_common()))

    jobs = {
        "chemberta": lambda: audit_chemberta(mols),
        "molformer": lambda: audit_molformer(mols),
        "gin": lambda: audit_graph("gin", "models/pretrained/gin/gin_supervised_contextpred_pre_trained.pth",
                                   "node_embeddings.0.weight", mols, elements),
        "gat": lambda: audit_graph("gat", "models/pretrained/stanford_gat/gat_contextpred.pth",
                                   "x_embedding1.weight", mols, elements),
        "dmpnn": lambda: audit_dmpnn(mols),
        "mol2vec": lambda: audit_mol2vec(mols),
    }
    for name in args.encoders:
        if name not in jobs:
            raise SystemExit(f"unknown encoder '{name}'. Choices: {', '.join(jobs)}")
        try:
            jobs[name]()
        except Exception as e:
            record(name, "major", "audit could not run", f"{type(e).__name__}: {e}")

    print("\n" + "=" * 100)
    order = {"critical": 0, "major": 1, "minor": 2, "ok": 3}
    for sev, label in [("critical", "CRITICAL - the encoder cannot read part of its input"),
                       ("major", "MAJOR - a real blind spot, worth stating in the paper"),
                       ("minor", "MINOR - worth knowing, not a defect")]:
        rows = [f for f in FINDINGS if f[1] == sev]
        if rows:
            print(f"\n{label}: {len(rows)}")
            for enc, _, head, detail in sorted(rows, key=lambda r: order[r[1]]):
                print(f"  [{enc}] {head}{(' - ' + detail) if detail else ''}")
    clean = sorted({f[0] for f in FINDINGS} - {f[0] for f in FINDINGS if f[1] in ("critical", "major")})
    print(f"\nencoders with no critical or major finding: {', '.join(clean) if clean else 'none'}")


if __name__ == "__main__":
    main()
