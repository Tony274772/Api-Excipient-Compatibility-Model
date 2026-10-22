"""
Loading, feature extraction, Morgan-bit decoding and importance tables for the final_project
Morgan-PGB SHAP analysis. Everything comes from final_project/src and the final_project checkpoint.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch.utils.data import DataLoader, Subset

from shap_analysis.feature_names import FEATURE_GROUPS, get_feature_group
from src.dataset import PGBDataset, PGBNormStats, pgb_collate_fn
from src.encoders import ENCODER_REGISTRY
from src.features import _parse_salt_aware
from src.model import pgb_model_from_config
from src.pgb_prior import ExcipientPriorTable


def load_checkpoint(ckpt_dir: str, device: str):
    """The trained model exactly as final_project/run.py loads it (config stored in best_model.pt)."""
    for f in ["best_model.pt", "prior_table.json", "pgb_norm_stats.npz"]:
        if not os.path.exists(os.path.join(ckpt_dir, f)):
            raise FileNotFoundError(f"{os.path.join(ckpt_dir, f)} not found")
    dev = torch.device(device)
    ckpt = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location=dev, weights_only=False)
    config = ckpt["config"]
    if config.encoder != "fixed_vector" or config.fixed_vector_source != "morgan":
        raise ValueError(f"this SHAP suite is for the Morgan-PGB model, checkpoint has {config.encoder}/{config.fixed_vector_source}")
    encoder = ENCODER_REGISTRY[config.encoder](source=config.fixed_vector_source, vector_path=None, device=str(dev))
    encoder.to(dev)
    model = pgb_model_from_config(config, encoder)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev).eval()
    prior_table = ExcipientPriorTable.load(os.path.join(ckpt_dir, "prior_table.json"))
    pgb_norm = PGBNormStats.load(os.path.join(ckpt_dir, "pgb_norm_stats.npz"))
    return model, config, encoder, prior_table, pgb_norm


def make_dataset(csv_path, config, prior_table, pgb_norm, encoder, row_priors=None) -> PGBDataset:
    return PGBDataset(csv_path, config.encoder, config.fixed_vector_source, prior_table, pgb_norm,
                      fixed_vector_encoder=encoder, row_priors=row_priors)


def extract_features(dataset: PGBDataset, model: torch.nn.Module, device: str, indices: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Flat feature matrix X [N, 2468] in FEATURE_GROUPS order, the tower-level matrix Z [N, 238],
    the model's own logits for the same rows, labels, raw descriptors and per-row metadata.
    Rows stay in dataset order (indices sorted), so X[i] belongs to meta.iloc[i].
    """
    idx = np.arange(len(dataset)) if indices is None else np.sort(np.asarray(indices))
    loader = DataLoader(Subset(dataset, idx.tolist()), batch_size=64, shuffle=False, collate_fn=pgb_collate_fn, num_workers=0)
    dev = torch.device(device)
    xs, zs, ys, logits, raw_api, raw_exc = [], [], [], [], [], []
    order = ["api_primary", "exc_primary", "api_secondary", "exc_secondary", "api_pgb_desc_norm", "exc_pgb_desc_norm",
             "api_salt", "exc_salt", "api_flags", "exc_flags", "prior_vec", "mech_flags"]
    with torch.no_grad():
        for batch in loader:
            if not (batch["api_prim_avail"].eq(1).all() and batch["exc_prim_avail"].eq(1).all()):
                raise RuntimeError("a molecule has no Morgan fingerprint (primary_avail = 0)")
            xs.append(np.concatenate([batch[k].cpu().numpy() for k in order], axis=1))
            ys.append(batch["label"].cpu().numpy())
            raw_api.append(batch["api_pgb_desc"].cpu().numpy())
            raw_exc.append(batch["exc_pgb_desc"].cpu().numpy())
            b = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            b.pop("label"); b.pop("sample_weight")
            logits.append(model(b).cpu().numpy())  # the model's own forward pass, for the fidelity check
            avail = torch.ones(len(batch["label"]), device=dev)
            h_a = model.api_tower(None, b["api_primary"], avail, b["api_secondary"], b["api_pgb_desc_norm"], b["api_salt"], b["api_flags"])
            h_e = model.exc_tower(None, b["exc_primary"], avail, b["exc_secondary"], b["exc_pgb_desc_norm"], b["exc_salt"], b["exc_flags"])
            zs.append(np.concatenate([h_a.cpu().numpy(), h_e.cpu().numpy(), batch["prior_vec"].numpy(),
                                      batch["api_flags"].numpy(), batch["exc_flags"].numpy(), batch["mech_flags"].numpy()], axis=1))
    X = np.concatenate(xs).astype(np.float32)
    if X.shape[1] != FEATURE_GROUPS["mech_flags"][1]:
        raise RuntimeError(f"feature matrix has {X.shape[1]} columns, feature names expect {FEATURE_GROUPS['mech_flags'][1]}")
    df = dataset.df.iloc[idx].reset_index(drop=True)
    meta = pd.DataFrame({
        "api_name": df["Api_name"] if "Api_name" in df else "", "exc_name": df["Excipient_name"] if "Excipient_name" in df else "",
        "api_cid": df["API_CID"], "exc_cid": df["Excipient_CID"], "api_smiles": df["API_Smiles"], "exc_smiles": df["Excipient_Smiles"],
        "label": df["Outcome1"].astype(int), "row_index": idx,
    })
    return {"X": X, "Z": np.concatenate(zs).astype(np.float32), "y": np.concatenate(ys).astype(np.float32),
            "model_logits": np.concatenate(logits).astype(np.float32), "meta": meta,
            "raw_desc_api": np.concatenate(raw_api).astype(np.float32), "raw_desc_exc": np.concatenate(raw_exc).astype(np.float32)}


def decode_morgan_bit(smiles_list: List[str], bit_idx: int, radius: int = 2, n_bits: int = 1024, max_examples: int = 5) -> Dict[str, Any]:
    """
    Substructures that set a folded Morgan bit, computed exactly like src.features.morgan_bits:
    each '.'-fragment fingerprinted separately (salt-aware), same generator, radius and size.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    counts: Dict[str, int] = {}
    examples, n_molecules = [], 0
    for smi in dict.fromkeys(s for s in smiles_list if isinstance(s, str) and s):
        hit = False
        for frag in _parse_salt_aware(smi):
            ao = rdFingerprintGenerator.AdditionalOutput()
            ao.AllocateBitInfoMap()
            gen.GetFingerprint(frag, additionalOutput=ao)
            for atom_idx, rad in ao.GetBitInfoMap().get(bit_idx, ()):
                if rad == 0:
                    a = frag.GetAtomWithIdx(atom_idx)
                    frag_smi = f"[{a.GetSymbol()}{'+' * a.GetFormalCharge() if a.GetFormalCharge() > 0 else '-' * -a.GetFormalCharge()}]"
                else:
                    env = Chem.FindAtomEnvironmentOfRadiusN(frag, rad, atom_idx)
                    frag_smi = Chem.MolToSmiles(Chem.PathToSubmol(frag, env)) if env else f"[{frag.GetAtomWithIdx(atom_idx).GetSymbol()}]"
                counts[frag_smi] = counts.get(frag_smi, 0) + 1
                hit = True
                if len(examples) < max_examples:
                    examples.append({"molecule_smiles": smi, "radius": rad, "fragment_smiles": frag_smi})
        n_molecules += hit
    top = max(counts.items(), key=lambda kv: kv[1])[0] if counts else f"Morgan_Bit_{bit_idx} (not set in these molecules)"
    return {"bit_index": bit_idx, "top_fragment_smiles": top, "occurrence_count": sum(counts.values()),
            "molecules_with_bit": n_molecules, "distinct_fragments": len(counts), "examples": examples}


def compute_feature_importance_df(shap_values: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature_index": np.arange(len(feature_names)), "feature_name": feature_names,
        "feature_group": [get_feature_group(i) for i in range(len(feature_names))],
        "mean_abs_shap": mean_abs, "percentage_importance": mean_abs / (mean_abs.sum() + 1e-12) * 100.0,
        "mean_shap": shap_values.mean(axis=0), "std_shap": shap_values.std(axis=0), "max_abs_shap": np.abs(shap_values).max(axis=0),
    })
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
