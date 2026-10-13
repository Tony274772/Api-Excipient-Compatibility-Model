"""
Utility functions for data preparation, model loading, bit decoding, and SHAP post-processing.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader

from experiments.pgb.config import PGBConfig
from experiments.pgb.dataset import PGBDataset, PGBNormStats, pgb_collate_fn
from experiments.pgb.model import pgb_model_from_config
from experiments.pgb.prior import ExcipientPriorTable
from shap_analysis.config import SHAPConfig
from shap_analysis.feature_names import FEATURE_GROUPS, FEATURE_NAMES, get_feature_group
from src.cross_validate import build_encoder


def load_pgb_morgan_model(
    config_obj: Optional[SHAPConfig] = None,
    checkpoint_path: Optional[str] = None,
    norm_path: Optional[str] = None,
    prior_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[torch.nn.Module, PGBNormStats, ExcipientPriorTable, PGBConfig, Any]:
    """
    Loads the trained PGB Morgan model, normalization statistics, and excipient priors.
    """
    cfg = config_obj or SHAPConfig()
    ckpt_file = checkpoint_path or cfg.checkpoint_path
    norm_file = norm_path or cfg.norm_stats_path
    prior_file = prior_path or cfg.prior_table_path
    dev_str = device or cfg.device

    dev = torch.device(dev_str)

    pgb_cfg = PGBConfig()
    pgb_cfg.apply_family_spec("morgan")

    encoder = build_encoder(pgb_cfg, dev)
    model = pgb_model_from_config(pgb_cfg, encoder)

    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_file}")
    if not os.path.exists(norm_file):
        raise FileNotFoundError(f"Norm stats not found at: {norm_file}")
    if not os.path.exists(prior_file):
        raise FileNotFoundError(f"Prior table not found at: {prior_file}")

    state = torch.load(ckpt_file, map_location=dev, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.to(dev)
    model.eval()

    pgb_norm = PGBNormStats.load(norm_file)
    prior_table = ExcipientPriorTable.load(prior_file)

    return model, pgb_norm, prior_table, pgb_cfg, encoder


def extract_features_from_dataset(
    dataset: PGBDataset,
    model: Optional[torch.nn.Module] = None,
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Extracts flat feature matrices [N, 2468], intermediate representations [N, 238],
    and sample metadata from a PGBDataset instance.
    """
    n_samples = len(dataset)
    if max_samples is not None and max_samples > 0 and max_samples < n_samples:
        indices = np.random.choice(n_samples, size=max_samples, replace=False)
        indices = np.sort(indices)
    else:
        indices = np.arange(n_samples)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        sampler=torch.utils.data.SubsetRandomSampler(indices) if len(indices) < n_samples else None,
        collate_fn=pgb_collate_fn,
        num_workers=0,
    )

    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    all_x = []
    all_y = []
    all_z = []
    meta_records = []
    raw_desc_api = []
    raw_desc_exc = []

    with torch.no_grad():
        for batch in loader:
            api_prim = batch["api_primary"].cpu().numpy()
            exc_prim = batch["exc_primary"].cpu().numpy()
            api_sec = batch["api_secondary"].cpu().numpy()
            exc_sec = batch["exc_secondary"].cpu().numpy()
            api_desc_norm = batch["api_pgb_desc_norm"].cpu().numpy()
            exc_desc_norm = batch["exc_pgb_desc_norm"].cpu().numpy()
            api_salt = batch["api_salt"].cpu().numpy()
            exc_salt = batch["exc_salt"].cpu().numpy()
            api_flags = batch["api_flags"].cpu().numpy()
            exc_flags = batch["exc_flags"].cpu().numpy()
            prior_vec = batch["prior_vec"].cpu().numpy()
            mech_flags = batch["mech_flags"].cpu().numpy()
            labels = batch["label"].cpu().numpy()

            raw_desc_api.append(batch["api_pgb_desc"].cpu().numpy())
            raw_desc_exc.append(batch["exc_pgb_desc"].cpu().numpy())

            # Concatenate to flat vector [B, 2468]
            flat_b = np.concatenate(
                [
                    api_prim,
                    exc_prim,
                    api_sec,
                    exc_sec,
                    api_desc_norm,
                    exc_desc_norm,
                    api_salt,
                    exc_salt,
                    api_flags,
                    exc_flags,
                    prior_vec,
                    mech_flags,
                ],
                axis=-1,
            )
            all_x.append(flat_b)
            all_y.append(labels)

            # Record metadata
            for i in range(len(batch["api_smiles"])):
                meta_records.append({
                    "api_smiles": batch["api_smiles"][i],
                    "exc_smiles": batch["exc_smiles"][i],
                    "label": float(labels[i]),
                })

            # Intermediate tower representations [B, 238] if model is provided
            if model is not None:
                avail = torch.ones(flat_b.shape[0], device=dev)
                h_a = model.api_tower(
                    None,
                    batch["api_primary"].to(dev),
                    avail,
                    batch["api_secondary"].to(dev),
                    batch["api_pgb_desc_norm"].to(dev),
                    batch["api_salt"].to(dev),
                    batch["api_flags"].to(dev),
                ).cpu().numpy()

                h_e = model.exc_tower(
                    None,
                    batch["exc_primary"].to(dev),
                    avail,
                    batch["exc_secondary"].to(dev),
                    batch["exc_pgb_desc_norm"].to(dev),
                    batch["exc_salt"].to(dev),
                    batch["exc_flags"].to(dev),
                ).cpu().numpy()

                z_b = np.concatenate(
                    [
                        h_a,
                        h_e,
                        prior_vec,
                        api_flags,
                        exc_flags,
                        mech_flags,
                    ],
                    axis=-1,
                )
                all_z.append(z_b)

    x_mat = np.concatenate(all_x, axis=0).astype(np.float32)
    y_arr = np.concatenate(all_y, axis=0).astype(np.float32)
    raw_api_mat = np.concatenate(raw_desc_api, axis=0).astype(np.float32)
    raw_exc_mat = np.concatenate(raw_desc_exc, axis=0).astype(np.float32)
    meta_df = pd.DataFrame(meta_records)
    if "Api_name" in dataset.df.columns and len(meta_df) == len(dataset.df):
        meta_df["api_name"] = dataset.df["Api_name"].values
    if "Excipient_name" in dataset.df.columns and len(meta_df) == len(dataset.df):
        meta_df["exc_name"] = dataset.df["Excipient_name"].values

    res = {
        "X": x_mat,
        "y": y_arr,
        "meta": meta_df,
        "raw_desc_api": raw_api_mat,
        "raw_desc_exc": raw_exc_mat,
    }
    if all_z:
        res["Z"] = np.concatenate(all_z, axis=0).astype(np.float32)

    return res


def decode_morgan_bit(
    smiles_list: List[str],
    bit_idx: int,
    radius: int = 2,
    n_bits: int = 1024,
    max_examples: int = 5,
) -> Dict[str, Any]:
    """
    Decodes an individual folded Morgan bit back to human-interpretable
    chemical substructure SMILES and atom environment descriptions.
    """
    matches = []
    substructure_smiles_counts: Dict[str, int] = {}

    for smi in smiles_list:
        if not smi or pd.isna(smi):
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue

        bit_info: Dict[int, List[Tuple[int, int]]] = {}
        _ = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, bitInfo=bit_info)

        if bit_idx in bit_info:
            for atom_idx, rad in bit_info[bit_idx]:
                atom = mol.GetAtomWithIdx(atom_idx)
                center_symbol = atom.GetSymbol()

                if rad == 0:
                    frag_smi = f"[{center_symbol}]"
                else:
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_idx)
                    amap = {}
                    submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                    try:
                        frag_smi = Chem.MolToSmiles(submol)
                    except Exception:
                        frag_smi = f"Radius-{rad} env around {center_symbol}"

                substructure_smiles_counts[frag_smi] = substructure_smiles_counts.get(frag_smi, 0) + 1

                if len(matches) < max_examples:
                    matches.append({
                        "molecule_smiles": smi,
                        "center_atom": center_symbol,
                        "radius": rad,
                        "fragment_smiles": frag_smi,
                    })

    top_fragment = (
        max(substructure_smiles_counts.items(), key=lambda item: item[1])[0]
        if substructure_smiles_counts
        else f"Morgan_Bit_{bit_idx}"
    )

    return {
        "bit_index": bit_idx,
        "top_fragment_smiles": top_fragment,
        "occurrence_count": sum(substructure_smiles_counts.values()),
        "distinct_fragments": len(substructure_smiles_counts),
        "examples": matches,
    }


def compute_feature_importance_df(
    shap_values: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Aggregates SHAP matrix [N, D] into a comprehensive ranking table.
    """
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    mean_shap = np.mean(shap_values, axis=0)
    std_shap = np.std(shap_values, axis=0)
    max_abs_shap = np.max(np.abs(shap_values), axis=0)

    total_importance = np.sum(mean_abs_shap)
    pct_importance = (mean_abs_shap / (total_importance + 1e-12)) * 100.0

    groups = [get_feature_group(i) for i in range(len(feature_names))]

    df = pd.DataFrame({
        "feature_index": np.arange(len(feature_names)),
        "feature_name": feature_names,
        "feature_group": groups,
        "mean_abs_shap": mean_abs_shap,
        "percentage_importance": pct_importance,
        "mean_shap": mean_shap,
        "std_shap": std_shap,
        "max_abs_shap": max_abs_shap,
    })

    df = df.sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    return df
