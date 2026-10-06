"""Inference script for the 11 models that achieved 12/12 incompatible recall.

Loads the 11 top-performing Fixed-Vector models (MACCS, Morgan, PubChem FP)
and runs inference on heldout.csv (or any specified CSV).

Outputs columns for each model matching the held_out_predictions.csv format:
    <model_name>_logit
    <model_name>_probability
    <model_name>_prediction
    <model_name>_threshold

Usage:
    python inference_11_models.py
    python inference_11_models.py --input new_heldout/heldout.csv --output new_heldout/heldout_predictions_11_models.csv
    python inference_11_models.py --input held_out_testset/held_out_test_set.csv
    python inference_11_models.py --device cuda   # or cpu
"""

import argparse
import base64
import json
import os
import sys

# Allow running from inside the inference/ folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import urllib.request
import urllib.parse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

# Project root path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.utils import seed_everything, get_device
from src.dataset import collate_fn
from src.descriptors import DESCRIPTOR_NAMES, _compute_descriptors, normalize_descriptors
from ensemble._common import build_model_from_checkpoint, load_config_for_checkpoint

# ---------------------------------------------------------------------------
# The 11 Target Models with 12/12 Incompatible Recall
# ---------------------------------------------------------------------------
TARGET_MODELS = [
    "fixed_vector_maccs_concat_asl",
    "fixed_vector_maccs_concat_bce",
    "fixed_vector_maccs_concat_focal",
    "fixed_vector_maccs_concat_weighted_bce",
    "fixed_vector_morgan_concat_asl",
    "fixed_vector_morgan_concat_bce",
    "fixed_vector_morgan_concat_weighted_bce",
    "fixed_vector_pubchemfp_concat_asl",
    "fixed_vector_pubchemfp_concat_bce",
    "fixed_vector_pubchemfp_concat_focal",
    "fixed_vector_pubchemfp_concat_weighted_bce",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference for 11 top-performing Fixed-Vector models on heldout data"
    )
    parser.add_argument(
        "--input",
        "--csv",
        type=str,
        default=None,
        help="Input CSV path (default: searches new_heldout/heldout.csv, heldout.csv, held_out_testset/held_out_test_set.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: <input_dir>/heldout_predictions_11_models.csv)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on: 'cuda', 'cpu', or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="checkpoints",
        help="Directory containing trained model checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="metrics",
        help="Directory containing saved validation metrics for thresholds (default: metrics)",
    )
    return parser.parse_args()


def resolve_input_csv(input_arg: Optional[str]) -> str:
    """Find input CSV from argument or sensible project defaults."""
    if input_arg and os.path.isfile(input_arg):
        return os.path.abspath(input_arg)

    candidates = [
        os.path.join(PROJECT_ROOT, "new_heldout", "heldout.csv"),
        os.path.join(PROJECT_ROOT, "heldout.csv"),
        os.path.join(PROJECT_ROOT, "held_out_testset", "held_out_test_set.csv"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    raise FileNotFoundError(
        "Could not find input heldout CSV. Please specify path explicitly via --input <path>"
    )


def detect_columns(df: pd.DataFrame) -> Tuple[str, str, Optional[str], Optional[str], Optional[str]]:
    """Auto-detect API smiles, excipient smiles, ground truth, and CID columns."""
    cols = df.columns.tolist()

    # API SMILES column
    api_candidates = ["API_Smiles", "Drugs", "drug", "smiles_api", "api_smiles", "API_SMILES", "Drug"]
    api_col = next((c for c in api_candidates if c in cols), None)
    if not api_col:
        # Fallback to first column with 'smiles' or 'drug'
        api_col = next((c for c in cols if "smiles" in c.lower() and "exc" not in c.lower()), None)
    if not api_col:
        raise ValueError(f"Could not identify API/Drug SMILES column in CSV. Found: {cols}")

    # Excipient SMILES column
    exc_candidates = ["Excipient_Smiles", "Excipients", "excipient", "smiles_exc", "excipient_smiles", "Excipient_SMILES", "Excipient"]
    exc_col = next((c for c in exc_candidates if c in cols), None)
    if not exc_col:
        exc_col = next((c for c in cols if "exc" in c.lower()), None)
    if not exc_col:
        raise ValueError(f"Could not identify Excipient SMILES column in CSV. Found: {cols}")

    # Ground truth column (optional)
    gt_candidates = ["ground_truth", "Output Value", "Outcome1", "label", "target", "groundtruth", "Output"]
    gt_col = next((c for c in gt_candidates if c in cols), None)

    # CID columns (optional)
    api_cid_col = next((c for c in ["API_CID", "api_cid", "drug_cid", "CID_API"] if c in cols), None)
    exc_cid_col = next((c for c in ["Excipient_CID", "excipient_cid", "exc_cid", "CID_Excipient"] if c in cols), None)

    return api_col, exc_col, gt_col, api_cid_col, exc_cid_col


class FingerprintManager:
    """Manages precomputed fingerprints and computes missing ones on the fly."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.maccs_dict: Dict[str, np.ndarray] = {}
        self.morgan_dict: Dict[str, np.ndarray] = {}
        self.pubchem_dict: Dict[str, np.ndarray] = {}

        self._load_precomputed("maccs", os.path.join(data_dir, "maccs_keys.csv"), self.maccs_dict)
        self._load_precomputed("morgan", os.path.join(data_dir, "morgan_fps.csv"), self.morgan_dict)
        self._load_precomputed("pubchem", os.path.join(data_dir, "pubchem_fps.csv"), self.pubchem_dict)

    def _load_precomputed(self, name: str, path: str, target_dict: Dict[str, np.ndarray]):
        if not os.path.isfile(path):
            print(f"[{name}] Precomputed vector file not found at {path}, will compute dynamically.")
            return

        df = pd.read_csv(path)
        feature_cols = [c for c in df.columns if c not in ["cid", "smiles", "API_CID", "Excipient_CID"]]
        for _, row in df.iterrows():
            vec = row[feature_cols].values.astype(np.float32)
            if "smiles" in row and pd.notna(row["smiles"]):
                target_dict[str(row["smiles"])] = vec
            if "cid" in row and pd.notna(row["cid"]):
                target_dict[str(row["cid"])] = vec

        print(f"[{name}] Loaded {len(target_dict)} precomputed vectors from {path}")

    def get_maccs(self, smiles: str, cid: Optional[str] = None) -> np.ndarray:
        if smiles in self.maccs_dict:
            return self.maccs_dict[smiles]
        if cid and str(cid) in self.maccs_dict:
            return self.maccs_dict[str(cid)]

        # Compute on the fly via RDKit
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        arr = np.zeros(166, dtype=np.float32)
        if mol is not None:
            fp = MACCSkeys.GenMACCSKeys(mol)
            for bit in fp.GetOnBits():
                if bit < 166:
                    arr[bit] = 1.0
        self.maccs_dict[smiles] = arr
        return arr

    def get_morgan(self, smiles: str, cid: Optional[str] = None) -> np.ndarray:
        if smiles in self.morgan_dict:
            return self.morgan_dict[smiles]
        if cid and str(cid) in self.morgan_dict:
            return self.morgan_dict[str(cid)]

        # Compute on the fly via RDKit (1024-bit, radius=2 / ECFP4)
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        arr = np.zeros(1024, dtype=np.float32)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            for bit in fp.GetOnBits():
                arr[bit] = 1.0
        self.morgan_dict[smiles] = arr
        return arr

    def get_pubchem(self, smiles: str, cid: Optional[str] = None) -> np.ndarray:
        if smiles in self.pubchem_dict:
            return self.pubchem_dict[smiles]
        if cid and str(cid) in self.pubchem_dict:
            return self.pubchem_dict[str(cid)]

        # Try to query NCBI PubChem by SMILES if not found
        arr = np.zeros(881, dtype=np.float32)
        if smiles:
            try:
                encoded_smi = urllib.parse.quote(smiles)
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_smi}/property/Fingerprint2D/JSON"
                req = urllib.request.Request(url, headers={"User-Agent": "API-Excipient-Inference/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    props = data.get("PropertyTable", {}).get("Properties", [])
                    if props:
                        fp_b64 = props[0].get("Fingerprint2D", "")
                        raw_bytes = base64.b64decode(fp_b64)
                        bit_bytes = raw_bytes[4:]
                        bits = []
                        for byte in bit_bytes:
                            for b in range(8):
                                bits.append((byte >> (7 - b)) & 1)
                        arr = np.array(bits[:881], dtype=np.float32)
            except Exception:
                pass  # Fall back to zero vector if offline/unreachable

        self.pubchem_dict[smiles] = arr
        return arr


class InferenceDataset(torch.utils.data.Dataset):
    """Dynamic dataset handling 21 descriptors and fingerprint embeddings."""

    def __init__(
        self,
        df: pd.DataFrame,
        api_col: str,
        exc_col: str,
        gt_col: Optional[str],
        api_cid_col: Optional[str],
        exc_cid_col: Optional[str],
        fp_mgr: FingerprintManager,
        norm_stats_path: str = "models/descriptor_norm_stats.json",
    ):
        self.df = df
        self.api_col = api_col
        self.exc_col = exc_col
        self.gt_col = gt_col
        self.api_cid_col = api_cid_col
        self.exc_cid_col = exc_cid_col
        self.fp_mgr = fp_mgr

        # Load normalization stats for 21 descriptors
        with open(norm_stats_path, "r") as f:
            stats = json.load(f)
        self.api_mean = np.array(stats["api_mean"], dtype=np.float32)
        self.api_std = np.array(stats["api_std"], dtype=np.float32)
        self.exc_mean = np.array(stats["exc_mean"], dtype=np.float32)
        self.exc_std = np.array(stats["exc_std"], dtype=np.float32)

        # Cache computed 21 descriptors per SMILES
        self._desc_cache: Dict[str, np.ndarray] = {}

    def _get_normalized_desc(self, smiles: str, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        if smiles not in self._desc_cache:
            raw_vals = _compute_descriptors(smiles)
            raw_arr = np.array(raw_vals, dtype=np.float32)
            norm_arr = normalize_descriptors(raw_arr, mean, std)
            self._desc_cache[smiles] = np.nan_to_num(norm_arr, nan=0.0)
        return self._desc_cache[smiles]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        api_smi = str(row[self.api_col]) if pd.notna(row[self.api_col]) else ""
        exc_smi = str(row[self.exc_col]) if pd.notna(row[self.exc_col]) else ""

        api_desc = self._get_normalized_desc(api_smi, self.api_mean, self.api_std)
        exc_desc = self._get_normalized_desc(exc_smi, self.exc_mean, self.exc_std)
        exc_available = 1.0 if exc_smi != "" else 0.0

        label = float(row[self.gt_col]) if (self.gt_col and pd.notna(row.get(self.gt_col))) else 0.0

        return {
            "api_smiles": api_smi,
            "exc_smiles": exc_smi,
            "api_desc": torch.tensor(api_desc, dtype=torch.float32),
            "exc_desc": torch.tensor(exc_desc, dtype=torch.float32),
            "exc_available": torch.tensor(exc_available, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "sample_weight": torch.tensor(1.0, dtype=torch.float32),
        }


def load_val_threshold(model_name: str, metrics_dir: str = "metrics") -> float:
    """Load the validation-tuned threshold from val_metrics.json."""
    val_json_path = os.path.join(metrics_dir, model_name, "val_metrics.json")
    if os.path.isfile(val_json_path):
        try:
            with open(val_json_path, "r") as f:
                metrics = json.load(f)
            return float(metrics.get("threshold", 0.5))
        except Exception:
            pass

    # Hardcoded fallbacks from documented validation runs
    defaults = {
        "fixed_vector_maccs_concat_asl": 0.747,
        "fixed_vector_maccs_concat_bce": 0.812,
        "fixed_vector_maccs_concat_focal": 0.519,
        "fixed_vector_maccs_concat_weighted_bce": 0.957,
        "fixed_vector_morgan_concat_asl": 0.750,
        "fixed_vector_morgan_concat_bce": 0.647,
        "fixed_vector_morgan_concat_weighted_bce": 0.852,
        "fixed_vector_pubchemfp_concat_asl": 0.784,
        "fixed_vector_pubchemfp_concat_bce": 0.961,
        "fixed_vector_pubchemfp_concat_focal": 0.484,
        "fixed_vector_pubchemfp_concat_weighted_bce": 0.971,
    }
    return defaults.get(model_name, 0.5)


def configure_encoder_for_model(model, model_name: str, fp_mgr: FingerprintManager):
    """Ensure the FixedVectorEncoder has access to dynamic on-the-fly fingerprints."""
    encoder = model.encoder
    if hasattr(encoder, "source"):
        if "maccs" in model_name:
            encoder._vectors = fp_mgr.maccs_dict
        elif "morgan" in model_name:
            encoder._vectors = fp_mgr.morgan_dict
        elif "pubchem" in model_name:
            encoder._vectors = fp_mgr.pubchem_dict


def run_inference_for_model(
    model_name: str,
    loader: torch.utils.data.DataLoader,
    fp_mgr: FingerprintManager,
    device: torch.device,
    checkpoints_dir: str = "checkpoints",
    metrics_dir: str = "metrics",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Execute inference for a single model and return (logits, probabilities, predictions, threshold)."""
    model, config = build_model_from_checkpoint(model_name, device, checkpoints_dir=checkpoints_dir)
    model.eval()

    configure_encoder_for_model(model, model_name, fp_mgr)
    threshold = load_val_threshold(model_name, metrics_dir=metrics_dir)

    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch_device = {
                "api_smiles": batch["api_smiles"],
                "exc_smiles": batch["exc_smiles"],
                "api_desc": batch["api_desc"].to(device),
                "exc_desc": batch["exc_desc"].to(device),
                "exc_available": batch["exc_available"].to(device),
            }
            logits = model(batch_device)
            all_logits.append(logits.cpu())

    logits = torch.cat(all_logits).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))  # Sigmoid
    predictions = (probs >= threshold).astype(int)

    return logits, probs, predictions, threshold


def main():
    args = parse_args()
    seed_everything(42)
    device = get_device(args.device)

    # 1. Resolve Input & Output Paths
    input_path = resolve_input_csv(args.input)
    default_out_name = "heldout_predictions_11_models.csv"
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        output_path = os.path.join(os.path.dirname(input_path), default_out_name)

    print("=" * 75)
    print("INFERENCE RUNNER: 11 MODELS WITH 12/12 INCOMPATIBLE RECALL")
    print(f"Device:           {device}")
    print(f"Input CSV:        {input_path}")
    print(f"Output CSV:       {output_path}")
    print(f"Checkpoints dir:  {args.checkpoints_dir}")
    print(f"Metrics dir:      {args.metrics_dir}")
    print("=" * 75)

    # 2. Load Input Data
    raw_df = pd.read_csv(input_path)
    print(f"\nLoaded {len(raw_df)} rows from {input_path}")

    api_col, exc_col, gt_col, api_cid_col, exc_cid_col = detect_columns(raw_df)
    print(f"Detected columns:")
    print(f"  API SMILES:       '{api_col}'")
    print(f"  Excipient SMILES: '{exc_col}'")
    print(f"  Ground Truth:     '{gt_col}'" if gt_col else "  Ground Truth:     None (Unlabeled)")

    # 3. Initialize Fingerprints & Descriptors
    print("\nInitializing Fingerprint Manager...")
    fp_mgr = FingerprintManager(data_dir=os.path.join(PROJECT_ROOT, "data"))

    # Pre-populate fingerprints for all unique SMILES in input CSV
    unique_smis = set(raw_df[api_col].dropna().unique()).union(set(raw_df[exc_col].dropna().unique()))
    print(f"Computing/caching fingerprints for {len(unique_smis)} unique molecules...")
    for s in unique_smis:
        fp_mgr.get_maccs(s)
        fp_mgr.get_morgan(s)
        fp_mgr.get_pubchem(s)
    print("Fingerprint caching complete.")

    # 4. Create Dataset & DataLoader
    norm_stats_path = os.path.join(PROJECT_ROOT, "models", "descriptor_norm_stats.json")
    dataset = InferenceDataset(
        raw_df,
        api_col=api_col,
        exc_col=exc_col,
        gt_col=gt_col,
        api_cid_col=api_cid_col,
        exc_cid_col=exc_cid_col,
        fp_mgr=fp_mgr,
        norm_stats_path=norm_stats_path,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=0,
    )

    # 5. Initialize Result DataFrame with original columns
    result_df = raw_df.copy()

    # 6. Run Inference Across the 11 Models
    summary_records = []
    print(f"\nRunning inference across {len(TARGET_MODELS)} models...")

    for idx, model_name in enumerate(TARGET_MODELS, 1):
        ckpt_path = os.path.join(args.checkpoints_dir, model_name, "best_model.pt")
        if not os.path.isfile(ckpt_path):
            print(f"[{idx}/{len(TARGET_MODELS)}] WARNING: Checkpoint not found for {model_name}, skipping.")
            continue

        t0 = time.time()
        logits, probs, preds, thresh = run_inference_for_model(
            model_name,
            loader,
            fp_mgr,
            device,
            checkpoints_dir=args.checkpoints_dir,
            metrics_dir=args.metrics_dir,
        )
        elapsed = time.time() - t0

        # Store in result dataframe exactly matching held_out_predictions.csv format
        result_df[f"{model_name}_logit"] = logits
        result_df[f"{model_name}_probability"] = probs
        result_df[f"{model_name}_prediction"] = preds
        result_df[f"{model_name}_threshold"] = thresh

        # Record summary
        rec = {
            "model": model_name,
            "threshold": round(thresh, 4),
            "positives": int(preds.sum()),
            "total": len(preds),
            "time_sec": round(elapsed, 2),
        }
        if gt_col:
            y_true = raw_df[gt_col].astype(int).values
            rec["accuracy"] = round(float((preds == y_true).mean()), 4)
            pos_mask = (y_true == 1)
            rec["incompat_recall"] = f"{int(preds[pos_mask].sum())}/{int(pos_mask.sum())}" if pos_mask.sum() > 0 else "N/A"
        summary_records.append(rec)

        print(f"[{idx:2d}/{len(TARGET_MODELS)}] {model_name:<42} | Pos: {preds.sum():3d}/{len(preds)} | Thresh: {thresh:.3f} | Time: {elapsed:.2f}s")

    # 7. Save Output CSV
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"\n{'='*75}")
    print(f"SUCCESS: Saved predictions to {output_path}")
    print(f"Total Columns: {len(result_df.columns)}")
    print(f"{'='*75}\n")

    # 8. Print Summary Table
    print(f"{'Model':<45} {'Thresh':<8} {'Pred Pos':<12} " + ("{'Accuracy':<10} {'Incompat Recall':<15}" if gt_col else ""))
    print("-" * 75)
    for r in summary_records:
        line = f"{r['model']:<45} {r['threshold']:<8.3f} {r['positives']:3d}/{r['total']:<8} "
        if gt_col:
            line += f"{r.get('accuracy', 0.0):<10.4f} {r.get('incompat_recall', ''):<15}"
        print(line)
    print("-" * 75)


if __name__ == "__main__":
    main()
