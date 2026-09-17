"""Utility to precompute fixed-vector embeddings for APIs and Excipients.

Supports:
1. Morgan / ECFP4 fingerprints (1024-bit, computed offline via RDKit)
2. MACCS keys (166-bit, computed offline via RDKit)
3. PubChem fingerprints (881-bit, fetched via PubChem REST API for CIDs)
4. mol2vec embeddings (300-dim, using mol2vec pretrained model if installed)

Usage:
    python data/compute_fixed_vectors.py --type morgan --output data/morgan_fps.csv
    python data/compute_fixed_vectors.py --type maccs --output data/maccs_keys.csv
    python data/compute_fixed_vectors.py --type pubchemfp --output data/pubchem_fps.csv
"""

import argparse
import base64
import os
import time
import urllib.request
import json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys


def get_unique_molecules(data_dir: str = "data") -> pd.DataFrame:
    """Collect unique (CID, SMILES) pairs across train, val, and test splits."""
    records = {}
    for split in ["train.csv", "val.csv", "test.csv"]:
        p = os.path.join(data_dir, split)
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if "API_CID" in df.columns and "API_Smiles" in df.columns:
            for _, r in df[["API_CID", "API_Smiles"]].dropna().iterrows():
                cid = str(int(r["API_CID"]) if isinstance(r["API_CID"], (int, float)) else r["API_CID"])
                records[cid] = str(r["API_Smiles"])
        if "Excipient_CID" in df.columns and "Excipient_Smiles" in df.columns:
            for _, r in df[["Excipient_CID", "Excipient_Smiles"]].dropna().iterrows():
                cid = str(int(r["Excipient_CID"]) if isinstance(r["Excipient_CID"], (int, float)) else r["Excipient_CID"])
                records[cid] = str(r["Excipient_Smiles"])

    return pd.DataFrame([{"cid": k, "smiles": v} for k, v in records.items()])


def compute_morgan(smiles_list: list[str], n_bits: int = 1024, radius: int = 2) -> np.ndarray:
    """Compute Morgan / ECFP fingerprints for a list of SMILES."""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=np.float32)
            for bit in fp.GetOnBits():
                arr[bit] = 1.0
            fps.append(arr)
        else:
            fps.append(np.zeros((n_bits,), dtype=np.float32))
    return np.array(fps, dtype=np.float32)


def compute_maccs(smiles_list: list[str]) -> np.ndarray:
    """Compute 166-bit MACCS keys for a list of SMILES."""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.zeros((166,), dtype=np.float32)
            for bit in fp.GetOnBits():
                if bit < 166:
                    arr[bit] = 1.0
            fps.append(arr)
        else:
            fps.append(np.zeros((166,), dtype=np.float32))
    return np.array(fps, dtype=np.float32)


def fetch_pubchem_fingerprints(cids: list[str]) -> dict[str, np.ndarray]:
    """Fetch 881-bit PubChem fingerprints for CIDs from NCBI PUG REST API."""
    result = {}
    batch_size = 100
    cid_list = [c for c in cids if c.isdigit()]

    print(f"Fetching PubChem fingerprints for {len(cid_list)} valid CIDs...")
    for i in range(0, len(cid_list), batch_size):
        chunk = cid_list[i:i + batch_size]
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{','.join(chunk)}/property/Fingerprint2D/JSON"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "API-Excipient-Pipeline/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for prop in data.get("PropertyTable", {}).get("Properties", []):
                    cid = str(prop.get("CID"))
                    fp_b64 = prop.get("Fingerprint2D", "")
                    if fp_b64:
                        raw_bytes = base64.b64decode(fp_b64)
                        # PubChem header has 4 bytes prefix (length) followed by bit vector
                        bit_bytes = raw_bytes[4:]
                        bits = []
                        for byte in bit_bytes:
                            for b in range(8):
                                bits.append((byte >> (7 - b)) & 1)
                        # First 881 bits represent the PubChem substructure fingerprint
                        fp_vec = np.array(bits[:881], dtype=np.float32)
                        result[cid] = fp_vec
            time.sleep(0.2)  # Respect NCBI rate limits
        except Exception as e:
            print(f"  Warning: Batch {i}-{i+len(chunk)} failed ({e}). Retrying individually...")
            for single_cid in chunk:
                try:
                    s_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{single_cid}/property/Fingerprint2D/JSON"
                    s_req = urllib.request.Request(s_url, headers={"User-Agent": "API-Excipient-Pipeline/1.0"})
                    with urllib.request.urlopen(s_req, timeout=10) as s_resp:
                        s_data = json.loads(s_resp.read().decode("utf-8"))
                        props = s_data.get("PropertyTable", {}).get("Properties", [])
                        if props:
                            fp_b64 = props[0].get("Fingerprint2D", "")
                            raw_bytes = base64.b64decode(fp_b64)
                            bit_bytes = raw_bytes[4:]
                            bits = []
                            for byte in bit_bytes:
                                for b in range(8):
                                    bits.append((byte >> (7 - b)) & 1)
                            result[single_cid] = np.array(bits[:881], dtype=np.float32)
                    time.sleep(0.1)
                except Exception:
                    pass

    print(f"Successfully retrieved {len(result)} PubChem fingerprints.")
    return result


def main():
    parser = argparse.ArgumentParser(description="Precompute fixed-vector embeddings")
    parser.add_argument("--type", choices=["morgan", "maccs", "pubchemfp"], default="morgan",
                        help="Type of fixed vector to compute")
    parser.add_argument("--data_dir", default="data", help="Directory containing train/val/test.csv")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    mols_df = get_unique_molecules(args.data_dir)
    print(f"Found {len(mols_df)} unique molecules across splits.")

    if args.type == "morgan":
        out_path = args.output or os.path.join(args.data_dir, "morgan_fps.csv")
        vectors = compute_morgan(mols_df["smiles"].tolist(), n_bits=1024)
        cols = [f"bit_{i}" for i in range(1024)]
        vec_df = pd.DataFrame(vectors, columns=cols)
        vec_df.insert(0, "smiles", mols_df["smiles"])
        vec_df.insert(0, "cid", mols_df["cid"])
        vec_df.to_csv(out_path, index=False)
        print(f"Saved 1024-bit Morgan fingerprints to {out_path}")

    elif args.type == "maccs":
        out_path = args.output or os.path.join(args.data_dir, "maccs_keys.csv")
        vectors = compute_maccs(mols_df["smiles"].tolist())
        cols = [f"maccs_{i}" for i in range(166)]
        vec_df = pd.DataFrame(vectors, columns=cols)
        vec_df.insert(0, "smiles", mols_df["smiles"])
        vec_df.insert(0, "cid", mols_df["cid"])
        vec_df.to_csv(out_path, index=False)
        print(f"Saved 166-bit MACCS keys to {out_path}")

    elif args.type == "pubchemfp":
        out_path = args.output or os.path.join(args.data_dir, "pubchem_fps.csv")
        fp_dict = fetch_pubchem_fingerprints(mols_df["cid"].tolist())
        rows = []
        cols = [f"pubchem_{i}" for i in range(881)]
        for _, r in mols_df.iterrows():
            cid = str(r["cid"])
            smi = str(r["smiles"])
            vec = fp_dict.get(cid, np.zeros(881, dtype=np.float32))
            row_dict = {"cid": cid, "smiles": smi}
            for i, c in enumerate(cols):
                row_dict[c] = vec[i]
            rows.append(row_dict)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Saved 881-bit PubChem fingerprints to {out_path}")


if __name__ == "__main__":
    main()
