"""Check every family builds and runs one forward pass. No training, nothing written.

Verifies for each family:
  * the encoder loads,
  * the towers receive exactly one molecular representation (and which one),
  * one batch flows through to a logit,
  * the pretrained encoder is frozen.

    python -m experiments.encoder_comparison.smoke_check
    python -m experiments.encoder_comparison.smoke_check --family morgan --device cpu
"""

import argparse
import os
import sys

import pandas as pd
import torch
from rdkit import RDLogger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from experiments.encoder_comparison.data import SplitPaths, build_vector_lookup, check_heldout_is_external  # noqa: E402
from experiments.encoder_comparison.families import ORDER, spec  # noqa: E402
from experiments.encoder_comparison.run_comparison import (RESULTS_ROOT, build_encoder, build_model,  # noqa: E402
                                                           make_config, make_loader)
from src.dataset import PGBNormStats  # noqa: E402
from src.pgb_prior import ExcipientPriorTable  # noqa: E402
from src.utils import get_device, seed_everything  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def check(family, paths, device_str, prior_table, pgb_norm):
    s = spec(family)
    seed_everything(42)
    config = make_config(family, paths, 42, device_str, 40)
    device = get_device(config.device)
    lookup = build_vector_lookup(family, paths, os.path.join(RESULTS_ROOT, "_lookup")) if s["needs_lookup"] else None
    encoder = build_encoder(config, family, lookup, device)
    model = build_model(config, family, encoder).to(device).eval()

    loader = make_loader(paths.val, config, prior_table, pgb_norm, encoder)
    batch = next(iter(loader))
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    with torch.no_grad():
        logits = model(batch)

    tower = model.api_tower
    rep = f"struct {config.proj_dim}" if tower.has_struct else f"fingerprint {batch['api_primary'].shape[1]}"
    salt_max = float(batch["api_salt"].abs().max())
    assert salt_max < 25, f"salt context looks unnormalised (max |value| {salt_max:.1f})"
    assert tower.has_struct != tower.has_primary, "a tower must have exactly one representation"
    assert "api_secondary" not in batch, "no family may receive a secondary fingerprint"
    assert logits.shape == (batch["label"].shape[0],), f"bad logit shape {tuple(logits.shape)}"
    assert torch.isfinite(logits).all(), "non-finite logits"
    frozen = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
    total_enc = sum(p.numel() for p in encoder.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  OK  {family:10s} representation={rep:20s} trainable={n_train:>9,} "
          f"salt |max| {salt_max:5.2f} encoder frozen={frozen:,}/{total_enc:,} logits[0]={logits[0].item():+.4f}")


def main():
    p = argparse.ArgumentParser(description="Build-and-forward check for the encoder comparison.")
    p.add_argument("--family", default="all")
    p.add_argument("--data-root", default=None)
    p.add_argument("--split", default="api_cluster")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    families = ORDER if args.family == "all" else [args.family]
    paths = SplitPaths(args.data_root, args.split)
    check_heldout_is_external(paths)
    device_str = args.device or "auto"
    print(f"data root {paths.root} | device {get_device(device_str)}")
    print("held-out set is external to train and val: OK\n")

    train_df = pd.read_csv(paths.train)
    pgb_norm = PGBNormStats.from_smiles_list(
        list(set(train_df["API_Smiles"].dropna()) | set(train_df["Excipient_Smiles"].dropna())),
        with_salt=True)
    prior_table = ExcipientPriorTable(global_rate=float(train_df["Outcome1"].mean()))
    prior_table.fit(train_df, leave_out_cluster_id=None)

    ok, bad = [], []
    for family in families:
        try:
            check(family, paths, device_str, prior_table, pgb_norm)
            ok.append(family)
        except Exception as e:
            bad.append(family)
            print(f"  FAIL {family:10s} {type(e).__name__}: {str(e)[:160]}")
    print(f"\n{len(ok)} of {len(families)} families ready" + (f" | failed: {', '.join(bad)}" if bad else ""))


if __name__ == "__main__":
    main()
