"""Do the 5 SMARTS mechanism flags change what a trained PGB model predicts? Inference only, no training.

For every seed's checkpoint, the val pairs and the 22 held-out pairs are scored four ways:

    as trained       the flags the model was trained with (src/lookup.py)
    all off          mech_flags set to 0 for every pair
    one off          one flag at a time set to 0 (only matters where that flag fired)
    one forced on    one flag at a time set to 1 for every pair (does the model react to it at all?)

and the change in probability, PR-AUC / ROC-AUC and the number of flipped decisions (at the run's own
threshold) is reported. The script also prints how large the first-layer weights on the 5 flag inputs
are next to the other inputs of the gate and the head MLP.

Default checkpoints: the control arm of experiments/reagent_ablation (the top5_seeds_depth6 recipe, with
the same results). Run on the machine that holds them:

    python experiments/mech_flags/knockout.py --device cuda
    python experiments/mech_flags/knockout.py --ckpt-root checkpoints/top5_seeds_depth6 --data data/grid_8020

Writes experiments/results/mech_flags/knockout_per_seed.csv and knockout_rows.csv (per pair, per seed).
"""

import argparse
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from rdkit import RDLogger  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.cross_validate import build_encoder  # noqa: E402
from src.dataset import PGBDataset, PGBNormStats, pgb_collate_fn  # noqa: E402
from src.lookup import get_mechanism_names  # noqa: E402
from src.model import pgb_model_from_config  # noqa: E402
from src.pgb_prior import ExcipientPriorTable  # noqa: E402
from src.utils import get_device  # noqa: E402

RDLogger.DisableLog("rdApp.*")
RUN = "pgb_dmpnn_chemprop_cross_attn_cls_asl"
NAMES = get_mechanism_names()
OUT = os.path.join(PROJECT_ROOT, "experiments", "results", "mech_flags")


def load(ckpt_dir, device):
    ck = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location="cpu", weights_only=False)
    config = ck["config"]
    config.device = str(device)
    encoder = build_encoder(config, device)
    if not encoder.is_sequence_capable:
        config.fusion = "concat"
    model = pgb_model_from_config(config, encoder).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    norm = PGBNormStats.load(os.path.join(ckpt_dir, "pgb_norm_stats.npz"))
    prior = ExcipientPriorTable.load(os.path.join(ckpt_dir, "prior_table.json"))
    fv = encoder if config.encoder == "fixed_vector" else None
    return model, config, norm, prior, fv


def batches(csv, config, norm, prior, fv):
    ds = PGBDataset(csv, config.encoder, getattr(config, "fixed_vector_source", "maccs"), prior, norm,
                    fixed_vector_encoder=fv)
    return list(DataLoader(ds, batch_size=config.pgb_batch_size, shuffle=False,
                           collate_fn=pgb_collate_fn, num_workers=0))


@torch.no_grad()
def score(model, cached, device, edit=None):
    probs, labels, flags = [], [], []
    for b in cached:
        b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        labels.append(b.pop("label").cpu())
        b.pop("sample_weight", None)
        flags.append(b["mech_flags"].cpu())
        if edit is not None:
            b["mech_flags"] = edit(b["mech_flags"].clone())
        probs.append(torch.sigmoid(model(b)).cpu())
    return torch.cat(probs).numpy(), torch.cat(labels).numpy().astype(int), torch.cat(flags).numpy()


def summarise(base, new, y, thr):
    d = new - base
    return {"pr_auc": average_precision_score(y, new), "roc_auc": roc_auc_score(y, new),
            "mean_abs_dprob": float(np.abs(d).mean()), "max_abs_dprob": float(np.abs(d).max()),
            "flipped": int(((base >= thr) != (new >= thr)).sum())}


def weight_report(model):
    head = model.head
    k = len(NAMES)
    for label, layer in (("gate", head.gate[0]), ("head MLP", head.mlp[0])):
        col = layer.weight.detach().norm(dim=0).cpu().numpy()        # L2 norm of each input column
        others = col[:-k]
        print(f"  {label:8s} first layer: mech-flag column norms "
              + " ".join(f"{n[:8]}={v:.3f}" for n, v in zip(NAMES, col[-k:]))
              + f" | other inputs median {np.median(others):.3f}, 90th pct {np.percentile(others, 90):.3f}")


def main():
    ap = argparse.ArgumentParser(description="Mechanism-flag knockout on trained PGB checkpoints.")
    ap.add_argument("--ckpt-root", default="checkpoints/reagent_ablation/control",
                    help="folder with seed_<s>/<run>/best_model.pt")
    ap.add_argument("--results-root", default="experiments/results/reagent_ablation/control",
                    help="folder with seed_<s>/<run>/val_metrics.json (threshold and PR-AUC check)")
    ap.add_argument("--data", default="data/grid_8020")
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    seeds = sorted(glob.glob(os.path.join(args.ckpt_root, "seed_*", args.run, "best_model.pt")))
    if not seeds:
        raise SystemExit(f"no checkpoints under {args.ckpt_root}/seed_*/{args.run}/")
    os.makedirs(OUT, exist_ok=True)
    per_seed, per_row = [], []
    for path in seeds:
        ckpt_dir = os.path.dirname(path)
        seed = int(ckpt_dir.split("seed_")[-1].split(os.sep)[0].split("/")[0])
        model, config, norm, prior, fv = load(ckpt_dir, device)
        saved = json.load(open(os.path.join(args.results_root, f"seed_{seed}", args.run, "val_metrics.json")))
        thr = saved["threshold"]
        print(f"\n=== seed {seed}  (threshold {thr})")
        weight_report(model)
        for split, csv in (("val", os.path.join(args.data, "val.csv")), ("heldout", os.path.join(args.data, "test.csv"))):
            cached = batches(csv, config, norm, prior, fv)
            base, y, F = score(model, cached, device)
            if split == "val" and abs(average_precision_score(y, base) - saved["pr_auc"]) > 1e-4:
                print(f"  WARNING: rebuilt model gives val PR-AUC {average_precision_score(y, base):.4f}, "
                      f"run saved {saved['pr_auc']:.4f}")
            fired = F.any(1)
            variants = {"all_off": lambda m: m.zero_()}
            for i, n in enumerate(NAMES):
                variants[f"off:{n}"] = (lambda i: lambda m: m.index_fill_(1, torch.tensor([i], device=m.device), 0.0))(i)
                variants[f"on:{n}"] = (lambda i: lambda m: m.index_fill_(1, torch.tensor([i], device=m.device), 1.0))(i)
            r0 = {"seed": seed, "split": split, "variant": "as_trained", "pr_auc": average_precision_score(y, base),
                  "roc_auc": roc_auc_score(y, base), "rows_with_any_flag": int(fired.sum())}
            per_seed.append(r0)
            for name, edit in variants.items():
                new, _, _ = score(model, cached, device, edit)
                r = {"seed": seed, "split": split, "variant": name, **summarise(base, new, y, thr),
                     "rows_with_any_flag": int(fired.sum())}
                per_seed.append(r)
                if name == "all_off":
                    per_row.append(pd.DataFrame({"seed": seed, "split": split, "label": y, "prob": base,
                                                 "prob_flags_off": new, **{n: F[:, i] for i, n in enumerate(NAMES)}}))
                    print(f"  {split:7s} all flags off: PR-AUC {r0['pr_auc']:.4f} -> {r['pr_auc']:.4f} | "
                          f"mean |dprob| {r['mean_abs_dprob']:.4f} (on {fired.sum()} flagged rows: "
                          f"{np.abs(new - base)[fired].mean() if fired.any() else 0:.4f}) | "
                          f"max {r['max_abs_dprob']:.4f} | flipped {r['flipped']}")

    ps = pd.DataFrame(per_seed)
    ps.to_csv(os.path.join(OUT, "knockout_per_seed.csv"), index=False)
    pd.concat(per_row).to_csv(os.path.join(OUT, "knockout_rows.csv"), index=False)
    cols = ["pr_auc", "roc_auc", "mean_abs_dprob", "max_abs_dprob", "flipped"]
    pd.set_option("display.width", 250)
    print("\nmean over seeds:")
    print(ps.groupby(["split", "variant"], sort=False)[cols].mean().round(4).to_string())
    print(f"\nwritten to {os.path.relpath(OUT, PROJECT_ROOT)}/knockout_per_seed.csv and knockout_rows.csv")


if __name__ == "__main__":
    main()
