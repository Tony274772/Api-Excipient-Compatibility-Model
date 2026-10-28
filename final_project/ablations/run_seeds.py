"""
Train several ablations on several seeds, then collect each one's seeds (collect_seeds.py).

Every run is a separate `run_ablation.py train` process, so each seed starts from a clean state. Runs that
already have a checkpoint are skipped, so the script can be stopped and started again.

Usage (from final_project/):
    python ablations/run_seeds.py                                   # all ablations below, seeds 42-46
    python ablations/run_seeds.py --ablations no_w no_prior --seeds 42 43 44 45 46
    python ablations/run_seeds.py --collect-only                    # only rebuild the <ablation>_5seeds folders
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL_PROJECT = os.path.dirname(HERE)

ABLATIONS = ["full", "no_w", "prior1", "no_gate", "no_bilinear", "no_prior", "no_salt", "no_avail", "no_mech",
             "no_flags", "no_flags_api", "no_flags_exc", "no_flags_metal", "prior1+no_family"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablations", nargs="+", default=ABLATIONS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args()

    log_dir = os.path.join(HERE, "results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    runs = [(a, s) for a in args.ablations for s in args.seeds]
    t0 = time.time()
    for i, (ablation, seed) in enumerate(runs, 1):
        if args.collect_only:
            break
        ckpt = os.path.join(HERE, "checkpoints", f"{ablation}_seed{seed}", "best_model.pt")
        if os.path.exists(ckpt):
            print(f"[{i}/{len(runs)}] {ablation} seed {seed}: exists, skipped")
            continue
        log = os.path.join(log_dir, f"{ablation}_seed{seed}.txt")
        print(f"[{i}/{len(runs)}] {ablation} seed {seed}: training ... (log {log})", flush=True)
        with open(log, "w") as f:
            code = subprocess.call([sys.executable, os.path.join(HERE, "run_ablation.py"), "train", "--ablation", ablation,
                                    "--seed", str(seed), "--device", args.device], stdout=f, stderr=subprocess.STDOUT,
                                   cwd=FINAL_PROJECT)
        print(f"    exit {code} | {(time.time() - t0) / 60:.1f} min so far", flush=True)
        if code != 0:
            print(f"    failed, see {log}")

    for ablation in args.ablations:
        have = [s for s in args.seeds
                if os.path.exists(os.path.join(HERE, "results", f"{ablation}_seed{s}", "predictions_heldout.csv"))]
        if len(have) < 2:
            print(f"{ablation}: {len(have)} seed(s) done, not collected")
            continue
        subprocess.call([sys.executable, os.path.join(HERE, "collect_seeds.py"), "--ablation", ablation,
                         "--seeds", *map(str, have)], cwd=FINAL_PROJECT, stdout=subprocess.DEVNULL)
        print(f"{ablation}: collected seeds {have} -> ablations/results/{ablation}_{len(have)}seeds/")


if __name__ == "__main__":
    main()
