"""Every file and folder location of the project, in one place.

All paths are anchored on the project root (the folder that holds run.py), never on the current
working directory, so every script works from any directory and the project folder can be moved
or copied anywhere.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- data (inputs, never written by training)
DATA = ROOT / "data"
RAW_DATASET = DATA / "raw" / "start_dataset.csv"              # all 3544 labelled pairs the splits are made from
HELDOUT_CSV = DATA / "heldout" / "held_out_test_set.csv"      # the 22 external test pairs (label column ground_truth)
SPLITS = DATA / "splits"                                      # one folder per train / val split
DEFAULT_SPLIT = "api_cluster"                                 # the split the final model was trained on
BENCHMARK_DATA = DATA / "benchmark"                           # extra validation pairs for benchmarking
BASELINES_DATA = DATA / "baselines"                           # held-out predictions of other published models

# --------------------------------------------------------------------------- trained models
CHECKPOINTS = ROOT / "checkpoints"
FINAL_CHECKPOINT = CHECKPOINTS / "morgan_pgb_8020_seed42"     # the shipped model (18/22), never overwritten
ABLATION_CHECKPOINTS = CHECKPOINTS / "ablations"

# --------------------------------------------------------------------------- outputs
RESULTS = ROOT / "results"
REFERENCE_RESULTS = RESULTS / "reference"                     # original metrics of the shipped model
EVALUATION_RESULTS = RESULTS / "evaluation"                   # run.py train / evaluate, one folder per checkpoint
PREDICTION_RESULTS = RESULTS / "predictions"                  # run.py predict
REBUILT_SPLIT = RESULTS / "rebuilt_split"                     # run.py split
ABLATION_RESULTS = RESULTS / "ablations"
ABLATION_RUNS = ABLATION_RESULTS / "runs"                     # one folder per ablation + seed
ABLATION_MULTI_SEED = ABLATION_RESULTS / "multi_seed"         # <ablation>_<n>seeds: seeds combined + ensemble
ABLATION_LOGS = ABLATION_RESULTS / "logs"                     # training logs of run_seeds.py
ABLATION_SUMMARY = ABLATION_RESULTS / "summary.csv"           # one row per ablation + seed
SHAP_RESULTS = RESULTS / "shap"
BENCHMARK_RESULTS = RESULTS / "benchmark"
FIGURES = RESULTS / "figures"


def split_dir(name: str) -> Path:
    """Folder of a named split, e.g. split_dir("random") -> data/splits/random."""
    return SPLITS / name


def available_splits() -> list[str]:
    return sorted(p.name for p in SPLITS.iterdir() if (p / "train.csv").exists())


def rel(path) -> str:
    """Path relative to the project root (forward slashes) for logs and run_info files; absolute if outside."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve(path) -> Path:
    """A path typed by the user. Absolute paths are kept. A relative path is taken from the current
    directory if it exists there, else from the project root if it exists there (so `checkpoints/...`
    works from any directory), else from the current directory (new output files)."""
    p = Path(path)
    if p.is_absolute():
        return p
    for base in (Path(os.getcwd()), ROOT):
        if (base / p).exists():
            return base / p
    return Path(os.getcwd()) / p
