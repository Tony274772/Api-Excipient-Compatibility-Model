"""Checks that the data, the shipped model and the entry points are consistent.

Run from any directory:  python -m pytest path/to/final_project/tests
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apiexc import paths  # noqa: E402
from apiexc.evaluation.metrics import metric_suite  # noqa: E402
from apiexc.pipeline import get_split, heldout_overlap, load_checkpoint  # noqa: E402


def test_paths_stay_inside_the_project():
    for name in dir(paths):
        value = getattr(paths, name)
        if isinstance(value, Path):
            assert value.resolve().is_relative_to(ROOT), f"paths.{name} = {value} is outside the project"


def test_heldout_set_has_22_pairs():
    ho = pd.read_csv(paths.HELDOUT_CSV)
    assert len(ho) == 22
    assert int(ho["ground_truth"].sum()) == 10
    assert not ho.duplicated(["API_CID", "Excipient_CID"]).any()


@pytest.mark.parametrize("split_name", paths.available_splits())
def test_no_heldout_pair_in_train_or_val(split_name):
    overlap = heldout_overlap(get_split(split_name))
    assert overlap == {"train": [], "val": []}


def test_every_checkpoint_split_exists():
    for f in paths.CHECKPOINTS.glob("*/split.txt"):
        assert get_split(f.read_text().strip()).train_csv.exists(), f


@pytest.fixture(scope="module")
def shipped_model():
    return load_checkpoint(paths.FINAL_CHECKPOINT, "cpu")


@pytest.mark.parametrize("which", ["val", "heldout"])
def test_shipped_model_reproduces_reference(shipped_model, which):
    split = get_split()
    probs, labels = shipped_model.score(split.frame(which))
    m = metric_suite(probs, labels, shipped_model.threshold)
    ref = json.loads((paths.REFERENCE_RESULTS / f"{which}_metrics.json").read_text())
    assert m["confusion_matrix"] == ref["confusion_matrix"]
    assert np.isclose(m["pr_auc"], ref["pr_auc"], atol=1e-6)
    if which == "heldout":
        assert m["correct"] == 18 and m["n"] == 22


def test_predict_runs_from_another_directory(tmp_path):
    out = tmp_path / "pred.csv"
    subprocess.run([sys.executable, str(ROOT / "run.py"), "predict", "--device", "cpu",
                    "--csv", str(paths.HELDOUT_CSV), "--out", str(out)],
                   cwd=tmp_path, check=True, capture_output=True)
    pred = pd.read_csv(out)
    assert len(pred) == 22 and int(pred["correct"].sum()) == 18
