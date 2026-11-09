"""The four entry points, run as a user would run them, on toy chips (tiny scale so a CPU does it in seconds)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def run(script: str, *args: str | Path) -> subprocess.CompletedProcess:
    done = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *map(str, args)],
        capture_output=True,
        text=True,
        cwd=SCRIPTS.parent,
    )
    assert done.returncode == 0, done.stderr[-1500:]
    return done


@pytest.mark.parametrize("script", ["finetune.py", "evaluate.py", "predict.py", "export_onnx.py"])
def test_every_script_prints_its_usage(script):
    assert run(script, "--help").stdout.startswith("usage:")


def test_finetune_then_evaluate_export_and_predict(labelled_chips, tmp_path):
    chips, labels, _ = labelled_chips
    dataset = chips.parent  # a dataset folder holds chips/ and labels/
    out = tmp_path / "run"
    run(
        "finetune.py", "--train-dir", dataset, "--val-dir", dataset, "--out", out, "--name", "TOY",
        "--model-gsd-m", "0.5", "--epochs", "1", "--batch-size", "2", "--device", "cpu",
    )  # fmt: skip
    assert {path.name for path in out.iterdir()} >= {
        "TOY.pt",
        "TOY.onnx",
        "history.json",
        "train.log",
        "epochs",
    }
    history = json.loads((out / "history.json").read_text())
    assert history["best_epoch"] == 1 and len(history["train_loss"]) == 1

    via_weights, via_onnx = tmp_path / "weights.json", tmp_path / "onnx.json"
    common = ["--chips", chips, "--labels", labels, "--model-gsd-m", "0.5", "--thresholds", "0.3", "0.5"]
    run("evaluate.py", "--weights", out / "TOY.pt", *common, "--device", "cpu", "--json", via_weights)
    run("evaluate.py", "--onnx", out / "TOY.onnx", *common, "--json", via_onnx)
    weights, onnx = json.loads(via_weights.read_text()), json.loads(via_onnx.read_text())
    assert [row["threshold"] for row in weights["sweep"]] == [0.3, 0.5] and set(weights["best"]) >= {
        "f1",
        "iou",
    }
    assert onnx["best"]["iou"] == pytest.approx(weights["best"]["iou"], abs=0.01)  # torch and ONNX agree

    run("export_onnx.py", out / "TOY.pt", tmp_path / "again.onnx")
    assert (tmp_path / "again.onnx").stat().st_size == (out / "TOY.onnx").stat().st_size

    geojson = tmp_path / "predictions.geojson"
    run("predict.py", "--onnx", out / "TOY.onnx", "--chips", chips, "--out", geojson,
        "--params", '{"threshold": 0.0, "model_gsd_m": 0.5, "min_area_px": 1}')  # fmt: skip
    collection = json.loads(geojson.read_text())
    assert (
        collection["type"] == "FeatureCollection" and len(collection["features"]) >= 4
    )  # threshold 0: all road
    assert collection["features"][0]["properties"]["class_name"] == "highway"


def test_finetune_refuses_a_folder_that_is_not_empty(labelled_chips):
    chips, _, _ = labelled_chips
    done = subprocess.run(
        [sys.executable, str(SCRIPTS / "finetune.py"), "--train-dir", str(chips.parent), "--out", str(chips.parent)],
        capture_output=True, text=True, cwd=SCRIPTS.parent,
    )  # fmt: skip
    assert done.returncode != 0 and "is not empty" in done.stderr
