"""The CPU inference image installs only the core dependencies: importing the package must not need torch."""

import subprocess
import sys

HEAVY = ("torch", "torchvision", "onnx", "datasets", "pandas", "pyarrow", "onnxruntime")


def test_import_hs_hot_loads_no_training_or_dataset_framework():
    code = (
        "import sys, hs_hot;"
        f"bad = [m for m in {HEAVY!r} if m in sys.modules];"
        "assert not bad, bad;"
        "assert callable(hs_hot.predict_session) and callable(hs_hot.decode_polygons)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-800:]


def test_serving_and_evaluation_modules_import_without_torch():
    code = (
        "import sys, hs_hot.serve, hs_hot.evaluate, hs_hot.export, hs_hot.dataset, hs_hot.tiles, hs_hot.decode, hs_hot.georef, hs_hot.postprocess;"
        f"bad = [m for m in {HEAVY!r} if m in sys.modules];"
        "assert not bad, bad"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-800:]
