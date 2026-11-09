"""End to end, offline and tiny: train a few epochs on toy chips, export, and score through torch and through ONNX."""

import pytest

pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

import torch  # noqa: E402

from hs_hot.evaluate import score_dataset, score_session, sweep_threshold  # noqa: E402
from hs_hot.export import build_unet, export_onnx_bytes  # noqa: E402
from hs_hot.params import PreprocessParams, TrainParams  # noqa: E402
from hs_hot.train import train  # noqa: E402

PRE = PreprocessParams(model_gsd_m=0.5)  # a zoom-20 chip becomes about 77 px: fast on a CPU
CFG = TrainParams(epochs=3, batch_size=2, learning_rate=1e-3, early_stop_patience=3, device="cpu", seed=0)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    root = tmp_path_factory.mktemp("smoke")
    chips = root / "chips"
    chips.mkdir()
    names = [f"OAM-{524288 + index}-524288-20.tif" for index in range(4)]
    rng = np.random.default_rng(0)
    band = {"kind": "road", "line": np.array([[-100.0, 64.0], [356.0, 64.0]]), "width_px": 128.0}
    labels = {}
    for name in names:
        data = rng.integers(0, 60, size=(3, 256, 256), dtype=np.uint8)
        data[:, :128, :] = 230  # a bright band: the "road" the model can learn
        with rasterio.open(
            chips / name, "w", driver="GTiff", height=256, width=256, count=3, dtype="uint8",
            crs="EPSG:3857", transform=from_origin(1_000_000.0, 2_000_000.0, 1.0, 1.0),
        ) as dst:  # fmt: skip
            dst.write(data)
        labels[name] = [band]
    epochs_seen: list[tuple[int, float, float | None]] = []
    checkpoints = root / "epochs"
    result = train(
        train_images_dir=chips, train_labels=labels, val_images_dir=chips, val_labels=labels,
        cfg=CFG, pre=PRE, train_names=names[:3], val_names=names[3:],
        on_epoch_end=lambda *seen: epochs_seen.append(seen), epoch_checkpoint_dir=checkpoints,
    )  # fmt: skip
    return result, chips, labels, names, epochs_seen, checkpoints


def test_training_reports_every_epoch_keeps_the_best_and_saves_checkpoints(trained):
    result, _, _, _, epochs_seen, checkpoints = trained
    assert [epoch for epoch, _, _ in epochs_seen] == [1, 2, 3]
    assert len(result.history["train_loss"]) == len(result.history["val_loss"]) == 3
    assert result.best_loss == pytest.approx(min(result.history["val_loss"]))
    assert result.history["train_loss"][-1] < result.history["train_loss"][0]  # it learned something
    assert sorted(path.name for path in checkpoints.glob("*.pt")) == [
        "epoch_01.pt",
        "epoch_02.pt",
        "epoch_03.pt",
    ]
    assert not result.model.training and next(result.model.parameters()).device.type == "cpu"


def test_a_saved_epoch_continues_training(trained):
    _, chips, labels, names, _, checkpoints = trained
    assert build_unet(weights=checkpoints / "epoch_03.pt")(torch.rand(1, 3, 40, 40)).shape == (1, 1, 40, 40)
    again = train(
        train_images_dir=chips, train_labels=labels, val_images_dir=None, val_labels=None,
        cfg=TrainParams(epochs=1, batch_size=2, device="cpu", seed=1), pre=PRE,
        checkpoint=checkpoints / "epoch_03.pt", train_names=names[:2],
    )  # fmt: skip
    assert len(again.history["train_loss"]) == 1 and again.history["val_loss"] == []


def test_onnx_scores_like_torch_and_the_sweep_lists_every_threshold(trained):
    result, chips, labels, names, _, _ = trained
    session = ort.InferenceSession(export_onnx_bytes(result.model), providers=["CPUExecutionProvider"])
    params = {"model_gsd_m": 0.5}
    via_onnx = score_session(session, images_dir=chips, labels=labels, names=names, params=params)[0.5]
    via_torch = score_dataset(result.model, images_dir=chips, labels=labels, names=names, params=params)[0.5]
    assert set(via_onnx) == {"precision", "recall", "f1", "iou", "tp", "fp", "fn"}
    assert via_onnx["iou"] == pytest.approx(via_torch["iou"], abs=0.01)
    rows = sweep_threshold(result.model, images_dir=chips, labels=labels, names=names, params=params)
    assert [row["threshold"] for row in rows] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert all(
        rows[i]["recall"] >= rows[i + 1]["recall"] for i in range(len(rows) - 1)
    )  # a higher bar loses recall
