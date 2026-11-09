"""Device choice and train mode: the encoder's BatchNorm statistics must stay those of the pretrained checkpoint."""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from hs_hot.train import _resolve_device, _set_train_mode  # noqa: E402


def test_cpu_is_always_honoured_and_auto_falls_back_without_cuda(monkeypatch):
    assert _resolve_device("cpu") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _resolve_device("auto") == "cpu" and _resolve_device("cuda") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert (
        _resolve_device("auto") == "cuda"
        and _resolve_device(" GPU ") == "cuda"
        and _resolve_device("1") == "cuda:1"
    )


class TwoPartModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4))
        self.decoder = torch.nn.Sequential(torch.nn.Conv2d(4, 4, 3), torch.nn.BatchNorm2d(4))


def test_train_mode_freezes_only_the_encoder_batchnorm():
    model = TwoPartModel().eval()
    _set_train_mode(model)
    assert model.training and model.encoder[0].training and model.decoder[1].training
    assert not model.encoder[1].training  # running mean and variance stay as pretrained
