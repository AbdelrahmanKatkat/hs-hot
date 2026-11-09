"""Learning-rate schedule (per batch, warmup then cosine) and the per-epoch random sample of the training tiles."""

import math

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from hs_hot.params import TrainParams, parse_train_params  # noqa: E402
from hs_hot.train import _epoch_loader, _lr_schedule  # noqa: E402


def lrs(total, warmup, cosine=True, base=1.0):
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=base)
    sched = _lr_schedule(opt, total, warmup, cosine)
    out = []
    for _ in range(total):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return out


def test_warmup_ramps_up_linearly_then_cosine_decays_to_zero():
    values = lrs(100, 10)
    assert values[0] == pytest.approx(0.1 * 0.5 * (1 + math.cos(0)))  # first step: 1/10 of the base rate
    assert values[9] == pytest.approx(0.5 * (1 + math.cos(math.pi * 9 / 100)))  # warmup done, cosine only
    assert all(a < b for a, b in zip(values[:9], values[1:10]))  # still rising during the warmup
    assert all(a >= b for a, b in zip(values[10:], values[11:]))  # then only falling
    assert values[-1] < 1e-3


def test_cosine_without_warmup_starts_at_the_base_rate_and_changes_every_batch():
    values = lrs(40, 0)
    assert values[0] == pytest.approx(1.0) and len(set(values)) == 40
    assert values[20] == pytest.approx(0.5)


def test_warmup_only_and_no_schedule():
    assert lrs(20, 5, cosine=False)[-1] == pytest.approx(1.0)
    assert _lr_schedule(torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0), 10, 0, False) is None


class Numbers(torch.utils.data.Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, i):
        return torch.full((3, 2, 2), float(i)), torch.zeros(
            2, 2, dtype=torch.uint8
        )  # an image whose value is its index


def _seen(ds, cfg, epoch):
    return sorted(int(x) for images, _ in _epoch_loader(ds, cfg, epoch) for x in images[:, 0, 0, 0])


def test_every_epoch_draws_a_fresh_random_fraction_of_the_tiles():
    ds, cfg = Numbers(), TrainParams(epoch_fraction=0.2, batch_size=4, seed=7)
    first, second = _seen(ds, cfg, 0), _seen(ds, cfg, 1)
    assert len(first) == len(second) == 20 and first != second and _seen(ds, cfg, 0) == first  # reproducible
    assert _seen(ds, TrainParams(epoch_fraction=1.0, batch_size=4), 0) == list(range(100))


def test_new_parameters_have_safe_defaults_and_are_validated():
    cfg = parse_train_params(None)
    assert (cfg.warmup_steps, cfg.epoch_fraction) == (0, 0.1)  # ten epochs of a tenth = one pass
    cfg = parse_train_params({"warmup_steps": "300", "epoch_fraction": 0.2})
    assert (cfg.warmup_steps, cfg.epoch_fraction) == (300, 0.2)
    for bad in ({"warmup_steps": -1}, {"epoch_fraction": 0}, {"epoch_fraction": 1.5}):
        with pytest.raises(ValueError):
            parse_train_params(bad)
