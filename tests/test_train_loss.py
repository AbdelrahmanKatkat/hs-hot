"""The loss: cross-entropy plus Dice over the pixels that are not ignored."""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from hs_hot.dataset import BACKGROUND, IGNORE, ROAD  # noqa: E402
from hs_hot.train import _loss_parts  # noqa: E402


def _mask() -> torch.Tensor:
    mask = torch.full((1, 8, 8), BACKGROUND, dtype=torch.uint8)
    mask[0, :4, :4] = ROAD
    return mask


def _logits_for(mask: torch.Tensor, confidence: float) -> torch.Tensor:
    return torch.where(mask == ROAD, confidence, -confidence).float()[:, None]


def test_a_confident_correct_prediction_has_almost_no_loss_and_a_wrong_one_a_lot():
    mask = _mask()
    right, wrong = _loss_parts(_logits_for(mask, 9.0), mask), _loss_parts(_logits_for(mask, -9.0), mask)
    assert right["bce"] < 1e-3 and right["dice"] < 0.05
    assert wrong["bce"] > 5 and wrong["dice"] > 0.9


def test_ignored_pixels_do_not_enter_the_loss():
    mask = _mask()
    mask[0, 6:, :] = IGNORE
    logits = _logits_for(mask, 2.0)
    wild = logits.clone()
    wild[0, 0, 6:, :] = 50.0  # a wild prediction on the ignored rows
    reference, changed = _loss_parts(logits, mask), _loss_parts(wild, mask)
    assert float(changed["bce"]) == pytest.approx(float(reference["bce"]))
    assert float(changed["dice"]) == pytest.approx(float(reference["dice"]))


def test_a_batch_without_any_tree_or_with_everything_ignored_has_a_defined_loss():
    parts = _loss_parts(torch.full((2, 1, 8, 8), -9.0), torch.zeros((2, 8, 8), dtype=torch.uint8))
    assert torch.isfinite(parts["bce"]) and float(parts["dice"]) < 0.05
    ignored = _loss_parts(torch.zeros((1, 1, 4, 4)), torch.full((1, 4, 4), IGNORE, dtype=torch.uint8))
    assert float(ignored["bce"]) == 0.0 and torch.isfinite(ignored["dice"])
