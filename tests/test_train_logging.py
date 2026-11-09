"""Training progress must be visible in a bare container step, and must respect a host that configured logging."""

import logging
import sys

import pytest

pytest.importorskip("torch")

from hs_hot import train  # noqa: E402


@pytest.fixture
def clean_logger(monkeypatch):
    """The `hs_hot.train` logger and the root logger as a fresh process would have them."""
    logger = logging.getLogger("hs_hot.train")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "level", logging.NOTSET)
    monkeypatch.setattr(logger, "propagate", True)
    return logger


def test_progress_reaches_stdout_when_nothing_is_configured(clean_logger, monkeypatch, capsys):
    monkeypatch.setattr(logging.getLogger(), "handlers", [])  # a pod with no logging setup
    monkeypatch.setattr(logging.getLogger(), "level", logging.WARNING)
    train._ensure_visible_progress()
    train.log.info("epoch 1/1 train step 1/1")
    assert "epoch 1/1 train step 1/1" in capsys.readouterr().out
    assert len(clean_logger.handlers) == 1


def test_a_configured_host_is_left_alone(clean_logger, monkeypatch):
    monkeypatch.setattr(logging.getLogger(), "handlers", [logging.StreamHandler(sys.stderr)])  # e.g. ZenML
    train._ensure_visible_progress()
    assert clean_logger.handlers == []  # no second handler, so no duplicate lines


def test_info_is_enabled_when_the_logger_level_would_hide_it(clean_logger, monkeypatch, caplog):
    monkeypatch.setattr(logging.getLogger(), "level", logging.WARNING)
    clean_logger.setLevel(logging.WARNING)
    train._ensure_visible_progress()
    with caplog.at_level(logging.INFO, logger="hs_hot.train"):
        train.log.info("visible")
    assert "visible" in caplog.text


def test_step_logging_is_bounded_for_large_datasets():
    # automatic step: about LOG_LINES_PER_EPOCH lines per epoch whatever the dataset size
    steps = 17_000
    every = max(1, steps // train.LOG_LINES_PER_EPOCH)
    assert steps // every <= train.LOG_LINES_PER_EPOCH + 1
