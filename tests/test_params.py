import pytest

from hs_hot.params import (
    parse_postprocess_params,
    parse_preprocess_params,
    parse_split_params,
    parse_train_params,
)


@pytest.mark.parametrize("empty", [None, {}, {"model_gsd_m": None}])
def test_preprocess_defaults_for_empty_params(empty):
    assert parse_preprocess_params(empty).model_gsd_m == 0.30


def test_max_input_px_default_override_and_bounds():
    assert parse_preprocess_params(None).max_input_px == 1024
    assert parse_preprocess_params({"max_input_px": "1333"}).max_input_px == 1333
    for bad in ({"max_input_px": 32}, {"max_input_px": 100000}):
        with pytest.raises(ValueError):
            parse_preprocess_params(bad)


def test_preprocess_params_override_and_bounds():
    assert parse_preprocess_params({"model_gsd_m": "0.1"}).model_gsd_m == 0.1
    for bad in ({"model_gsd_m": 0}, {"model_gsd_m": 5}, {"model_gsd_m": True}):
        with pytest.raises(ValueError):
            parse_preprocess_params(bad)


def test_split_params():
    s = parse_split_params(None)
    assert (s.val_ratio, s.split_seed, s.block_size) == (0.2, 42, 4)
    assert parse_split_params({"val_ratio": 0.3, "split_seed": 7}).split_seed == 7
    with pytest.raises(ValueError):
        parse_split_params({"val_ratio": 0.0})


def test_train_params_defaults():
    cfg = parse_train_params({})
    assert (cfg.learning_rate, cfg.weight_decay, cfg.max_grad_norm, cfg.scheduler) == (
        1e-3,
        1e-4,
        1.0,
        "cosine",
    )
    assert cfg.freeze_encoder is False and cfg.epochs >= 1 and cfg.batch_size >= 1  # the whole encoder trains
    with pytest.raises(ValueError):
        parse_train_params({"epochs": 0})


def test_train_params_overrides_alias_and_bounds():
    cfg = parse_train_params(
        {"lr": "0.01", "freeze_encoder": "false", "epochs": 3, "seed": None, "scheduler": "None"}
    )
    assert cfg.learning_rate == 0.01 and cfg.freeze_encoder is False and cfg.epochs == 3 and cfg.seed == 42
    assert cfg.scheduler == "none"
    assert (
        parse_train_params({"freeze_backbone": False}).freeze_encoder is False
    )  # earlier name still accepted
    for bad in (
        {"learning_rate": 0},
        {"early_stop_patience": 0},
        {"freeze_encoder": "maybe"},
        {"scheduler": "step"},
        {"max_grad_norm": -1},
    ):
        with pytest.raises(ValueError):
            parse_train_params(bad)


def test_partial_unfreeze_params():
    cfg = parse_train_params({"unfreeze_from": "Layer4", "encoder_lr_factor": "0.1"})
    assert (cfg.freeze_encoder, cfg.unfreeze_from, cfg.encoder_lr_factor) == (True, "layer4", 0.1)
    defaults = parse_train_params({})  # the best Guru run: the whole encoder trains at a tenth of the rate
    assert (defaults.freeze_encoder, defaults.unfreeze_from, defaults.encoder_lr_factor) == (False, "none", 0.1)
    frozen = parse_train_params({"freeze_encoder": "true"})  # a frozen encoder still trains layer4
    assert (frozen.freeze_encoder, frozen.unfreeze_from) == (True, "layer4")
    assert parse_train_params({"unfreeze_from": "none"}).unfreeze_from == "none"
    for bad in ({"unfreeze_from": "layer9"}, {"encoder_lr_factor": 0}, {"encoder_lr_factor": 2}):
        with pytest.raises(ValueError):
            parse_train_params(bad)
    with pytest.raises(ValueError, match="needs freeze_encoder=True"):
        parse_train_params({"freeze_encoder": False, "unfreeze_from": "layer4"})


def test_postprocess_defaults_alias_and_bounds():
    cfg = parse_postprocess_params(None)
    assert (cfg.threshold, cfg.min_area_px, cfg.simplify_px) == (0.3, 400, 1.0)
    assert (
        parse_postprocess_params({"confidence_threshold": "0.7"}).threshold == 0.7
    )  # name used by other fAIr models
    assert (
        parse_postprocess_params({"threshold": 0.3, "confidence_threshold": 0.9}).threshold == 0.3
    )  # canonical wins
    for bad in ({"threshold": 1.5}, {"threshold": True}, {"min_area_px": -1}, {"simplify_px": -0.5}):
        with pytest.raises(ValueError):
            parse_postprocess_params(bad)


def test_the_loss_weight_has_a_default_override_and_bounds():
    assert parse_train_params({}).dice_weight == 1.0
    assert parse_train_params({"dice_weight": 0}).dice_weight == 0.0  # cross-entropy only
    with pytest.raises(ValueError):
        parse_train_params({"dice_weight": -1})
