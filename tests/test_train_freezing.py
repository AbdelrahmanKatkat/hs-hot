"""Which weights train: frozen encoder, the deepest stages only, or everything; the encoder's BatchNorm never."""

import pytest

pytest.importorskip("torch")

from torch.nn.modules.batchnorm import _BatchNorm  # noqa: E402

from hs_hot.export import build_unet  # noqa: E402
from hs_hot.train import _freeze, _optimizer_groups  # noqa: E402


def trainable(module):
    return {name for name, param in module.named_parameters() if param.requires_grad}


def batchnorm_parameters(module):
    return [param for part in module.modules() if isinstance(part, _BatchNorm) for param in part.parameters()]


def test_a_frozen_encoder_leaves_the_whole_decoder_training():
    model = build_unet()
    _freeze(model, freeze_encoder=True)
    assert not trainable(model.encoder)
    assert all(param.requires_grad for param in model.up_blocks.parameters())
    assert all(param.requires_grad for param in model.head.parameters())
    assert all(
        param.requires_grad for param in batchnorm_parameters(model.up_blocks)
    )  # the new BatchNorm trains


@pytest.mark.parametrize("stage, expected", [("layer4", {"layer4"}), ("layer3", {"layer3", "layer4"})])
def test_unfreeze_from_trains_the_named_stage_and_the_deeper_ones(stage, expected):
    model = build_unet()
    _freeze(model, freeze_encoder=True, unfreeze_from=stage)
    assert {name.split(".")[0] for name in trainable(model.encoder)} == expected


def test_encoder_batchnorm_never_trains_even_when_the_encoder_does():
    model = build_unet()
    _freeze(model, freeze_encoder=False)
    assert trainable(model.encoder)  # the convolutions train
    assert not any(param.requires_grad for param in batchnorm_parameters(model.encoder))


def test_the_encoder_weights_get_their_own_learning_rate():
    model = build_unet()
    _freeze(model, freeze_encoder=True, unfreeze_from="layer4")
    decoder_group, encoder_group = _optimizer_groups(model, lr=1e-3, encoder_lr_factor=0.1)
    assert decoder_group["lr"] == 1e-3 and encoder_group["lr"] == pytest.approx(1e-4)
    frozen = build_unet()
    _freeze(frozen, freeze_encoder=True)
    assert len(_optimizer_groups(frozen, 1e-3, 0.1)) == 1  # nothing in the encoder trains: one group
