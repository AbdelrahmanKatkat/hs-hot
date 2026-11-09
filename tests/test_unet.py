"""The highway U-Net: any input size, normalisation inside, and the two ways to load weights."""

import io

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
import torchvision  # noqa: E402

from hs_hot.export import build_unet  # noqa: E402
from hs_hot.unet import HighwayUNet  # noqa: E402


def _resnet50_state() -> dict[str, torch.Tensor]:
    """What `resnet50-*.pth` holds: torchvision's ResNet-50 state dict, `fc` included (random weights here)."""
    return torchvision.models.resnet50(weights=None).state_dict()


def _as_stream(state: dict[str, torch.Tensor]) -> io.BytesIO:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return buffer


@pytest.mark.parametrize("height, width", [(64, 64), (97, 130), (33, 40)])
def test_output_has_the_input_size_whatever_it_is(height, width):
    logits = build_unet()(torch.rand(1, 3, height, width))
    assert logits.shape == (1, 1, height, width)


def test_normalisation_is_inside_and_not_part_of_the_saved_weights():
    model = build_unet()
    assert not any(key in model.state_dict() for key in ("mean", "std"))
    assert model.encoder.conv1.weight.shape == (64, 3, 7, 7) and not model.training


def test_the_encoder_is_initialised_from_a_torchvision_imagenet_state_dict_without_its_fc_layer():
    import torchvision

    state = torchvision.models.resnet50(
        weights=None
    ).state_dict()  # what `resnet50-*.pth` holds, `fc` included
    model = build_unet(encoder_checkpoint=_as_stream(state))
    assert torch.equal(model.encoder.conv1.weight, state["conv1.weight"])
    assert torch.equal(model.encoder.layer4[2].conv3.weight, state["layer4.2.conv3.weight"])


def test_a_saved_unet_loads_back_whole():
    source = build_unet()
    restored = build_unet(weights=_as_stream(source.state_dict()))
    assert all(
        torch.equal(a, b) for a, b in zip(source.state_dict().values(), restored.state_dict().values())
    )


def test_bad_checkpoints_and_both_arguments_are_rejected():
    with pytest.raises(ValueError, match="not a torchvision ResNet-50 state dict"):
        build_unet(encoder_checkpoint=_as_stream({"something": torch.zeros(1)}))
    with pytest.raises(ValueError, match="not both"):
        build_unet(encoder_checkpoint=_as_stream(_resnet50_state()), weights=_as_stream({}))
    assert isinstance(build_unet(), HighwayUNet)
