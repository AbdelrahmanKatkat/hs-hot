"""Building the highway U-Net and its ONNX export (needs the `train` extra; torch is imported inside the functions).

The exported graph is the contract the serving side reads:

    input  `image`        (3, H, W) float32 in [0, 1], RGB, already scaled by `preprocess.py`; any H and W
    output `probability`  (H, W) float32 in [0, 1], the road probability of every pixel

ImageNet normalisation, the network and the sigmoid are inside the graph; the threshold, the tracing of polygons and
their cleanup (`decode.py`) stay outside so they remain tunable.
"""

import logging
from pathlib import Path
from typing import BinaryIO

log = logging.getLogger(__name__)

# ONNX operator-set version written into the file. The runtime must support 17 or newer.
OPSET = 17
INPUT_NAME = "image"
OUTPUT_NAME = "probability"
EXAMPLE_SIDE_PX = 160  # size of the example input used to trace the graph; the exported axes are dynamic


def build_unet(
    *, encoder_checkpoint: str | Path | BinaryIO | None = None, weights: str | Path | BinaryIO | None = None
):
    """A `HighwayUNet` in eval mode, from random weights or from one of two checkpoints (each a path or a binary stream).

    `encoder_checkpoint` is a torchvision ResNet-50 state dict (`resnet50-*.pth`, ImageNet; its `fc` layer is dropped)
    that initialises the encoder while the decoder stays random. `weights` is a state dict saved from a `HighwayUNet`,
    loaded whole.
    The checkpoints store weights only: normalisation and the architecture come from `HighwayUNet`, so every generation
    (base, fine-tuned, re-fine-tuned) is rebuilt with the same constants.
    """
    import torch

    from .unet import HighwayUNet

    if encoder_checkpoint is not None and weights is not None:
        raise ValueError(
            "give either encoder_checkpoint (to start a new U-Net) or weights (to continue one), not both"
        )
    model = HighwayUNet()
    if encoder_checkpoint is not None:
        model.encoder.load_state_dict(
            _encoder_weights(torch.load(encoder_checkpoint, map_location="cpu", weights_only=True)),
            strict=True,
        )
    elif weights is not None:
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return model.eval()


def _encoder_weights(state: dict) -> dict:
    """The ResNet-50 weights (`conv1`, `bn1`, `layer1` to `layer4`) of a torchvision state dict, without `fc`."""
    body = {key: value for key, value in state.items() if not key.startswith("fc.")}
    if "conv1.weight" not in body or "layer4.2.conv3.weight" not in body:
        raise ValueError("not a torchvision ResNet-50 state dict: no `conv1` / `layer4` weights")
    return body


def probability_network(model):
    """Wrap `model` (switched to eval mode) as `image (3, H, W) -> probability (H, W)`, the exported contract."""
    import torch

    class ProbabilityNetwork(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, image):
            return torch.sigmoid(self.model(image[None]))[0, 0]

    return ProbabilityNetwork(model).eval()


def export_onnx_bytes(model, *, opset: int = OPSET) -> bytes:
    """Export `probability_network(model)` with dynamic height and width; validate and return the file's bytes."""
    import tempfile

    import onnx
    import torch

    network = probability_network(model)
    example = torch.rand(
        3, EXAMPLE_SIDE_PX, EXAMPLE_SIDE_PX + 8
    )  # unequal sides, so a swapped axis cannot hide
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "hs_hot.onnx"
        torch.onnx.export(
            network,
            (example,),
            str(path),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={INPUT_NAME: {1: "height", 2: "width"}, OUTPUT_NAME: {0: "height", 1: "width"}},
            opset_version=opset,
            dynamo=False,
        )
        onnx.checker.check_model(onnx.load(str(path)))
        data = path.read_bytes()
    log.info("exported ONNX (opset %d, %s -> %s): %.1f MB", opset, INPUT_NAME, OUTPUT_NAME, len(data) / 1e6)
    return data
