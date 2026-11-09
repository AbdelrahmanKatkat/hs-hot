"""The exported ONNX gives the same probabilities as the torch model, at sizes it was not traced with."""

import numpy as np
import pytest

pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

import torch  # noqa: E402

from hs_hot.export import INPUT_NAME, OUTPUT_NAME, build_unet, export_onnx_bytes, probability_network  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_unet()


@pytest.fixture(scope="module")
def session(model):
    return ort.InferenceSession(export_onnx_bytes(model), providers=["CPUExecutionProvider"])


def test_the_contract_names_shapes_and_dtypes(session):
    (image,), (probability,) = session.get_inputs(), session.get_outputs()
    assert (image.name, image.type, len(image.shape)) == (INPUT_NAME, "tensor(float)", 3)
    assert (probability.name, probability.type, len(probability.shape)) == (OUTPUT_NAME, "tensor(float)", 2)
    assert (
        image.shape[0] == 3
        and image.shape[1:] == ["height", "width"]
        and probability.shape == ["height", "width"]
    )


@pytest.mark.parametrize("height, width", [(160, 168), (97, 130), (64, 64), (33, 200)])
def test_onnx_matches_torch_at_any_size(model, session, height, width):
    image = torch.rand(3, height, width)
    expected = probability_network(model)(image).detach().numpy()
    (actual,) = session.run(None, {INPUT_NAME: image.numpy()})
    assert actual.shape == (height, width) and actual.dtype == np.float32
    assert 0.0 <= actual.min() and actual.max() <= 1.0
    np.testing.assert_allclose(actual, expected, atol=1e-4)
