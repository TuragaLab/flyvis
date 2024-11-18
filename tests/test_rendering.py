import numpy as np
import pytest
import torch

from flyvis.datasets import rendering


@pytest.fixture(scope="module")
def boxeye() -> rendering.BoxEye:
    return rendering.BoxEye(15, 13)


def test_init(boxeye: rendering.BoxEye):
    assert boxeye.extent == 15
    assert boxeye.kernel_size == 13
    assert boxeye.receptor_centers.shape == (boxeye.hexals, 2)
    assert boxeye.min_frame_size.shape == (2,)
    assert boxeye.pad == (6, 6, 6, 6)


def test_call(boxeye: rendering.BoxEye):
    sequence = torch.ones((2, 2, 100, 100))

    rendered = boxeye(sequence, ftype="mean", hex_sample=True)
    assert rendered.shape == (2, 2, 1, boxeye.hexals)
    assert np.isclose(rendered.cpu().numpy().mean(), 1, atol=0.05)

    rendered = boxeye(sequence, ftype="sum", hex_sample=True)
    assert rendered.shape == (2, 2, 1, boxeye.hexals)
    assert np.isclose(rendered.cpu().numpy().mean(), boxeye.kernel_size**2, atol=5)

    sequence = torch.ones((2, 2, 100, 100)).random_(0, 11)

    rendered = boxeye(sequence, ftype="median", hex_sample=True)
    assert rendered.shape == (2, 2, 1, boxeye.hexals)
    assert np.isclose(rendered.cpu().numpy().mean(), 5, atol=0.05)

    rendered = boxeye(sequence.clone(), ftype="median", hex_sample=False)
    assert rendered.shape == (*sequence.shape[:2], *boxeye.min_frame_size.cpu().numpy())
    assert np.isclose(rendered.cpu().numpy().mean(), 5, atol=0.05)

    with pytest.raises(ValueError):
        boxeye(sequence, ftype="invalid", hex_sample=True)


def test_hex_render(boxeye: rendering.BoxEye):
    sequence = torch.ones((2, 2, 100, 100))
    rendered = boxeye.hex_render(sequence)
    assert rendered.shape == (2, 2, 1, boxeye.hexals)
