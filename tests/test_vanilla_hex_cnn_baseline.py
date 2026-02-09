import torch

from flyvis.baselines.vanilla_hex_cnn.models import (
    VanillaHexCNNBaseline,
    VanillaHexCNNBaselineHexSpace,
)


def test_vanilla_hex_cnn_baseline_forward_shape():
    model = VanillaHexCNNBaseline(n_frames=2)
    x = torch.randn(3, 2, 1, 721)
    y = model(x)
    assert y.shape == (3, 1, 2, 721)


def test_vanilla_hex_cnn_baseline_hex_space_forward_shape():
    model = VanillaHexCNNBaselineHexSpace(n_frames=2)
    x = torch.randn(3, 2, 1, 721)
    y = model(x)
    assert y.shape == (3, 1, 2, 721)

