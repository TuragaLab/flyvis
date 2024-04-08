import pytest

import torch
import numpy as np

from datamate import set_root_context

from flyvision.datasets.sintel import RenderedSintel, MultiTaskSintel, sintel_meta


def test_rendering(tmp_path_factory):
    with set_root_context(tmp_path_factory.mktemp("tmp")):
        rendered = RenderedSintel(
            tasks=["flow"],
            boxfilter=dict(extent=1, kernel_size=13),
            vertical_splits=3,
            n_frames=2,
            gamma=1,
            center_crop_fraction=0.7,
            unittest=True,
        )
    assert len(rendered) == 3
    split_1 = rendered(0)
    assert split_1["flow"].shape == (3, 2, 7)
    assert split_1["lum"].shape == (3, 1, 7)
    assert rendered.config == dict(
        type="RenderedSintel",
        tasks=["flow"],
        boxfilter=dict(extent=1, kernel_size=13),
        vertical_splits=3,
        n_frames=2,
        gamma=1,
        center_crop_fraction=0.7,
        unittest=True,
    )


@pytest.fixture(scope="module")
def dataset():
    return MultiTaskSintel(
        tasks=["flow"],
        boxfilter=dict(extent=15, kernel_size=13),
        vertical_splits=3,
        n_frames=19,
        center_crop_fraction=0.7,
        dt=1 / 50,
        augment=True,
        random_temporal_crop=True,
        all_frames=False,
        resampling=True,
        interpolate=True,
        p_flip=0.5,
        p_rot=5 / 6,
        contrast_std=0.2,
        brightness_std=0.1,
        gaussian_white_noise=0.08,
        gamma_std=None,
        _init_cache=True,
        unittest=True,
        flip_axes=[
            0,
            1,
            2,
            3,
        ],  # 2 and 3 with all rotation axes lead to redundant transforms
        task_weights=None,
    )


@pytest.fixture(
    scope="module",
    params=[
        ["lum"],
        ["flow"],
        ["depth"],
        ["lum", "flow"],
        ["lum", "depth"],
        ["flow", "depth"],
        ["lum", "flow", "depth"],
    ],
)
def tasks(request):
    return request.param


def test_init(tasks):
    dataset = MultiTaskSintel(tasks=tasks, unittest=True)
    assert hasattr(dataset, "tasks")
    assert "lum" in dataset.data_keys
    assert hasattr(dataset, "config")
    assert hasattr(dataset, "meta")
    assert hasattr(dataset, "cached_sequences")
    assert hasattr(dataset, "arg_df")
    assert set(dataset[0].keys()) == set(["lum", *tasks])
    dataset = MultiTaskSintel(tasks=tasks, unittest=True, _init_cache=False)
    assert not hasattr(dataset, "cached_sequences")


def test_init_augmentation(dataset):
    assert hasattr(dataset, "temporal_crop")
    assert hasattr(dataset, "jitter")
    assert hasattr(dataset, "rotate")
    assert hasattr(dataset, "flip")
    assert hasattr(dataset, "noise")
    assert hasattr(dataset, "piecewise_resample")
    assert hasattr(dataset, "linear_interpolate")
    assert hasattr(dataset, "gamma_correct")


def test_set_augmentation_parameters(dataset):
    dataset.p_rot = 5 / 6
    dataset.p_flip = 3 / 4

    def sample(total_sequence_length=None):
        rotations = []
        flips = []
        contrast_factors = []
        brightness_factors = []
        gammas = []
        start_frames = []
        for _ in range(10_000):
            dataset.set_augmentation_params(total_sequence_length=total_sequence_length)
            rotations.append(dataset.rotate.n_rot)
            flips.append(dataset.flip.axis)
            contrast_factors.append(dataset.jitter.contrast_factor)
            brightness_factors.append(dataset.jitter.brightness_factor)
            gammas.append(dataset.gamma_correct.gamma)
            start_frames.append(dataset.temporal_crop.start)
        return (
            rotations,
            flips,
            contrast_factors,
            brightness_factors,
            gammas,
            start_frames,
        )

    (
        rotations,
        flips,
        contrast_factors,
        brightness_factors,
        gammas,
        start_frames,
    ) = sample()

    rots, counts = np.unique(rotations, return_counts=True)
    assert len(rots) == 6
    assert np.allclose(counts / np.sum(counts), 1 / 6, atol=0.01)
    flips, counts = np.unique(flips, return_counts=True)
    assert len(flips) == 4
    assert np.allclose(counts / np.sum(counts), 1 / 4, atol=0.01)
    assert len(set(contrast_factors)) != 1
    assert len(set(brightness_factors)) != 1
    assert all(v == 1 for v in gammas)
    assert all(v is None for v in start_frames)

    dataset.gamma_std = 0.1
    (
        _,
        _,
        _,
        _,
        gammas,
        start_frames,
    ) = sample(total_sequence_length=50)
    brightness_factors, counts = np.unique(brightness_factors, return_counts=True)
    assert len(set(gammas)) != 1
    assert len(set(start_frames)) != 1
    assert all(v < 50 - dataset.n_frames for v in start_frames)


def test_getitem(dataset):
    dataset.dt = 1 / 24

    dataset.augment = False
    data0 = dataset[0]
    assert set(data0.keys()) == set(["lum", "flow"])
    assert data0["lum"].shape == (3, 1, 721)
    assert data0["flow"].shape == (3, 2, 721)

    # switched off augmentation results in equal data
    data01 = dataset[0]
    assert (data0["lum"] == data01["lum"]).all()
    assert (data0["flow"] == data01["flow"]).all()

    # switch on augmentation results in different data
    dataset.augment = True
    data1 = dataset[0]
    assert set(data1.keys()) == set(["lum", "flow"])
    assert (data0["lum"] != data1["lum"]).any()
    assert (data0["flow"] != data1["flow"]).any()
    assert data1["lum"].shape == (3, 1, 721)
    assert data1["flow"].shape == (3, 2, 721)

    # change dt to 1/50
    dataset.dt = 1 / 50
    data2 = dataset[0]
    assert data2["lum"].shape == (7, 1, 721)
    assert data2["flow"].shape == (7, 2, 721)


def test_apply_augmentation(dataset):
    dataset.augment = False
    dataset.dt = 1 / dataset.framerate
    data = dataset[0]
    data1 = dataset.apply_augmentation(data)
    assert set(data1.keys()) == set(data.keys())
    assert (data["lum"] != data1["lum"]).any()
    assert (data["flow"] != data1["flow"]).any()


def test_original_sequence_index(dataset):
    assert dataset.vertical_splits == 3
    assert dataset.original_sequence_index(0) == 0
    assert dataset.original_sequence_index(2) == 0
    assert dataset.original_sequence_index(3) == 1


def test_cartesian(dataset):
    dataset.meta = sintel_meta(
        dataset.rendered,
        dataset.sintel_path,
        dataset.n_frames,
        dataset.vertical_splits,
        True,
    )
    cartesian = {
        "lum": dataset.cartesian_sequence(0),
        "flow": dataset.cartesian_flow(0),
        "depth": dataset.cartesian_depth(0),
    }
    assert len(cartesian["lum"].shape) == 4
    assert cartesian["lum"].shape[0] == dataset.vertical_splits
    assert len(cartesian["flow"].shape) == 5
    assert cartesian["flow"].shape[0] == dataset.vertical_splits
    assert len(cartesian["depth"].shape) == 4
    assert cartesian["depth"].shape[0] == dataset.vertical_splits


def test_original_train_and_validation_indices(dataset):
    train_indices, val_indices = dataset.original_train_and_validation_indices()
    # because unittest only renders 3 splits of first scene
    assert train_indices == [0, 1, 2]
    assert val_indices == []
