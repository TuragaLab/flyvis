import math

import numpy as np
import torch

from flyvis.datasets.augmentation.augmentation import Augmentation
from flyvis.datasets.augmentation.hex import (
    ContrastBrightness,
    HexFlip,
    HexRotate,
    PixelNoise,
)
from flyvis.datasets.augmentation.temporal import (
    CropFrames,
    # InterpolateFrames,
    Interpolate,
)


def test_base_class():
    class Test(Augmentation):
        def transform(self, sequence: torch.Tensor):
            return sequence * 2

    class Test2(Augmentation):
        def transform(self, sequence: torch.Tensor):
            return sequence * 3

    class Test3(Test2):
        def transform(self, sequence: torch.Tensor):
            return sequence * 4

    test0 = Augmentation()
    test1 = Test()
    test2 = Test2()
    test3 = Test3()
    sequence = torch.tensor([0, 1])

    assert test0(sequence).tolist() == [0, 1]
    test0.augment = False
    assert test0(sequence).tolist() == [0, 1]

    assert test1(sequence).tolist() == [0, 2]
    test1.augment = False
    assert test1(sequence).tolist() == [0, 1]

    assert test2(sequence).tolist() == [0, 3]
    test2.augment = False
    assert test2(sequence).tolist() == [0, 1]

    assert test3(sequence).tolist() == [0, 4]
    test3.augment = False
    assert test3(sequence).tolist() == [0, 1]


# -- hex spatial augmentation


def test_hex_rotate():
    hexrotate = HexRotate(extent=1, n_rot=0)
    sequence = torch.tensor(np.arange(7)[None, None], dtype=torch.float32)
    assert_equals = {
        0: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        1: [1.0, 4.0, 0.0, 3.0, 6.0, 2.0, 5.0],
        2: [4.0, 6.0, 1.0, 3.0, 5.0, 0.0, 2.0],
        3: [6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        4: [5.0, 2.0, 6.0, 3.0, 0.0, 4.0, 1.0],
        5: [2.0, 0.0, 5.0, 3.0, 1.0, 6.0, 4.0],
    }

    for n_rot, expected in assert_equals.items():
        assert np.allclose(hexrotate(sequence, n_rot).cpu().numpy().flatten(), expected)

    def _angle(flow):
        return (np.degrees(np.arctan2(flow[1], flow[0])).mean() + 360) % 360

    flow = torch.tensor(
        np.repeat([[np.cos(np.radians(0))], [np.sin(np.radians(0))]], 7, axis=1)[None],
        dtype=torch.float32,
    )
    assert_equals = {0: 0, 1: 60, 2: 120, 3: 180, 4: 240, 5: 300}
    for n_rot, expected in assert_equals.items():
        hexrotate.n_rot = n_rot
        assert np.allclose(_angle(hexrotate(flow).cpu().numpy().squeeze()), expected)

    hexrotate.augment = False
    assert np.allclose(hexrotate(flow).tolist(), flow.tolist())


def test_hex_flip():
    hexflip = HexFlip(extent=1, axis=0)
    sequence = torch.tensor(np.arange(7)[None, None], dtype=torch.float32)
    assert_equals = {
        0: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        1: [0.0, 2.0, 1.0, 3.0, 5.0, 4.0, 6.0],
        2: [5.0, 6.0, 2.0, 3.0, 4.0, 0.0, 1.0],
        3: [4.0, 1.0, 6.0, 3.0, 0.0, 5.0, 2.0],
    }

    for axis, expected in assert_equals.items():
        assert np.allclose(hexflip(sequence, axis).cpu().numpy().flatten(), expected)

    def _angle(flow):
        return (np.degrees(np.arctan2(flow[1], flow[0])).mean() + 360) % 360

    flow = torch.tensor(
        np.repeat([[np.cos(np.radians(0))], [np.sin(np.radians(0))]], 7, axis=1)[None],
        dtype=torch.float32,
    )
    assert_equals = {0: 0, 1: 180, 2: 300, 3: 60}
    for axis, expected in assert_equals.items():
        assert np.allclose(_angle(hexflip(flow, axis).cpu().numpy().squeeze()), expected)
    hexflip.augment = False
    assert np.allclose(hexflip(flow).tolist(), flow.tolist())


def test_jitter():
    jitter = ContrastBrightness(0.5, 0.5)
    sequence = torch.tensor(
        np.array([0, 1, 1, 1, 1, 0, 0])[None, None], dtype=torch.float32
    )
    assert np.allclose(
        jitter(sequence).cpu().numpy().flatten(),
        [0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
    )
    sequence = torch.tensor(
        np.array([-1, 1, 1, 1, 1, 0, 0])[None, None], dtype=torch.float32
    )
    assert np.allclose(
        jitter(sequence).cpu().numpy().flatten(),
        [0.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
    )
    jitter.augment = False
    assert np.allclose(jitter(sequence).tolist(), sequence.tolist())


def test_noise():
    noise = PixelNoise(0.5)
    sequence = torch.tensor(
        np.random.random_sample(size=[1, 1, 721]), dtype=torch.float32
    )
    transformed = noise(sequence)
    snr_light = torch.mean(transformed[sequence > 0.5]) ** 2 / torch.var(
        transformed[sequence > 0.5]
    )
    snr_dark = torch.mean(transformed[sequence <= 0.5]) ** 2 / torch.var(
        transformed[sequence <= 0.5]
    )
    assert snr_light > snr_dark
    noise.augment = False
    assert np.allclose(noise(sequence).tolist(), sequence.tolist())


# -- temporal augmentation


def test_random_crop():
    def _is_monotonous(sequence):
        return np.all(np.diff(sequence) >= 0)

    random_crop = CropFrames(10, start=0, all_frames=False, random=True)
    sequence = torch.arange(1000)

    # fixed start frame at 0
    cropped1 = random_crop(sequence).cpu().numpy().flatten()
    cropped2 = random_crop(sequence).cpu().numpy().flatten()
    assert len(cropped1) == 10 == len(cropped2)
    assert _is_monotonous(cropped1) and _is_monotonous(cropped2)
    assert cropped1.min() in sequence and cropped1.max() in sequence
    assert cropped2.min() in sequence and cropped2.max() in sequence
    assert np.allclose(cropped1, cropped2)

    # random start frame
    random_crop.set_or_sample(start=None, total_sequence_length=len(sequence))
    cropped3 = random_crop(sequence).cpu().numpy().flatten()
    assert not np.allclose(cropped1, cropped3)

    # provided start frame
    random_crop.set_or_sample(start=42, total_sequence_length=len(sequence))
    cropped4 = random_crop(sequence).cpu().numpy().flatten()
    assert not np.allclose(cropped1, cropped4)

    # turn augmentation on and off
    random_crop.augment = False
    assert np.allclose(random_crop(sequence).tolist(), sequence.tolist())
    random_crop.augment = True
    random_crop.all_frames = True
    assert np.allclose(random_crop(sequence).tolist(), sequence.tolist())
    random_crop.random = False
    random_crop.all_frames = False
    assert np.allclose(random_crop(sequence).tolist(), list(range(10)))
    random_crop.random = True
    random_crop.set_or_sample(start=None, total_sequence_length=len(sequence))
    random_crop.all_frames = False
    assert not np.allclose(random_crop(sequence).tolist(), list(range(10)))


def test_resample():
    resample = Interpolate(24, 50, mode="nearest-exact")
    sequence = torch.arange(10)[None, None]
    resampled = resample(sequence, dim=2).flatten().tolist()
    assert np.allclose(
        resampled,
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
    )
    assert np.allclose(
        resampled,
        sequence.flatten()[
            resample.piecewise_constant_indices(len(sequence.flatten()))
        ].tolist(),
    )
    assert len(resampled) == math.ceil(10 * 50 / 24)
    resample.augment = False
    assert np.allclose(resample(sequence).flatten().tolist(), sequence.tolist())


def test_interpolate():
    interpolate = Interpolate(24, 50, mode="linear")
    flow = torch.tensor(
        [
            [np.cos(np.radians(0)), np.sin(np.radians(0))],
            [np.cos(np.radians(90)), np.sin(np.radians(90))],
            [np.cos(np.radians(180)), np.sin(np.radians(180))],
        ],
        dtype=torch.float32,
    )[:, :, None]
    interpolated = interpolate(flow)
    target = [
        [1, 0],
        [2 / 3, 1 / 3],
        [1 / 3, 2 / 3],
        [0, 1],
        [-1 / 3, 2 / 3],
        [-2 / 3, 1 / 3],
        [-1, 0],
    ]
    assert len(interpolated) == 7
    assert np.allclose(interpolated.cpu().numpy().squeeze(), target)
    interpolate.augment = False
    assert np.allclose(interpolate(flow).tolist(), flow.tolist())
