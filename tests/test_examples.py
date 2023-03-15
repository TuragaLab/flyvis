from os import PathLike

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import urllib
import pytest

import torch
import numpy as np
from datamate import Directory

import flyvision
from flyvision import connectome_file
from flyvision import ConnectomeDir, ConnectomeView
from flyvision.datasets.base import SequenceDataset
from flyvision.ensemble import EnsembleView
from flyvision.utils.activity_utils import LayerActivity


class RenderedData(Directory):
    class Config(dict):
        path: PathLike

    def __init__(self, config: Config):

        sequences = np.load(self.config.path)
        sequences = np.transpose(sequences, (1, 0, 2, 3)) / 255.0
        receptors = flyvision.rendering.BoxEye()

        rendered_sequences = []
        for sequence in sequences:
            rendered_sequences.append(receptors(sequence[None]).cpu().numpy())
            # break to only create one sequence
            break

        rendered_sequences = np.array(rendered_sequences)
        self.sequences = rendered_sequences


class CustomStimuli(SequenceDataset):

    dt = 1 / 50
    framerate = 24
    t_pre = 0.5
    t_post = 0.5
    n_sequences = None
    augment = False

    def __init__(self, raw_data_path, rendered_path="auto"):
        if rendered_path == "auto":
            self.dir = RenderedData(dict(path=raw_data_path))
        else:
            self.dir = RenderedData(rendered_path, dict(path=raw_data_path))
        self.sequences = torch.permute(
            torch.Tensor(self.dir.sequences[:]), (0, 2, 1, 3)
        )
        self.n_sequences = self.sequences.shape[0]

    def get_item(self, key):
        sequence = self.sequences[key]
        resample = self.get_temporal_sample_indices(
            sequence.shape[0], sequence.shape[0]
        )
        return sequence[resample]


def test_connectome(tmp_path):
    connectome = ConnectomeDir(
        tmp_path / "test", dict(file=connectome_file, extent=15, n_syn_fill=1)
    )
    assert isinstance(connectome, Directory)
    assert connectome.path.name == "test"
    assert connectome.path.parent == tmp_path


def test_connectome_view(tmp_path):

    connectome = ConnectomeDir(
        tmp_path / "test", dict(file=connectome_file, extent=15, n_syn_fill=1)
    )
    connectome_view = ConnectomeView(connectome)
    assert isinstance(connectome, Directory)
    assert isinstance(connectome_view, ConnectomeView)

    fig = connectome_view.connectivity_matrix("n_syn")
    assert len(fig.axes) == 2
    assert isinstance(fig, Figure)

    fig = connectome_view.network_layout()
    assert len(fig.axes) == len(connectome.unique_cell_types) + 1
    assert isinstance(fig, Figure)

    fig = connectome_view.receptive_fields_grid("T4c")
    assert (
        len([ax for ax in fig.axes if ax.get_visible()])
        == len(connectome_view.receptive_fields_df("T4c"))
        == len(connectome_view.sources_list("T4c"))
    )
    assert isinstance(fig, Figure)

    fig = connectome_view.projective_fields_grid("T4c")
    assert (
        len([ax for ax in fig.axes if ax.get_visible()])
        == len(connectome_view.projective_fields_df("T4c"))
        == len(connectome_view.targets_list("T4c"))
    )
    assert isinstance(fig, Figure)


def test_imshow():
    # path to public moving mnist mirror
    moving_mnist_url = (
        "https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
    )

    # path to where moving mnist will be stored for this example
    moving_mnist_path = flyvision.root_dir / "mnist_test_seq.npy"

    if not moving_mnist_path.exists():
        urllib.request.urlretrieve(moving_mnist_url, moving_mnist_path)

    sequences = np.load(moving_mnist_path)
    sequences = np.transpose(sequences, (1, 0, 2, 3)) / 255.0
    example_sequence_ids = np.random.randint(0, high=sequences.shape[0], size=10)
    animation = flyvision.animations.Imshow(sequences, cmap=plt.cm.binary_r)
    animation.notebook_animation(samples=example_sequence_ids, frames=[0])


def test_boxeye():
    receptors = flyvision.rendering.BoxEye()
    fig = receptors.illustrate()

    assert isinstance(fig, Figure)

    moving_mnist_path = flyvision.root_dir / "mnist_test_seq.npy"

    sequences = np.load(moving_mnist_path)
    sequences = np.transpose(sequences, (1, 0, 2, 3)) / 255.0
    single_frame = sequences[0, 0]
    single_frame = torch.Tensor(single_frame)
    single_frame = single_frame[None, None]
    rendered = receptors(single_frame)

    assert rendered.shape == (1, 1, 721)


def test_rendering(tmp_path):

    moving_mnist_path = flyvision.root_dir / "mnist_test_seq.npy"
    moving_mnist_rendered = RenderedData(
        tmp_path / "test", dict(path=moving_mnist_path)
    )
    rendered_sequences = moving_mnist_rendered.sequences[:]
    # to stick to our convention for dimensions (samples, frames, 1, hexals)
    rendered_sequences = np.transpose(rendered_sequences, (0, 2, 1, 3))

    # just checking the visualzation runs
    animation = flyvision.animations.HexScatter(rendered_sequences, vmin=0, vmax=1)
    animation.notebook_animation(samples="all", frames=[0])


def test_sequence_dataset(tmp_path):
    moving_mnist_path = flyvision.root_dir / "mnist_test_seq.npy"

    custom_stimuli_dataset = CustomStimuli(moving_mnist_path, tmp_path / "test")

    assert custom_stimuli_dataset[0].shape == (42, 1, 721)


def test_model_responses(tmp_path):

    ensemble = EnsembleView(flyvision.results_dir / "opticflow/000")

    moving_mnist_path = flyvision.root_dir / "mnist_test_seq.npy"

    custom_stimuli_dataset = CustomStimuli(moving_mnist_path, tmp_path / "test")

    movie_input = custom_stimuli_dataset[0]

    responses = np.array(
        list(ensemble.simulate(movie_input[None], custom_stimuli_dataset.dt))
    )

    assert responses.shape == (50, 1, 42, 45669)

    voltages = LayerActivity(responses, ensemble[0].ctome, keepref=True)

    cell_type = "T4c"
    assert voltages[cell_type].shape == (50, 1, 42, 721)

    central_voltages = voltages.central

    assert central_voltages[cell_type].shape == (50, 1, 42)


def test_cluster():

    from datamate import namespacify

    cell_type = "T4c"
    ensemble = EnsembleView(flyvision.results_dir / "opticflow/000")

    cluster_indices = ensemble.cluster_indices(cell_type)

    assert namespacify(cluster_indices) == namespacify(
        {
            0: np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    6,
                    9,
                    23,
                    24,
                    13,
                    12,
                    21,
                    22,
                    37,
                    19,
                    35,
                    30,
                    27,
                    17,
                    29,
                    16,
                    36,
                    14,
                    20,
                    7,
                    18,
                    31,
                    42,
                    11,
                    47,
                    44,
                    48,
                ]
            ),
            1: np.array([5, 26, 4, 33, 38, 40, 43]),
            2: np.array([8, 25, 10, 32, 34, 28, 45, 15, 39, 41, 46, 49]),
        }
    )
