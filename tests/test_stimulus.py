import numpy as np
import pytest
import torch

from flyvis.network.stimulus import Stimulus


@pytest.fixture
def stimulus(connectome) -> Stimulus:
    return Stimulus(connectome)


def test_stimulus_init(stimulus):
    assert stimulus.buffer.shape == (1, 1, stimulus.n_nodes)
    assert stimulus.buffer.sum() == 0.0
    assert hasattr(stimulus, "layer_index")
    assert hasattr(stimulus, "central_cells_index")
    assert hasattr(stimulus, "input_index")


def test_add_input(stimulus: Stimulus):
    # single batch, single frame
    batch_size = 1
    frames = 1
    x = torch.ones((batch_size, frames, 1, stimulus.n_input_elements))
    stimulus.add_input(x)
    assert stimulus.buffer.shape == (batch_size, frames, stimulus.n_nodes)
    assert stimulus.buffer.sum() == batch_size * frames * np.prod(
        stimulus.input_index.shape
    )

    # multi batch, multi frame
    batch_size = 2
    frames = 2
    x = torch.ones((batch_size, frames, 1, stimulus.n_input_elements))
    stimulus.add_input(x)
    assert stimulus.buffer.shape == (batch_size, frames, stimulus.n_nodes)
    assert stimulus.buffer.sum() == batch_size * frames * np.prod(
        stimulus.input_index.shape
    )

    # single batch, temporal start
    batch_size = 1
    frames = 20
    start = 10
    x = torch.ones((batch_size, frames - start, 1, stimulus.n_input_elements))
    stimulus.add_input(x, start=start, n_frames_buffer=frames)
    assert stimulus.buffer.shape == (batch_size, frames, stimulus.n_nodes)
    assert stimulus.buffer.sum() == batch_size * (frames - start) * np.prod(
        stimulus.input_index.shape
    )

    # single batch, temporal start and stop
    batch_size = 1
    frames = 20
    start = 5
    stop = 15
    x = torch.ones((batch_size, stop - start, 1, stimulus.n_input_elements))
    stimulus.add_input(x, start=start, stop=stop, n_frames_buffer=frames)
    assert stimulus.buffer.shape == (batch_size, frames, stimulus.n_nodes)
    assert stimulus.buffer.sum() == batch_size * (stop - start) * np.prod(
        stimulus.input_index.shape
    )

    # cumulate
    stimulus.add_input(x, start=start, stop=stop, n_frames_buffer=frames, cumulate=True)
    assert stimulus.buffer.sum() == 2 * batch_size * (stop - start) * np.prod(
        stimulus.input_index.shape
    )

    # don't cumulate
    stimulus.add_input(x, start=start, stop=stop, n_frames_buffer=frames, cumulate=False)
    assert stimulus.buffer.sum() == batch_size * (stop - start) * np.prod(
        stimulus.input_index.shape
    )

    # fixed_frames - actual frames mismatch raises error
    batch_size = 1
    frames = 20
    x = torch.ones((batch_size, frames + 1, 1, stimulus.n_input_elements))
    with pytest.raises(RuntimeError):
        stimulus.add_input(x, n_frames_buffer=frames)

    # without specifying fixed_frames, resizes buffer
    stimulus.add_input(x)
    assert stimulus.buffer.shape == (batch_size, frames + 1, stimulus.n_nodes)
    assert stimulus.buffer.sum() == batch_size * (frames + 1) * np.prod(
        stimulus.input_index.shape
    )
    # since cumulate is False by default and the buffer has nonzero elements
    # the buffer should be initialized to zero
    assert stimulus.nonzero
    stimulus.add_input(x)
    assert stimulus.buffer.sum() == batch_size * (frames + 1) * np.prod(
        stimulus.input_index.shape
    )

    # setting the _nonzero attribute to False should allow the buffer to
    # accumulate even though cumulate is False by default
    stimulus._nonzero = False
    stimulus.add_input(x)
    assert stimulus.buffer.sum() == 2 * batch_size * (frames + 1) * np.prod(
        stimulus.input_index.shape
    )


def test_add_pre_stim(stimulus: Stimulus):
    # adding single value
    stimulus.add_pre_stim(1.0)
    assert stimulus.buffer.shape == (1, 1, stimulus.n_nodes)
    assert stimulus.buffer.sum() == np.prod(stimulus.input_index.shape)

    # adding tensor but not matching implicit buffer
    x = torch.ones((10))
    with pytest.raises(RuntimeError):
        stimulus.add_pre_stim(x)

    # adding tensor and not matching explicit buffer
    stimulus.zero(1, 20)
    with pytest.raises(RuntimeError):
        stimulus.add_pre_stim(x)

    # adding tensor and matching explicit buffer
    stimulus.zero(1, 10)
    stimulus.add_pre_stim(x)
    assert stimulus.buffer.shape == (1, 10, stimulus.n_nodes)
    assert stimulus.buffer.sum() == 10 * np.prod(stimulus.input_index.shape)

    # adding tensor and matching explicit buffer using start, stop and n_frames_buffer
    stimulus.add_pre_stim(x, start=0, stop=10, n_frames_buffer=20)
    assert stimulus.buffer.shape == (1, 20, stimulus.n_nodes)
    assert stimulus.buffer.sum() == 10 * np.prod(stimulus.input_index.shape)
