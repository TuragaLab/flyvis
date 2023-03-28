from flyvision.stimulus import Stimulus



def test_stimulus_init(connectome):

    stimulus = Stimulus(connectome)
    assert stimulus.buffer.shape == (1, 1, len(connectome.nodes.type))
    assert stimulus.buffer.sum() == 0.0
    assert hasattr(stimulus, "layer_index")
    assert hasattr(stimulus, "central_cells_index")
    assert hasattr(stimulus, "input_index")
