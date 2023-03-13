from matplotlib.figure import Figure

import pytest

from datamate import Directory

import flyvision
from flyvision import connectome_file
from flyvision import Connectome, ConnectomeView


def test_connectome():
    connectome = Connectome(dict(file="fib25-fib19_v2.2.json", extent=15, n_syn_fill=1))
    assert isinstance(connectome, Directory)
    assert connectome.path.parent.name == "connectome"
    assert connectome.path.parent.parent.name == "data"


def test_connectome_view():

    connectome = Connectome(dict(file=connectome_file, extent=15, n_syn_fill=1))
    connectome_view = ConnectomeView(connectome)
    assert isinstance(connectome, Directory)
    assert isinstance(connectome_view, ConnectomeView)

    fig = connectome_view.connectivity_matrix("n_syn")
    assert len(fig.axes) == 2
    assert isinstance(fig, Figure)

    fig = connectome_view.network_layout()
    assert len(fig.axes) == len(connectome.unique_node_types) + 1
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
