# Figure 3


```python
%load_ext autoreload
%autoreload 2

from flyvision import EnsembleView
```


```python
ensemble = EnsembleView("flow/0000")
```


    Loading ensemble:   0%|          | 0/50 [00:00<?, ?it/s]


    [2024-09-28 04:19:39] ensemble:138 Loaded 50 networks.


## a


```python
task_error = ensemble.task_error()
```


```python
embedding_and_clustering = ensemble.clustering("T4c")
```

    [2024-09-28 04:19:42] clustering:640 Loaded T4c embedding and clustering from /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/umap_and_clustering.



```python
embeddingplot = embedding_and_clustering.plot(task_error=task_error.values, colors=task_error.colors)
```



![png](figure_03_naturalistic_stimuli_responses_files/figure_03_naturalistic_stimuli_responses_6_0.png)



## b


```python
import numpy as np
import matplotlib.pyplot as plt

from flyvision.analysis.clustering import check_markers
from flyvision.utils.activity_utils import CellTypeArray
from flyvision.plots import plt_utils
from flyvision.plots.plt_utils import get_marker, add_cluster_marker
from flyvision.plots import plots
from flyvision.datasets import MovingEdge
from flyvision.analysis.moving_bar_responses import MovingEdgeResponseView, plot_angular_tuning
from flyvision.utils.groundtruth_utils import tuning_curves
```


```python
cluster_indices = ensemble.cluster_indices("T4c")
```


```python
r = ensemble.moving_edge_responses()
r['responses'] /= np.abs(r['responses']).max(dim=('frame', 'sample'))
```


```python
cluster_indices = ensemble.cluster_indices("T4c")
```


```python
colors = ensemble.task_error().colors
```


```python
fig, axes = plt.subplots(
    1, len(cluster_indices), subplot_kw={"projection": "polar"}, figsize=[2, 1]
)
for cluster_id, indices in cluster_indices.items():
    plot_angular_tuning(
        r.sel(network_id=indices),
        "T4c",
        intensity=1,
        colors=colors[indices],
        zorder=ensemble.zorder()[indices],
        groundtruth=True if cluster_id == 0 else False,
        fig=fig,
        ax=axes[cluster_id],
    )
    add_cluster_marker(fig, axes[cluster_id], marker=get_marker(cluster_id))
```



![png](figure_03_naturalistic_stimuli_responses_files/figure_03_naturalistic_stimuli_responses_13_0.png)



## e


```python
for cluster_id, indices in cluster_indices.items():
    with ensemble.select_items(indices):
        fig, ax = ensemble.flash_response_index(cell_types=["Mi1", "Tm3", "Mi4", "Mi9", "CT1(M10)"],
                                                figsize=[1, 1])
```

    [2024-09-28 04:29:58] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/000', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 102bbcd0dbfd7bfa6847ec1a64bd245a)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:29:58] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/__cache__/flyvision/analysis/stimulus_responses/compute_responses/102bbcd0dbfd7bfa6847ec1a64bd245a/output.h5
    [2024-09-28 04:29:58] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/001/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/001', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/001/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 01bc24ee52f66d92df6dda7a54fbb512)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/001/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:29:58] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/001/__cache__/flyvision/analysis/stimulus_responses/compute_responses/01bc24ee52f66d92df6dda7a54fbb512/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:29:59] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/002/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/002', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/002/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 1be5fe2784c35f3799f9c52e4c1f4ab1)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/002/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:29:59] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/002/__cache__/flyvision/analysis/stimulus_responses/compute_responses/1be5fe2784c35f3799f9c52e4c1f4ab1/output.h5
    [2024-09-28 04:29:59] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/003/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/003', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/003/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 3a32cdc078ada6ec77ef312d81a5a3d3)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/003/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:29:59] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/003/__cache__/flyvision/analysis/stimulus_responses/compute_responses/3a32cdc078ada6ec77ef312d81a5a3d3/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:29:59] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/006/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/006', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/006/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 22e55668eb1321c0343e492d14827f15)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/006/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:29:59] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/006/__cache__/flyvision/analysis/stimulus_responses/compute_responses/22e55668eb1321c0343e492d14827f15/output.h5
    [2024-09-28 04:29:59] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/007/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/007', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/007/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 37749baa331d7b1fe7d6cb50f19136e2)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/007/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:29:59] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/007/__cache__/flyvision/analysis/stimulus_responses/compute_responses/37749baa331d7b1fe7d6cb50f19136e2/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:29:59] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/009/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/009', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/009/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash bb93e66c28a9572b3f9a66ec21c1a8fb)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/009/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:29:59] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/009/__cache__/flyvision/analysis/stimulus_responses/compute_responses/bb93e66c28a9572b3f9a66ec21c1a8fb/output.h5
    [2024-09-28 04:29:59] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/011/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/011', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/011/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 9a6457c1d9d4757b5c8d72581fe2c4a6)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/011/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:29:59] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/011/__cache__/flyvision/analysis/stimulus_responses/compute_responses/9a6457c1d9d4757b5c8d72581fe2c4a6/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:29:59] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/012/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/012', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/012/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 2b81eec90842579ef5cf5fc368ee5fef)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/012/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:29:59] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/012/__cache__/flyvision/analysis/stimulus_responses/compute_responses/2b81eec90842579ef5cf5fc368ee5fef/output.h5
    [2024-09-28 04:30:00] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/013/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/013', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/013/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 0be4cfd5ec972f5ecb5e3d5c1fc3fef8)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/013/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:00] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/013/__cache__/flyvision/analysis/stimulus_responses/compute_responses/0be4cfd5ec972f5ecb5e3d5c1fc3fef8/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:30:00] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/014/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/014', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/014/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 78f41007e53b1d6503476220d9dd4db9)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/014/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:00] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/014/__cache__/flyvision/analysis/stimulus_responses/compute_responses/78f41007e53b1d6503476220d9dd4db9/output.h5
    [2024-09-28 04:30:00] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/016/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/016', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/016/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 55046ad197d2c798dff879afc6a8dc2f)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/016/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:00] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/016/__cache__/flyvision/analysis/stimulus_responses/compute_responses/55046ad197d2c798dff879afc6a8dc2f/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:30:00] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/017/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/017', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/017/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 1ed79aec45a0a6fbdb1f2916e5b3ab37)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/017/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:00] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/017/__cache__/flyvision/analysis/stimulus_responses/compute_responses/1ed79aec45a0a6fbdb1f2916e5b3ab37/output.h5
    [2024-09-28 04:30:00] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/018/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/018', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/018/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash fdfe90bb92c33852b686cc05968bc0ee)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/018/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:00] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/018/__cache__/flyvision/analysis/stimulus_responses/compute_responses/fdfe90bb92c33852b686cc05968bc0ee/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:30:00] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/019/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/019', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/019/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 2ecb8848e11c25e962824a63b4845af0)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/019/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:00] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/019/__cache__/flyvision/analysis/stimulus_responses/compute_responses/2ecb8848e11c25e962824a63b4845af0/output.h5
    [2024-09-28 04:30:00] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/020/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/020', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/020/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash f20b34d2152c35a3d4a89cf78e990ed5)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/020/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:00] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/020/__cache__/flyvision/analysis/stimulus_responses/compute_responses/f20b34d2152c35a3d4a89cf78e990ed5/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:30:00] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/021/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/021', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/021/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 360d0bc7deb1d81f2ab3f5356583f315)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/021/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:00] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/021/__cache__/flyvision/analysis/stimulus_responses/compute_responses/360d0bc7deb1d81f2ab3f5356583f315/output.h5
    [2024-09-28 04:30:01] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/022/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/022', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/022/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash ae08c1dba87ee4799be3030b4c5aad27)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/022/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:01] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/022/__cache__/flyvision/analysis/stimulus_responses/compute_responses/ae08c1dba87ee4799be3030b4c5aad27/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:30:01] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/023/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/023', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/023/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 27dc64e2a8b3b41db36b500aa5e52a96)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/023/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:01] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/023/__cache__/flyvision/analysis/stimulus_responses/compute_responses/27dc64e2a8b3b41db36b500aa5e52a96/output.h5
    [2024-09-28 04:30:01] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/024/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/024', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/024/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 6d8f016a87d7e080d3edfd7bb0a7898f)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/024/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:01] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/024/__cache__/flyvision/analysis/stimulus_responses/compute_responses/6d8f016a87d7e080d3edfd7bb0a7898f/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:30:01] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/027/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/027', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/027/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash fb032b83a94391508f0254a2f2f872c2)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/027/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:01] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/027/__cache__/flyvision/analysis/stimulus_responses/compute_responses/fb032b83a94391508f0254a2f2f872c2/output.h5
    [2024-09-28 04:30:01] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/029/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/029', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/029/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash e84bb56decd6a3f09a23bd2cae2f5506)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/029/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:01] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/029/__cache__/flyvision/analysis/stimulus_responses/compute_responses/e84bb56decd6a3f09a23bd2cae2f5506/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:30:01] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/030/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/030', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/030/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash c4e18ac3fee1bd1be999fb76dce2de5b)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/030/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:01] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/030/__cache__/flyvision/analysis/stimulus_responses/compute_responses/c4e18ac3fee1bd1be999fb76dce2de5b/output.h5
    [2024-09-28 04:30:01] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/031/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/031', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/031/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash e45ab1c2f5e3e64ddce642cf3e497ee2)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/031/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:01] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/031/__cache__/flyvision/analysis/stimulus_responses/compute_responses/e45ab1c2f5e3e64ddce642cf3e497ee2/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:30:02] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/035/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/035', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/035/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash b7a4bfa4d3087222643fab31b62f51fe)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/035/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:02] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/035/__cache__/flyvision/analysis/stimulus_responses/compute_responses/b7a4bfa4d3087222643fab31b62f51fe/output.h5
    [2024-09-28 04:30:02] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/036/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/036', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/036/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash de3691d1258fd5a1635a52271dec143b)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/036/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:02] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/036/__cache__/flyvision/analysis/stimulus_responses/compute_responses/de3691d1258fd5a1635a52271dec143b/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:30:02] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/037/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/037', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/037/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash f6d87334dddf4b5055f783c1bd4a918f)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/037/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:02] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/037/__cache__/flyvision/analysis/stimulus_responses/compute_responses/f6d87334dddf4b5055f783c1bd4a918f/output.h5
    [2024-09-28 04:30:02] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/042/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/042', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/042/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 57d59f63bb56aa8899f2a0c061c04b8d)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/042/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:02] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/042/__cache__/flyvision/analysis/stimulus_responses/compute_responses/57d59f63bb56aa8899f2a0c061c04b8d/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:30:02] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/044/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/044', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/044/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 8ec6a78ed981b7f3927a8b525b371192)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/044/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:02] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/044/__cache__/flyvision/analysis/stimulus_responses/compute_responses/8ec6a78ed981b7f3927a8b525b371192/output.h5
    [2024-09-28 04:30:02] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/047/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/047', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/047/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 100134dba58e8ff8df01b926651d8cc1)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/047/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:02] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/047/__cache__/flyvision/analysis/stimulus_responses/compute_responses/100134dba58e8ff8df01b926651d8cc1/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:30:02] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/048/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/048', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/048/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 969ef58fdb1292ee59fd0b9c97ed99c4)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/048/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:02] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/048/__cache__/flyvision/analysis/stimulus_responses/compute_responses/969ef58fdb1292ee59fd0b9c97ed99c4/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:30:04] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/004/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/004', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/004/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 73395eed18b129d61ea168228b6dcbce)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/004/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:04] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/004/__cache__/flyvision/analysis/stimulus_responses/compute_responses/73395eed18b129d61ea168228b6dcbce/output.h5
    [2024-09-28 04:30:04] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/005/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/005', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/005/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 78e98f930ff18637efa20e1d548d3a8c)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/005/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:04] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/005/__cache__/flyvision/analysis/stimulus_responses/compute_responses/78e98f930ff18637efa20e1d548d3a8c/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:30:04] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/026/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/026', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/026/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash c987788d6c25740a69e7ec1a679ff258)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/026/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:04] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/026/__cache__/flyvision/analysis/stimulus_responses/compute_responses/c987788d6c25740a69e7ec1a679ff258/output.h5
    [2024-09-28 04:30:05] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/033/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/033', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/033/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 26df16b1c43ef99adf6817123c0c8218)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/033/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:05] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/033/__cache__/flyvision/analysis/stimulus_responses/compute_responses/26df16b1c43ef99adf6817123c0c8218/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:30:05] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/038/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/038', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/038/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash c21fa6c75abbb1492269f408de9df256)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/038/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:05] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/038/__cache__/flyvision/analysis/stimulus_responses/compute_responses/c21fa6c75abbb1492269f408de9df256/output.h5
    [2024-09-28 04:30:05] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/040/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/040', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/040/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 0d3fcc066332bc7dd4678d04b6b4af61)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/040/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:05] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/040/__cache__/flyvision/analysis/stimulus_responses/compute_responses/0d3fcc066332bc7dd4678d04b6b4af61/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:30:05] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/043/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/043', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/043/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 4af2ad30b7fabf3fdb6ed9e1b9176480)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/043/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:05] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/043/__cache__/flyvision/analysis/stimulus_responses/compute_responses/4af2ad30b7fabf3fdb6ed9e1b9176480/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:30:06] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/008/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/008', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/008/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 23d0249da454f978d09c5205e8b0136e)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/008/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:06] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/008/__cache__/flyvision/analysis/stimulus_responses/compute_responses/23d0249da454f978d09c5205e8b0136e/output.h5
    [2024-09-28 04:30:06] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/010/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/010', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/010/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash a9576a365d53c7f625f018b285fdc5b6)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/010/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:06] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/010/__cache__/flyvision/analysis/stimulus_responses/compute_responses/a9576a365d53c7f625f018b285fdc5b6/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:30:06] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/015/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/015', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/015/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 47e8f5d6314ae3e74978ca0c4eb4ca3e)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/015/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:06] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/015/__cache__/flyvision/analysis/stimulus_responses/compute_responses/47e8f5d6314ae3e74978ca0c4eb4ca3e/output.h5
    [2024-09-28 04:30:06] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/025/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/025', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/025/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 1aadf43eef81276244c6817a58eacb4d)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/025/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:06] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/025/__cache__/flyvision/analysis/stimulus_responses/compute_responses/1aadf43eef81276244c6817a58eacb4d/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:30:06] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/028/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/028', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/028/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 03f1ffa516a4c51d7ac3ef725c377377)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/028/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:06] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/028/__cache__/flyvision/analysis/stimulus_responses/compute_responses/03f1ffa516a4c51d7ac3ef725c377377/output.h5
    [2024-09-28 04:30:06] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/032/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/032', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/032/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash b4d629a0b5686c66d2386c4805b2737b)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/032/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:06] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/032/__cache__/flyvision/analysis/stimulus_responses/compute_responses/b4d629a0b5686c66d2386c4805b2737b/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:30:07] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/034/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/034', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/034/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 7676dffb17edcbb69828b60827d4e4a1)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/034/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:07] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/034/__cache__/flyvision/analysis/stimulus_responses/compute_responses/7676dffb17edcbb69828b60827d4e4a1/output.h5
    [2024-09-28 04:30:07] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/039/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/039', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/039/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 70698f9b9a7eed88297f1d5d2a055e42)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/039/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:07] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/039/__cache__/flyvision/analysis/stimulus_responses/compute_responses/70698f9b9a7eed88297f1d5d2a055e42/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:30:07] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/041/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/041', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/041/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 17090a11c0b9256830b8b7d83cd153ef)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/041/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:07] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/041/__cache__/flyvision/analysis/stimulus_responses/compute_responses/17090a11c0b9256830b8b7d83cd153ef/output.h5
    [2024-09-28 04:30:07] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/045/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/045', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/045/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash c1804d08f4c39bdbc0db962817c1d1c0)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/045/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:07] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/045/__cache__/flyvision/analysis/stimulus_responses/compute_responses/c1804d08f4c39bdbc0db962817c1d1c0/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 04:30:07] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/046/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/046', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/046/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash 2a16de239bc0d1c84e0e34668cb060fa)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/046/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:07] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/046/__cache__/flyvision/analysis/stimulus_responses/compute_responses/2a16de239bc0d1c84e0e34668cb060fa/output.h5
    [2024-09-28 04:30:07] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f521be851f0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/049/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/049', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/049/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f5212dd2ee0>, network=None),
    <class 'flyvision.datasets.flashes.Flashes'>, { 'alternations': (0, 1, 0),
      'dt': 0.005,
      'dynamic_range': [0, 1],
      'radius': (-1, 6),
      't_pre': 1.0,
      't_stim': 1},
    4, 1.0, 0.0).

                            (argument hash dfcaf12160e6417247e694d35b314425)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/049/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 04:30:07] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/049/__cache__/flyvision/analysis/stimulus_responses/compute_responses/dfcaf12160e6417247e694d35b314425/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min




![png](figure_03_naturalistic_stimuli_responses_files/figure_03_naturalistic_stimuli_responses_15_52.png)





![png](figure_03_naturalistic_stimuli_responses_files/figure_03_naturalistic_stimuli_responses_15_53.png)





![png](figure_03_naturalistic_stimuli_responses_files/figure_03_naturalistic_stimuli_responses_15_54.png)
