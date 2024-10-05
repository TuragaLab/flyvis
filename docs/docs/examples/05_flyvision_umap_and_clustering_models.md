```python
%load_ext autoreload
%autoreload 2
```

# Cluster analysis based on naturalistic stimuli responses

This notebook illustrates how to cluster the models of an ensemble after nonlinear dimensionality reduction on their predicted responses to naturalistic stimuli. This can be done for any cell type. Here we provide a detailed example focusing on clustering based on T4c responses.

**Select GPU runtime**

To run the notebook on a GPU select Menu -> Runtime -> Change runtime type -> GPU.


```python
# @markdown **Check access to GPU**

try:
    import google.colab

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    import torch

    try:
        cuda_name = torch.cuda.get_device_name()
        print(f"Name of the assigned GPU / CUDA device: {cuda_name}")
    except RuntimeError:
        import warnings

        warnings.warn(
            "You have not selected Runtime Type: 'GPU' or Google could not assign you one. Please revisit the settings as described above or proceed on CPU (slow)."
        )
```

**Install Flyvis**

The notebook requires installing our package `flyvis`. You may need to restart your session after running the code block below with Menu -> Runtime -> Restart session. Then, imports from `flyvis` should succeed without issue.


```python
if IN_COLAB:
    #@markdown **Install Flyvis**
    %%capture
    !git clone https://github.com/flyvis/flyvis-dev.git
    %cd /content/flyvis-dev
    !pip install -e .
```


```python
# basic imports
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams['figure.dpi'] = 200
```

# Naturalistic stimuli dataset (Sintel)
We load the dataset with our custom augmentations. The dataset contains movie sequences from the publicly available computer-animated movie Sintel rendered to the hexagonal lattice structure of the fly eye. For a more detailed introduction to the dataset class and parameters see the notebook on the optic flow task.


```python
import flyvision
from flyvision.datasets.sintel import AugmentedSintel
import numpy as np
```


```python
dt = 1 / 100  # can be changed for other temporal resolutions
dataset = AugmentedSintel(
    tasks=["lum"],
    interpolate=False,
    boxfilter={'extent': 15, 'kernel_size': 13},
    temporal_split=True,
    dt=dt,
)
```


```python
# view stimulus parameters
dataset.arg_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>original_index</th>
      <th>vertical_split_index</th>
      <th>temporal_split_index</th>
      <th>frames</th>
      <th>flip_ax</th>
      <th>n_rot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sequence_00_alley_1_split_00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sequence_00_alley_1_split_00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sequence_00_alley_1_split_00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sequence_00_alley_1_split_00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sequence_00_alley_1_split_00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2263</th>
      <td>sequence_22_temple_3_split_02</td>
      <td>22</td>
      <td>68</td>
      <td>188</td>
      <td>19</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2264</th>
      <td>sequence_22_temple_3_split_02</td>
      <td>22</td>
      <td>68</td>
      <td>188</td>
      <td>19</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2265</th>
      <td>sequence_22_temple_3_split_02</td>
      <td>22</td>
      <td>68</td>
      <td>188</td>
      <td>19</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2266</th>
      <td>sequence_22_temple_3_split_02</td>
      <td>22</td>
      <td>68</td>
      <td>188</td>
      <td>19</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2267</th>
      <td>sequence_22_temple_3_split_02</td>
      <td>22</td>
      <td>68</td>
      <td>188</td>
      <td>19</td>
      <td>1</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>2268 rows × 7 columns</p>
</div>




```python
sequence = dataset[0]["lum"]
```


```python
# one sequence contains 80 frames with 721 hexals each
sequence.shape
```




    torch.Size([80, 1, 721])




```python
animation = flyvision.animations.HexScatter(sequence[None], vmin=0, vmax=1)
animation.animate_in_notebook(frames=np.arange(5))
```



![png](05_flyvision_umap_and_clustering_models_files/05_flyvision_umap_and_clustering_models_13_0.png)



# Ensemble responses to a single sequence
We compute the responses of all models in the stored ensemble to the first sequence of the augmented Sintel dataset.


```python
from flyvision import results_dir
```


```python
# We load the ensemble trained on the optic flow task
ensemble = flyvision.EnsembleView(results_dir / "flow/0000")
```


    Loading ensemble:   0%|          | 0/50 [00:00<?, ?it/s]


    [2024-09-28 03:44:52] ensemble:138 Loaded 50 networks.


We use `ensemble.naturalistic_stimuli_responses` to return responses of all networks within the ensemble.


```python
stims_and_resps = ensemble.naturalistic_stimuli_responses()
```

    [2024-09-28 03:44:52] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/000', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash bef8ab0b1f1206171b1e38bf67305c6f)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:52] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/__cache__/flyvision/analysis/stimulus_responses/compute_responses/bef8ab0b1f1206171b1e38bf67305c6f/output.h5
    [2024-09-28 03:44:52] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/001/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/001', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/001/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 5dfcd34808931599c6e27e5ad94909ca)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/001/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:52] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/001/__cache__/flyvision/analysis/stimulus_responses/compute_responses/5dfcd34808931599c6e27e5ad94909ca/output.h5


    ___________________________________compute_responses cache loaded - 0.1s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:44:52] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/002/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/002', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/002/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 34b1b21c2f716dc063336dbecbacd359)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/002/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:52] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/002/__cache__/flyvision/analysis/stimulus_responses/compute_responses/34b1b21c2f716dc063336dbecbacd359/output.h5
    [2024-09-28 03:44:52] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/003/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/003', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/003/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 5ad5a3d978297c698566ab6886508aab)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/003/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:52] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/003/__cache__/flyvision/analysis/stimulus_responses/compute_responses/5ad5a3d978297c698566ab6886508aab/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:44:53] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/004/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/004', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/004/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 31d213c49b6e1131d94d309a37bba639)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/004/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:53] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/004/__cache__/flyvision/analysis/stimulus_responses/compute_responses/31d213c49b6e1131d94d309a37bba639/output.h5
    [2024-09-28 03:44:53] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/005/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/005', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/005/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 4dabb28b09b5a639c029a18ce380d46e)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/005/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:53] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/005/__cache__/flyvision/analysis/stimulus_responses/compute_responses/4dabb28b09b5a639c029a18ce380d46e/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:44:53] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/006/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/006', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/006/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash a38274cdad5b947ed2e463bb2f899add)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/006/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:53] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/006/__cache__/flyvision/analysis/stimulus_responses/compute_responses/a38274cdad5b947ed2e463bb2f899add/output.h5
    [2024-09-28 03:44:53] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/007/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/007', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/007/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 24ae2d0f3817eeb1c1747a513378f4d2)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/007/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:53] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/007/__cache__/flyvision/analysis/stimulus_responses/compute_responses/24ae2d0f3817eeb1c1747a513378f4d2/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:44:53] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/008/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/008', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/008/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 0123be8edce88ec2f710d1fd9dfff14a)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/008/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:53] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/008/__cache__/flyvision/analysis/stimulus_responses/compute_responses/0123be8edce88ec2f710d1fd9dfff14a/output.h5
    [2024-09-28 03:44:53] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/009/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/009', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/009/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 718237993c1fb18418d915b2968ff3c2)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/009/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:53] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/009/__cache__/flyvision/analysis/stimulus_responses/compute_responses/718237993c1fb18418d915b2968ff3c2/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:44:53] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/010/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/010', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/010/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash ff2491ccb529684a8e9a7df2d7b79bf9)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/010/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:53] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/010/__cache__/flyvision/analysis/stimulus_responses/compute_responses/ff2491ccb529684a8e9a7df2d7b79bf9/output.h5
    [2024-09-28 03:44:54] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/011/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/011', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/011/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 83252c4949e38c210f3aa72e7ef22069)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/011/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:54] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/011/__cache__/flyvision/analysis/stimulus_responses/compute_responses/83252c4949e38c210f3aa72e7ef22069/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:44:54] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/012/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/012', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/012/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 1538b2fc8f46ad04617796f998332762)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/012/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:54] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/012/__cache__/flyvision/analysis/stimulus_responses/compute_responses/1538b2fc8f46ad04617796f998332762/output.h5
    [2024-09-28 03:44:54] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/013/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/013', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/013/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 585cfb92bccb448a82653108abb70ff3)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/013/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:54] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/013/__cache__/flyvision/analysis/stimulus_responses/compute_responses/585cfb92bccb448a82653108abb70ff3/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:44:54] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/014/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/014', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/014/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash a52998e5072aa7b3f390b8dfd1c6889c)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/014/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:54] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/014/__cache__/flyvision/analysis/stimulus_responses/compute_responses/a52998e5072aa7b3f390b8dfd1c6889c/output.h5
    [2024-09-28 03:44:54] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/015/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/015', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/015/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash a402afd786b7a1463e110a73ff0d319e)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/015/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:54] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/015/__cache__/flyvision/analysis/stimulus_responses/compute_responses/a402afd786b7a1463e110a73ff0d319e/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:44:54] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/016/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/016', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/016/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash f0efdcc46faa20ab4ce5c963b0ca7d8d)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/016/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:54] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/016/__cache__/flyvision/analysis/stimulus_responses/compute_responses/f0efdcc46faa20ab4ce5c963b0ca7d8d/output.h5
    [2024-09-28 03:44:54] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/017/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/017', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/017/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash b12b5469a747e95fc2ce4e7b5bf73b9a)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/017/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:54] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/017/__cache__/flyvision/analysis/stimulus_responses/compute_responses/b12b5469a747e95fc2ce4e7b5bf73b9a/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:44:55] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/018/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/018', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/018/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 72b6cc7f982a87e845f2e393a185ef96)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/018/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:55] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/018/__cache__/flyvision/analysis/stimulus_responses/compute_responses/72b6cc7f982a87e845f2e393a185ef96/output.h5
    [2024-09-28 03:44:55] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/019/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/019', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/019/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 55e904c1132161ff29d7476e20b2cbea)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/019/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:55] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/019/__cache__/flyvision/analysis/stimulus_responses/compute_responses/55e904c1132161ff29d7476e20b2cbea/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:44:55] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/020/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/020', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/020/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 5fcbf0ff71289c927d78ab82b8b658e1)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/020/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:55] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/020/__cache__/flyvision/analysis/stimulus_responses/compute_responses/5fcbf0ff71289c927d78ab82b8b658e1/output.h5
    [2024-09-28 03:44:55] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/021/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/021', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/021/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 8206028404679417dccf5dada182bcce)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/021/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:55] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/021/__cache__/flyvision/analysis/stimulus_responses/compute_responses/8206028404679417dccf5dada182bcce/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:44:55] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/022/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/022', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/022/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 399d668f254864b6559e7efb3769bd0a)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/022/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:55] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/022/__cache__/flyvision/analysis/stimulus_responses/compute_responses/399d668f254864b6559e7efb3769bd0a/output.h5
    [2024-09-28 03:44:55] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/023/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/023', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/023/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 23cc9baf9d310beceb3262e6c4da1743)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/023/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:55] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/023/__cache__/flyvision/analysis/stimulus_responses/compute_responses/23cc9baf9d310beceb3262e6c4da1743/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:44:55] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/024/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/024', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/024/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 60b4790394985973a5e808cb0a7869c0)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/024/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:55] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/024/__cache__/flyvision/analysis/stimulus_responses/compute_responses/60b4790394985973a5e808cb0a7869c0/output.h5
    [2024-09-28 03:44:55] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/025/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/025', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/025/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 6d866edcf636ca2c5de322d90262cc52)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/025/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:56] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/025/__cache__/flyvision/analysis/stimulus_responses/compute_responses/6d866edcf636ca2c5de322d90262cc52/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:44:56] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/026/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/026', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/026/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash f7c36c5c21211df0f419b5284403d88b)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/026/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:56] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/026/__cache__/flyvision/analysis/stimulus_responses/compute_responses/f7c36c5c21211df0f419b5284403d88b/output.h5
    [2024-09-28 03:44:56] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/027/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/027', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/027/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 459546e87c028c04222599bc967a4a61)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/027/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:56] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/027/__cache__/flyvision/analysis/stimulus_responses/compute_responses/459546e87c028c04222599bc967a4a61/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:44:56] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/028/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/028', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/028/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash b1ca46217250705cd80c3356d9084f7e)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/028/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:56] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/028/__cache__/flyvision/analysis/stimulus_responses/compute_responses/b1ca46217250705cd80c3356d9084f7e/output.h5
    [2024-09-28 03:44:56] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/029/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/029', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/029/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 7907dbe75d0ae5ecc252d7abc2fc21ba)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/029/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:56] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/029/__cache__/flyvision/analysis/stimulus_responses/compute_responses/7907dbe75d0ae5ecc252d7abc2fc21ba/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:44:56] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/030/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/030', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/030/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash bb38c7354699e2a69caae52e10193d6c)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/030/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:56] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/030/__cache__/flyvision/analysis/stimulus_responses/compute_responses/bb38c7354699e2a69caae52e10193d6c/output.h5
    [2024-09-28 03:44:56] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/031/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/031', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/031/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash a48b16337cba74c9740c8892c0d694ed)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/031/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:56] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/031/__cache__/flyvision/analysis/stimulus_responses/compute_responses/a48b16337cba74c9740c8892c0d694ed/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:44:56] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/032/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/032', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/032/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 3f4fec610bd93e077866f41e9f14be97)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/032/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:56] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/032/__cache__/flyvision/analysis/stimulus_responses/compute_responses/3f4fec610bd93e077866f41e9f14be97/output.h5
    [2024-09-28 03:44:57] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/033/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/033', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/033/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 7af457beb2a5448a382a401f87c6c43b)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/033/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:57] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/033/__cache__/flyvision/analysis/stimulus_responses/compute_responses/7af457beb2a5448a382a401f87c6c43b/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:44:57] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/034/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/034', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/034/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash f4d5d3e124a1707086a4b9571dcf88fd)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/034/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:57] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/034/__cache__/flyvision/analysis/stimulus_responses/compute_responses/f4d5d3e124a1707086a4b9571dcf88fd/output.h5
    [2024-09-28 03:44:57] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/035/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/035', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/035/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 066e0e5617f86c1a995fd9593f6ae50d)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/035/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:57] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/035/__cache__/flyvision/analysis/stimulus_responses/compute_responses/066e0e5617f86c1a995fd9593f6ae50d/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:44:57] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/036/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/036', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/036/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 6dc79549ae41d127ad03473dd19d7191)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/036/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:57] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/036/__cache__/flyvision/analysis/stimulus_responses/compute_responses/6dc79549ae41d127ad03473dd19d7191/output.h5
    [2024-09-28 03:44:57] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/037/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/037', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/037/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 11242a4bd086271b04d8b381fe536d34)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/037/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:57] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/037/__cache__/flyvision/analysis/stimulus_responses/compute_responses/11242a4bd086271b04d8b381fe536d34/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:44:57] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/038/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/038', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/038/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 4de539f0328a62b7b7b2e2f406ac512f)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/038/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:57] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/038/__cache__/flyvision/analysis/stimulus_responses/compute_responses/4de539f0328a62b7b7b2e2f406ac512f/output.h5
    [2024-09-28 03:44:57] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/039/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/039', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/039/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 54037e45c494136e3ec0df5c68b144b9)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/039/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:57] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/039/__cache__/flyvision/analysis/stimulus_responses/compute_responses/54037e45c494136e3ec0df5c68b144b9/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:44:58] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/040/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/040', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/040/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash aee145f08d47e04be77524fff9b97a0e)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/040/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:58] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/040/__cache__/flyvision/analysis/stimulus_responses/compute_responses/aee145f08d47e04be77524fff9b97a0e/output.h5
    [2024-09-28 03:44:58] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/041/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/041', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/041/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 8d00acd269369e38751d2eebc739cfad)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/041/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:58] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/041/__cache__/flyvision/analysis/stimulus_responses/compute_responses/8d00acd269369e38751d2eebc739cfad/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:44:58] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/042/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/042', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/042/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 1df07a8c6046bb4b1e015fddfdc522bd)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/042/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:58] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/042/__cache__/flyvision/analysis/stimulus_responses/compute_responses/1df07a8c6046bb4b1e015fddfdc522bd/output.h5
    [2024-09-28 03:44:58] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/043/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/043', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/043/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 118907b7b0db15c9a12ebe485363b677)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/043/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:58] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/043/__cache__/flyvision/analysis/stimulus_responses/compute_responses/118907b7b0db15c9a12ebe485363b677/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:44:58] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/044/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/044', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/044/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash b7cbac573ab7f8613e90b80217af416a)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/044/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:58] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/044/__cache__/flyvision/analysis/stimulus_responses/compute_responses/b7cbac573ab7f8613e90b80217af416a/output.h5
    [2024-09-28 03:44:58] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/045/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/045', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/045/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 47d58ea044d58bb33a1e19bb6d6da424)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/045/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:58] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/045/__cache__/flyvision/analysis/stimulus_responses/compute_responses/47d58ea044d58bb33a1e19bb6d6da424/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:44:58] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/046/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/046', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/046/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 4934fc3026c637ce814154f98aaee621)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/046/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:58] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/046/__cache__/flyvision/analysis/stimulus_responses/compute_responses/4934fc3026c637ce814154f98aaee621/output.h5
    [2024-09-28 03:44:59] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/047/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/047', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/047/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 35cb033dfacc17e90bc630ef8cd25ebc)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/047/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:59] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/047/__cache__/flyvision/analysis/stimulus_responses/compute_responses/35cb033dfacc17e90bc630ef8cd25ebc/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:44:59] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/048/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/048', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/048/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 0cd0d09b9f5f4a64c8e753aca75bbdb0)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/048/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:59] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/048/__cache__/flyvision/analysis/stimulus_responses/compute_responses/0cd0d09b9f5f4a64c8e753aca75bbdb0/output.h5
    [2024-09-28 03:44:59] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/049/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/049', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/049/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True},
    4, 0.0, 2.0).

                            (argument hash 22e903adf1516c4f5bcf3461281e707d)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/049/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:44:59] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/049/__cache__/flyvision/analysis/stimulus_responses/compute_responses/22e903adf1516c4f5bcf3461281e707d/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min



```python
norm = ensemble.responses_norm()
```


```python
responses = stims_and_resps["responses"] / (norm + 1e-6)
```


```python
responses.custom.where(cell_type="T4c", u=0, v=0, sample=0).custom.plot_traces(
    x="time", plot_kwargs=dict(color="tab:blue", add_legend=False)
)
ax = plt.gca()
ax.set_title("T4c responses to naturalistic stimuli")
```




    Text(0.5, 1.0, 'T4c responses to naturalistic stimuli')





![png](05_flyvision_umap_and_clustering_models_files/05_flyvision_umap_and_clustering_models_21_1.png)



We see that the across models of the ensemble the predictions for T4c vary. Our goal is to understand the underlying structure in those variations.

## Nonlinear dimensionality reduction (UMAP) and Gaussian Mixtures


```python
from flyvision.analysis.clustering import EnsembleEmbedding, get_cluster_to_indices
from flyvision.utils.activity_utils import CentralActivity
```


```python
# specify parameters for umap embedding

embedding_kwargs = {
    "min_dist": 0.105,
    "spread": 9.0,
    "n_neighbors": 5,
    "random_state": 42,
    "n_epochs": 1500,
}
```

We compute the UMAP embedding of the ensemble based on the T4c responses of the single models to the single sequence for illustration.


```python
central_responses = CentralActivity(responses.values,
                                    connectome=ensemble.connectome)
```


```python
embedding = EnsembleEmbedding(central_responses)
t4c_embedding = embedding("T4c", embedding_kwargs=embedding_kwargs)
```

    [2024-09-28 03:45:09] clustering:346 reshaped X from (50, 2268, 80) to (50, 181440)


    /home/lappalainenj@hhmi.org/miniconda3/envs/flyvision/lib/python3.9/site-packages/umap/umap_.py:1356: RuntimeWarning: divide by zero encountered in power
      return 1.0 / (1.0 + a * x ** (2 * b))



```python
task_error = ensemble.task_error()
```


```python
embeddingplot = t4c_embedding.plot(colors=task_error.colors)
```



![png](05_flyvision_umap_and_clustering_models_files/05_flyvision_umap_and_clustering_models_30_0.png)



Each of these scatterpoints in 2d represents a single time series plotted above.

We fit a Gaussian Mixture of 2 to 5 components to this embedding to label the clusters. We select the final number of Gaussian Mixture components that minimize the Bayesian Information Criterion (BIC).


```python
# specifiy parameters for Gaussian Mixture

gm_kwargs = {
    "range_n_clusters": [1, 2, 3, 4, 5],
    "n_init": 100,
    "max_iter": 1000,
    "random_state": 42,
    "tol": 0.001,
}
```


```python
gm_clustering = t4c_embedding.cluster.gaussian_mixture(**gm_kwargs)
```


```python
embeddingplot = gm_clustering.plot(
    task_error=task_error.values, colors=task_error.colors
)
```



![png](05_flyvision_umap_and_clustering_models_files/05_flyvision_umap_and_clustering_models_34_0.png)



We can use the labels to disambiguate the time series data that we plotted above. We expect that these labels aggregate similar time series together and different time series separately.


```python
import matplotlib.colors as mcolors
```


```python
cluster_to_indices = get_cluster_to_indices(
    embeddingplot.cluster.embedding.mask,
    embeddingplot.cluster.labels,
    ensemble.task_error(),
)
```


```python
fig, axes = plt.subplots(1, len(cluster_to_indices), figsize=(6, 2))
colors = {i: color for i, color in enumerate(mcolors.TABLEAU_COLORS.values())}
for cluster_id, indices in cluster_to_indices.items():
    responses.sel(network_id=indices, sample=[0]).custom.where(
        cell_type="T4c"
    ).custom.plot_traces(
        x="time",
        plot_kwargs=dict(
            color=colors[cluster_id], add_legend=False, ax=axes[cluster_id]
        ),
    )
    axes[cluster_id].set_title(f"Cluster {cluster_id + 1}")
plt.subplots_adjust(wspace=0.3)
```



![png](05_flyvision_umap_and_clustering_models_files/05_flyvision_umap_and_clustering_models_38_0.png)



The clustering has led us to three qualitatively distinct predictions from the ensemble for this cell and sequence. This is a first lead for an underlying structure in these predictions. We will get an even better estimate once we use more sequences for the clustering.

# Using the clustering to discover tuning predictions in responses to simple stimuli

We expect that the clustering based on naturalistic stimuli will also disambiguate the different tuning predictions from different models for simple stimuli.


```python
cluster_to_indices = get_cluster_to_indices(
    embeddingplot.cluster.embedding.mask,
    embeddingplot.cluster.labels,
    ensemble.task_error(),
)
```


```python
# define different colormaps for clusters
cluster_colors = {}
CMAPS = ["Blues_r", "Reds_r", "Greens_r", "Oranges_r", "Purples_r"]

for cluster_id in cluster_to_indices:
    cluster_colors[cluster_id] = ensemble.task_error(cmap=CMAPS[cluster_id]).colors
```

## Clustered voltage responses to moving edges


```python
from flyvision.analysis.moving_bar_responses import plot_angular_tuning
from flyvision.plots.plt_utils import add_cluster_marker, get_marker
from flyvision.utils.color_utils import color_to_cmap
```


```python
stims_and_resps_moving_edge = ensemble.moving_edge_responses()
```

    [2024-09-28 03:45:27] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/000', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 77be693699aefc8a65b21a471879e324)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:27] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/__cache__/flyvision/analysis/stimulus_responses/compute_responses/77be693699aefc8a65b21a471879e324/output.h5
    [2024-09-28 03:45:27] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/001/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/001', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/001/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 1d2c73d297130e58bf05418b7385227e)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/001/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:27] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/001/__cache__/flyvision/analysis/stimulus_responses/compute_responses/1d2c73d297130e58bf05418b7385227e/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:45:27] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/002/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/002', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/002/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash a492cccda41a28f65f6051985f0cd6fc)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/002/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:27] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/002/__cache__/flyvision/analysis/stimulus_responses/compute_responses/a492cccda41a28f65f6051985f0cd6fc/output.h5
    [2024-09-28 03:45:27] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/003/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/003', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/003/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 86e7a776a281aa3c6274211ff33649db)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/003/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:27] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/003/__cache__/flyvision/analysis/stimulus_responses/compute_responses/86e7a776a281aa3c6274211ff33649db/output.h5
    [2024-09-28 03:45:27] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/004/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/004', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/004/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash cbb23a582ff5ce0c06961764a957f0d0)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/004/__cache__/flyvision/analysis/stimulus_responses/compute_responses.



    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:45:27] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/004/__cache__/flyvision/analysis/stimulus_responses/compute_responses/cbb23a582ff5ce0c06961764a957f0d0/output.h5
    [2024-09-28 03:45:27] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/005/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/005', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/005/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash ef1adddb6ce7d96ba6e3dba901c05dd7)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/005/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:27] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/005/__cache__/flyvision/analysis/stimulus_responses/compute_responses/ef1adddb6ce7d96ba6e3dba901c05dd7/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:45:28] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/006/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/006', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/006/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 17643fb00254698b3a84bd93a2e0903d)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/006/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:28] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/006/__cache__/flyvision/analysis/stimulus_responses/compute_responses/17643fb00254698b3a84bd93a2e0903d/output.h5
    [2024-09-28 03:45:28] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/007/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/007', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/007/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash c6e2af646c29f24a4c697b4c066d8916)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/007/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:28] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/007/__cache__/flyvision/analysis/stimulus_responses/compute_responses/c6e2af646c29f24a4c697b4c066d8916/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:45:28] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/008/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/008', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/008/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 88dff80cbdda9c384ba9384e600e62a6)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/008/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:28] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/008/__cache__/flyvision/analysis/stimulus_responses/compute_responses/88dff80cbdda9c384ba9384e600e62a6/output.h5
    [2024-09-28 03:45:28] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/009/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/009', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/009/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 395a0cfb766f550893c991f34a74ce3d)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/009/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:28] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/009/__cache__/flyvision/analysis/stimulus_responses/compute_responses/395a0cfb766f550893c991f34a74ce3d/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:45:28] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/010/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/010', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/010/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 1b50a7069c7e4554c7d0eb916570aef1)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/010/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:28] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/010/__cache__/flyvision/analysis/stimulus_responses/compute_responses/1b50a7069c7e4554c7d0eb916570aef1/output.h5
    [2024-09-28 03:45:28] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/011/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/011', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/011/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 79f834efef624b1e9d60c43d17276867)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/011/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:28] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/011/__cache__/flyvision/analysis/stimulus_responses/compute_responses/79f834efef624b1e9d60c43d17276867/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:45:28] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/012/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/012', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/012/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash e06aa1ed77434be8547758c8c99f9e46)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/012/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:28] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/012/__cache__/flyvision/analysis/stimulus_responses/compute_responses/e06aa1ed77434be8547758c8c99f9e46/output.h5
    [2024-09-28 03:45:28] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/013/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/013', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/013/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 925a6e44c1bab90e390b5ad9c2b6981c)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/013/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:28] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/013/__cache__/flyvision/analysis/stimulus_responses/compute_responses/925a6e44c1bab90e390b5ad9c2b6981c/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:45:29] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/014/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/014', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/014/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 9e020caeb163c1d9ba05f97942ae5dcf)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/014/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:29] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/014/__cache__/flyvision/analysis/stimulus_responses/compute_responses/9e020caeb163c1d9ba05f97942ae5dcf/output.h5
    [2024-09-28 03:45:29] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/015/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/015', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/015/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 0d688cabf8fc136bfd108ba052e269aa)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/015/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:29] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/015/__cache__/flyvision/analysis/stimulus_responses/compute_responses/0d688cabf8fc136bfd108ba052e269aa/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:45:29] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/016/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/016', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/016/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 288293f54c15e1f1a01bb3fc172da208)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/016/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:29] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/016/__cache__/flyvision/analysis/stimulus_responses/compute_responses/288293f54c15e1f1a01bb3fc172da208/output.h5
    [2024-09-28 03:45:29] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/017/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/017', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/017/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 5f9706358964c27cab147f777ab3532d)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/017/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:29] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/017/__cache__/flyvision/analysis/stimulus_responses/compute_responses/5f9706358964c27cab147f777ab3532d/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:45:29] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/018/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/018', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/018/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 91cbd372e15242b67cc575d217793c77)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/018/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:29] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/018/__cache__/flyvision/analysis/stimulus_responses/compute_responses/91cbd372e15242b67cc575d217793c77/output.h5
    [2024-09-28 03:45:29] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/019/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/019', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/019/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 83afbe6c8322f3ae8730b5a6ef08810c)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/019/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:29] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/019/__cache__/flyvision/analysis/stimulus_responses/compute_responses/83afbe6c8322f3ae8730b5a6ef08810c/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:45:29] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/020/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/020', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/020/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 39a733fede1ec19c0cf72d18c654c9ea)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/020/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:29] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/020/__cache__/flyvision/analysis/stimulus_responses/compute_responses/39a733fede1ec19c0cf72d18c654c9ea/output.h5
    [2024-09-28 03:45:30] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/021/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/021', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/021/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 23b9b479ad5e7a60ac2968d5be088b01)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/021/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:30] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/021/__cache__/flyvision/analysis/stimulus_responses/compute_responses/23b9b479ad5e7a60ac2968d5be088b01/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:45:30] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/022/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/022', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/022/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 09cccac26f2c0bdfd79875647c896f91)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/022/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:30] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/022/__cache__/flyvision/analysis/stimulus_responses/compute_responses/09cccac26f2c0bdfd79875647c896f91/output.h5
    [2024-09-28 03:45:30] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/023/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/023', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/023/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 3eb89a8b21395702eeeb5688fa041b25)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/023/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:30] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/023/__cache__/flyvision/analysis/stimulus_responses/compute_responses/3eb89a8b21395702eeeb5688fa041b25/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:45:30] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/024/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/024', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/024/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash fffb6b84d25a66672dcf7e1c01337efe)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/024/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:30] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/024/__cache__/flyvision/analysis/stimulus_responses/compute_responses/fffb6b84d25a66672dcf7e1c01337efe/output.h5
    [2024-09-28 03:45:30] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/025/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/025', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/025/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 30a8def9735fa87e11508ff3667b9657)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/025/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:30] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/025/__cache__/flyvision/analysis/stimulus_responses/compute_responses/30a8def9735fa87e11508ff3667b9657/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:45:30] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/026/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/026', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/026/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 67b400f5402ea2331da87fbbc98819a9)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/026/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:30] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/026/__cache__/flyvision/analysis/stimulus_responses/compute_responses/67b400f5402ea2331da87fbbc98819a9/output.h5
    [2024-09-28 03:45:30] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/027/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/027', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/027/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 311992d8d68da98c9765828039f1e273)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/027/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:30] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/027/__cache__/flyvision/analysis/stimulus_responses/compute_responses/311992d8d68da98c9765828039f1e273/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:45:31] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/028/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/028', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/028/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash dfb813caf89e1b3d059c9919f59268f6)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/028/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:31] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/028/__cache__/flyvision/analysis/stimulus_responses/compute_responses/dfb813caf89e1b3d059c9919f59268f6/output.h5
    [2024-09-28 03:45:31] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/029/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/029', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/029/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash fa256c3c26d9ee5218c9f04a7b1a1d4a)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/029/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:31] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/029/__cache__/flyvision/analysis/stimulus_responses/compute_responses/fa256c3c26d9ee5218c9f04a7b1a1d4a/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:45:31] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/030/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/030', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/030/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 2c4e1d3435c7d860f2a28d085ebe60b8)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/030/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:31] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/030/__cache__/flyvision/analysis/stimulus_responses/compute_responses/2c4e1d3435c7d860f2a28d085ebe60b8/output.h5
    [2024-09-28 03:45:31] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/031/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/031', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/031/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash fba723386be5ff554eeee43e00ed9864)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/031/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:31] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/031/__cache__/flyvision/analysis/stimulus_responses/compute_responses/fba723386be5ff554eeee43e00ed9864/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:45:31] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/032/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/032', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/032/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 2f3d76e84ab24b947c7f702e13e2c492)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/032/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:31] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/032/__cache__/flyvision/analysis/stimulus_responses/compute_responses/2f3d76e84ab24b947c7f702e13e2c492/output.h5
    [2024-09-28 03:45:31] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/033/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/033', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/033/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 8b7b59496e349519b4e73a33521bc534)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/033/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:31] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/033/__cache__/flyvision/analysis/stimulus_responses/compute_responses/8b7b59496e349519b4e73a33521bc534/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:45:31] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/034/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/034', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/034/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash a41b1afbd9774383cf1d9c27a872c18f)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/034/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:31] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/034/__cache__/flyvision/analysis/stimulus_responses/compute_responses/a41b1afbd9774383cf1d9c27a872c18f/output.h5
    [2024-09-28 03:45:32] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/035/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/035', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/035/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 008ad9d6c9d8e45c8bd399b790db37b7)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/035/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:32] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/035/__cache__/flyvision/analysis/stimulus_responses/compute_responses/008ad9d6c9d8e45c8bd399b790db37b7/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:45:32] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/036/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/036', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/036/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 73de418ea2de244810e4551b90c7d0c4)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/036/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:32] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/036/__cache__/flyvision/analysis/stimulus_responses/compute_responses/73de418ea2de244810e4551b90c7d0c4/output.h5
    [2024-09-28 03:45:32] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/037/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/037', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/037/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 6bcd62a2a8814c0fe7b141d7cc7a697b)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/037/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:32] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/037/__cache__/flyvision/analysis/stimulus_responses/compute_responses/6bcd62a2a8814c0fe7b141d7cc7a697b/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:45:32] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/038/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/038', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/038/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 35f979e9a8e3b715c48e2c77dc19b7e7)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/038/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:32] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/038/__cache__/flyvision/analysis/stimulus_responses/compute_responses/35f979e9a8e3b715c48e2c77dc19b7e7/output.h5
    [2024-09-28 03:45:32] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/039/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/039', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/039/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash c00f70d0ec34030129daf8d8e590c9af)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/039/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:32] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/039/__cache__/flyvision/analysis/stimulus_responses/compute_responses/c00f70d0ec34030129daf8d8e590c9af/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:45:32] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/040/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/040', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/040/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 6df6b75392955b45e5eb38727672181f)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/040/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:32] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/040/__cache__/flyvision/analysis/stimulus_responses/compute_responses/6df6b75392955b45e5eb38727672181f/output.h5
    [2024-09-28 03:45:32] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/041/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/041', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/041/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 3c4cbbdfafe431a0b4d4fba40eea79ae)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/041/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:32] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/041/__cache__/flyvision/analysis/stimulus_responses/compute_responses/3c4cbbdfafe431a0b4d4fba40eea79ae/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:45:33] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/042/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/042', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/042/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 7b6ff84c5542a888c65a2d9deb92e590)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/042/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:33] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/042/__cache__/flyvision/analysis/stimulus_responses/compute_responses/7b6ff84c5542a888c65a2d9deb92e590/output.h5
    [2024-09-28 03:45:33] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/043/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/043', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/043/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 307ee913c057f0dd1174186d199d5206)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/043/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:33] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/043/__cache__/flyvision/analysis/stimulus_responses/compute_responses/307ee913c057f0dd1174186d199d5206/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:45:33] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/044/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/044', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/044/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 1bf587096948fa0bc762bf8ea1cdc812)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/044/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:33] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/044/__cache__/flyvision/analysis/stimulus_responses/compute_responses/1bf587096948fa0bc762bf8ea1cdc812/output.h5
    [2024-09-28 03:45:33] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/045/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/045', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/045/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash fa0575c9275e23ba7b3f6c3e200f7439)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/045/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:33] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/045/__cache__/flyvision/analysis/stimulus_responses/compute_responses/fa0575c9275e23ba7b3f6c3e200f7439/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:45:34] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/046/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/046', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/046/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 43205473c0733c1640466cae80b7ef1b)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/046/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:34] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/046/__cache__/flyvision/analysis/stimulus_responses/compute_responses/43205473c0733c1640466cae80b7ef1b/output.h5
    [2024-09-28 03:45:34] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/047/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/047', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/047/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 68b9c9f4751b092c82426454cd0c68ca)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/047/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:34] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/047/__cache__/flyvision/analysis/stimulus_responses/compute_responses/68b9c9f4751b092c82426454cd0c68ca/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min


    [2024-09-28 03:45:34] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/048/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/048', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/048/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash 8b2c68af76beb5bea7f8c1643bc93a77)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/048/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:34] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/048/__cache__/flyvision/analysis/stimulus_responses/compute_responses/8b2c68af76beb5bea7f8c1643bc93a77/output.h5
    [2024-09-28 03:45:34] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fda303570d0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/049/__cache__)]:
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/049', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/049/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7fda27385d30>, network=None),
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'device': 'cuda',
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': (-10, 11),
      'post_pad_mode': 'continue',
      'speeds': (2.4, 4.8, 9.7, 13, 19, 25),
      't_post': 1.0,
      't_pre': 1.0},
    4, 1.0, 0.0).

                            (argument hash c7036d5becf2c39915e36f3a2a1adfd4)

                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/049/__cache__/flyvision/analysis/stimulus_responses/compute_responses.

    [2024-09-28 03:45:34] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/049/__cache__/flyvision/analysis/stimulus_responses/compute_responses/c7036d5becf2c39915e36f3a2a1adfd4/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    ___________________________________compute_responses cache loaded - 0.0s, 0.0min



```python
# invariant to different magnitudes of responses, only to assess direction tuning
stims_and_resps_moving_edge["responses"] /= np.abs(stims_and_resps_moving_edge["responses"]).max(dim=("sample", "frame"))

# relative to the norm of the responses to naturalistic stimuli (used for averaging)
# stims_and_resps_moving_edge['responses'] /= (norm + 1e-6)
```


```python
fig, axes = plt.subplots(1, len(cluster_to_indices), figsize=(6, 2))
colors = {i: color for i, color in enumerate(mcolors.TABLEAU_COLORS.values())}
for cluster_id, indices in cluster_to_indices.items():
    stims_and_resps_moving_edge['responses'].sel(network_id=indices).custom.where(
        cell_type="T4c", intensity=1, speed=19, angle=90
    ).custom.plot_traces(
        x="time",
        plot_kwargs=dict(
            color=colors[cluster_id], add_legend=False, ax=axes[cluster_id]
        ),
    )
    axes[cluster_id].set_title(f"Cluster {cluster_id + 1}")
plt.subplots_adjust(wspace=0.3)
```



![png](05_flyvision_umap_and_clustering_models_files/05_flyvision_umap_and_clustering_models_48_0.png)




```python
plot_angular_tuning(
        stims_and_resps_moving_edge,
        "T4c",
        intensity=1,
        )
```




    (<Figure size 300x300 with 1 Axes>, <PolarAxes: >)





![png](05_flyvision_umap_and_clustering_models_files/05_flyvision_umap_and_clustering_models_49_1.png)




```python
tabcolors = list(mcolors.TABLEAU_COLORS.values())
colors = [ensemble.task_error(cmap=color_to_cmap(tabcolors[cluster_id]).reversed()).colors[indices] for cluster_id, indices in cluster_to_indices.items()]
fig, axes = plt.subplots(
    1, len(cluster_to_indices), subplot_kw={"projection": "polar"}, figsize=[2, 1]
)
for cluster_id, indices in cluster_to_indices.items():
    plot_angular_tuning(
        stims_and_resps_moving_edge.sel(network_id=indices),
        "T4c",
        intensity=1,
        colors=colors[cluster_id],
        zorder=ensemble.zorder()[indices],
        groundtruth=True if cluster_id == 0 else False,
        fig=fig,
        ax=axes[cluster_id],
    )
    add_cluster_marker(fig, axes[cluster_id], marker=get_marker(cluster_id))
    axes[cluster_id].set_title(f"Cluster {cluster_id + 1}")
plt.subplots_adjust(wspace=0.5)
```



![png](05_flyvision_umap_and_clustering_models_files/05_flyvision_umap_and_clustering_models_50_0.png)



As we can see here, the models predict clustered neural responses.

# Load precomputed umap and clustering

Due to the computational requirement of recording and embedding all responses and for consistency we also show how to use the precomputed embeddings and clusterings from the paper.


```python
cell_type = "T4c"
clustering = ensemble.clustering(cell_type)
```

    [2024-09-28 03:45:53] clustering:640 Loaded T4c embedding and clustering from /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/umap_and_clustering.



```python
task_error = ensemble.task_error()
```


```python
embeddingplot = clustering.plot(task_error=task_error.values,
                                colors=task_error.colors)
```



![png](05_flyvision_umap_and_clustering_models_files/05_flyvision_umap_and_clustering_models_56_0.png)



With this embedding and clustering one can proceed in the same way as above to plot the tunings.
