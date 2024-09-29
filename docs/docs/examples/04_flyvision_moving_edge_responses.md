```python
%load_ext autoreload
%autoreload 2
```

# Moving edge responses

This notebook introduces moving edge responses and the direction selectivity index (DSI). The DSI measures motion selectivity of cells to visual input.

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

plt.rcParams["figure.dpi"] = 200
```

## Moving edge stimuli

To elicit moving edge responses and characterise the motion selectivity of neurons, experimenters show an ON or OFF edge moving in different cardinal directions. We generate and render these stimuli with the `MovingEdge` dataset.


```python
# import dataset and visualization helper
from flyvision.datasets.moving_bar import MovingEdge
from flyvision.animations.hexscatter import HexScatter
```


```python
# initialize dataset
# make the dataset
dataset = MovingEdge(
    offsets=[-10, 11],  # offset of bar from center in 1 * radians(2.25) led size
    intensities=[0, 1],  # intensity of bar
    speeds=[19],  # speed of bar in 1 * radians(5.8) / s
    height=80,  # height of moving bar in 1 * radians(2.25) led size
    post_pad_mode="continue",  # for post-stimulus period, continue with the last frame of the stimulus
    t_pre=1.0,  # duration of pre-stimulus period
    t_post=1.0,  # duration of post-stimulus period
    dt=1 / 200,  # temporal resolution of rendered video
    angles=list(np.arange(0, 360, 30)),  # motion direction (orthogonal to edge)
)
```


```python
# view stimulus parameters
dataset.arg_df
# the dataset has four samples, one corresponding to each row
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
      <th>angle</th>
      <th>width</th>
      <th>intensity</th>
      <th>t_stim</th>
      <th>speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>80</td>
      <td>0</td>
      <td>0.428766</td>
      <td>19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>80</td>
      <td>1</td>
      <td>0.428766</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>80</td>
      <td>0</td>
      <td>0.428766</td>
      <td>19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>80</td>
      <td>1</td>
      <td>0.428766</td>
      <td>19</td>
    </tr>
    <tr>
      <th>4</th>
      <td>60</td>
      <td>80</td>
      <td>0</td>
      <td>0.428766</td>
      <td>19</td>
    </tr>
    <tr>
      <th>5</th>
      <td>60</td>
      <td>80</td>
      <td>1</td>
      <td>0.428766</td>
      <td>19</td>
    </tr>
    <tr>
      <th>6</th>
      <td>90</td>
      <td>80</td>
      <td>0</td>
      <td>0.428766</td>
      <td>19</td>
    </tr>
    <tr>
      <th>7</th>
      <td>90</td>
      <td>80</td>
      <td>1</td>
      <td>0.428766</td>
      <td>19</td>
    </tr>
    <tr>
      <th>8</th>
      <td>120</td>
      <td>80</td>
      <td>0</td>
      <td>0.428766</td>
      <td>19</td>
    </tr>
    <tr>
      <th>9</th>
      <td>120</td>
      <td>80</td>
      <td>1</td>
      <td>0.428766</td>
      <td>19</td>
    </tr>
    <tr>
      <th>10</th>
      <td>150</td>
      <td>80</td>
      <td>0</td>
      <td>0.428766</td>
      <td>19</td>
    </tr>
    <tr>
      <th>11</th>
      <td>150</td>
      <td>80</td>
      <td>1</td>
      <td>0.428766</td>
      <td>19</td>
    </tr>
    <tr>
      <th>12</th>
      <td>180</td>
      <td>80</td>
      <td>0</td>
      <td>0.428766</td>
      <td>19</td>
    </tr>
    <tr>
      <th>13</th>
      <td>180</td>
      <td>80</td>
      <td>1</td>
      <td>0.428766</td>
      <td>19</td>
    </tr>
    <tr>
      <th>14</th>
      <td>210</td>
      <td>80</td>
      <td>0</td>
      <td>0.428766</td>
      <td>19</td>
    </tr>
    <tr>
      <th>15</th>
      <td>210</td>
      <td>80</td>
      <td>1</td>
      <td>0.428766</td>
      <td>19</td>
    </tr>
    <tr>
      <th>16</th>
      <td>240</td>
      <td>80</td>
      <td>0</td>
      <td>0.428766</td>
      <td>19</td>
    </tr>
    <tr>
      <th>17</th>
      <td>240</td>
      <td>80</td>
      <td>1</td>
      <td>0.428766</td>
      <td>19</td>
    </tr>
    <tr>
      <th>18</th>
      <td>270</td>
      <td>80</td>
      <td>0</td>
      <td>0.428766</td>
      <td>19</td>
    </tr>
    <tr>
      <th>19</th>
      <td>270</td>
      <td>80</td>
      <td>1</td>
      <td>0.428766</td>
      <td>19</td>
    </tr>
    <tr>
      <th>20</th>
      <td>300</td>
      <td>80</td>
      <td>0</td>
      <td>0.428766</td>
      <td>19</td>
    </tr>
    <tr>
      <th>21</th>
      <td>300</td>
      <td>80</td>
      <td>1</td>
      <td>0.428766</td>
      <td>19</td>
    </tr>
    <tr>
      <th>22</th>
      <td>330</td>
      <td>80</td>
      <td>0</td>
      <td>0.428766</td>
      <td>19</td>
    </tr>
    <tr>
      <th>23</th>
      <td>330</td>
      <td>80</td>
      <td>1</td>
      <td>0.428766</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>




```python
# visualize single sample
# %#matplotlib notebook
animation = HexScatter(
    dataset[3][None, ::25, None], vmin=0, vmax=1
)  # intensity=1, radius=6
animation.animate_in_notebook()
```


    
![png](04_flyvision_moving_edge_responses_files/04_flyvision_moving_edge_responses_11_0.png)
    


## Moving edge response

Now that we have generated the stimulus, we can use it to drive a trained connectome-constrained network.


```python
from flyvision import results_dir
from flyvision.network import NetworkView

# model are already sorted by task error
# we take the best task-performing model from the pre-sorted ensemble
network_view = NetworkView(results_dir / "flow/0000/000")
```

    [2024-09-25 15:35:49] network:1005 Initialized network view at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000.



```python
stims_and_resps = network_view.movingedge_responses(dataset)
```

    [2024-09-25 15:35:49] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f3082758790>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/__cache__)]: 
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/000', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f30798ac430>, network=None), 
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'angles': [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
      'bar_loc_horizontal': 0.0,
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': [-10, 11],
      'post_pad_mode': 'continue',
      'shuffle_offsets': False,
      'speeds': [19],
      't_post': 1.0,
      't_pre': 1.0,
      'widths': [80]}, 
    4, 1.0, 0.0).
    
                            (argument hash 3b098d8fe94f37eebf7369247554b311)
    
                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/__cache__/flyvision/analysis/stimulus_responses/compute_responses.
    
    [2024-09-25 15:35:49] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/__cache__/flyvision/analysis/stimulus_responses/compute_responses/3b098d8fe94f37eebf7369247554b311/output.h5


    ___________________________________compute_responses cache loaded - 0.1s, 0.0min


We've now computed network moving edge responses for all cells in the network.

### Response traces
We can plot single-cell response traces with `stims_and_resps['responses'].custom.plot_traces()`. Here, we plot responses of T4c cells to edges with intensity 1 (ON edges).


```python
stims_and_resps["responses"].custom.where(
    cell_type="T4c", intensity=1, time=">-0.5,<1.0"
).custom.plot_traces(x="time", legend_labels=["angle"])
ax = plt.gca()
ax.set_title("T4c responses to moving edge")
```




    Text(0.5, 1.0, 'T4c responses to moving edge')




    
![png](04_flyvision_moving_edge_responses_files/04_flyvision_moving_edge_responses_17_1.png)
    


### Direction selectivity index (DSI)

The **Direction Selectivity Index (DSI)** quantifies a cell's preference for stimuli moving in a particular direction.

The DSI is derived from the following steps:
1. Obtain the neuron's peak responses to stimuli moving in different directions $\theta$ and at different speeds $S$.
2. Rectify these peak responses to ensure they are non-negative.
3. Compute the DSI using the equation:

$$
DSI_{t_i}(I) = \frac{1}{\lvert S \rvert} \sum_{S \in S} \left\lvert \frac{\sum_{\theta \in \Theta} r^{peak}_{t_{central}}(I, S, \theta) e^{i\theta}}{\max_{I \in I} \left\lvert \sum_{\theta \in \Theta} r^{peak}_{t_{central}}(I, S, \theta) \right\rvert} \right\rvert
$$

Where:
- $DSI_{t_i}(I)$ is the Direction Selectivity Index for cell type $t_i$ at stimulus intensity $I$.
- $\lvert S \rvert$ is the number of different speeds at which stimuli are moved.
- $r^{peak}_{t_{central}}(I, S, \theta)$ represents the rectified peak response of the central cell in hexagonal space of a cell type, for a given stimulus intensity $I$, speed $S$, and direction $\theta$.
- $\theta$ is varied across all tested directions $\Theta$.
- $e^{i\theta}$ introduces the directional component by weighting the response by the complex exponential of the angle of movement.
- The denominator normalizes the responses, ensuring that DSI values range from 0 to 1.

The DSI values range from 0 to 1. A DSI of 0 indicates no directional preference, while a DSI of 1 indicates a strong preference for a specific direction.

For the T4c cell plotted before, we can see that it preferentially responds to ON edges moving at an angle of 60 degrees, so we expect to see a large DSI.

 We compute the DSI with `flyvision.analysis.direction_selectivity_index`.


```python
from flyvision.analysis import direction_selectivity_index
```


```python
# get DSI for T4c cell
dsis = direction_selectivity_index(stims_and_resps)
print(f"T4c DSI: {dsis.custom.where(cell_type='T4c', intensity=1).item():.2f}")
```

    T4c DSI: 0.63


We compute the preferred direction of the cell with `flyvision.analysis.preferred_direction` (this is the direction that the tuning lobe points towards). We would expect the preferred direction to be around 60 degrees based on the response traces.


```python
from flyvision.analysis import preferred_direction
```


```python
pds = preferred_direction(stims_and_resps)
print(
    f"T4c preferred direction: {pds.custom.where(cell_type='T4c', intensity=1).item() / np.pi * 180:.2f} degrees"
)
```

    T4c preferred direction: 56.24 degrees


We can also inspect the direction selecity of a cell type visually, by plotting the angular tuning with `plot_angular_tuning`.

Here we see clearly how the cell is tuned to stimuli moving at a 60 degree angle.


```python
from flyvision.analysis import plot_angular_tuning
```


```python
plot_angular_tuning(stims_and_resps, cell_type="T4c", intensity=1)
```




    (<Figure size 300x300 with 1 Axes>, <PolarAxes: >)




    
![png](04_flyvision_moving_edge_responses_files/04_flyvision_moving_edge_responses_27_1.png)
    


### DSI  and tuning curve correlation

With the `dsi()` function we can also compute DSIs for every cell type at once. Since the selectivity of some cell types have been determined experimentally, we can then compare our model to experimental findings by computing the correlation between the model DSIs for known cell types with their expected motion selectivity.


```python
from flyvision.analysis import dsi_correlation_to_known
```


```python
dsi_corr = dsi_correlation_to_known(
    direction_selectivity_index(stims_and_resps)
).median()
```


```python
print(f"DSI correlation = {dsi_corr.item(): .2f}")
```

    DSI correlation =  0.65


Further, for certain cell types, their actual tuning curves have also been measured experimentally, so we can correlate our model cell's tuning to the true values. For T4c, the cell is known to tune to stimuli moving at 90 degrees, so the correlation should be relatively high.


```python
from flyvision.analysis import correlation_to_known_tuning_curves
```


```python
corrs = correlation_to_known_tuning_curves(stims_and_resps)
```


```python
print(
    f"T4c tuning curve correlation = {corrs.custom.where(cell_type='T4c', intensity=1).squeeze().item():.2f}"
)
```

    T4c tuning curve correlation = 0.54


In fact, tuning curves for all T4 and T5 cells have been measured, so we can compute the correlation for all 8 cell types.


```python
t4_corrs = corrs.custom.where(cell_type=["T4a", "T4b", "T4c", "T4d"], intensity=1)
t5_corrs = corrs.custom.where(cell_type=["T5a", "T5b", "T5c", "T5d"], intensity=0)
```


```python
print(
    f"T4 tuning curve correlations: {t4_corrs.cell_type.values}\n{t4_corrs.squeeze().values}"
)
```

    T4 tuning curve correlations: ['T4a' 'T4b' 'T4c' 'T4d']
    [0.93699976 0.71944943 0.53721794 0.85661069]



```python
print(
    f"T5 tuning curve correlations: {t5_corrs.cell_type.values}\n{t5_corrs.squeeze().values}"
)
```

    T5 tuning curve correlations: ['T5a' 'T5b' 'T5c' 'T5d']
    [0.84125435 0.90320946 0.94956466 0.90100504]


So, the model yields accurate predictions for all T4 and T5 cell types.

## Ensemble responses

Now we can compare motion selectivity properties across an ensemble of trained models. First we need to again simulate the network responses.


```python
from flyvision import EnsembleView

ensemble = EnsembleView(results_dir / "flow/0000")
# choose best 10
ensemble = ensemble[ensemble.argsort()[:10]]
```


    Loading ensemble:   0%|          | 0/50 [00:00<?, ?it/s]


    [2024-09-25 15:36:10] ensemble:138 Loaded 50 networks.



    Loading ensemble:   0%|          | 0/10 [00:00<?, ?it/s]


    [2024-09-25 15:36:15] ensemble:138 Loaded 10 networks.



```python
%%capture
stims_and_resps = ensemble.movingedge_responses(dataset=dataset)
```

    [2024-09-25 15:36:16] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f3082758790>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/__cache__)]: 
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/000', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f30798ac430>, network=None), 
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'angles': [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
      'bar_loc_horizontal': 0.0,
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': [-10, 11],
      'post_pad_mode': 'continue',
      'shuffle_offsets': False,
      'speeds': [19],
      't_post': 1.0,
      't_pre': 1.0,
      'widths': [80]}, 
    4, 1.0, 0.0).
    
                            (argument hash 3b098d8fe94f37eebf7369247554b311)
    
                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/__cache__/flyvision/analysis/stimulus_responses/compute_responses.
    
    [2024-09-25 15:36:16] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/__cache__/flyvision/analysis/stimulus_responses/compute_responses/3b098d8fe94f37eebf7369247554b311/output.h5
    [2024-09-25 15:36:16] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f3082758790>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/001/__cache__)]: 
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/001', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/001/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f30798ac430>, network=None), 
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'angles': [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
      'bar_loc_horizontal': 0.0,
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': [-10, 11],
      'post_pad_mode': 'continue',
      'shuffle_offsets': False,
      'speeds': [19],
      't_post': 1.0,
      't_pre': 1.0,
      'widths': [80]}, 
    4, 1.0, 0.0).
    
                            (argument hash 325e88a715551d0a149b5dae673c8f83)
    
                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/001/__cache__/flyvision/analysis/stimulus_responses/compute_responses.
    
    [2024-09-25 15:36:16] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/001/__cache__/flyvision/analysis/stimulus_responses/compute_responses/325e88a715551d0a149b5dae673c8f83/output.h5
    [2024-09-25 15:36:16] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f3082758790>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/002/__cache__)]: 
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/002', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/002/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f30798ac430>, network=None), 
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'angles': [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
      'bar_loc_horizontal': 0.0,
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': [-10, 11],
      'post_pad_mode': 'continue',
      'shuffle_offsets': False,
      'speeds': [19],
      't_post': 1.0,
      't_pre': 1.0,
      'widths': [80]}, 
    4, 1.0, 0.0).
    
                            (argument hash 8912ba93ac77a2d8a372a9aedbce0d4b)
    
                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/002/__cache__/flyvision/analysis/stimulus_responses/compute_responses.
    
    [2024-09-25 15:36:16] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/002/__cache__/flyvision/analysis/stimulus_responses/compute_responses/8912ba93ac77a2d8a372a9aedbce0d4b/output.h5
    [2024-09-25 15:36:16] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f3082758790>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/003/__cache__)]: 
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/003', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/003/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f30798ac430>, network=None), 
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'angles': [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
      'bar_loc_horizontal': 0.0,
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': [-10, 11],
      'post_pad_mode': 'continue',
      'shuffle_offsets': False,
      'speeds': [19],
      't_post': 1.0,
      't_pre': 1.0,
      'widths': [80]}, 
    4, 1.0, 0.0).
    
                            (argument hash 34abd51c78f6409009acc6ec30fb6e54)
    
                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/003/__cache__/flyvision/analysis/stimulus_responses/compute_responses.
    
    [2024-09-25 15:36:16] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/003/__cache__/flyvision/analysis/stimulus_responses/compute_responses/34abd51c78f6409009acc6ec30fb6e54/output.h5
    [2024-09-25 15:36:16] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f3082758790>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/004/__cache__)]: 
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/004', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/004/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f30798ac430>, network=None), 
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'angles': [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
      'bar_loc_horizontal': 0.0,
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': [-10, 11],
      'post_pad_mode': 'continue',
      'shuffle_offsets': False,
      'speeds': [19],
      't_post': 1.0,
      't_pre': 1.0,
      'widths': [80]}, 
    4, 1.0, 0.0).
    
                            (argument hash 757735e00ab721a6bcd14c8ac50f65c9)
    
                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/004/__cache__/flyvision/analysis/stimulus_responses/compute_responses.
    
    [2024-09-25 15:36:16] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/004/__cache__/flyvision/analysis/stimulus_responses/compute_responses/757735e00ab721a6bcd14c8ac50f65c9/output.h5
    [2024-09-25 15:36:16] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f3082758790>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/005/__cache__)]: 
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/005', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/005/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f30798ac430>, network=None), 
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'angles': [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
      'bar_loc_horizontal': 0.0,
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': [-10, 11],
      'post_pad_mode': 'continue',
      'shuffle_offsets': False,
      'speeds': [19],
      't_post': 1.0,
      't_pre': 1.0,
      'widths': [80]}, 
    4, 1.0, 0.0).
    
                            (argument hash 761b503cb11d83b87ccca4437485a05c)
    
                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/005/__cache__/flyvision/analysis/stimulus_responses/compute_responses.
    
    [2024-09-25 15:36:16] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/005/__cache__/flyvision/analysis/stimulus_responses/compute_responses/761b503cb11d83b87ccca4437485a05c/output.h5
    [2024-09-25 15:36:17] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f3082758790>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/006/__cache__)]: 
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/006', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/006/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f30798ac430>, network=None), 
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'angles': [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
      'bar_loc_horizontal': 0.0,
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': [-10, 11],
      'post_pad_mode': 'continue',
      'shuffle_offsets': False,
      'speeds': [19],
      't_post': 1.0,
      't_pre': 1.0,
      'widths': [80]}, 
    4, 1.0, 0.0).
    
                            (argument hash 2ef227637edf3421104836b79b8dd061)
    
                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/006/__cache__/flyvision/analysis/stimulus_responses/compute_responses.
    
    [2024-09-25 15:36:17] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/006/__cache__/flyvision/analysis/stimulus_responses/compute_responses/2ef227637edf3421104836b79b8dd061/output.h5
    [2024-09-25 15:36:17] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f3082758790>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/007/__cache__)]: 
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/007', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/007/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f30798ac430>, network=None), 
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'angles': [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
      'bar_loc_horizontal': 0.0,
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': [-10, 11],
      'post_pad_mode': 'continue',
      'shuffle_offsets': False,
      'speeds': [19],
      't_post': 1.0,
      't_pre': 1.0,
      'widths': [80]}, 
    4, 1.0, 0.0).
    
                            (argument hash e2e8db05a1a5046ed0ec8793ea1bf1ec)
    
                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/007/__cache__/flyvision/analysis/stimulus_responses/compute_responses.
    
    [2024-09-25 15:36:17] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/007/__cache__/flyvision/analysis/stimulus_responses/compute_responses/e2e8db05a1a5046ed0ec8793ea1bf1ec/output.h5
    [2024-09-25 15:36:17] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f3082758790>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/008/__cache__)]: 
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/008', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/008/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f30798ac430>, network=None), 
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'angles': [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
      'bar_loc_horizontal': 0.0,
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': [-10, 11],
      'post_pad_mode': 'continue',
      'shuffle_offsets': False,
      'speeds': [19],
      't_post': 1.0,
      't_pre': 1.0,
      'widths': [80]}, 
    4, 1.0, 0.0).
    
                            (argument hash 93e70ecb95c60d98cdc493ee50ba177b)
    
                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/008/__cache__/flyvision/analysis/stimulus_responses/compute_responses.
    
    [2024-09-25 15:36:17] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/008/__cache__/flyvision/analysis/stimulus_responses/compute_responses/93e70ecb95c60d98cdc493ee50ba177b/output.h5
    [2024-09-25 15:36:17] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7f3082758790>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/009/__cache__)]: 
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'groupby': ['source_type', 'target_type']}, 'syn_count': {'type': 'SynapseCount', 'initial_dist': 'Lognormal', 'mode': 'mean', 'requires_grad': False, 'std': 1.0, 'groupby': ['source_type', 'target_type', 'du', 'dv']}, 'syn_strength': {'type': 'SynapseCountScaling', 'initial_dist': 'Value', 'requires_grad': True, 'scale_elec': 0.01, 'scale_chem': 0.01, 'clamp': 'non_negative', 'groupby': ['source_type', 'target_type', 'edge_type']}}}, name='flow/0000/009', checkpoint=PosixPath('/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/009/chkpts/chkpt_00000'), recover_fn=<function recover_network at 0x7f30798ac430>, network=None), 
    <class 'flyvision.datasets.moving_bar.MovingEdge'>, { 'angles': [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
      'bar_loc_horizontal': 0.0,
      'dt': 0.005,
      'height': 80,
      'intensities': [0, 1],
      'offsets': [-10, 11],
      'post_pad_mode': 'continue',
      'shuffle_offsets': False,
      'speeds': [19],
      't_post': 1.0,
      't_pre': 1.0,
      'widths': [80]}, 
    4, 1.0, 0.0).
    
                            (argument hash bf2f8033eea3ff6c92e1e5f355f18096)
    
                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/009/__cache__/flyvision/analysis/stimulus_responses/compute_responses.
    
    [2024-09-25 15:36:17] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/009/__cache__/flyvision/analysis/stimulus_responses/compute_responses/bf2f8033eea3ff6c92e1e5f355f18096/output.h5


### Response traces

We can once again plot response traces for a single cell type. 

We subtract the initial value of each trace and divide by the max as the network neuron activities are in arbitrary units. 

We plot only T4c responses to ON edges moving at a 90-degree angle.


```python
responses = (
    stims_and_resps["responses"]
    - stims_and_resps["responses"].custom.where(time=0).values
)
```


```python
responses = responses / np.abs(responses).max(("sample", "frame"))
```


```python
responses.custom.where(
    cell_type="T4c",
    intensity=1,
    time=">-0.5,<1.0",
    angle=90,
).custom.plot_traces(
    x="time", plot_kwargs=dict(color="tab:blue"), legend_labels=["network_id"]
)
```




    <Axes: xlabel='time', ylabel='responses'>




    
![png](04_flyvision_moving_edge_responses_files/04_flyvision_moving_edge_responses_47_1.png)
    


Though for most networks T4c responses are correctly predicted to the stimuli, there are some networks in the ensemble with different responses.

### Direction selectivity index (DSI)

We can also compute direction selectivity indices for each network in the ensemble.


```python
dsis = direction_selectivity_index(stims_and_resps)
```


```python
dsis.custom.where(cell_type="T4c", intensity=1).plot.hist()
ax = plt.gca()
ax.set_title("T4c DSI distribution")
ax.set_ylabel("Number of networks")
```




    Text(0, 0.5, 'Number of networks')




    
![png](04_flyvision_moving_edge_responses_files/04_flyvision_moving_edge_responses_51_1.png)
    


Most networks in this group recover some direction selectivity for T4c. We can also plot the distribution of DSIs per cell type for both ON and OFF-edge stimuli across the ensemble.


```python
from flyvision.analysis.moving_bar_responses import plot_dsis

fig, ax = plot_dsis(
    dsis,
    dsis.cell_type,
    bold_output_type_labels=True,
    output_cell_types=ensemble[ensemble.names[0]]
    .connectome.output_cell_types[:]
    .astype(str),
    figsize=[10, 1.2],
    color_known_types=True,
    fontsize=6,
    scatter_best_index=0,
    scatter_best_color=plt.get_cmap("Blues")(1.0),
)
```


    
![png](04_flyvision_moving_edge_responses_files/04_flyvision_moving_edge_responses_53_0.png)
    


### DSI correlation

Lastly, we look at the correlations to ground-truth DSIs and tuning curves across the ensemble. This provides us with a high-level understanding of the accuracy of known motion tuning predictions. 


```python
dsi_corr = dsi_correlation_to_known(
    direction_selectivity_index(stims_and_resps)
).median("intensity")
```


```python
tuning_corrs = correlation_to_known_tuning_curves(stims_and_resps)
```


```python
t4_corrs = tuning_corrs.custom.where(cell_type=["T4a", "T4b", "T4c", "T4d"], intensity=1).median("neuron").squeeze()
t5_corrs = tuning_corrs.custom.where(cell_type=["T5a", "T5b", "T5c", "T5d"], intensity=0).median("neuron").squeeze()
```


```python
dsi_corr.shape, t4_corrs.shape, t5_corrs.shape
```




    ((10,), (10,), (10,))




```python
from flyvision.plots.plots import violin_groups

fig, ax, *_ = violin_groups(
    np.stack([dsi_corr.values, t4_corrs.values, t5_corrs.values], axis=0)[:, None, :],
    ["DSI", "T4 tuning", "T5 tuning"],
    ylabel="correlation",
    figsize=(1.8, 1.5),
    ylim=(-1, 1),
    colors=[
        plt.get_cmap("Dark2")(0.125),
        plt.get_cmap("Dark2")(0),
        plt.get_cmap("Dark2")(0.25),
    ],
    color_by="experiments",
    scatter_edge_color="gray",
    scatter_radius=5,
    violin_alpha=0.8,
)
```


    
![png](04_flyvision_moving_edge_responses_files/04_flyvision_moving_edge_responses_59_0.png)
    


<!-- ... Models in general have very good match to known single-neuron tuning properties, with median correlation around $0.8$. -->
