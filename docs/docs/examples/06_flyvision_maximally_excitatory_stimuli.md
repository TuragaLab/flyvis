```python
%load_ext autoreload
%autoreload 2
```

# Maximally excitatory stimuli from trained models

This notebook illustrates how to compute the stimuli that maximally excite a specific neuron.

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

# Optimal naturalistic stimuli

We first find the optimal naturalistic stimuli. To do that, we simulate the responses of 
the network (including the neuron of interest) to all stimuli from a fixed dataset of stimuli. 
The optimal, or here maximally exctitatory naturalistic stimulus to be precise, is the stimulus 
for which the response of the chosen neuron is maximal. Finding this is simple and does not require numerical optimization with gradients.
We find the stimulus per cell type based on its cell in the central column. At least in our coarse model, 
the offset version of this stimulus would also maximally excite the equivalently offset neighboring cells of the same type.



```python

from flyvision import NetworkView, results_dir
from flyvision.datasets.sintel import AugmentedSintel
from flyvision.analysis.optimal_stimuli import FindOptimalStimuli, GenerateOptimalStimuli,  plot_stim_response
```


```python
# let's load the dataset and the pretrained network
dataset = AugmentedSintel(tasks=["lum"], temporal_split=True)
network_view = NetworkView(results_dir / "flow/0000/000")
```

    [2024-09-25 18:08:10] network:1005 Initialized network view at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000.



```python
findoptstim = FindOptimalStimuli(network_view, dataset)
```

    [2024-09-25 18:08:20] network:252 Initialized network with NumberOfParams(free=734, fixed=2959) parameters.
    [2024-09-25 18:08:20] chkpt_utils:72 Recovered network state.


For the T4c neuron, we would expect that the maximally excitatory stimulus is an ON-edge
moving upward.


```python
optstim = network_view.optimal_stimulus_responses("T4c")
```

    [2024-09-25 18:08:21] logger:83 [MemorizedFunc(func=<function compute_optimal_stimulus_responses at 0x7fba9d1b1dc0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/__cache__)]: 
                            Querying compute_optimal_stimulus_responses with signature
                            compute_optimal_stimulus_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'group..., 
    'T4c', { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True}, 
    <class 'flyvision.datasets.sintel.AugmentedSintel'>).
    
                            (argument hash 2281c4887178168b7aaa6dc2c0005a0b)
    
                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/__cache__/flyvision/analysis/stimulus_responses/compute_optimal_stimulus_responses.
    
    [2024-09-25 18:08:21] logger:80 [MemorizedFunc(func=<function compute_optimal_stimulus_responses at 0x7fba9d1b1dc0>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/__cache__)]: Computing func compute_optimal_stimulus_responses, argument hash 2281c4887178168b7aaa6dc2c0005a0b in location /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/__cache__/flyvision/analysis/stimulus_responses/compute_optimal_stimulus_responses


    ________________________________________________________________________________
    [Memory] Calling flyvision.analysis.stimulus_responses.compute_optimal_stimulus_responses...
    compute_optimal_stimulus_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'group..., 
    'T4c', { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True}, 
    <class 'flyvision.datasets.sintel.AugmentedSintel'>)


    [2024-09-25 18:08:21] network:1005 Initialized network view at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000.
    [2024-09-25 18:08:34] network:252 Initialized network with NumberOfParams(free=734, fixed=2959) parameters.
    [2024-09-25 18:08:34] chkpt_utils:72 Recovered network state.
    [2024-09-25 18:08:34] logger:83 [MemorizedFunc(func=<function compute_responses at 0x7fba9d193670>, location=/groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/__cache__)]: 
                            Querying compute_responses with signature
                            compute_responses(CheckpointedNetwork(network_class=<class 'flyvision.network.Network'>, config={'connectome': {'type': 'ConnectomeDir', 'file': 'fib25-fib19_v2.2.json', 'extent': 15, 'n_syn_fill': 1}, 'dynamics': {'type': 'PPNeuronIGRSynapses', 'activation': {'type': 'relu'}}, 'node_config': {'bias': {'type': 'RestingPotential', 'groupby': ['type'], 'initial_dist': 'Normal', 'mode': 'sample', 'requires_grad': True, 'mean': 0.5, 'std': 0.05, 'penalize': {'activity': True}, 'seed': 0}, 'time_const': {'type': 'TimeConstant', 'groupby': ['type'], 'initial_dist': 'Value', 'value': 0.05, 'requires_grad': True}}, 'edge_config': {'sign': {'type': 'SynapseSign', 'initial_dist': 'Value', 'requires_grad': False, 'group..., 
    <class 'flyvision.datasets.sintel.AugmentedSintel'>, { 'boxfilter': {'extent': 15, 'kernel_size': 13},
      'dt': 0.01,
      'interpolate': False,
      'tasks': ['lum'],
      'temporal_split': True}, 
    4, 0.0, 2.0).
    
                            (argument hash bef8ab0b1f1206171b1e38bf67305c6f)
    
                            The store location is /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/__cache__/flyvision/analysis/stimulus_responses/compute_responses.
    
    [2024-09-25 18:08:34] xarray_joblib_backend:582 Loading Dataset from NetCDF at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/__cache__/flyvision/analysis/stimulus_responses/compute_responses/bef8ab0b1f1206171b1e38bf67305c6f/output.h5


    ___________________________________compute_responses cache loaded - 0.0s, 0.0min
    Persisting in /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000/__cache__/flyvision/analysis/stimulus_responses/compute_optimal_stimulus_responses/2281c4887178168b7aaa6dc2c0005a0b
    ______________________________compute_optimal_stimulus_responses - 43.0s, 0.7min



```python
stim_resp_plot = plot_stim_response(optstim.stimulus.stimulus, optstim.stimulus.response, 1/100, *network_view.connectome_view.get_uv("T4c"), figsize=[5, 1.6], ylabel=None, label_peak_response=False)
```


    
![png](06_flyvision_maximally_excitatory_stimuli_files/06_flyvision_maximally_excitatory_stimuli_14_0.png)
    


We see that the the stimulus indeed contains an ON-edge component moving upward and this is the portion of the stimulus that T4c cells respond most to.
What's unclear is whether the other parts of the stimulus have an influence on the response.

# Regularized optimal stimuli

We can regularize the optimal stimuli with the objective to keep the response of the
cell intact while bringing the stimulus pixels to a neutral grey value.


```python
stim_resp_plot = plot_stim_response(optstim.regularized_stimulus, optstim.response, 1/100, *network_view.connectome_view.get_uv("T4c"), figsize=[5, 1.6], ylabel=None, label_peak_response=False)
```


    
![png](06_flyvision_maximally_excitatory_stimuli_files/06_flyvision_maximally_excitatory_stimuli_17_0.png)
    


This looks remarkably different! Now only a central black portion follow by the ON-edge
moving upward remains in the stimulus. Let's make sure that the central cell response is
really the same as before! This is the entire time trace.


```python
fig = plt.figure(figsize=[2, 1])
time = np.arange(len(optstim.central_target_response)) / 100
plt.plot(time, optstim.central_target_response)
plt.plot(time, optstim.central_predicted_response)
plt.xlabel("time (s)")
plt.ylabel("response")
```




    Text(0, 0.5, 'response')




    
![png](06_flyvision_maximally_excitatory_stimuli_files/06_flyvision_maximally_excitatory_stimuli_19_1.png)
    


This looks quite similar! One can play with the regularization optimization and its parameters of the function `regularized_optimal_stimuli`.

# Generate artificial optimal stimuli from scratch
If one is able to optimize the naturalistic stimulus with the gradient, why don't we
use the gradient to generate an optimal stimulus from scratch (or rather random noise).
We do that in the following. Again for T4c, we would expect that it would have some sort
of ON-edge moving upwards.


```python
genoptstim = GenerateOptimalStimuli(network_view)
```

    [2024-09-25 18:09:34] chkpt_utils:72 Recovered network state.



```python
artoptstim = genoptstim.artificial_optimal_stimuli("T4c", t_stim=0.8)
```


```python
stim_resp_plot = plot_stim_response(artoptstim.stimulus, artoptstim.response, 1/100, *network_view.connectome_view.get_uv("T4c"), figsize=[5, 1.6], ylabel=None, label_peak_response=False)
```


    
![png](06_flyvision_maximally_excitatory_stimuli_files/06_flyvision_maximally_excitatory_stimuli_24_0.png)
    


Wow! This stimulus is contains very similar components to the one before and is much more
saturated! It also contains new ON-components already from the beginning!

Last, let's compare which stimulus excited the neuron the most.


```python
fig = plt.figure(figsize=[2, 1])
time = np.arange(len(optstim.central_target_response)) / 100
plt.plot(time, optstim.central_target_response, label='naturalistic')
plt.plot(time, optstim.central_predicted_response, label='regularized naturalistic')
plt.plot(time, artoptstim.response[:, :, artoptstim.response.shape[-1]//2].flatten(), label='artificial')
plt.xlabel("time (s)")
plt.ylabel("response")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fba75d0d7f0>




    
![png](06_flyvision_maximally_excitatory_stimuli_files/06_flyvision_maximally_excitatory_stimuli_26_1.png)
    



```python

```
