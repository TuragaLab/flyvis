# Flash responses

This notebook introduces flash responses and the flash response index (FRI). 

The FRI measures whether a cell depolarizes to bright or to dark increments in a visual input.

##### You can skip the next cells if you are not on google colab but run this locally

**Select GPU runtime**

Only for usage on google colab: to run the notebook on a GPU select Menu -> Runtime -> Change runtime type -> GPU.


```
%load_ext autoreload
%autoreload 2
```


```
#@markdown **Check access to GPU**

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
      warnings.warn("You have not selected Runtime Type: 'GPU' or Google could not assign you one. Please revisit the settings as described above or proceed on CPU (slow).")
```

**Install Flyvis**

The notebook requires installing our package `flyvis`. You may need to restart your session after running the code block below with Menu -> Runtime -> Restart session. Then, imports from `flyvis` should succeed without issue.


```
if IN_COLAB:
    #@markdown **Install Flyvis**
    %%capture
    !git clone https://github.com/flyvis/flyvis-dev.git
    %cd /content/flyvis-dev
    !pip install -e .
```


```
# basic imports
import matplotlib.pyplot as plt
import numpy as np
import logging
logging.disable(100)

plt.rcParams['figure.dpi'] = 200
```

## Flash stimuli

To elicit flash responses, experimenters show a flashing dot to the subject in the center of their field of view. We generate and render these stimuli with the `Flashes` dataset.


```
# import dataset and visualization helper
from flyvision.animations.hexscatter import HexScatter
from flyvision.datasets.flashes import Flashes
```


```
# initialize dataset
dataset = Flashes(
    dynamic_range=[0, 1],  # min and max pixel intensity values, must be in range [0, 1]
    t_stim=1.0,  # duration of flash
    t_pre=1.0,  # duration of period between flashes
    dt=1 / 200,  # temporal resolution of rendered video
    radius=[-1, 6],  # radius of flashing dot. -1 fills entire field of view
    alternations=(0, 1, 0),  # flashing pattern, off - on - off
)
```


```
# view stimulus parameters
dataset.arg_df
# the dataset has four samples, one corresponding to each row
```


```
# visualize single sample
animation = HexScatter(
    dataset[3][None, ::50, None], vmin=0, vmax=1
)  # intensity=1, radius=6
animation.animate_in_notebook()
```

## Network flash response

Now that we have generated the stimulus, we can use it to drive a trained connectome-constrained network.


```
from flyvision import results_dir
from flyvision.network import NetworkView

# model are already sorted by task error
# we take the best task-performing model from the pre-sorted ensemble
network_view = NetworkView(results_dir / "flow/0000/000")
```


```
stims_and_resps = network_view.flash_responses(dataset=dataset)
```


```

stims_and_resps['responses'].custom.where(cell_type="L1", radius=6).custom.plot_traces(x='time')
fig = plt.gcf()
fig.axes[-1].set_title("L1 flash responses")
```

### Flash response index (FRI)

The flash response index (FRI) is a measure of the strength of contrast tuning of a particular cell. It is computed as the difference between the cell's peak voltage in response to on-flashes (intensity = 1) and off-flashes (intensity = 0), divided by the sum of those peak values.

That is, given a single neuron's response to on-flashes `r_on` and off-flashes `r_off` (both of `shape=(T,)`), we can compute the flash response index with

```
r_on_max = max(r_on)
r_off_max = max(r_off)
fri = (r_on_max - r_off_max) / (r_on_max + r_off_max + 1e-16)
```

with the additional `1e-16` simply for numerical stability. Before this calculation, the response traces are shifted to be non-negative.

The flash response index can take on values between $-1$, when the off response is much stronger (or more positive) than the on response, to $1$, when the on response is much stronger (or more positive) than the off response.

For the L1 cell plotted before, we can see that it displays a positive response to off flashes and a negative response to on flashes, so we expect a negative flash response index.


```
from flyvision.analysis.flash_responses import flash_response_index
```


```
fris = flash_response_index(stims_and_resps, radius=6)
```


```
fris.custom.where(cell_type="L1")
```

### FRI correlation

Since the tuning of some cell types have been determined experimentally, we can then compare our model to experimental findings by computing the correlation between the model FRIs for known cell types with their expected tuning.


```
from flyvision.analysis.flash_responses import fri_correlation_to_known
from flyvision.utils.groundtruth_utils import polarity

```


```
fri_corr = fri_correlation_to_known(fris)
```


```
# manually extract model and true FRIs for plotting
known_cell_types = [k for k, v in polarity.items() if v != 0]
model_fris = [fris.custom.where(cell_type=k).item() for k in known_cell_types]
true_fris = [polarity[k] for k in known_cell_types]
# plot
plt.figure(figsize=[2, 1])
plt.scatter(model_fris, true_fris, color="k", s=10)
plt.xlabel("predicted FRI")
plt.ylabel("putative FRI (true tuning)")
plt.axvline(0, linestyle="--", color="black")
plt.axhline(0, linestyle="--", color="black")

plt.axhspan(0, 2, 0, 0.5, color="red", zorder=-10)
plt.axhspan(0, 2, 0.5, 1.0, color="green", zorder=-10)
plt.axhspan(-2, 0, 0, 0.5, color="green", zorder=-10)
plt.axhspan(-2, 0, 0.5, 1.0, color="red", zorder=-10)

plt.xlim(-1.05, 1.05)
plt.ylim(-2, 2)
plt.title(f"Correlation = {fri_corr[0].item():.2g}")
plt.yticks([-1, 1], ["OFF", "ON"])
plt.show()
```

As we can see, for all except two cell types, the model correctly predicts the cell's tuning (positive or negative).

## Ensemble responses

Now we can compare tuning properties across an ensemble of trained models. First we need to again simulate the network responses.


```
from flyvision import EnsembleView

ensemble = EnsembleView(results_dir / "flow/0000")
```


```
stims_and_resps = ensemble.flash_responses(dataset=dataset)
```

### Response traces

We can once again plot response traces for a single cell type. We subtract the initial value of each trace to center the data before plotting, as the network neuron activities are in arbitrary units.


```
centered = stims_and_resps['responses'] - stims_and_resps['responses'].custom.where(time=0.0).values
```


```
centered.sel(network_id=ensemble.argsort()[:10]).custom.where(cell_type="L1", radius=6, intensity=1).custom.plot_traces(x='time', plot_kwargs=dict(color='orange', linewidth=0.5))
ax = plt.gca()
centered.sel(network_id=ensemble.argsort()[:10]).custom.where(cell_type="L1", radius=6, intensity=0).custom.plot_traces(x='time', plot_kwargs=dict(ax=ax, color='blue', linewidth=0.5))
ax.set_title("L1 flash responses")
```

Though the scaling varies, all networks predict depolarization to OFF-flashes for L1.

### Flash response index (FRI)

We can also compute flash response indices for each network in the ensemble.


```
# get FRI for L1 cell
fri_l1 = flash_response_index(stims_and_resps, radius=6).sel(network_id=ensemble.argsort()[:10]).custom.where(cell_type="L1")
print(fri_l1.squeeze().values)
```

All models recover similar flash response indices for this cell type. We can also plot the distribution of FRIs per cell type across the ensemble.


```
with ensemble.select_items(ensemble.argsort()[:10]):
    ensemble.flash_response_index()
```

### FRI correlation

Lastly, we look at the correlations to ground-truth tuning across the ensemble.


```
from flyvision.analysis.flash_responses import flash_response_index
```


```
fris = flash_response_index(stims_and_resps, radius=6)
```


```
from flyvision.plots.plots import violin_groups

# compute correlation
fri_corr = fri_correlation_to_known(fris)

fig, ax, *_ = violin_groups(
    np.array(fri_corr)[None, None, :].squeeze(-1),
    ylabel="FRI correlation",
    figsize=(2, 2),
    xlim=(0, 1),
    xticklabels=[],
    colors=[plt.get_cmap("Pastel1")(0.0)],
    scatter_edge_color="gray",
    scatter_radius=10,
)
```

Models in general have very good match to known single-neuron tuning properties, with median correlation around 0.8.
