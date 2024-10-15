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
    # @markdown **Install Flyvis**
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
from flyvision.analysis.animations import HexScatter
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
animation = HexScatter(sequence[None], vmin=0, vmax=1)
animation.animate_in_notebook(frames=np.arange(5))
```



![png](05_flyvision_umap_and_clustering_models_files/05_flyvision_umap_and_clustering_models_13_0.png)



# Ensemble responses to naturalistic sequences
We compute the responses of all models in the stored ensemble to the augmented Sintel dataset.


```python
from flyvision import results_dir
```


```python
# We load the ensemble trained on the optic flow task
ensemble = flyvision.EnsembleView(results_dir / "flow/0000")
```


    Loading ensemble:   0%|          | 0/50 [00:00<?, ?it/s]


    [2024-10-04 22:47:37] ensemble:142 Loaded 50 networks.


We use `ensemble.naturalistic_stimuli_responses` to return responses of all networks within the ensemble.


```python
stims_and_resps = ensemble.naturalistic_stimuli_responses()
```


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
central_responses = CentralActivity(responses.values, connectome=ensemble.connectome)
```


```python
embedding = EnsembleEmbedding(central_responses)
t4c_embedding = embedding("T4c", embedding_kwargs=embedding_kwargs)
```

    [2024-10-04 22:47:52] clustering:349 reshaped X from (50, 2268, 80) to (50, 181440)
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
embeddingplot = gm_clustering.plot(task_error=task_error.values, colors=task_error.colors)
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
        plot_kwargs=dict(color=colors[cluster_id], add_legend=False, ax=axes[cluster_id]),
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
from flyvision.analysis.visualization import plt_utils
from flyvision.utils.color_utils import color_to_cmap
```


```python
stims_and_resps_moving_edge = ensemble.moving_edge_responses()
```


```python
# invariant to different magnitudes of responses, only to assess direction tuning
stims_and_resps_moving_edge["responses"] /= np.abs(
    stims_and_resps_moving_edge["responses"]
).max(dim=("sample", "frame"))

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
        plot_kwargs=dict(color=colors[cluster_id], add_legend=False, ax=axes[cluster_id]),
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
colors = [
    ensemble.task_error(cmap=color_to_cmap(tabcolors[cluster_id]).reversed()).colors[
        indices
    ]
    for cluster_id, indices in cluster_to_indices.items()
]
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
    plt_utils.add_cluster_marker(
        fig, axes[cluster_id], marker=plt_utils.get_marker(cluster_id)
    )
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

    [2024-10-04 22:49:49] clustering:643 Loaded T4c embedding and clustering from /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/umap_and_clustering.



```python
task_error = ensemble.task_error()
```


```python
embeddingplot = clustering.plot(task_error=task_error.values, colors=task_error.colors)
```



![png](05_flyvision_umap_and_clustering_models_files/05_flyvision_umap_and_clustering_models_56_0.png)



With this embedding and clustering one can proceed in the same way as above to plot the tunings.
