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


    [2024-10-14 23:36:32] ensemble:166 Loaded 50 networks.


## a


```python
task_error = ensemble.task_error()
```


```python
embedding_and_clustering = ensemble.clustering("T4c")
```

    [2024-10-14 23:36:38] clustering:835 Loaded T4c embedding and clustering from /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/umap_and_clustering



```python
embeddingplot = embedding_and_clustering.plot(
    task_error=task_error.values, colors=task_error.colors
)
```



![png](figure_03_naturalistic_stimuli_responses_files/figure_03_naturalistic_stimuli_responses_6_0.png)



## b


```python
import numpy as np
import matplotlib.pyplot as plt

from flyvision.analysis.visualization import plt_utils
from flyvision.analysis.moving_bar_responses import plot_angular_tuning
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
    plt_utils.add_cluster_marker(
        fig, axes[cluster_id], marker=plt_utils.get_marker(cluster_id)
    )
```



![png](figure_03_naturalistic_stimuli_responses_files/figure_03_naturalistic_stimuli_responses_13_0.png)



## e


```python
for cluster_id, indices in cluster_indices.items():
    with ensemble.select_items(indices):
        fig, ax = ensemble.flash_response_index(
            cell_types=["Mi1", "Tm3", "Mi4", "Mi9", "CT1(M10)"], figsize=[1, 1]
        )
```



![png](figure_03_naturalistic_stimuli_responses_files/figure_03_naturalistic_stimuli_responses_15_0.png)





![png](figure_03_naturalistic_stimuli_responses_files/figure_03_naturalistic_stimuli_responses_15_1.png)





![png](figure_03_naturalistic_stimuli_responses_files/figure_03_naturalistic_stimuli_responses_15_2.png)
