Main notebook containing relevant analysis steps, run for each ensemble.

The script `notebook_per_ensemble.py' automatically copies this notebook to an ensemble directory and executes it for newly trained ensembles using papermill.

**Warning:** You can loose your work! Don't edit automatically created copies of this notebook within an ensemble directory. Those will be overwritten at a rerun. Create a copy instead.

**Warning:** This notebook is not intended for standalone use. It is automatically copied to an ensemble directory and executed for newly trained ensembles using papermill. Adapt mindfully.



```
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt

from flyvision import EnsembleView
from flyvision.analysis.moving_bar_responses import plot_angular_tuning
from flyvision.analysis.visualization.plt_utils import add_cluster_marker, get_marker

logging.disable()


mpl.rcParams["figure.dpi"] = 300

%load_ext autoreload
%autoreload 2
```


```
ensemble_name = "flow/0001"  # type: str
```


```
validation_subdir = "validation"
loss_file_name = "loss"
```


```
ensemble = EnsembleView(
    ensemble_name,
    best_checkpoint_fn_kwargs={
        "validation_subdir": validation_subdir,
        "loss_file_name": loss_file_name,
    },
)
```


    Loading ensemble:   0%|          | 0/50 [00:00<?, ?it/s]



```
print(f"Description of experiment: {getattr(ensemble[0].dir.config, 'description', '')}")
```

    Description of experiment: test


# Task performance

## Training and validation losses


```
fig, ax = ensemble.training_loss()
```



![png](__main___files/__main___8_0.png)




```
fig, ax = ensemble.validation_loss()
```



![png](__main___files/__main___9_0.png)




```
fig, ax = ensemble.task_error_histogram()
```



![png](__main___files/__main___10_0.png)



## Learned parameter marginals


```
fig, axes = ensemble.node_parameters("bias")
```



![png](__main___files/__main___12_0.png)




```
fig, axes = ensemble.node_parameters("time_const")
```



![png](__main___files/__main___13_0.png)




```
fig, axes = ensemble.edge_parameters("syn_strength")
```



![png](__main___files/__main___14_0.png)



## Dead or alive


```
fig, ax, cbar, matrix = ensemble.dead_or_alive()
```


    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]




![png](__main___files/__main___16_50.png)



## Contrast selectivity and flash response indices (FRI)

#### 20% best task-performing models


```
with ensemble.ratio(best=0.2):
    ensemble.flash_response_index()
```



![png](__main___files/__main___19_0.png)



#### 100% models


```
fig, ax = ensemble.flash_response_index()
```


    Batch:   0%|          | 0/1 [00:00<?, ?it/s]




![png](__main___files/__main___21_1.png)



## Motion selectivity and direction selectivity index (DSI)

#### 20% best task-performing models


```
with ensemble.ratio(best=0.2):
    ensemble.direction_selectivity_index()
```



![png](__main___files/__main___24_0.png)



#### 100% models


```
ensemble.direction_selectivity_index()
```


    Batch:   0%|          | 0/36 [00:00<?, ?it/s]



    Batch:   0%|          | 0/36 [00:00<?, ?it/s]





    (<Figure size 3000x360 with 2 Axes>, array([<Axes: >, <Axes: >], dtype=object))





![png](__main___files/__main___26_3.png)



## Clustering of models based on responses to naturalistic stimuli

#### T4c


```
task_error = ensemble.task_error()
embeddingplot = ensemble.clustering("T4c").plot(
    task_error=task_error.values, colors=task_error.colors
)
```

    /home/lappalainenj@hhmi.org/miniconda3/envs/flyvision/lib/python3.9/site-packages/umap/umap_.py:1356: RuntimeWarning: divide by zero encountered in power
      return 1.0 / (1.0 + a * x ** (2 * b))




![png](__main___files/__main___29_1.png)




```
r = ensemble.moving_edge_responses()
```


```
cluster_indices = ensemble.cluster_indices("T4c")
```


```
colors = ensemble.task_error().colors
```


```
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



![png](__main___files/__main___33_0.png)




```

```
