Main notebook containing relevant analysis steps, run for each model.

The script `notebook_per_model.py' automatically copies this notebook to an ensemble directory and executes it for newly trained ensembles using papermill.

**Warning:** You can loose your work! Don't edit automatically created copies of this notebook within a model directory. Those will be overwritten at a rerun. Create a copy instead.

**Warning:** This notebook is not intended for standalone use. It is automatically copied to an ensemble directory and executed for newly trained models using papermill. Adapt mindfully.



```python
import flyvision
```


```python
ensemble_and_network_id = "0000/000"  # type: str
task_name = "flow"  # type: str
```


```python
network_view = flyvision.NetworkView(f"{task_name}/{ensemble_and_network_id}")
```

    [2024-10-15 02:34:30] network_view:125 Initialized network view at /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/000
