Main notebook containing relevant analysis steps, run for each model.

The script `notebook_per_model.py' automatically copies this notebook to an ensemble directory and executes it for newly trained ensembles using papermill.

**Warning:** You can loose your work! Don't edit automatically created copies of this notebook within a model directory. Those will be overwritten at a rerun. Create a copy instead.

**Warning:** This notebook is not intended for standalone use. It is automatically copied to an ensemble directory and executed for newly trained models using papermill. Adapt mindfully.



```python
import flyvis
```


```python
ensemble_and_network_id = "0000/000"  # type: str
task_name = "flow"  # type: str
```


```python
network_view = flyvis.NetworkView(f"{task_name}/{ensemble_and_network_id}")
```

    [2024-12-08 19:51:02] network_view:122 Initialized network view at ../flyvis/data/results/flow/0000/000
