# Cluster analysis based on naturalistic stimuli responses

This notebook illustrates how to cluster the models of an ensemble after nonlinear dimensionality reduction on their predicted responses to naturalistic stimuli. This can be done for any cell type. Here we provide a detailed example focusing on clustering based on T4c responses.


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



![png](05_flyvision_umap_and_clustering_models_files/05_flyvision_umap_and_clustering_models_9_0.png)



# Ensemble responses to naturalistic sequences
We compute the responses of all models in the stored ensemble to the augmented Sintel dataset.


```python
# We load the ensemble trained on the optic flow task
ensemble = flyvision.EnsembleView("flow/0000")
```


    Loading ensemble:   0%|          | 0/50 [00:00<?, ?it/s]


    [2024-10-14 21:02:31] ensemble:166 Loaded 50 networks.


We use `ensemble.naturalistic_stimuli_responses` to return responses of all networks within the ensemble.


```python
# alternatively, specify indices of sequences to load
# stims_and_resps = ensemble.naturalistic_stimuli_responses(indices=np.arange(5))
# or load all sequences
stims_and_resps = ensemble.naturalistic_stimuli_responses()
```

    [2024-10-14 21:02:41] network:222 Initialized network with NumberOfParams(free=734, fixed=2959) parameters.
    [2024-10-14 21:02:41] chkpt_utils:35 Recovered network state.
    [2024-10-14 21:02:46] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 21:06:00] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/007/__cache__/flyvision/analysis/stimulus_responses/compute_responses/62f12e60e448f187eb7c3c597ad40084/output.h5
    [2024-10-14 21:06:07] chkpt_utils:35 Recovered network state.
    [2024-10-14 21:06:11] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 21:09:23] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/008/__cache__/flyvision/analysis/stimulus_responses/compute_responses/57f443b19940e708cd13ea4c8c285770/output.h5
    [2024-10-14 21:09:29] chkpt_utils:35 Recovered network state.
    [2024-10-14 21:09:34] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 21:12:50] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/009/__cache__/flyvision/analysis/stimulus_responses/compute_responses/8c63fb5782e9c87881de4ef9f68a9794/output.h5
    [2024-10-14 21:12:56] chkpt_utils:35 Recovered network state.
    [2024-10-14 21:13:00] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 21:16:16] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/010/__cache__/flyvision/analysis/stimulus_responses/compute_responses/b70bd43513f96a8ca2149b4b707cf55b/output.h5
    [2024-10-14 21:16:22] chkpt_utils:35 Recovered network state.
    [2024-10-14 21:16:27] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 21:19:43] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/011/__cache__/flyvision/analysis/stimulus_responses/compute_responses/0a2bc829a7f7a299dd597db764831509/output.h5
    [2024-10-14 21:19:49] chkpt_utils:35 Recovered network state.
    [2024-10-14 21:19:53] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 21:23:09] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/012/__cache__/flyvision/analysis/stimulus_responses/compute_responses/89d8b0766a56e2e8c9ab4019b752b186/output.h5
    [2024-10-14 21:23:15] chkpt_utils:35 Recovered network state.
    [2024-10-14 21:23:19] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 21:26:32] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/013/__cache__/flyvision/analysis/stimulus_responses/compute_responses/8e4ab5f546dddc67d4e5d818a5ce98fb/output.h5
    [2024-10-14 21:26:38] chkpt_utils:35 Recovered network state.
    [2024-10-14 21:26:43] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 21:29:55] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/014/__cache__/flyvision/analysis/stimulus_responses/compute_responses/ec463cad10d3570261974dd49b2f39b4/output.h5
    [2024-10-14 21:30:01] chkpt_utils:35 Recovered network state.
    [2024-10-14 21:30:06] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 21:33:19] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/015/__cache__/flyvision/analysis/stimulus_responses/compute_responses/9f12efdac08bb4ec43d6cfa2b410cb84/output.h5
    [2024-10-14 21:33:25] chkpt_utils:35 Recovered network state.
    [2024-10-14 21:33:29] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 21:36:42] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/016/__cache__/flyvision/analysis/stimulus_responses/compute_responses/b766af72495270e849c173fe33059a52/output.h5
    [2024-10-14 21:36:48] chkpt_utils:35 Recovered network state.
    [2024-10-14 21:36:52] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 21:40:05] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/017/__cache__/flyvision/analysis/stimulus_responses/compute_responses/547d8627bc643f80104e95aa70946f87/output.h5
    [2024-10-14 21:40:11] chkpt_utils:35 Recovered network state.
    [2024-10-14 21:40:16] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 21:43:29] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/018/__cache__/flyvision/analysis/stimulus_responses/compute_responses/e8623ea69603f974a7b3b8d5eadb3673/output.h5
    [2024-10-14 21:43:35] chkpt_utils:35 Recovered network state.
    [2024-10-14 21:43:39] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 21:46:51] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/019/__cache__/flyvision/analysis/stimulus_responses/compute_responses/25481e4909d43c99cf38eadd2587310e/output.h5
    [2024-10-14 21:46:57] chkpt_utils:35 Recovered network state.
    [2024-10-14 21:47:02] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 21:50:15] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/020/__cache__/flyvision/analysis/stimulus_responses/compute_responses/fbfc463c7bc43fd02c29dd9d621674ac/output.h5
    [2024-10-14 21:50:21] chkpt_utils:35 Recovered network state.
    [2024-10-14 21:50:25] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 21:53:39] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/021/__cache__/flyvision/analysis/stimulus_responses/compute_responses/47879cefa1cb64fad137fa3abcef9f72/output.h5
    [2024-10-14 21:53:45] chkpt_utils:35 Recovered network state.
    [2024-10-14 21:53:49] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 21:57:01] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/022/__cache__/flyvision/analysis/stimulus_responses/compute_responses/a95a2b7ddef91af37031848d7003b1f3/output.h5
    [2024-10-14 21:57:08] chkpt_utils:35 Recovered network state.
    [2024-10-14 21:57:12] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 22:00:25] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/023/__cache__/flyvision/analysis/stimulus_responses/compute_responses/48f1e15295ea1bd1ecd10d16406b2290/output.h5
    [2024-10-14 22:00:31] chkpt_utils:35 Recovered network state.
    [2024-10-14 22:00:36] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 22:03:48] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/024/__cache__/flyvision/analysis/stimulus_responses/compute_responses/6174edb2564849ef913daa0a8b68b9ec/output.h5
    [2024-10-14 22:03:54] chkpt_utils:35 Recovered network state.
    [2024-10-14 22:03:58] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 22:07:11] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/025/__cache__/flyvision/analysis/stimulus_responses/compute_responses/dc7c7535c7861e19b3ecab2cf7122136/output.h5
    [2024-10-14 22:07:17] chkpt_utils:35 Recovered network state.
    [2024-10-14 22:07:22] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 22:10:36] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/026/__cache__/flyvision/analysis/stimulus_responses/compute_responses/0f442e5e698f56a7df6a03ed8f7000a9/output.h5
    [2024-10-14 22:10:42] chkpt_utils:35 Recovered network state.
    [2024-10-14 22:10:46] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 22:14:00] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/027/__cache__/flyvision/analysis/stimulus_responses/compute_responses/cca57252f75a3f13eb6d01f1bf8d47c7/output.h5
    [2024-10-14 22:14:06] chkpt_utils:35 Recovered network state.
    [2024-10-14 22:14:11] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 22:17:24] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/028/__cache__/flyvision/analysis/stimulus_responses/compute_responses/438a5c4c26cf00e96e592fa3b05b8d46/output.h5
    [2024-10-14 22:17:30] chkpt_utils:35 Recovered network state.
    [2024-10-14 22:17:35] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 22:20:48] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/029/__cache__/flyvision/analysis/stimulus_responses/compute_responses/04c9bf7d4e1e29bd6c0e192faa3be95e/output.h5
    [2024-10-14 22:20:54] chkpt_utils:35 Recovered network state.
    [2024-10-14 22:20:58] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 22:24:12] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/030/__cache__/flyvision/analysis/stimulus_responses/compute_responses/f8ad0bbdc84d745bd24435f409659e18/output.h5
    [2024-10-14 22:24:18] chkpt_utils:35 Recovered network state.
    [2024-10-14 22:24:24] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 22:27:38] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/031/__cache__/flyvision/analysis/stimulus_responses/compute_responses/feb22aa77c3a671a7742cd26d19729b5/output.h5
    [2024-10-14 22:27:44] chkpt_utils:35 Recovered network state.
    [2024-10-14 22:27:48] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 22:31:02] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/032/__cache__/flyvision/analysis/stimulus_responses/compute_responses/3b7c08a1383d3077532a4435c16e9132/output.h5
    [2024-10-14 22:31:08] chkpt_utils:35 Recovered network state.
    [2024-10-14 22:31:12] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 22:34:26] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/033/__cache__/flyvision/analysis/stimulus_responses/compute_responses/3e014da9460072b9e468d5d4db901c36/output.h5
    [2024-10-14 22:34:32] chkpt_utils:35 Recovered network state.
    [2024-10-14 22:34:36] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 22:37:49] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/034/__cache__/flyvision/analysis/stimulus_responses/compute_responses/2f053e975330c272ea7e10e3f096a66f/output.h5
    [2024-10-14 22:37:55] chkpt_utils:35 Recovered network state.
    [2024-10-14 22:38:00] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 22:41:12] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/035/__cache__/flyvision/analysis/stimulus_responses/compute_responses/ff0bbe4777ee636da504d0c4edfd5846/output.h5
    [2024-10-14 22:41:19] chkpt_utils:35 Recovered network state.
    [2024-10-14 22:41:23] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 22:44:35] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/036/__cache__/flyvision/analysis/stimulus_responses/compute_responses/a8e6ccb3a83d8ea15533bd088bd59ef2/output.h5
    [2024-10-14 22:44:41] chkpt_utils:35 Recovered network state.
    [2024-10-14 22:44:46] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 22:47:59] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/037/__cache__/flyvision/analysis/stimulus_responses/compute_responses/e7a08d9c9dc1504228836eb499c33fb1/output.h5
    [2024-10-14 22:48:05] chkpt_utils:35 Recovered network state.
    [2024-10-14 22:48:09] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 22:51:22] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/038/__cache__/flyvision/analysis/stimulus_responses/compute_responses/1ff54cbc099f81f052a5661fc26d000a/output.h5
    [2024-10-14 22:51:28] chkpt_utils:35 Recovered network state.
    [2024-10-14 22:51:32] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 22:54:46] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/039/__cache__/flyvision/analysis/stimulus_responses/compute_responses/50bc9a814ddfb1fd6f4ccb5a8093534d/output.h5
    [2024-10-14 22:54:52] chkpt_utils:35 Recovered network state.
    [2024-10-14 22:54:56] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 22:58:10] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/040/__cache__/flyvision/analysis/stimulus_responses/compute_responses/ac5a53daa316bd164892a45b48fbcbf3/output.h5
    [2024-10-14 22:58:16] chkpt_utils:35 Recovered network state.
    [2024-10-14 22:58:21] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 23:01:33] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/041/__cache__/flyvision/analysis/stimulus_responses/compute_responses/dea4830713214750326ad2c0f74f43b4/output.h5
    [2024-10-14 23:01:39] chkpt_utils:35 Recovered network state.
    [2024-10-14 23:01:44] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 23:04:57] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/042/__cache__/flyvision/analysis/stimulus_responses/compute_responses/28f107cb005216c0c70c85156602badb/output.h5
    [2024-10-14 23:05:03] chkpt_utils:35 Recovered network state.
    [2024-10-14 23:05:07] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 23:08:24] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/043/__cache__/flyvision/analysis/stimulus_responses/compute_responses/9a536865e85de3f3bf6c7c0f51c48a68/output.h5
    [2024-10-14 23:08:30] chkpt_utils:35 Recovered network state.
    [2024-10-14 23:08:35] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 23:11:47] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/044/__cache__/flyvision/analysis/stimulus_responses/compute_responses/9dadf5ce41ea13977a03bf250f47182a/output.h5
    [2024-10-14 23:11:53] chkpt_utils:35 Recovered network state.
    [2024-10-14 23:11:58] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 23:15:12] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/045/__cache__/flyvision/analysis/stimulus_responses/compute_responses/7c5005bd1893429e6cf5fbd05351bd85/output.h5
    [2024-10-14 23:15:18] chkpt_utils:35 Recovered network state.
    [2024-10-14 23:15:22] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 23:18:37] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/046/__cache__/flyvision/analysis/stimulus_responses/compute_responses/cf1f47a50bfb0f93400c60652dd8a353/output.h5
    [2024-10-14 23:18:43] chkpt_utils:35 Recovered network state.
    [2024-10-14 23:18:48] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 23:22:01] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/047/__cache__/flyvision/analysis/stimulus_responses/compute_responses/babe9286fd9940a1d8f3910e35d166cb/output.h5
    [2024-10-14 23:22:07] chkpt_utils:35 Recovered network state.
    [2024-10-14 23:22:12] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 23:25:24] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/048/__cache__/flyvision/analysis/stimulus_responses/compute_responses/e2331bf2e28ee218ea340438d3b3e966/output.h5
    [2024-10-14 23:25:30] chkpt_utils:35 Recovered network state.
    [2024-10-14 23:25:35] network:757 Computing 2268 stimulus responses.



    Batch:   0%|          | 0/567 [00:00<?, ?it/s]


    [2024-10-14 23:28:47] xarray_joblib_backend:54 Store item /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/049/__cache__/flyvision/analysis/stimulus_responses/compute_responses/2a0503ab856cbb6d9206e77684949b55/output.h5



```python
# recommended to only run with precomputed responses using the record script
norm = ensemble.responses_norm()
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





![png](05_flyvision_umap_and_clustering_models_files/05_flyvision_umap_and_clustering_models_15_1.png)



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

    [2024-10-14 23:29:06] clustering:482 reshaped X from (50, 2268, 80) to (50, 181440)
    /home/lappalainenj@hhmi.org/miniconda3/envs/flyvision/lib/python3.9/site-packages/umap/umap_.py:1356: RuntimeWarning: divide by zero encountered in power
      return 1.0 / (1.0 + a * x ** (2 * b))



```python
task_error = ensemble.task_error()
```


```python
embeddingplot = t4c_embedding.plot(colors=task_error.colors)
```



![png](05_flyvision_umap_and_clustering_models_files/05_flyvision_umap_and_clustering_models_24_0.png)



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



![png](05_flyvision_umap_and_clustering_models_files/05_flyvision_umap_and_clustering_models_28_0.png)



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



![png](05_flyvision_umap_and_clustering_models_files/05_flyvision_umap_and_clustering_models_32_0.png)



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



![png](05_flyvision_umap_and_clustering_models_files/05_flyvision_umap_and_clustering_models_42_0.png)




```python
plot_angular_tuning(
    stims_and_resps_moving_edge,
    "T4c",
    intensity=1,
)
```




    (<Figure size 300x300 with 1 Axes>, <PolarAxes: >)





![png](05_flyvision_umap_and_clustering_models_files/05_flyvision_umap_and_clustering_models_43_1.png)




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



![png](05_flyvision_umap_and_clustering_models_files/05_flyvision_umap_and_clustering_models_44_0.png)



As we can see here, the models predict clustered neural responses.

# Load precomputed umap and clustering

Due to the computational requirement of recording and embedding all responses and for consistency we also show how to use the precomputed embeddings and clusterings from the paper.


```python
cell_type = "T4c"
clustering = ensemble.clustering(cell_type)
```

    [2024-10-14 23:29:47] clustering:835 Loaded T4c embedding and clustering from /groups/turaga/home/lappalainenj/FlyVis/private/flyvision/data/results/flow/0000/umap_and_clustering



```python
task_error = ensemble.task_error()
```


```python
embeddingplot = clustering.plot(task_error=task_error.values, colors=task_error.colors)
```



![png](05_flyvision_umap_and_clustering_models_files/05_flyvision_umap_and_clustering_models_50_0.png)



With this embedding and clustering one can proceed in the same way as above to plot the tunings.
