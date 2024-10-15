```
import flyvision
from flyvision import connectome_file
from flyvision import ConnectomeDir, ConnectomeView
```

# Connectome
This notebook illustrates the constructed spatially invariant connectome from local reconstructions that builds the
scaffold of the network.

**Select GPU runtime**

To run the notebook on a GPU select Menu -> Runtime -> Change runtime type -> GPU.


```
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


```
if IN_COLAB:
    # @markdown **Install Flyvis**
    %%capture
    !git clone https://github.com/flyvis/flyvis-dev.git
    %cd /content/flyvis-dev
    !pip install -e .
```


```
# The ConnectomeDir class compiles the network graph from `data/connectome/fib25-fib19_v2.2.json`.
# This json-file includes a list of cell types (`nodes`) and average convolutional filters
# (anatomical receptive fields) (`edges`) that are scattered across a regular hexagonal lattice
# of 15 column extent and stored on the hierarchical filesystem as h5-files.
config = dict(file=connectome_file, extent=15, n_syn_fill=1)
connectome = ConnectomeDir(config)
```


```
# our network models 45,669 cells represented in this table of nodes
connectome.nodes.to_df()
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
      <th>role</th>
      <th>type</th>
      <th>u</th>
      <th>v</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>input</td>
      <td>R1</td>
      <td>-15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>input</td>
      <td>R1</td>
      <td>-15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>input</td>
      <td>R1</td>
      <td>-15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>input</td>
      <td>R1</td>
      <td>-15</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>input</td>
      <td>R1</td>
      <td>-15</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>45664</th>
      <td>output</td>
      <td>TmY18</td>
      <td>15</td>
      <td>-4</td>
    </tr>
    <tr>
      <th>45665</th>
      <td>output</td>
      <td>TmY18</td>
      <td>15</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>45666</th>
      <td>output</td>
      <td>TmY18</td>
      <td>15</td>
      <td>-2</td>
    </tr>
    <tr>
      <th>45667</th>
      <td>output</td>
      <td>TmY18</td>
      <td>15</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>45668</th>
      <td>output</td>
      <td>TmY18</td>
      <td>15</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>45669 rows × 4 columns</p>
</div>




```
# our network models 1,513,231 synapses represented in this table of edges
connectome.edges.to_df()
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
      <th>du</th>
      <th>dv</th>
      <th>edge_type</th>
      <th>n_syn</th>
      <th>n_syn_certainty</th>
      <th>sign</th>
      <th>source_index</th>
      <th>source_type</th>
      <th>source_u</th>
      <th>source_v</th>
      <th>target_index</th>
      <th>target_type</th>
      <th>target_u</th>
      <th>target_v</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>chem</td>
      <td>40.0</td>
      <td>5.859477</td>
      <td>-1.0</td>
      <td>0</td>
      <td>R1</td>
      <td>-15</td>
      <td>0</td>
      <td>5768</td>
      <td>L1</td>
      <td>-15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>chem</td>
      <td>40.0</td>
      <td>5.859477</td>
      <td>-1.0</td>
      <td>1</td>
      <td>R1</td>
      <td>-15</td>
      <td>1</td>
      <td>5769</td>
      <td>L1</td>
      <td>-15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>chem</td>
      <td>40.0</td>
      <td>5.859477</td>
      <td>-1.0</td>
      <td>2</td>
      <td>R1</td>
      <td>-15</td>
      <td>2</td>
      <td>5770</td>
      <td>L1</td>
      <td>-15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>chem</td>
      <td>40.0</td>
      <td>5.859477</td>
      <td>-1.0</td>
      <td>3</td>
      <td>R1</td>
      <td>-15</td>
      <td>3</td>
      <td>5771</td>
      <td>L1</td>
      <td>-15</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>chem</td>
      <td>40.0</td>
      <td>5.859477</td>
      <td>-1.0</td>
      <td>4</td>
      <td>R1</td>
      <td>-15</td>
      <td>4</td>
      <td>5772</td>
      <td>L1</td>
      <td>-15</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1513226</th>
      <td>0</td>
      <td>0</td>
      <td>chem</td>
      <td>1.0</td>
      <td>2.239571</td>
      <td>1.0</td>
      <td>45664</td>
      <td>TmY18</td>
      <td>15</td>
      <td>-4</td>
      <td>45664</td>
      <td>TmY18</td>
      <td>15</td>
      <td>-4</td>
    </tr>
    <tr>
      <th>1513227</th>
      <td>0</td>
      <td>0</td>
      <td>chem</td>
      <td>1.0</td>
      <td>2.239571</td>
      <td>1.0</td>
      <td>45665</td>
      <td>TmY18</td>
      <td>15</td>
      <td>-3</td>
      <td>45665</td>
      <td>TmY18</td>
      <td>15</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>1513228</th>
      <td>0</td>
      <td>0</td>
      <td>chem</td>
      <td>1.0</td>
      <td>2.239571</td>
      <td>1.0</td>
      <td>45666</td>
      <td>TmY18</td>
      <td>15</td>
      <td>-2</td>
      <td>45666</td>
      <td>TmY18</td>
      <td>15</td>
      <td>-2</td>
    </tr>
    <tr>
      <th>1513229</th>
      <td>0</td>
      <td>0</td>
      <td>chem</td>
      <td>1.0</td>
      <td>2.239571</td>
      <td>1.0</td>
      <td>45667</td>
      <td>TmY18</td>
      <td>15</td>
      <td>-1</td>
      <td>45667</td>
      <td>TmY18</td>
      <td>15</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>1513230</th>
      <td>0</td>
      <td>0</td>
      <td>chem</td>
      <td>1.0</td>
      <td>2.239571</td>
      <td>1.0</td>
      <td>45668</td>
      <td>TmY18</td>
      <td>15</td>
      <td>0</td>
      <td>45668</td>
      <td>TmY18</td>
      <td>15</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1513231 rows × 14 columns</p>
</div>



## Connectivity between identified cell types

Identified connectivity between 64 cell types, represented by total number of input synapses from all neurons of a given presynaptic cell type to a single postsynaptic of a given cell type. Blue color indicates putative hyperpolarizing inputs, red putative depolarizing inputs as inferred from neurotransmitter and receptor profiling. Size of squares indicates number of input synapses.


```
# the ConnectomeView class provides visualizations of the connectome data
connectome_view = ConnectomeView(connectome)
```


```
fig = connectome_view.connectivity_matrix("n_syn")
```



![png](01_flyvision_connectome_files/01_flyvision_connectome_11_0.png)



## Example receptive fields
Example of convolutional filter, representing inputs onto cells of the target cell type. Values represent the average number of synapses projecting from presynaptic cells in columns with indicated offset onto the postsynaptic dendrite. Values indicate connection strength derived from electron microscopy data.


```
fig = connectome_view.receptive_fields_grid("T4c")
```



![png](01_flyvision_connectome_files/01_flyvision_connectome_13_0.png)



## Example projective fields
Example of projective fields, representing outputs of a source cell type onto target cells. Values represent the average number of synapses projecting from the presynaptic cell onto postsynaptic dendrites in columns with indicated offset. Values indicate connection strength derived from electron microscopy data.


```
fig = connectome_view.projective_fields_grid("T4c")
```



![png](01_flyvision_connectome_files/01_flyvision_connectome_15_0.png)



## Network layout

Our retinotopic hexagonal lattice network organizes cells of each cell type into visual columns corresponding to photoreceptor locations to capture the crystalline, hexagonal structure of the fly eye. Some cell types are non-columnar, i.e. their cells occur only in every other column---here Lawf1 and Lawf2 cell types---as estimated by our connectome construction algorithm. The edges represent pairs of connected cell types. For the task, we decoded from T-shaped and transmedullary cells (within the black box).


```
# cause the layout is spatially periodic it suffices to visualize a few columns
# to get the gist of the layout which can be controlled using max_extent
fig = connectome_view.network_layout(max_extent=6)
```



![png](01_flyvision_connectome_files/01_flyvision_connectome_18_0.png)
