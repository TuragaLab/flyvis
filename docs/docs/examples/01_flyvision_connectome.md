# Connectome
This notebook illustrates the constructed spatially invariant connectome from local reconstructions that builds the
scaffold of the network.

# The connectome from average local reconstructions


```python
from flyvis import connectome_file
from flyvis import ConnectomeFromAvgFilters, ConnectomeView
```


```python
# The ConnectomeFromAvgFilters class compiles the network graph from `data/connectome/fib25-fib19_v2.2.json`.
# This json-file includes a list of cell types (`nodes`) and average convolutional filters
# (anatomical receptive fields) (`edges`) that are scattered across a regular hexagonal lattice
# of 15 column extent and stored on the hierarchical filesystem as h5-files.
config = dict(file=connectome_file.name, extent=15, n_syn_fill=1)
connectome = ConnectomeFromAvgFilters(config)
```


```python
# our network models 45,669 cells represented in this table of nodes
print(connectome.nodes.to_df().iloc[0:20].to_markdown())
```

    |    |   index | role   | type   |   u |   v |
    |---:|--------:|:-------|:-------|----:|----:|
    |  0 |       0 | input  | R1     | -15 |   0 |
    |  1 |       1 | input  | R1     | -15 |   1 |
    |  2 |       2 | input  | R1     | -15 |   2 |
    |  3 |       3 | input  | R1     | -15 |   3 |
    |  4 |       4 | input  | R1     | -15 |   4 |
    |  5 |       5 | input  | R1     | -15 |   5 |
    |  6 |       6 | input  | R1     | -15 |   6 |
    |  7 |       7 | input  | R1     | -15 |   7 |
    |  8 |       8 | input  | R1     | -15 |   8 |
    |  9 |       9 | input  | R1     | -15 |   9 |
    | 10 |      10 | input  | R1     | -15 |  10 |
    | 11 |      11 | input  | R1     | -15 |  11 |
    | 12 |      12 | input  | R1     | -15 |  12 |
    | 13 |      13 | input  | R1     | -15 |  13 |
    | 14 |      14 | input  | R1     | -15 |  14 |
    | 15 |      15 | input  | R1     | -15 |  15 |
    | 16 |      16 | input  | R1     | -14 |  -1 |
    | 17 |      17 | input  | R1     | -14 |   0 |
    | 18 |      18 | input  | R1     | -14 |   1 |
    | 19 |      19 | input  | R1     | -14 |   2 |



```python
# our network models 1,513,231 synapses represented in this table of edges
print(connectome.edges.to_df().iloc[0:20].to_markdown())
```

    |    |   du |   dv |   n_syn |   n_syn_certainty |   sign |   source_index | source_type   |   source_u |   source_v |   target_index | target_type   |   target_u |   target_v |
    |---:|-----:|-----:|--------:|------------------:|-------:|---------------:|:--------------|-----------:|-----------:|---------------:|:--------------|-----------:|-----------:|
    |  0 |    0 |    0 |      40 |           5.85948 |     -1 |              0 | R1            |        -15 |          0 |           5768 | L1            |        -15 |          0 |
    |  1 |    0 |    0 |      40 |           5.85948 |     -1 |              1 | R1            |        -15 |          1 |           5769 | L1            |        -15 |          1 |
    |  2 |    0 |    0 |      40 |           5.85948 |     -1 |              2 | R1            |        -15 |          2 |           5770 | L1            |        -15 |          2 |
    |  3 |    0 |    0 |      40 |           5.85948 |     -1 |              3 | R1            |        -15 |          3 |           5771 | L1            |        -15 |          3 |
    |  4 |    0 |    0 |      40 |           5.85948 |     -1 |              4 | R1            |        -15 |          4 |           5772 | L1            |        -15 |          4 |
    |  5 |    0 |    0 |      40 |           5.85948 |     -1 |              5 | R1            |        -15 |          5 |           5773 | L1            |        -15 |          5 |
    |  6 |    0 |    0 |      40 |           5.85948 |     -1 |              6 | R1            |        -15 |          6 |           5774 | L1            |        -15 |          6 |
    |  7 |    0 |    0 |      40 |           5.85948 |     -1 |              7 | R1            |        -15 |          7 |           5775 | L1            |        -15 |          7 |
    |  8 |    0 |    0 |      40 |           5.85948 |     -1 |              8 | R1            |        -15 |          8 |           5776 | L1            |        -15 |          8 |
    |  9 |    0 |    0 |      40 |           5.85948 |     -1 |              9 | R1            |        -15 |          9 |           5777 | L1            |        -15 |          9 |
    | 10 |    0 |    0 |      40 |           5.85948 |     -1 |             10 | R1            |        -15 |         10 |           5778 | L1            |        -15 |         10 |
    | 11 |    0 |    0 |      40 |           5.85948 |     -1 |             11 | R1            |        -15 |         11 |           5779 | L1            |        -15 |         11 |
    | 12 |    0 |    0 |      40 |           5.85948 |     -1 |             12 | R1            |        -15 |         12 |           5780 | L1            |        -15 |         12 |
    | 13 |    0 |    0 |      40 |           5.85948 |     -1 |             13 | R1            |        -15 |         13 |           5781 | L1            |        -15 |         13 |
    | 14 |    0 |    0 |      40 |           5.85948 |     -1 |             14 | R1            |        -15 |         14 |           5782 | L1            |        -15 |         14 |
    | 15 |    0 |    0 |      40 |           5.85948 |     -1 |             15 | R1            |        -15 |         15 |           5783 | L1            |        -15 |         15 |
    | 16 |    0 |    0 |      40 |           5.85948 |     -1 |             16 | R1            |        -14 |         -1 |           5784 | L1            |        -14 |         -1 |
    | 17 |    0 |    0 |      40 |           5.85948 |     -1 |             17 | R1            |        -14 |          0 |           5785 | L1            |        -14 |          0 |
    | 18 |    0 |    0 |      40 |           5.85948 |     -1 |             18 | R1            |        -14 |          1 |           5786 | L1            |        -14 |          1 |
    | 19 |    0 |    0 |      40 |           5.85948 |     -1 |             19 | R1            |        -14 |          2 |           5787 | L1            |        -14 |          2 |


## Connectivity between identified cell types

Identified connectivity between 64 cell types, represented by total number of input synapses from all neurons of a given presynaptic cell type to a single postsynaptic of a given cell type. Blue color indicates putative hyperpolarizing inputs, red putative depolarizing inputs as inferred from neurotransmitter and receptor profiling. Size of squares indicates number of input synapses.


```python
# the ConnectomeView class provides visualizations of the connectome data
connectome_view = ConnectomeView(connectome)
```


```python
fig = connectome_view.connectivity_matrix("n_syn")
```



![png](01_flyvision_connectome_files/01_flyvision_connectome_8_0.png)



## Example receptive fields
Example of convolutional filter, representing inputs onto cells of the target cell type. Values represent the average number of synapses projecting from presynaptic cells in columns with indicated offset onto the postsynaptic dendrite. Values indicate connection strength derived from electron microscopy data.


```python
fig = connectome_view.receptive_fields_grid("T4c")
```



![png](01_flyvision_connectome_files/01_flyvision_connectome_10_0.png)



## Example projective fields
Example of projective fields, representing outputs of a source cell type onto target cells. Values represent the average number of synapses projecting from the presynaptic cell onto postsynaptic dendrites in columns with indicated offset. Values indicate connection strength derived from electron microscopy data.


```python
fig = connectome_view.projective_fields_grid("T4c")
```



![png](01_flyvision_connectome_files/01_flyvision_connectome_12_0.png)



## Network layout

Our retinotopic hexagonal lattice network organizes cells of each cell type into visual columns corresponding to photoreceptor locations to capture the crystalline, hexagonal structure of the fly eye. Some cell types are non-columnar, i.e. their cells occur only in every other column---here Lawf1 and Lawf2 cell types---as estimated by our connectome construction algorithm. The edges represent pairs of connected cell types. For the task, we decoded from T-shaped and transmedullary cells (within the black box).


```python
# cause the layout is spatially periodic it suffices to visualize a few columns
# to get the gist of the layout which can be controlled using max_extent
fig = connectome_view.network_layout(max_extent=6)
```



![png](01_flyvision_connectome_files/01_flyvision_connectome_15_0.png)




```python

```
