```python
from flyvision.initialization import (
    Parameter,
    deepcopy_config,
    RestingPotential,
    TimeConstant,
    SynapseCount,
    SynapseCountScaling,
    SynapseSign,
)
from flyvision.connectome import ConnectomeDir
from datamate import Namespace
```


```python
import pandas as pd
import torch
```


```python
from flyvision import connectome_file
from flyvision.initialization import symmetry_mask_for_edges
```


```python
def cast(x):
    if np.issubdtype(x.dtype, np.dtype("S")):
        return x.astype("U")
    return x
```


```python
def get_scatter_indices(
    dataframe, grouped_dataframe, groupby
):
    """Returns the grouped dataframe and the reference indices for shared parameters.

    Args:
        dataframe (pd.DataFrame): dataframe of nodes or edges, can also
                contain synapse counts and signs.
        groupby (list): list of columns to group the dataframe by.
        group (method): groupby method, e.g. first, mean, sum.

    Returns:
        pd.DataFrame: first entries per group.
        tensor: indices for parameter sharing
    """
    ungrouped_elements = zip(*[dataframe[k][:] for k in groupby])
    grouped_elements = zip(*[grouped_dataframe[k][:] for k in groupby])
    to_index = {k: i for i, k in enumerate(grouped_elements)}
    return torch.tensor([to_index[k] for k in ungrouped_elements])
```


```python
from numbers import Number
```


```python
def byte_to_str(obj):
    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.dtype("S")):
            return obj.astype("U")
        return obj
    if isinstance(obj, list):
        obj = [byte_to_str(item) for item in obj]
        return obj
    elif isinstance(obj, tuple):
        obj = tuple([byte_to_str(item) for item in obj])
        return obj
    elif isinstance(obj, bytes):
        return obj.decode()
    elif isinstance(obj, (str, Number)):
        return obj
    else:
        raise TypeError
```


```python
def compare_param(param1, param2):
    print(f"comparing {param1}, {param2}")
    raw_v = (param1.raw_values == param2.raw_values).all()
    print(f"raw values {raw_v}, {param2.raw_values.shape}")
    ind = (param1.indices == param2.indices).all()
    print(f"indices {ind}, {param2.indices}")
    keys = byte_to_str(param1.keys) == byte_to_str(param2.keys)
    print(f"keys {keys}, {np.array(param2.keys).shape}")
    masks = all([
        all([a[b].all(), a.sum() == len(b)])
        for a, b in zip(param1.symmetry_masks, param2.symmetry_masks)
    ])
    print(f"masks {masks}, {len(param2.symmetry_masks)}")
    if all([raw_v, ind, keys, masks]):
        print("all passed")
        return True
    return False
```


```python
from flyvision.initialization import InitialDistribution, symmetry_mask_for_nodes
```


```python
import numpy as np
```


```python
connectome = ConnectomeDir(dict(file=connectome_file, extent=15, n_syn_fill=1))
```


```python
from typing import List
from numpy.typing import NDArray
```


```python
np.random.normal(size=[4, 4]).astype('|S64')
```




    array([[b'-0.9409330618360031', b'1.0359303347565043',
            b'0.7579351187895884', b'1.4303009058439506'],
           [b'1.6960079452744552', b'0.36587744106980996',
            b'0.36973241134944446', b'-0.4087418488265296'],
           [b'0.1919436271958159', b'-0.7850875645753341',
            b'-0.6310774064104027', b'0.5065377202272231'],
           [b'1.1692414813317535', b'1.041828059256901',
            b'1.2154525911366734', b'-1.1017525548382592']], dtype='|S64')




```python
def atleast_column_vector(array):
    array = np.array(array)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def matrix_mask_by_sub(sub_matrix, matrix) -> NDArray[bool]:
    """Mask of rows in matrix that are contained in sub_matrix.

    Args:
        sub_matrix (array): shape (#rows1, #columns)
        matrix (array): shape (#rows2, #columns)

    Returns:
        array: 1D boolean array of length #rows2

    Note: #rows1 !<= #rows2

    Example:
        sub_matrix = np.array([[1, 2, 3],
                                [4, 3, 1]])
        matrix = np.array([[3, 4, 1],
                            [4, 3, 1],
                            [1, 2, 3]])
        matrix_mask_by_sub(sub_matrix, matrix)
        array([False, True, True])

    Typically, indexing a tensor with indices instead of booleans is
    faster. Therefore, see also where_equal_rows.
    """
    from functools import reduce

    n_rows, n_columns = sub_matrix.shape
    n_rows2 = matrix.shape[0]
    if not n_rows <= n_rows2:
        raise ValueError
    row_mask = []
    for i in range(n_rows):
        column_mask = []
        for j in range(n_columns):
            column_mask.append(sub_matrix[i, j] == matrix[:, j])
        row_mask.append(reduce(np.logical_and, column_mask))
    return reduce(np.logical_or, row_mask)


def where_equal_rows(matrix1, matrix2, as_mask=False, astype='|S64') -> NDArray[int]:
    """Indices where matrix1 rows are in matrix2.

    Example:
        matrix1 = np.array([[1, 2, 3],
                            [4, 3, 1]])
        matrix2 = np.array([[3, 4, 1],
                            [4, 3, 1],
                            [1, 2, 3],
                            [0, 0, 0]])
        where_equal_rows(matrix1, matrix2)
        array([2, 1])
        matrix2[where_equal_rows(matrix1, matrix2)]
        array([[1, 2, 3],
               [4, 3, 1]])

    See also: matrix_mask_by_sub.
    """
    matrix1 = atleast_column_vector(matrix1)
    matrix2 = atleast_column_vector(matrix2)
    matrix1 = matrix1.astype(astype)
    matrix2 = matrix2.astype(astype)
    
    if as_mask:
        return matrix_mask_by_sub(matrix1, matrix2)

    n_rows1, n_cols1 = matrix1.shape
    n_rows2, n_cols2 = matrix2.shape

    if not n_rows1 <= n_rows2:
        raise ValueError("matrix1 must have less or" " equal as many rows as matrix2")
    if not n_cols1 == n_cols2:
        raise ValueError("cannot compare matrices with different number of columns")

    where = []
    rows = np.arange(matrix2.shape[0])
    for row in matrix1:
        equal_rows = (row == matrix2).all(axis=1)
        for index in rows[equal_rows]:
            where.append(index)
    return np.array(where)


def symmetry_masks(symmetric, keys, as_mask=False) -> List[torch.Tensor]:
    """One additional way to constrain network elements to have the same
    parameter values.

    Note, this method stores one mask per shared tuple of the size of the
    parameter and should thus be used sparsely because its not very memory
    friendly. The bulk part of parameter sharing is achieved through scatter
    and gather operations.

    Args:
        symmetric: list of tuples of cell types that share parameters.
        nodes: DataFrame containing 'cell_type' column.

    Returns:
        list of masks List[torch.BoolTensor]

    Example:
        symmetric = [["T4a", "T4b", "T4c", "T4d"], ["T5a", "T5b", "T5c", "T5d"]]
        would return two masks, one to constrain all T4 subtypes to the same
        parameter and another to constrain all T5 subtypes to the same parameter.
    """
    if not symmetric:
        return []
    symmetry_masks = []  # type: List[torch.Tensor]
    keys = atleast_column_vector(keys)
    for i, identifiers in enumerate(symmetric):
        identifiers = atleast_column_vector(identifiers)
        # to allow identifiers like [None, "A", None, 0]
        # for parameters that have multi-keys
        columns = np.arange(identifiers.shape[1] + 1)[
            np.where((identifiers != None).all(axis=0))
        ]
        try:
            symmetry_masks.append(
                torch.tensor(
                    where_equal_rows(identifiers[:, columns],
                                     keys[:, columns], as_mask=as_mask)
                )
            )
        except Exception as e:
            raise ValueError(
                f"{identifiers} cannot be a symmetry constraint"
                f" for parameter with keys {keys}: {e}"
            ) from e
    return symmetry_masks
```

# RestingPotential


```python
config = Namespace(
                type="RestingPotential",
                groupby=["type"],
                initial_dist="Normal",
                mode="sample",
                requires_grad=True,
                mean=0.5,
                std=0.05,
                penalize=Namespace(activity=True),
                seed=0,
    symmetric = [["T4a", "T4b", "T4c", "T4d"], ["T5a", "T5b", "T5c", "T5d"]]
            )
```


```python
from flyvision.initialization import Parameter
```


```python
class RestingPotentialv2(Parameter):
    """Initialize resting potentials a.k.a. biases for cell types."""

    @deepcopy_config
    def __init__(self, param_config: Namespace, connectome: ConnectomeDir):
        nodes_dir = connectome.nodes

        nodes = pd.DataFrame(
            {k: byte_to_str(nodes_dir[k][:]) for k in param_config.groupby}
        )
        grouped_nodes = nodes.groupby(
            param_config.groupby, as_index=False, sort=False
        ).first()

        param_config["type"] = grouped_nodes["type"].values
        param_config["mean"] = np.repeat(param_config["mean"], len(grouped_nodes))
        param_config["std"] = np.repeat(param_config["std"], len(grouped_nodes))

        self.parameter = InitialDistribution(param_config)
        self.indices = get_scatter_indices(nodes, grouped_nodes, param_config.groupby)
        self.keys = param_config["type"].tolist()
        self.symmetry_masks = symmetry_masks(
            param_config.get("symmetric", []), self.keys
        )
```


```python
param = Parameter(config, connectome)
```


```python
np.array(param.keys)[param.symmetry_masks[0].cpu().numpy()]
```




    array(['T4a', 'T4b', 'T4c', 'T4d'], dtype='<U8')




```python
config2 = Namespace(
                type="RestingPotentialv2",
                groupby=["type"],
                initial_dist="Normal",
                mode="sample",
                requires_grad=True,
                mean=0.5,
                std=0.05,
                penalize=Namespace(activity=True),
                seed=0,
    symmetric = [["T4a", "T4b", "T4c", "T4d"], ["T5a", "T5b", "T5c", "T5d"]]
            )
```


```python
nodes = connectome.nodes.to_df()
```


```python
param2 = Parameter(config2, connectome)
```


```python
param.keys[0]
```




    'R1'




```python
param2.symmetry_masks
```




    [tensor([35, 36, 37, 38]), tensor([39, 40, 41, 42])]




```python
compare_param(param, param2)
```

    comparing RestingPotential(Namespace(
      type = 'RestingPotential',
      groupby = ['type'],
      initial_dist = 'Normal',
      mode = 'sample',
      requires_grad = True,
      mean = 0.5,
      std = 0.05,
      penalize = Namespace(activity=True),
      seed = 0,
      symmetric = [['T4a', 'T4b', 'T4c', 'T4d'], ['T5a', 'T5b', 'T5c', 'T5d']]
    ), ConnectomeDir), RestingPotentialv2(Namespace(
      type = 'RestingPotentialv2',
      groupby = ['type'],
      initial_dist = 'Normal',
      mode = 'sample',
      requires_grad = True,
      mean = 0.5,
      std = 0.05,
      penalize = Namespace(activity=True),
      seed = 0,
      symmetric = [['T4a', 'T4b', 'T4c', 'T4d'], ['T5a', 'T5b', 'T5c', 'T5d']]
    ), ConnectomeDir)
    raw values True, torch.Size([65])
    indices True, tensor([ 0,  0,  0,  ..., 64, 64, 64])
    keys True, (65,)
    masks True, 2
    all passed





    True



# Time constants


```python
from flyvision.initialization import Parameter
```


```python
class TimeConstantv2(Parameter):
    """Initialize time constants for cell types."""

    @deepcopy_config
    def __init__(self, param_config: Namespace, connectome: ConnectomeDir):
        nodes_dir = connectome.nodes

        nodes = pd.DataFrame({k: byte_to_str(nodes_dir[k][:]) for k in param_config.groupby})
        grouped_nodes = nodes.groupby(param_config.groupby, as_index=False, sort=False).first()

        param_config["type"] = grouped_nodes["type"].values
        param_config["value"] = np.repeat(param_config["value"], len(grouped_nodes))

        self.indices = get_scatter_indices(nodes, grouped_nodes, param_config.groupby)
        self.parameter = InitialDistribution(param_config)
        self.keys = param_config["type"].tolist()
        self.symmetry_masks = symmetry_masks(
            param_config.get("symmetric", []), self.keys
        )
```


```python
config1 = Namespace(
                type="TimeConstant",
                groupby=["type"],
                initial_dist="Value",
                value=0.05,
                requires_grad=True,
        symmetric = [["T4a", "T4b", "T4c", "T4d"], ["T5a", "T5b", "T5c", "T5d"]]

            )

config2 = Namespace(
                type="TimeConstantv2",
                groupby=["type"],
                initial_dist="Value",
                value=0.05,
                requires_grad=True,
        symmetric = [["T4a", "T4b", "T4c", "T4d"], ["T5a", "T5b", "T5c", "T5d"]]

            )
```


```python
param = Parameter(config1, connectome)
```


```python
param2 = Parameter(config2, connectome)
```


```python
param2.symmetry_masks
```




    [tensor([35, 36, 37, 38]), tensor([39, 40, 41, 42])]




```python
compare_param(param, param2)
```

    comparing TimeConstant(Namespace(
      type = 'TimeConstant',
      groupby = ['type'],
      initial_dist = 'Value',
      value = 0.05,
      requires_grad = True,
      symmetric = [['T4a', 'T4b', 'T4c', 'T4d'], ['T5a', 'T5b', 'T5c', 'T5d']]
    ), ConnectomeDir), TimeConstantv2(Namespace(
      type = 'TimeConstantv2',
      groupby = ['type'],
      initial_dist = 'Value',
      value = 0.05,
      requires_grad = True,
      symmetric = [['T4a', 'T4b', 'T4c', 'T4d'], ['T5a', 'T5b', 'T5c', 'T5d']]
    ), ConnectomeDir)
    raw values True, torch.Size([65])
    indices True, tensor([ 0,  0,  0,  ..., 64, 64, 64])
    keys True, (65,)
    masks True, 2
    all passed





    True



# SynapseSign


```python
class SynapseSignv2(Parameter):
    """Initialize synapse signs for edge types."""

    @deepcopy_config
    def __init__(self, param_config: Namespace, connectome: ConnectomeDir) -> None:
        edges_dir = connectome.edges

        edges = pd.DataFrame(
            {k: byte_to_str(edges_dir[k][:]) for k in [*param_config.groupby, "sign"]}
        )
        grouped_edges = edges.groupby(param_config.groupby, as_index=False, sort=False).first()

        param_config.source_type = grouped_edges.source_type.values
        param_config.target_type = grouped_edges.target_type.values
        param_config.value = grouped_edges.sign.values
        
        self.indices = get_scatter_indices(edges, grouped_edges, param_config.groupby)
        self.parameter = InitialDistribution(param_config)
        self.keys = list(
            zip(
                param_config.source_type.tolist(),
                param_config.target_type.tolist(),
            )
        )
        self.symmetry_masks = symmetry_masks(
            param_config.get("symmetric", []), self.keys
        )
```


```python
config = Namespace(
    type="SynapseSign",
    initial_dist="Value",
    requires_grad=False,
    groupby=["source_type", "target_type"],
    attributes=["sign"],
    symmetric=[[("R1", "L1"), ("R4", "L1")]],
)
```


```python
param = Parameter(config, connectome)
```


```python
param.symmetry_masks
```




    []




```python
config2 = Namespace(
                type="SynapseSignv2",
                initial_dist="Value",
                requires_grad=False,
                groupby=["source_type", "target_type"],
                attributes=["sign"],
    symmetric=[[("R1", "L1"),
                ("R4", "L1")],
              [("Mi9", "T4a"),
              ("Mi9", "T4b")]],
            )
```


```python
param2 = Parameter(config2, connectome)
```


```python
compare_param(param, param2)
```

    comparing SynapseSign(Namespace(
      type = 'SynapseSign',
      initial_dist = 'Value',
      requires_grad = False,
      groupby = ['source_type', 'target_type'],
      attributes = ['sign'],
      symmetric = [[('R1', 'L1'), ('R4', 'L1')]]
    ), ConnectomeDir), SynapseSignv2(Namespace(
      type = 'SynapseSignv2',
      initial_dist = 'Value',
      requires_grad = False,
      groupby = ['source_type', 'target_type'],
      attributes = ['sign'],
      symmetric = [[('R1', 'L1'), ('R4', 'L1')], [('Mi9', 'T4a'), ('Mi9', 'T4b')]]
    ), ConnectomeDir)
    raw values True, torch.Size([604])
    indices True, tensor([  0,   0,   0,  ..., 603, 603, 603])
    keys True, (604, 2)
    masks True, 2
    all passed





    True



# SynapseCount


```python
class SynapseCountv2(Parameter):
    """Initialize synapse counts for edge types."""

    @deepcopy_config
    def __init__(self, param_config: Namespace, connectome: ConnectomeDir) -> None:
        mode = param_config.get("mode", "")
        if mode != "mean":
            raise NotImplementedError(
                f"SynapseCount does not implement {mode}. Implement "
                "a custom Parameter subclass."
            )

        edges_dir = connectome.edges

        edges = pd.DataFrame(
            {k: byte_to_str(edges_dir[k][:]) for k in [*param_config.groupby, "n_syn"]}
        )
        grouped_edges = edges.groupby(
            param_config.groupby, as_index=False, sort=False
        ).mean()

        param_config.source_type = grouped_edges.source_type.values
        param_config.target_type = grouped_edges.target_type.values
        param_config.du = grouped_edges.du.values
        param_config.dv = grouped_edges.dv.values

        param_config.mode = "mean"
        param_config.mean = np.log(grouped_edges.n_syn.values)

        self.indices = get_scatter_indices(edges, grouped_edges, param_config.groupby)
        self.parameter = InitialDistribution(param_config)
        self.keys = list(
            zip(
                param_config.source_type.tolist(),
                param_config.target_type.tolist(),
                param_config.du.tolist(),
                param_config.dv.tolist(),
            )
        )
        self.symmetry_masks = symmetry_masks(
            param_config.get("symmetric", []), self.keys
        )
        
    def symmetry(self):
        keys = np.array(self.keys)
        return [keys[mask.cpu()] for mask in self.symmetry_masks]
```


```python
config = Namespace(
                type="SynapseCount",
                initial_dist="Lognormal",
                mode="mean",
                requires_grad=False,
                std=1.0,
                groupby=["source_type", "target_type", "dv", "du"],
#         symmetric=[[("R1", "L1"),
#                 ("R4", "L1")],
#               [("Mi9", "T4a"),
#               ("Mi9", "T4b")]],
            )
```


```python
param = Parameter(config, connectome)
```


```python
config2 = Namespace(
    type="SynapseCountv2",
    initial_dist="Lognormal",
    mode="mean",
    requires_grad=False,
    std=1.0,
    groupby=["source_type", "target_type", "du", "dv"],
    attributes=["n_syn"],
#     symmetric=[[("R1", "L1"),
#                 ("R4", "L1")],
#               [("Mi9", "T4a"),
#               ("Mi9", "T4b")], 
#               [(None, None, 0, 0),
#               (None, None, 0, 0)]],
)
```


```python
config2 = Namespace(
    type="SynapseCountv2",
    initial_dist="Lognormal",
    mode="mean",
    requires_grad=False,
    std=1.0,
    groupby=["source_type", "target_type", "dv", "du"],
    attributes=["n_syn"],
    symmetric=[
        [("R2",), ("R3",)],
        [("R1", "L1", 0), ("R4", "L1", 0)],
        [("Mi9", "T4a", 0, 0), ("Mi9", "T4b", 0, 0)],
        [("Mi9", "T4a", 0, 0, 1), ("Mi9", "T4b", 0, 0, 1)],
    ],
)
```


```python
param2 = Parameter(config2, connectome)
```


```python
compare_param(param, param2)
```

    comparing SynapseCount(Namespace(
      type = 'SynapseCount',
      initial_dist = 'Lognormal',
      mode = 'mean',
      requires_grad = False,
      std = 1.0,
      groupby = ['source_type', 'target_type', 'dv', 'du']
    ), ConnectomeDir), SynapseCountv2(Namespace(
      type = 'SynapseCountv2',
      initial_dist = 'Lognormal',
      mode = 'mean',
      requires_grad = False,
      std = 1.0,
      groupby = ['source_type', 'target_type', 'du', 'dv'],
      attributes = ['n_syn']
    ), ConnectomeDir)
    raw values True, torch.Size([2355])
    indices True, tensor([   0,    0,    0,  ..., 2354, 2354, 2354])
    keys True, (2355, 4)
    masks True, 0
    all passed





    True




```python
param2.symmetry()
```




    [array([['R1', 'L1', '0', '0'],
            ['R4', 'L1', '0', '0']], dtype='<U21'),
     array([['Mi9', 'T4a', '-1', '0'],
            ['Mi9', 'T4a', '0', '1'],
            ['Mi9', 'T4a', '-1', '1'],
            ['Mi9', 'T4a', '-1', '-1'],
            ['Mi9', 'T4a', '0', '-1'],
            ['Mi9', 'T4a', '0', '-2'],
            ['Mi9', 'T4a', '1', '-2'],
            ['Mi9', 'T4a', '1', '-1'],
            ['Mi9', 'T4a', '0', '0'],
            ['Mi9', 'T4a', '2', '2'],
            ['Mi9', 'T4a', '-4', '0'],
            ['Mi9', 'T4a', '-3', '0'],
            ['Mi9', 'T4a', '-2', '-1'],
            ['Mi9', 'T4a', '-2', '0'],
            ['Mi9', 'T4a', '1', '0'],
            ['Mi9', 'T4a', '1', '1'],
            ['Mi9', 'T4b', '0', '1'],
            ['Mi9', 'T4b', '-1', '1'],
            ['Mi9', 'T4b', '-1', '2'],
            ['Mi9', 'T4b', '0', '-1'],
            ['Mi9', 'T4b', '0', '0']], dtype='<U21'),
     array([['R1', 'L1', '0', '0'],
            ['R1', 'L2', '0', '0'],
            ['R1', 'L3', '0', '0'],
            ...,
            ['TmY18', 'TmY5a', '0', '0'],
            ['TmY18', 'TmY15', '0', '0'],
            ['TmY18', 'TmY18', '0', '0']], dtype='<U21')]



# SynapseCountScaling


```python
class SynapseCountScalingv2(Parameter):
    """Initialize synapse count scaling for edge types."""

    @deepcopy_config
    def __init__(self, param_config: Namespace, connectome: ConnectomeDir) -> None:
        edges_dir = connectome.edges

        edges = pd.DataFrame(
            {
                k: byte_to_str(edges_dir[k][:])
                for k in [*param_config.groupby, "n_syn"]
            }
        )
        grouped_edges = edges.groupby(
            param_config.groupby, as_index=False, sort=False
        ).mean()

        # to initialize synapse strengths with 1/<N>_rf
        syn_strength = 1 / grouped_edges.n_syn.values  # 1/<N>_rf

        # scale synapse strengths of chemical and electrical synapses
        # individually
        syn_strength[grouped_edges[grouped_edges.edge_type == "chem"].index] *= getattr(
            param_config, "scale_chem", 0.01
        )
        syn_strength[grouped_edges[grouped_edges.edge_type == "elec"].index] *= getattr(
            param_config, "scale_elec", 0.01
        )

        param_config.target_type = grouped_edges.target_type.values
        param_config.source_type = grouped_edges.source_type.values
        param_config.value = syn_strength

        self.indices = get_scatter_indices(edges, grouped_edges, param_config.groupby)
        self.parameter = InitialDistribution(param_config)
        self.keys = list(
            zip(
                param_config.source_type.tolist(),
                param_config.target_type.tolist(),
            )
        )
        self.symmetry_masks = symmetry_masks(
            param_config.get("symmetric", []), self.keys
        )
```


```python
config = Namespace(
                type="SynapseCountScaling",
                initial_dist="Value",
                requires_grad=True,
                scale_elec=0.01,
                scale_chem=0.01,
                clamp="non_negative",
                groupby=["source_type", "target_type"],
                attributes=["edge_type", "n_syn"],
            symmetric=[[("R1", "L1"),
                ("R4", "L1")],
              [("Mi9", "T4a"),
              ("Mi9", "T4b")]],
            )
```


```python
param = Parameter(config, connectome)
```


```python
config2 = Namespace(
                type="SynapseCountScalingv2",
                initial_dist="Value",
                requires_grad=True,
                scale_elec=0.01,
                scale_chem=0.01,
                clamp="non_negative",
                groupby=["source_type", "target_type", "edge_type"],
                attributes=["n_syn"],
            symmetric=[[("R1", "L1"),
                ("R4", "L1")],
              [("Mi9", "T4a"),
              ("Mi9", "T4b")]],
            )
```


```python
compare_param(param, param2)
```

    comparing SynapseCountScaling(Namespace(
      type = 'SynapseCountScaling',
      initial_dist = 'Value',
      requires_grad = True,
      scale_elec = 0.01,
      scale_chem = 0.01,
      clamp = 'non_negative',
      groupby = ['source_type', 'target_type'],
      attributes = ['edge_type', 'n_syn'],
      symmetric = [[('R1', 'L1'), ('R4', 'L1')], [('Mi9', 'T4a'), ('Mi9', 'T4b')]]
    ), ConnectomeDir), SynapseCountScalingv2(Namespace(
      type = 'SynapseCountScalingv2',
      initial_dist = 'Value',
      requires_grad = True,
      scale_elec = 0.01,
      scale_chem = 0.01,
      clamp = 'non_negative',
      groupby = ['source_type', 'target_type', 'edge_type'],
      attributes = ['n_syn'],
      symmetric = [[('R1', 'L1'), ('R4', 'L1')], [('Mi9', 'T4a'), ('Mi9', 'T4b')]]
    ), ConnectomeDir)
    raw values True, torch.Size([604])
    indices True, tensor([  0,   0,   0,  ..., 603, 603, 603])
    keys True, (604, 2)
    masks False, 2





    False




```python
from typing import Mapping
```


```python

def byte_to_str(obj):
    """Cast byte elements to string types.

    Note, this function is recursive and will cast all byte elements in a nested
    list or tuple.
    """
    if isinstance(obj, Mapping):
        return type(obj)({k: byte_to_str(v) for k, v in obj.items()})
    elif isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.dtype("S")):
            return obj.astype("U")
        return obj
    elif isinstance(obj, list):
        obj = [byte_to_str(item) for item in obj]
        return obj
    elif isinstance(obj, tuple):
        obj = tuple([byte_to_str(item) for item in obj])
        return obj
    elif isinstance(obj, bytes):
        return obj.decode()
    elif isinstance(obj, (str, Number)):
        return obj
    else:
        raise TypeError(f"can't cast {obj} of type {type(obj)} to str")
```


```python
a = Namespace(a=Namespace(a=Namespace(a=b"a")))
```


```python
a
```




    Namespace(a=Namespace(a=Namespace(a=b'a')))




```python
byte_to_str([[[[[(b"a", b"b")]]]]])
```




    [[[[[('a', 'b')]]]]]




```python
edges = connectome.edges.to_df()
```


```python
edges.groupby(["source_type", "target_type"], as_index=False, sort=False).mean(["n_syn"])
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
      <th>source_type</th>
      <th>target_type</th>
      <th>source_u</th>
      <th>target_u</th>
      <th>n_syn</th>
      <th>du</th>
      <th>source_index</th>
      <th>dv</th>
      <th>source_v</th>
      <th>target_index</th>
      <th>target_v</th>
      <th>sign</th>
      <th>n_syn_certainty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>R1</td>
      <td>L1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
      <td>0.000000</td>
      <td>360.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6128.000000</td>
      <td>0.000000</td>
      <td>-1.0</td>
      <td>5.859477</td>
    </tr>
    <tr>
      <th>1</th>
      <td>R1</td>
      <td>L2</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>46.000000</td>
      <td>0.000000</td>
      <td>360.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6849.000000</td>
      <td>0.000000</td>
      <td>-1.0</td>
      <td>5.859477</td>
    </tr>
    <tr>
      <th>2</th>
      <td>R1</td>
      <td>L3</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>360.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7570.000000</td>
      <td>0.000000</td>
      <td>-1.0</td>
      <td>5.859477</td>
    </tr>
    <tr>
      <th>3</th>
      <td>R1</td>
      <td>Am</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>36.000000</td>
      <td>0.000000</td>
      <td>360.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9979.000000</td>
      <td>0.000000</td>
      <td>-1.0</td>
      <td>5.859477</td>
    </tr>
    <tr>
      <th>4</th>
      <td>R1</td>
      <td>T1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>360.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>21515.000000</td>
      <td>0.000000</td>
      <td>-1.0</td>
      <td>5.859477</td>
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
    </tr>
    <tr>
      <th>599</th>
      <td>TmY18</td>
      <td>T2</td>
      <td>0.244507</td>
      <td>-0.244507</td>
      <td>2.364281</td>
      <td>-0.489015</td>
      <td>45313.822112</td>
      <td>0.489015</td>
      <td>-0.244507</td>
      <td>22230.177888</td>
      <td>0.244507</td>
      <td>1.0</td>
      <td>1.587606</td>
    </tr>
    <tr>
      <th>600</th>
      <td>TmY18</td>
      <td>TmY5a</td>
      <td>-0.244507</td>
      <td>0.244507</td>
      <td>1.656361</td>
      <td>0.489015</td>
      <td>45302.177888</td>
      <td>-0.489015</td>
      <td>0.244507</td>
      <td>40987.822112</td>
      <td>-0.244507</td>
      <td>1.0</td>
      <td>5.517218</td>
    </tr>
    <tr>
      <th>601</th>
      <td>TmY18</td>
      <td>TmY9</td>
      <td>1.500000</td>
      <td>-1.500000</td>
      <td>1.000000</td>
      <td>-3.000000</td>
      <td>45344.787302</td>
      <td>2.000000</td>
      <td>-1.000000</td>
      <td>41666.212698</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>2.782612</td>
    </tr>
    <tr>
      <th>602</th>
      <td>TmY18</td>
      <td>TmY15</td>
      <td>-0.220461</td>
      <td>0.220461</td>
      <td>2.230947</td>
      <td>0.440922</td>
      <td>45302.296781</td>
      <td>0.426213</td>
      <td>-0.213107</td>
      <td>44592.703219</td>
      <td>0.213107</td>
      <td>1.0</td>
      <td>0.864511</td>
    </tr>
    <tr>
      <th>603</th>
      <td>TmY18</td>
      <td>TmY18</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.292609</td>
      <td>0.000000</td>
      <td>45308.099109</td>
      <td>-0.198219</td>
      <td>0.099109</td>
      <td>45307.900891</td>
      <td>-0.099109</td>
      <td>1.0</td>
      <td>2.239571</td>
    </tr>
  </tbody>
</table>
<p>604 rows × 13 columns</p>
</div>




```python
edges.columns
```




    Index(['source_u', 'target_u', 'n_syn', 'du', 'source_type', 'target_type',
           'edge_type', 'source_index', 'dv', 'source_v', 'target_index',
           'target_v', 'sign', 'n_syn_certainty'],
          dtype='object')




```python
edges
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
      <th>source_u</th>
      <th>target_u</th>
      <th>n_syn</th>
      <th>du</th>
      <th>source_type</th>
      <th>target_type</th>
      <th>edge_type</th>
      <th>source_index</th>
      <th>dv</th>
      <th>source_v</th>
      <th>target_index</th>
      <th>target_v</th>
      <th>sign</th>
      <th>n_syn_certainty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-15</td>
      <td>-15</td>
      <td>40.0</td>
      <td>0</td>
      <td>R1</td>
      <td>L1</td>
      <td>chem</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5768</td>
      <td>0</td>
      <td>-1.0</td>
      <td>5.859477</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-15</td>
      <td>-15</td>
      <td>40.0</td>
      <td>0</td>
      <td>R1</td>
      <td>L1</td>
      <td>chem</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>5769</td>
      <td>1</td>
      <td>-1.0</td>
      <td>5.859477</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-15</td>
      <td>-15</td>
      <td>40.0</td>
      <td>0</td>
      <td>R1</td>
      <td>L1</td>
      <td>chem</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>5770</td>
      <td>2</td>
      <td>-1.0</td>
      <td>5.859477</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-15</td>
      <td>-15</td>
      <td>40.0</td>
      <td>0</td>
      <td>R1</td>
      <td>L1</td>
      <td>chem</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>5771</td>
      <td>3</td>
      <td>-1.0</td>
      <td>5.859477</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-15</td>
      <td>-15</td>
      <td>40.0</td>
      <td>0</td>
      <td>R1</td>
      <td>L1</td>
      <td>chem</td>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>5772</td>
      <td>4</td>
      <td>-1.0</td>
      <td>5.859477</td>
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
      <td>15</td>
      <td>15</td>
      <td>1.0</td>
      <td>0</td>
      <td>TmY18</td>
      <td>TmY18</td>
      <td>chem</td>
      <td>45664</td>
      <td>0</td>
      <td>-4</td>
      <td>45664</td>
      <td>-4</td>
      <td>1.0</td>
      <td>2.239571</td>
    </tr>
    <tr>
      <th>1513227</th>
      <td>15</td>
      <td>15</td>
      <td>1.0</td>
      <td>0</td>
      <td>TmY18</td>
      <td>TmY18</td>
      <td>chem</td>
      <td>45665</td>
      <td>0</td>
      <td>-3</td>
      <td>45665</td>
      <td>-3</td>
      <td>1.0</td>
      <td>2.239571</td>
    </tr>
    <tr>
      <th>1513228</th>
      <td>15</td>
      <td>15</td>
      <td>1.0</td>
      <td>0</td>
      <td>TmY18</td>
      <td>TmY18</td>
      <td>chem</td>
      <td>45666</td>
      <td>0</td>
      <td>-2</td>
      <td>45666</td>
      <td>-2</td>
      <td>1.0</td>
      <td>2.239571</td>
    </tr>
    <tr>
      <th>1513229</th>
      <td>15</td>
      <td>15</td>
      <td>1.0</td>
      <td>0</td>
      <td>TmY18</td>
      <td>TmY18</td>
      <td>chem</td>
      <td>45667</td>
      <td>0</td>
      <td>-1</td>
      <td>45667</td>
      <td>-1</td>
      <td>1.0</td>
      <td>2.239571</td>
    </tr>
    <tr>
      <th>1513230</th>
      <td>15</td>
      <td>15</td>
      <td>1.0</td>
      <td>0</td>
      <td>TmY18</td>
      <td>TmY18</td>
      <td>chem</td>
      <td>45668</td>
      <td>0</td>
      <td>0</td>
      <td>45668</td>
      <td>0</td>
      <td>1.0</td>
      <td>2.239571</td>
    </tr>
  </tbody>
</table>
<p>1513231 rows × 14 columns</p>
</div>




```python
elements = ["A", "A", "A", "B", "B", "C", "D", "D", "E"]
groups = ["A", "B", "C", "D", "E"]
parameter = [1, 2, 3, 4, 5]
# get_scatter_indices would return
scatter_indices = [0, 0, 0, 1, 1, 2, 3, 3, 4]
scattered_parameters = [parameter[idx] for idx in scatter_indices] 
```


```python
scattered_parameters
```




    [1, 1, 1, 2, 2, 3, 4, 4, 5]




```python
param2.keys
```




    [('R1', 'L1'),
     ('R1', 'L2'),
     ('R1', 'L3'),
     ('R1', 'Am'),
     ('R1', 'T1'),
     ('R2', 'L1'),
     ('R2', 'L2'),
     ('R2', 'L3'),
     ('R2', 'Am'),
     ('R2', 'T1'),
     ('R3', 'L1'),
     ('R3', 'L2'),
     ('R3', 'L3'),
     ('R3', 'Am'),
     ('R3', 'T1'),
     ('R4', 'L1'),
     ('R4', 'L2'),
     ('R4', 'L3'),
     ('R4', 'Am'),
     ('R4', 'T1'),
     ('R5', 'L1'),
     ('R5', 'L2'),
     ('R5', 'L3'),
     ('R5', 'Am'),
     ('R5', 'T1'),
     ('R6', 'L1'),
     ('R6', 'L2'),
     ('R6', 'L3'),
     ('R6', 'Am'),
     ('R6', 'T1'),
     ('R7', 'R8'),
     ('R7', 'Mi9'),
     ('R7', 'Mi15'),
     ('R7', 'Tm5a'),
     ('R7', 'Tm5b'),
     ('R8', 'R7'),
     ('R8', 'L1'),
     ('R8', 'L3'),
     ('R8', 'Mi1'),
     ('R8', 'Mi4'),
     ('R8', 'Mi9'),
     ('R8', 'Mi15'),
     ('R8', 'Tm5c'),
     ('R8', 'Tm20'),
     ('L1', 'L5'),
     ('L1', 'Lawf2'),
     ('L1', 'C2'),
     ('L1', 'C3'),
     ('L1', 'Mi1'),
     ('L1', 'Tm3'),
     ('L2', 'R1'),
     ('L2', 'R2'),
     ('L2', 'L1'),
     ('L2', 'L4'),
     ('L2', 'L5'),
     ('L2', 'Lawf2'),
     ('L2', 'C3'),
     ('L2', 'Mi1'),
     ('L2', 'Mi4'),
     ('L2', 'Mi9'),
     ('L2', 'T1'),
     ('L2', 'Tm1'),
     ('L2', 'Tm2'),
     ('L2', 'Tm4'),
     ('L2', 'Tm20'),
     ('L2', 'TmY3'),
     ('L3', 'Lawf1'),
     ('L3', 'C3'),
     ('L3', 'Mi1'),
     ('L3', 'Mi2'),
     ('L3', 'Mi9'),
     ('L3', 'Mi13'),
     ('L3', 'Mi15'),
     ('L3', 'Tm3'),
     ('L3', 'Tm5Y'),
     ('L3', 'Tm5a'),
     ('L3', 'Tm5c'),
     ('L3', 'Tm9'),
     ('L3', 'Tm20'),
     ('L3', 'Tm28'),
     ('L3', 'TmY9'),
     ('L3', 'TmY10'),
     ('L4', 'R1'),
     ('L4', 'R2'),
     ('L4', 'R3'),
     ('L4', 'R4'),
     ('L4', 'R5'),
     ('L4', 'R6'),
     ('L4', 'L2'),
     ('L4', 'L3'),
     ('L4', 'L4'),
     ('L4', 'L5'),
     ('L4', 'Am'),
     ('L4', 'Mi9'),
     ('L4', 'T2'),
     ('L4', 'Tm2'),
     ('L4', 'Tm4'),
     ('L4', 'Tm9'),
     ('L4', 'TmY3'),
     ('L4', 'TmY13'),
     ('L5', 'L1'),
     ('L5', 'L2'),
     ('L5', 'L5'),
     ('L5', 'C2'),
     ('L5', 'C3'),
     ('L5', 'Mi1'),
     ('L5', 'Mi2'),
     ('L5', 'Mi4'),
     ('L5', 'Mi15'),
     ('L5', 'T1'),
     ('L5', 'T2'),
     ('L5', 'Tm3'),
     ('L5', 'Tm20'),
     ('L5', 'TmY3'),
     ('L5', 'TmY13'),
     ('L5', 'TmY14'),
     ('L5', 'TmY18'),
     ('Lawf1', 'Tm3'),
     ('Lawf2', 'L3'),
     ('Lawf2', 'Lawf2'),
     ('Lawf2', 'Am'),
     ('Am', 'R1'),
     ('Am', 'R2'),
     ('Am', 'R4'),
     ('Am', 'R5'),
     ('Am', 'L1'),
     ('Am', 'L2'),
     ('Am', 'L3'),
     ('Am', 'L4'),
     ('Am', 'C2'),
     ('Am', 'C3'),
     ('Am', 'T1'),
     ('C2', 'L1'),
     ('C2', 'L2'),
     ('C2', 'L3'),
     ('C2', 'L4'),
     ('C2', 'L5'),
     ('C2', 'Lawf2'),
     ('C2', 'Am'),
     ('C2', 'Mi1'),
     ('C2', 'T1'),
     ('C2', 'T2'),
     ('C2', 'T4a'),
     ('C2', 'Tm1'),
     ('C2', 'Tm9'),
     ('C2', 'TmY14'),
     ('C3', 'R3'),
     ('C3', 'L1'),
     ('C3', 'L2'),
     ('C3', 'L3'),
     ('C3', 'L5'),
     ('C3', 'Lawf2'),
     ('C3', 'Am'),
     ('C3', 'CT1(M10)'),
     ('C3', 'Mi1'),
     ('C3', 'Mi4'),
     ('C3', 'Mi9'),
     ('C3', 'T1'),
     ('C3', 'T2'),
     ('C3', 'T4a'),
     ('C3', 'T4b'),
     ('C3', 'T4c'),
     ('C3', 'T4d'),
     ('C3', 'Tm1'),
     ('C3', 'Tm2'),
     ('C3', 'Tm4'),
     ('C3', 'Tm9'),
     ('C3', 'Tm20'),
     ('CT1(Lo1)', 'T5a'),
     ('CT1(Lo1)', 'T5b'),
     ('CT1(Lo1)', 'T5c'),
     ('CT1(Lo1)', 'T5d'),
     ('CT1(Lo1)', 'Tm1'),
     ('CT1(Lo1)', 'Tm9'),
     ('CT1(M10)', 'C3'),
     ('CT1(M10)', 'Mi1'),
     ('CT1(M10)', 'T4a'),
     ('CT1(M10)', 'T4b'),
     ('CT1(M10)', 'T4c'),
     ('CT1(M10)', 'T4d'),
     ('Mi1', 'L1'),
     ('Mi1', 'L5'),
     ('Mi1', 'C2'),
     ('Mi1', 'C3'),
     ('Mi1', 'CT1(M10)'),
     ('Mi1', 'Mi2'),
     ('Mi1', 'Mi4'),
     ('Mi1', 'Mi9'),
     ('Mi1', 'Mi10'),
     ('Mi1', 'Mi12'),
     ('Mi1', 'Mi13'),
     ('Mi1', 'Mi14'),
     ('Mi1', 'Mi15'),
     ('Mi1', 'T2'),
     ('Mi1', 'T2a'),
     ('Mi1', 'T3'),
     ('Mi1', 'T4a'),
     ('Mi1', 'T4b'),
     ('Mi1', 'T4c'),
     ('Mi1', 'T4d'),
     ('Mi1', 'Tm1'),
     ('Mi1', 'Tm3'),
     ('Mi1', 'Tm20'),
     ('Mi1', 'TmY3'),
     ('Mi1', 'TmY9'),
     ('Mi1', 'TmY13'),
     ('Mi1', 'TmY14'),
     ('Mi1', 'TmY15'),
     ('Mi1', 'TmY18'),
     ('Mi2', 'Mi1'),
     ('Mi2', 'T2'),
     ('Mi2', 'T2a'),
     ('Mi2', 'T3'),
     ('Mi2', 'Tm1'),
     ('Mi2', 'TmY18'),
     ('Mi3', 'T2'),
     ('Mi3', 'Tm5b'),
     ('Mi4', 'L2'),
     ('Mi4', 'C3'),
     ('Mi4', 'Mi1'),
     ('Mi4', 'Mi4'),
     ('Mi4', 'Mi9'),
     ('Mi4', 'T1'),
     ('Mi4', 'T4a'),
     ('Mi4', 'T4b'),
     ('Mi4', 'T4c'),
     ('Mi4', 'T4d'),
     ('Mi4', 'Tm1'),
     ('Mi4', 'Tm2'),
     ('Mi4', 'Tm5Y'),
     ('Mi4', 'Tm5a'),
     ('Mi4', 'Tm5b'),
     ('Mi4', 'Tm5c'),
     ('Mi4', 'Tm9'),
     ('Mi4', 'Tm16'),
     ('Mi4', 'Tm20'),
     ('Mi4', 'Tm28'),
     ('Mi4', 'TmY3'),
     ('Mi4', 'TmY4'),
     ('Mi4', 'TmY5a'),
     ('Mi4', 'TmY10'),
     ('Mi4', 'TmY13'),
     ('Mi4', 'TmY15'),
     ('Mi4', 'TmY18'),
     ('Mi9', 'L3'),
     ('Mi9', 'Lawf1'),
     ('Mi9', 'C3'),
     ('Mi9', 'CT1(M10)'),
     ('Mi9', 'Mi1'),
     ('Mi9', 'Mi4'),
     ('Mi9', 'Mi10'),
     ('Mi9', 'Mi15'),
     ('Mi9', 'T2'),
     ('Mi9', 'T2a'),
     ('Mi9', 'T4a'),
     ('Mi9', 'T4b'),
     ('Mi9', 'T4c'),
     ('Mi9', 'T4d'),
     ('Mi9', 'Tm1'),
     ('Mi9', 'Tm2'),
     ('Mi9', 'Tm4'),
     ('Mi9', 'Tm5a'),
     ('Mi9', 'Tm5b'),
     ('Mi9', 'Tm16'),
     ('Mi9', 'TmY3'),
     ('Mi9', 'TmY4'),
     ('Mi9', 'TmY5a'),
     ('Mi9', 'TmY13'),
     ('Mi9', 'TmY18'),
     ('Mi10', 'Lawf1'),
     ('Mi10', 'Mi9'),
     ('Mi10', 'Mi14'),
     ('Mi10', 'T4a'),
     ('Mi10', 'T4b'),
     ('Mi10', 'T4c'),
     ('Mi10', 'T4d'),
     ('Mi10', 'Tm5b'),
     ('Mi10', 'TmY3'),
     ('Mi10', 'TmY5a'),
     ('Mi10', 'TmY14'),
     ('Mi12', 'Mi1'),
     ('Mi12', 'Mi9'),
     ('Mi12', 'Tm1'),
     ('Mi12', 'Tm2'),
     ('Mi12', 'Tm3'),
     ('Mi12', 'Tm4'),
     ('Mi13', 'L5'),
     ('Mi13', 'Mi1'),
     ('Mi13', 'T2'),
     ('Mi13', 'Tm1'),
     ('Mi13', 'Tm2'),
     ('Mi13', 'Tm3'),
     ('Mi13', 'Tm4'),
     ('Mi13', 'Tm9'),
     ('Mi14', 'Tm1'),
     ('Mi14', 'Tm4'),
     ('Mi14', 'TmY3'),
     ('Mi15', 'C3'),
     ('Mi15', 'Mi4'),
     ('Mi15', 'Mi10'),
     ('Mi15', 'Mi15'),
     ('Mi15', 'Tm5c'),
     ('T1', 'L2'),
     ('T1', 'L5'),
     ('T1', 'Tm20'),
     ('T2', 'Lawf2'),
     ('T2', 'Mi1'),
     ('T2', 'T2'),
     ('T2', 'Tm5c'),
     ('T2', 'TmY5a'),
     ('T2', 'TmY15'),
     ('T2a', 'Mi1'),
     ('T2a', 'T3'),
     ('T2a', 'Tm3'),
     ('T2a', 'Tm5Y'),
     ('T2a', 'Tm5b'),
     ('T2a', 'Tm28'),
     ('T2a', 'TmY4'),
     ('T2a', 'TmY5a'),
     ('T2a', 'TmY9'),
     ('T3', 'T2'),
     ('T3', 'T2a'),
     ('T3', 'Tm3'),
     ('T4a', 'C3'),
     ('T4a', 'CT1(M10)'),
     ('T4a', 'Mi9'),
     ('T4a', 'Mi12'),
     ('T4a', 'T4a'),
     ('T4a', 'T4c'),
     ('T4a', 'T5a'),
     ('T4a', 'TmY14'),
     ('T4a', 'TmY18'),
     ('T4b', 'CT1(M10)'),
     ('T4b', 'Mi9'),
     ('T4b', 'T4b'),
     ('T4b', 'T4c'),
     ('T4b', 'T5b'),
     ('T4b', 'TmY15'),
     ('T4c', 'CT1(M10)'),
     ('T4c', 'Mi9'),
     ('T4c', 'T4c'),
     ('T4c', 'T5c'),
     ('T4c', 'TmY4'),
     ('T4c', 'TmY14'),
     ('T4c', 'TmY15'),
     ('T4d', 'CT1(M10)'),
     ('T4d', 'T4a'),
     ('T4d', 'T4d'),
     ('T4d', 'T5d'),
     ('T4d', 'TmY4'),
     ('T4d', 'TmY15'),
     ('T5a', 'CT1(Lo1)'),
     ('T5a', 'T4a'),
     ('T5a', 'T5a'),
     ('T5a', 'T5b'),
     ('T5a', 'TmY9'),
     ('T5a', 'TmY14'),
     ('T5a', 'TmY15'),
     ('T5b', 'CT1(Lo1)'),
     ('T5b', 'T2'),
     ('T5b', 'T4b'),
     ('T5b', 'T5b'),
     ('T5b', 'T5d'),
     ('T5b', 'Tm2'),
     ('T5b', 'TmY14'),
     ('T5b', 'TmY15'),
     ('T5c', 'CT1(Lo1)'),
     ('T5c', 'T4c'),
     ('T5c', 'T5c'),
     ('T5c', 'Tm2'),
     ('T5c', 'TmY4'),
     ('T5c', 'TmY14'),
     ('T5c', 'TmY15'),
     ('T5d', 'CT1(Lo1)'),
     ('T5d', 'T2'),
     ('T5d', 'T4d'),
     ('T5d', 'T5d'),
     ('T5d', 'TmY4'),
     ('T5d', 'TmY15'),
     ('Tm1', 'L2'),
     ('Tm1', 'L5'),
     ('Tm1', 'Lawf2'),
     ('Tm1', 'C3'),
     ('Tm1', 'CT1(Lo1)'),
     ('Tm1', 'Mi1'),
     ('Tm1', 'Mi2'),
     ('Tm1', 'Mi4'),
     ('Tm1', 'Mi9'),
     ('Tm1', 'Mi12'),
     ('Tm1', 'Mi13'),
     ('Tm1', 'T1'),
     ('Tm1', 'T2'),
     ('Tm1', 'T2a'),
     ('Tm1', 'T3'),
     ('Tm1', 'T5a'),
     ('Tm1', 'T5b'),
     ('Tm1', 'T5c'),
     ('Tm1', 'T5d'),
     ('Tm1', 'Tm2'),
     ('Tm1', 'Tm4'),
     ('Tm1', 'Tm5Y'),
     ('Tm1', 'Tm9'),
     ('Tm1', 'Tm20'),
     ('Tm1', 'Tm28'),
     ('Tm1', 'TmY4'),
     ('Tm1', 'TmY9'),
     ('Tm1', 'TmY15'),
     ('Tm2', 'L2'),
     ('Tm2', 'L5'),
     ('Tm2', 'Lawf1'),
     ('Tm2', 'C3'),
     ('Tm2', 'CT1(Lo1)'),
     ('Tm2', 'Mi4'),
     ('Tm2', 'Mi9'),
     ('Tm2', 'T2'),
     ('Tm2', 'T2a'),
     ('Tm2', 'T5a'),
     ('Tm2', 'T5b'),
     ('Tm2', 'T5c'),
     ('Tm2', 'T5d'),
     ('Tm2', 'Tm1'),
     ('Tm2', 'Tm4'),
     ('Tm2', 'Tm9'),
     ('Tm2', 'Tm28'),
     ('Tm2', 'TmY3'),
     ('Tm2', 'TmY4'),
     ('Tm2', 'TmY13'),
     ('Tm2', 'TmY15'),
     ('Tm2', 'TmY18'),
     ('Tm3', 'L1'),
     ('Tm3', 'Lawf1'),
     ('Tm3', 'Lawf2'),
     ('Tm3', 'Mi1'),
     ('Tm3', 'Mi11'),
     ('Tm3', 'Mi14'),
     ('Tm3', 'T2'),
     ('Tm3', 'T2a'),
     ('Tm3', 'T3'),
     ('Tm3', 'T4a'),
     ('Tm3', 'T4b'),
     ('Tm3', 'T4c'),
     ('Tm3', 'T4d'),
     ('Tm3', 'Tm3'),
     ('Tm3', 'Tm4'),
     ('Tm3', 'Tm30'),
     ('Tm3', 'TmY3'),
     ('Tm3', 'TmY5a'),
     ('Tm3', 'TmY13'),
     ('Tm3', 'TmY14'),
     ('Tm3', 'TmY15'),
     ('Tm3', 'TmY18'),
     ('Tm4', 'Lawf2'),
     ('Tm4', 'C3'),
     ('Tm4', 'Mi4'),
     ('Tm4', 'Mi10'),
     ('Tm4', 'Mi12'),
     ('Tm4', 'Mi13'),
     ('Tm4', 'Mi14'),
     ('Tm4', 'T2'),
     ('Tm4', 'T2a'),
     ('Tm4', 'T3'),
     ('Tm4', 'T5a'),
     ('Tm4', 'T5b'),
     ('Tm4', 'T5c'),
     ('Tm4', 'T5d'),
     ('Tm4', 'Tm3'),
     ('Tm4', 'Tm4'),
     ('Tm4', 'TmY3'),
     ('Tm4', 'TmY4'),
     ('Tm4', 'TmY5a'),
     ('Tm4', 'TmY13'),
     ('Tm4', 'TmY14'),
     ('Tm4', 'TmY15'),
     ('Tm5Y', 'Tm5a'),
     ('Tm5Y', 'Tm5b'),
     ('Tm5Y', 'Tm20'),
     ('Tm5Y', 'TmY5a'),
     ('Tm5Y', 'TmY10'),
     ('Tm5a', 'Tm5a'),
     ('Tm5a', 'Tm5b'),
     ('Tm5b', 'Mi3'),
     ('Tm5b', 'Mi10'),
     ('Tm5b', 'Tm5a'),
     ('Tm5b', 'Tm5b'),
     ('Tm5b', 'Tm5c'),
     ('Tm5c', 'Mi4'),
     ('Tm5c', 'Mi15'),
     ('Tm5c', 'Tm5Y'),
     ('Tm5c', 'Tm5b'),
     ('Tm5c', 'Tm16'),
     ('Tm9', 'CT1(Lo1)'),
     ('Tm9', 'Mi9'),
     ('Tm9', 'T5a'),
     ('Tm9', 'T5b'),
     ('Tm9', 'T5c'),
     ('Tm9', 'T5d'),
     ('Tm9', 'Tm1'),
     ('Tm9', 'Tm2'),
     ('Tm9', 'Tm5Y'),
     ('Tm9', 'Tm28'),
     ('Tm9', 'TmY4'),
     ('Tm9', 'TmY9'),
     ('Tm9', 'TmY15'),
     ('Tm16', 'Lawf1'),
     ('Tm16', 'Lawf2'),
     ('Tm16', 'Mi9'),
     ('Tm16', 'T2'),
     ('Tm16', 'Tm1'),
     ('Tm16', 'Tm2'),
     ('Tm16', 'Tm4'),
     ('Tm16', 'Tm5Y'),
     ('Tm16', 'Tm5c'),
     ('Tm16', 'Tm9'),
     ('Tm16', 'TmY3'),
     ('Tm16', 'TmY5a'),
     ('Tm16', 'TmY10'),
     ('Tm16', 'TmY13'),
     ('Tm16', 'TmY14'),
     ('Tm20', 'Mi2'),
     ('Tm20', 'Mi9'),
     ('Tm20', 'Tm5Y'),
     ('Tm20', 'Tm9'),
     ('Tm20', 'Tm20'),
     ('Tm28', 'Tm5Y'),
     ('Tm28', 'TmY4'),
     ('Tm28', 'TmY9'),
     ('TmY3', 'T2'),
     ('TmY3', 'Tm4'),
     ('TmY3', 'TmY3'),
     ('TmY3', 'TmY4'),
     ('TmY3', 'TmY5a'),
     ('TmY3', 'TmY13'),
     ('TmY3', 'TmY14'),
     ('TmY3', 'TmY15'),
     ('TmY4', 'Mi3'),
     ('TmY4', 'Mi4'),
     ('TmY4', 'Mi13'),
     ('TmY4', 'T2'),
     ('TmY4', 'T5d'),
     ('TmY4', 'TmY4'),
     ('TmY4', 'TmY5a'),
     ('TmY4', 'TmY9'),
     ('TmY4', 'TmY14'),
     ('TmY5a', 'Mi2'),
     ('TmY5a', 'Mi3'),
     ('TmY5a', 'T2'),
     ('TmY5a', 'T2a'),
     ('TmY5a', 'Tm16'),
     ('TmY5a', 'TmY3'),
     ('TmY5a', 'TmY4'),
     ('TmY5a', 'TmY14'),
     ('TmY5a', 'TmY15'),
     ('TmY5a', 'TmY18'),
     ('TmY9', 'Mi13'),
     ('TmY9', 'TmY4'),
     ('TmY9', 'TmY9'),
     ('TmY10', 'Mi4'),
     ('TmY10', 'T2a'),
     ('TmY10', 'Tm5b'),
     ('TmY10', 'Tm5c'),
     ('TmY10', 'Tm9'),
     ('TmY10', 'Tm16'),
     ('TmY10', 'Tm20'),
     ('TmY10', 'Tm28'),
     ('TmY10', 'TmY4'),
     ('TmY10', 'TmY9'),
     ('TmY13', 'Mi4'),
     ('TmY13', 'T2'),
     ('TmY13', 'Tm5b'),
     ('TmY13', 'Tm16'),
     ('TmY13', 'TmY5a'),
     ('TmY14', 'C3'),
     ('TmY14', 'Mi12'),
     ('TmY14', 'Mi13'),
     ('TmY14', 'Tm9'),
     ('TmY14', 'Tm16'),
     ('TmY14', 'TmY4'),
     ('TmY15', 'Mi4'),
     ('TmY15', 'Mi9'),
     ('TmY15', 'T2'),
     ('TmY15', 'T2a'),
     ('TmY15', 'T4a'),
     ('TmY15', 'T4b'),
     ('TmY15', 'T4c'),
     ('TmY15', 'T4d'),
     ('TmY15', 'T5a'),
     ('TmY15', 'T5b'),
     ('TmY15', 'T5c'),
     ('TmY15', 'T5d'),
     ('TmY15', 'Tm1'),
     ('TmY15', 'Tm2'),
     ('TmY15', 'Tm3'),
     ('TmY15', 'Tm4'),
     ('TmY15', 'Tm9'),
     ('TmY15', 'TmY3'),
     ('TmY15', 'TmY14'),
     ('TmY15', 'TmY18'),
     ('TmY18', 'Mi4'),
     ('TmY18', 'Mi10'),
     ('TmY18', 'T2'),
     ('TmY18', 'TmY5a'),
     ('TmY18', 'TmY9'),
     ('TmY18', 'TmY15'),
     ('TmY18', 'TmY18')]




```python
from flyvision.utils.tensor_utils import matrix_mask_by_sub, where_equal_rows
```


```python
keys = np.array(param2.keys)
```


```python
keys[matrix_mask_by_sub(np.array([("R4", "Am"), ("Mi9", "T4a")]), np.array(param2.keys))]
```




    array([['R4', 'Am'],
           ['Mi9', 'T4a']], dtype='<U8')




```python
matrix_mask_by_sub(np.array([("R4", "Am"), ("Mi9", "T4a")]), np.array(param2.keys))
```




    array([False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
            True, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False,  True, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False])




```python
np.array([("R4", "Am"), ("Mi9", "T4a")]).shape
```




    (2, 2)




```python
np.array(param2.keys).shape
```




    (65,)




```python
np.array(param2.keys)[where_equal_rows(np.array([("R4", "Am"), ("Mi9", "T4a")]), np.array(param2.keys))]
```




    array([['R4', 'Am'],
           ['Mi9', 'T4a']], dtype='<U8')




```python
where_equal_rows??
```


```python
edges.groupby(["source_type", "target_type"], sort=False, as_index=False, group_keys=["source_type", "target_type"], observed=["source_type", "target_type", "edge_type"]).mean(["n_syn"])
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
      <th>source_type</th>
      <th>target_type</th>
      <th>source_u</th>
      <th>target_u</th>
      <th>n_syn</th>
      <th>du</th>
      <th>source_index</th>
      <th>dv</th>
      <th>source_v</th>
      <th>target_index</th>
      <th>target_v</th>
      <th>sign</th>
      <th>n_syn_certainty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>R1</td>
      <td>L1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
      <td>0.000000</td>
      <td>360.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6128.000000</td>
      <td>0.000000</td>
      <td>-1.0</td>
      <td>5.859477</td>
    </tr>
    <tr>
      <th>1</th>
      <td>R1</td>
      <td>L2</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>46.000000</td>
      <td>0.000000</td>
      <td>360.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6849.000000</td>
      <td>0.000000</td>
      <td>-1.0</td>
      <td>5.859477</td>
    </tr>
    <tr>
      <th>2</th>
      <td>R1</td>
      <td>L3</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>360.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7570.000000</td>
      <td>0.000000</td>
      <td>-1.0</td>
      <td>5.859477</td>
    </tr>
    <tr>
      <th>3</th>
      <td>R1</td>
      <td>Am</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>36.000000</td>
      <td>0.000000</td>
      <td>360.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9979.000000</td>
      <td>0.000000</td>
      <td>-1.0</td>
      <td>5.859477</td>
    </tr>
    <tr>
      <th>4</th>
      <td>R1</td>
      <td>T1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>360.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>21515.000000</td>
      <td>0.000000</td>
      <td>-1.0</td>
      <td>5.859477</td>
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
    </tr>
    <tr>
      <th>599</th>
      <td>TmY18</td>
      <td>T2</td>
      <td>0.244507</td>
      <td>-0.244507</td>
      <td>2.364281</td>
      <td>-0.489015</td>
      <td>45313.822112</td>
      <td>0.489015</td>
      <td>-0.244507</td>
      <td>22230.177888</td>
      <td>0.244507</td>
      <td>1.0</td>
      <td>1.587606</td>
    </tr>
    <tr>
      <th>600</th>
      <td>TmY18</td>
      <td>TmY5a</td>
      <td>-0.244507</td>
      <td>0.244507</td>
      <td>1.656361</td>
      <td>0.489015</td>
      <td>45302.177888</td>
      <td>-0.489015</td>
      <td>0.244507</td>
      <td>40987.822112</td>
      <td>-0.244507</td>
      <td>1.0</td>
      <td>5.517218</td>
    </tr>
    <tr>
      <th>601</th>
      <td>TmY18</td>
      <td>TmY9</td>
      <td>1.500000</td>
      <td>-1.500000</td>
      <td>1.000000</td>
      <td>-3.000000</td>
      <td>45344.787302</td>
      <td>2.000000</td>
      <td>-1.000000</td>
      <td>41666.212698</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>2.782612</td>
    </tr>
    <tr>
      <th>602</th>
      <td>TmY18</td>
      <td>TmY15</td>
      <td>-0.220461</td>
      <td>0.220461</td>
      <td>2.230947</td>
      <td>0.440922</td>
      <td>45302.296781</td>
      <td>0.426213</td>
      <td>-0.213107</td>
      <td>44592.703219</td>
      <td>0.213107</td>
      <td>1.0</td>
      <td>0.864511</td>
    </tr>
    <tr>
      <th>603</th>
      <td>TmY18</td>
      <td>TmY18</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.292609</td>
      <td>0.000000</td>
      <td>45308.099109</td>
      <td>-0.198219</td>
      <td>0.099109</td>
      <td>45307.900891</td>
      <td>-0.099109</td>
      <td>1.0</td>
      <td>2.239571</td>
    </tr>
  </tbody>
</table>
<p>604 rows × 13 columns</p>
</div>




```python
Parameter.__subclasses__()
```




    [flyvision.initialization.RestingPotential,
     flyvision.initialization.TimeConstant,
     flyvision.initialization.SynapseSign,
     flyvision.initialization.SynapseCount,
     flyvision.initialization.SynapseCountScaling,
     __main__.RestingPotentialv2,
     __main__.RestingPotentialv2,
     __main__.RestingPotentialv2,
     __main__.TimeConstantv2,
     __main__.TimeConstantv2,
     __main__.TimeConstantv2]




```python
class A:
    pass
```


```python
class B(A):
    pass
```


```python
for cls in Parameter.__subclasses__():
    cls.__bases__ = (B,)
```


```python
Parameter.__subclasses__()
```




    []




```python
from flyvision import results_dir
from flyvision.ensemble import EnsembleView, EnsembleDir
```


```python
ensemble = EnsembleDir(results_dir / "opticflow/000")
```


```python
old_config = Namespace(
  type = 'NetworkDir',
  network = Namespace(
    connectome = Namespace(
      type = 'ConnectomeDir',
      file = 'fib25-fib19_v2.2.json',
      extent = 15,
      n_syn_fill = 1
    ),
    dynamics = Namespace(
      type = 'PPNeuronIGRSynapses',
      activation = Namespace(type='relu')
    ),
    node_config = Namespace(
      bias = Namespace(
        type = 'RestingPotential',
        keys = ['type'],
        initial_dist = 'Normal',
        mode = 'sample',
        requires_grad = True,
        mean = 0.5,
        std = 0.05,
        penalize = Namespace(activity=True),
        seed = 0
      ),
      time_const = Namespace(
        type = 'TimeConstant',
        keys = ['type'],
        initial_dist = 'Value',
        value = 0.05,
        requires_grad = True
      )
    ),
    edge_config = Namespace(
      sign = Namespace(
        type = 'SynapseSign',
        initial_dist = 'Value',
        requires_grad = False
      ),
      syn_count = Namespace(
        type = 'SynapseCount',
        initial_dist = 'Lognormal',
        mode = 'mean',
        requires_grad = False,
        std = 1.0
      ),
      syn_strength = Namespace(
        type = 'SynapseCountScaling',
        initial_dist = 'Value',
        requires_grad = True,
        scale_elec = 0.01,
        scale_chem = 0.01,
        clamp = 'non_negative'
      )
    )
  ),
  task_dataset = Namespace(
    type = 'MultiTaskSintel',
    tasks = ['flow'],
    n_frames = 19,
    dt_sampling = True,
    interpolate = True,
    boxfilter = Namespace(extent=15, kernel_size=13),
    vertical_splits = 3,
    p_flip = 0.5,
    p_rot = 0.5,
    contrast = 0.2,
    brightness = 0.1,
    noise = 0.08,
    cache = 'gpu',
    gamma = 1.0
  ),
  task_decoder = Namespace(
    type = 'DecoderGAVP',
    shape = [8, 2],
    kernel_size = 5,
    const_weight = 0.001,
    n_out_features = None,
    p_dropout = 0.5
  ),
  task_loss = Namespace(type='mean_root_norm')
)
```


```python
new_config = Namespace(
  type = 'NetworkDir',
  network = Namespace(
    connectome = Namespace(
      type = 'ConnectomeDir',
      file = 'fib25-fib19_v2.2.json',
      extent = 15,
      n_syn_fill = 1
    ),
    dynamics = Namespace(
      type = 'PPNeuronIGRSynapses',
      activation = Namespace(type='relu')
    ),
    node_config = Namespace(
      bias = Namespace(
        type = 'RestingPotential',
        groupby=["type"],
        initial_dist = 'Normal',
        mode = 'sample',
        requires_grad = True,
        mean = 0.5,
        std = 0.05,
        penalize = Namespace(activity=True),
        seed = 0
      ),
      time_const = Namespace(
        type = 'TimeConstant',
        groupby=["type"],
        initial_dist = 'Value',
        value = 0.05,
        requires_grad = True
      )
    ),
    edge_config = Namespace(
      sign = Namespace(
        type = 'SynapseSign',
        initial_dist = 'Value',
        requires_grad = False,
        groupby=["source_type", "target_type"],
      ),
      syn_count = Namespace(
        type = 'SynapseCount',
        initial_dist = 'Lognormal',
        mode = 'mean',
        requires_grad = False,
        std = 1.0,
        groupby=["source_type", "target_type", "du", "dv"],
      ),
      syn_strength = Namespace(
        type = 'SynapseCountScaling',
        initial_dist = 'Value',
        requires_grad = True,
        scale_elec = 0.01,
        scale_chem = 0.01,
        clamp = 'non_negative',
        groupby=["source_type", "target_type", "edge_type"],
      )
    )
  ),
  task_dataset = Namespace(
    type = 'MultiTaskSintel',
    tasks = ['flow'],
    n_frames = 19,
    dt_sampling = True,
    interpolate = True,
    boxfilter = Namespace(extent=15, kernel_size=13),
    vertical_splits = 3,
    p_flip = 0.5,
    p_rot = 0.5,
    contrast = 0.2,
    brightness = 0.1,
    noise = 0.08,
    cache = 'gpu',
    gamma = 1.0
  ),
  task_decoder = Namespace(
    type = 'DecoderGAVP',
    shape = [8, 2],
    kernel_size = 5,
    const_weight = 0.001,
    n_out_features = None,
    p_dropout = 0.5
  ),
  task_loss = Namespace(type='norm')
)
```


```python
config = None
for name, _dir in ensemble.items():
    if name.isnumeric():
        print(name)
        if config is None:
            config = _dir.config
        assert _dir.config == config
        _dir._override_config(new_config)
```

    0003
    0004
    0032
    0035
    0034
    0033
    0005
    0002
    0020
    0018
    0027
    0011
    0029
    0016
    0042
    0045
    0028
    0017
    0010
    0019
    0026
    0021
    0044
    0043
    0007
    0038
    0000
    0036
    0009
    0031
    0030
    0037
    0008
    0001
    0006
    0039
    0046
    0041
    0048
    0024
    0023
    0015
    0012
    0049
    0040
    0047
    0013
    0014
    0022
    0025


    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)
    /var/folders/b1/s01d0fxj4r56mc22xx01vnd9c7405m/T/ipykernel_70259/2707961450.py:8: ConfigWarning: Overriding config. Diff is:Namespace(
      passed = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type', 'edge_type']"
      ],
      stored = [
        "≠network.edge_config.syn_strength.groupby: ['source_type', 'target_type']"
      ]
    )
      _dir._override_config(new_config)



```python
_dir._override_config()
```




    0025/ - Last modified: March 23, 2023 11:22:29
    ├── _meta.yaml
    ├── best_chkpt
    └── validation_loss.h5
    
    displaying: 1 directory, 3 files




```python
x = np.arange(10)
```


```python
x[slice(-1, -2, -1)]
```




    array([9])




```python
x.tolist()[slice(-1, -1, -1)]
```




    []




```python

```




    slice(-1, -2, -1)




```python
import wget
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Input In [164], in <cell line: 1>()
    ----> 1 import wget


    ModuleNotFoundError: No module named 'wget'



```python
import tqdm
```


```python

```
