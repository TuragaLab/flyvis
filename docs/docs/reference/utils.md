# Utils

Utility modules.

## flyvision.utils.activity_utils

This module contains classes for handling various types of neural activity.

### Classes

::: flyvision.utils.activity_utils.CellTypeActivity
    options:
      heading_level: 4

::: flyvision.utils.activity_utils.CentralActivity
    options:
      heading_level: 4

::: flyvision.utils.activity_utils.LayerActivity
    options:
      heading_level: 4

::: flyvision.utils.activity_utils.SourceCurrentView
    options:
      heading_level: 4

## flyvision.utils.cache_utils

This module provides utility functions for caching.

### Functions

::: flyvision.utils.cache_utils.context_aware_cache
    options:
      heading_level: 4

::: flyvision.utils.cache_utils.make_hashable
    options:
      heading_level: 4

## flyvision.utils.chkpt_utils

This module contains utilities for handling checkpoints.

### Classes

::: flyvision.utils.chkpt_utils.Checkpoints
    options:
      heading_level: 4

### Functions

::: flyvision.utils.chkpt_utils.recover_network
    options:
      heading_level: 4

::: flyvision.utils.chkpt_utils.recover_decoder
    options:
      heading_level: 4

::: flyvision.utils.chkpt_utils.recover_optimizer
    options:
      heading_level: 4

::: flyvision.utils.chkpt_utils.recover_penalty_optimizers
    options:
      heading_level: 4

::: flyvision.utils.chkpt_utils.get_from_state_dict
    options:
      heading_level: 4

::: flyvision.utils.chkpt_utils.resolve_checkpoints
    options:
      heading_level: 4

::: flyvision.utils.chkpt_utils.checkpoint_index_to_path_map
    options:
      heading_level: 4

::: flyvision.utils.chkpt_utils.best_checkpoint_default_fn
    options:
      heading_level: 4

::: flyvision.utils.chkpt_utils.check_loss_name
    options:
      heading_level: 4

## flyvision.utils.class_utils

This module provides utility functions for working with classes.

### Functions

::: flyvision.utils.class_utils.find_subclass
    options:
      heading_level: 4

::: flyvision.utils.class_utils.forward_subclass
    options:
      heading_level: 4

## flyvision.utils.color_utils

This module contains utilities for color manipulation and mapping.

### Classes

::: flyvision.utils.color_utils.cmap_iter
    options:
      heading_level: 4

### Functions

::: flyvision.utils.color_utils.is_hex
    options:
      heading_level: 4

::: flyvision.utils.color_utils.is_integer_rgb
    options:
      heading_level: 4

::: flyvision.utils.color_utils.single_color_cmap
    options:
      heading_level: 4

::: flyvision.utils.color_utils.color_to_cmap
    options:
      heading_level: 4

::: flyvision.utils.color_utils.get_alpha_colormap
    options:
      heading_level: 4

::: flyvision.utils.color_utils.adapt_color_alpha
    options:
      heading_level: 4

::: flyvision.utils.color_utils.flash_response_color_labels
    options:
      heading_level: 4

::: flyvision.utils.color_utils.truncate_colormap
    options:
      heading_level: 4

## flyvision.utils.compute_cloud_utils

This module provides utilities for managing cluster computations.

### Classes

::: flyvision.utils.compute_cloud_utils.ClusterManager
    options:
      heading_level: 4

::: flyvision.utils.compute_cloud_utils.LSFManager
    options:
      heading_level: 4

::: flyvision.utils.compute_cloud_utils.SLURMManager
    options:
      heading_level: 4

### Functions

::: flyvision.utils.compute_cloud_utils.get_cluster_manager
    options:
      heading_level: 4

::: flyvision.utils.compute_cloud_utils.run_job
    options:
      heading_level: 4

::: flyvision.utils.compute_cloud_utils.is_running
    options:
      heading_level: 4

::: flyvision.utils.compute_cloud_utils.kill_job
    options:
      heading_level: 4

::: flyvision.utils.compute_cloud_utils.wait_for_single
    options:
      heading_level: 4

::: flyvision.utils.compute_cloud_utils.wait_for_many
    options:
      heading_level: 4

::: flyvision.utils.compute_cloud_utils.check_valid_host
    options:
      heading_level: 4

::: flyvision.utils.compute_cloud_utils.launch_range
    options:
      heading_level: 4

::: flyvision.utils.compute_cloud_utils.launch_single
    options:
      heading_level: 4

## flyvision.utils.config_utils

This module provides utilities for configuration management.

### Classes

::: flyvision.utils.config_utils.HybridArgumentParser
    options:
      heading_level: 4

### Functions

::: flyvision.utils.config_utils.get_default_config
    options:
      heading_level: 4

::: flyvision.utils.config_utils.parse_kwargs_to_dict
    options:
      heading_level: 4

::: flyvision.utils.config_utils.safe_cast
    options:
      heading_level: 4

## flyvision.utils.dataset_utils

This module contains utilities for dataset manipulation and loading.

### Classes

::: flyvision.utils.dataset_utils.CrossValIndices
    options:
      heading_level: 4

::: flyvision.utils.dataset_utils.IndexSampler
    options:
      heading_level: 4

### Functions

::: flyvision.utils.dataset_utils.random_walk_of_blocks
    options:
      heading_level: 4

::: flyvision.utils.dataset_utils.load_moving_mnist
    options:
      heading_level: 4

::: flyvision.utils.dataset_utils.download_sintel
    options:
      heading_level: 4

::: flyvision.utils.dataset_utils.get_random_data_split
    options:
      heading_level: 4

## flyvision.utils.df_utils

This module provides utilities for working with pandas DataFrames.

### Functions

::: flyvision.utils.df_utils.filter_by_column_values
    options:
      heading_level: 4

::: flyvision.utils.df_utils.where_dataframe
    options:
      heading_level: 4

## flyvision.utils.hex_utils

This module contains utilities for working with hexagonal grids.

### Classes

::: flyvision.utils.hex_utils.Hexal
    options:
      heading_level: 4

::: flyvision.utils.hex_utils.HexArray
    options:
      heading_level: 4

::: flyvision.utils.hex_utils.HexLattice
    options:
      heading_level: 4

::: flyvision.utils.hex_utils.LatticeMask
    options:
      heading_level: 4

### Functions

::: flyvision.utils.hex_utils.get_hex_coords
    options:
      heading_level: 4

::: flyvision.utils.hex_utils.hex_to_pixel
    options:
      heading_level: 4

::: flyvision.utils.hex_utils.hex_rows
    options:
      heading_level: 4

::: flyvision.utils.hex_utils.pixel_to_hex
    options:
      heading_level: 4

::: flyvision.utils.hex_utils.pad_to_regular_hex
    options:
      heading_level: 4

::: flyvision.utils.hex_utils.max_extent_index
    options:
      heading_level: 4

::: flyvision.utils.hex_utils.get_num_hexals
    options:
      heading_level: 4

::: flyvision.utils.hex_utils.get_hextent
    options:
      heading_level: 4

::: flyvision.utils.hex_utils.sort_u_then_v
    options:
      heading_level: 4

::: flyvision.utils.hex_utils.sort_u_then_v_index
    options:
      heading_level: 4

::: flyvision.utils.hex_utils.get_extent
    options:
      heading_level: 4

## flyvision.utils.log_utils

This module provides utilities for logging and status tracking.

### Classes

::: flyvision.utils.log_utils.Status
    options:
      heading_level: 4

### Functions

::: flyvision.utils.log_utils.find_host
    options:
      heading_level: 4

::: flyvision.utils.log_utils.get_exclude_host_part
    options:
      heading_level: 4

::: flyvision.utils.log_utils.get_status
    options:
      heading_level: 4

::: flyvision.utils.log_utils.flatten_list
    options:
      heading_level: 4

## flyvision.utils.logging_utils

This module contains additional logging utilities.

### Functions

::: flyvision.utils.logging_utils.warn_once
    options:
      heading_level: 4

::: flyvision.utils.logging_utils.save_conda_environment
    options:
      heading_level: 4

::: flyvision.utils.logging_utils.all_logging_disabled
    options:
      heading_level: 4

## flyvision.utils.nn_utils

This module provides utilities for neural network operations.

### Classes

::: flyvision.utils.nn_utils.NumberOfParams
    options:
      heading_level: 4

### Functions

::: flyvision.utils.nn_utils.simulation
    options:
      heading_level: 4

::: flyvision.utils.nn_utils.n_params
    options:
      heading_level: 4

## flyvision.utils.nodes_edges_utils

This module contains utilities for working with graph nodes and edges.

### Classes

::: flyvision.utils.nodes_edges_utils.NodeIndexer
    options:
      heading_level: 4

::: flyvision.utils.nodes_edges_utils.CellTypeArray
    options:
      heading_level: 4

### Functions

::: flyvision.utils.nodes_edges_utils.order_node_type_list
    options:
      heading_level: 4

::: flyvision.utils.nodes_edges_utils.get_index_mapping_lists
    options:
      heading_level: 4

::: flyvision.utils.nodes_edges_utils.sort_by_mapping_lists
    options:
      heading_level: 4

::: flyvision.utils.nodes_edges_utils.nodes_list_sorting_on_off_unknown
    options:
      heading_level: 4

## flyvision.utils.tensor_utils

This module provides utilities for tensor operations.

### Classes

::: flyvision.utils.tensor_utils.RefTensor
    options:
      heading_level: 4

::: flyvision.utils.tensor_utils.AutoDeref
    options:
      heading_level: 4

### Functions

::: flyvision.utils.tensor_utils.detach
    options:
      heading_level: 4

::: flyvision.utils.tensor_utils.clone
    options:
      heading_level: 4

::: flyvision.utils.tensor_utils.to_numpy
    options:
      heading_level: 4

::: flyvision.utils.tensor_utils.atleast_column_vector
    options:
      heading_level: 4

::: flyvision.utils.tensor_utils.matrix_mask_by_sub
    options:
      heading_level: 4

::: flyvision.utils.tensor_utils.where_equal_rows
    options:
      heading_level: 4

::: flyvision.utils.tensor_utils.broadcast
    options:
      heading_level: 4

::: flyvision.utils.tensor_utils.scatter_reduce
    options:
      heading_level: 4

::: flyvision.utils.tensor_utils.scatter_mean
    options:
      heading_level: 4

::: flyvision.utils.tensor_utils.scatter_add
    options:
      heading_level: 4

::: flyvision.utils.tensor_utils.select_along_axes
    options:
      heading_level: 4

::: flyvision.utils.tensor_utils.asymmetric_weighting
    options:
      heading_level: 4

## flyvision.utils.type_utils

This module provides utilities for type conversions.

### Functions

::: flyvision.utils.type_utils.byte_to_str
    options:
      heading_level: 4

## flyvision.utils.xarray_joblib_backend

This module contains backend utilities for xarray and joblib integration.

### Classes

::: flyvision.utils.xarray_joblib_backend.H5XArrayDatasetStoreBackend
    options:
      heading_level: 4

## flyvision.utils.xarray_utils

This module provides utilities for working with xarray objects.

### Classes

::: flyvision.utils.xarray_utils.CustomAccessor
    options:
      heading_level: 4

### Functions

::: flyvision.utils.xarray_utils.where_xarray
    options:
      heading_level: 4

::: flyvision.utils.xarray_utils.plot_traces
    options:
      heading_level: 4
