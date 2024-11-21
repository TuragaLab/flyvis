# Configuration System

`flyvis` uses [Hydra](https://hydra.cc/) for configuration management of network and training. The configuration system is organized hierarchically, allowing for modular configuration of different components.

## Configuration Components

### Configuration Components

#### Network Configuration

* `network/`: Core network architecture and parameters
* `connectome/`: Neural connectivity specification
* `dynamics/`: Neural dynamics parameters
* `edge_config/`: Synapse-related configurations (sign, strength, count)
* `node_config/`: Neuron-related configurations (bias, time constants)
* `stimulus_config/`: Input stimulus parameters

#### Training Configuration

* `optim/`: Optimization parameters
* `penalizer/`: Loss function penalties
* `scheduler/`: Learning rate scheduling
* `task/`: Task-specific settings and parameters

#### Top-level Configuration
* `solver.yaml`: Training loop parameters


## Default Configuration Structure

The default configuration is structured as follows:

```bash
config
├── network                         # Network configuration
│   ├── connectome                  # Connectome configurations
│   │   └── connectome.yaml         # Default connectome configuration
│   ├── dynamics                    # Dynamics configurations
│   │   └── dynamics.yaml           # Default dynamics configuration
│   ├── edge_config                 # Edge configuration
│   │   ├── edge_config.yaml        # Default edge configuration
│   │   ├── sign                    # Signaling configurations
│   │   │   └── sign.yaml           # Default signaling configuration
│   │   ├── syn_count               # Synaptic count configurations
│   │   │   └── syn_count.yaml      # Default synaptic count configuration
│   │   └── syn_strength            # Synaptic strength configurations
│   │       └── syn_strength.yaml   # Default synaptic strength configuration
│   ├── network.yaml                # Default network configuration
│   ├── node_config                 # Node configuration
│   │   ├── bias                    # Bias configurations
│   │   │   └── bias.yaml           # Default bias configuration
│   │   ├── node_config.yaml        # Default node configuration
│   │   └── time_const              # Time constant configurations
│   │       └── time_const.yaml     # Default time constant configuration
│   └── stimulus_config             # Stimulus configuration
│       └── stimulus_config.yaml    # Default stimulus configuration
├── optim                           # Optimizer configuration
│   └── optim.yaml                  # Default optimizer configuration
├── penalizer                       # Penalizer configuration
│   └── penalizer.yaml              # Default penalizer configuration
├── scheduler                       # Scheduler configuration
│   └── scheduler.yaml              # Default scheduler configuration
├── solver.yaml                     # Default solver configuration
└── task                            # Task configuration
    └── task.yaml                   # Default task configuration

15 directories, 16 files
```

## Customizing Configurations

### Overriding Default Parameters

For quick parameter changes you can use hydra command-line overrides.

```bash
# Example: Change the scale of the synaptic strength to 0.5
flyvis train-single network.edge_config.syn_strength.scale=0.5 ensemble_and_network_id=0042/007 task_name=flow

# Example: Add a custom parameter to an existing config
flyvis train-single +network.edge_config.syn_strength.custom_param=100 ensemble_and_network_id=0042/007 task_name=flow
```

More information on hydra command-line overrides can be found [here](https://hydra.cc/docs/advanced/override_grammar/basic/).

### Using Custom Configurations

In many situations you will want to create custom configuration files, to either break
the existing config structure for new code or to keep track of different sets of parameters.

This particularly applies when installing `flyvis` as a package instead of using the
developer mode, where one can directly modify the source code configuration.

To create and maintain your own configuration files proceed as follows:

1. Initialize a local config directory:

    ```bash
    # Copy the default config structure into flyvis_config
    flyvis init-config
    ```

2. Create your custom configurations in the generated `flyvis_config` directory.

3. Use your custom configs:

    ```bash
    # Using the full custom config directory
    flyvis train-single --config-path $(pwd)/flyvis_config [other options]

    # Using a specific custom config
    flyvis train-single network/edge_config/syn_strength=custom_syn_strength --config-dir $(pwd)/flyvis_config
    ```

    Notice that **`config-path`** overrides the entire default config directory, while **`config-dir`** only adds another search path for resolving configs.

    **Important:** It is recommended to use different file names than the defaults.
    This is because for partial usage with `config-dir` hydra resolves the config names in the default config directory first. I.e., your changes reflected in files of the same name in the custom config directory might not have the intended effect. E.g., name your config file `custom_syn_strength.yaml` instead of `syn_strength.yaml`.


More information on hydra configuration can be found [here](https://hydra.cc/docs/advanced/search_path/).
