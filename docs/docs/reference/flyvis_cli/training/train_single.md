# Run Training for Single Model


::: flyvis_cli.training.train_single
    options:
      heading_level: 4


```

== train single flyvis network ==

Train the visual system model using the specified configuration.

  This script initializes and runs the training process for the model.
  It uses the configuration provided through Hydra to set up the solver, manage the
  training process, and handle checkpoints.

  Example:
      Train a network for 1000 iterations (and add description 'test'):

      flyvis train-single \
          ensemble_and_network_id=0045/000 \
          task_name=flow \
          train=true \
          resume=false \
          task.n_iters=1000
          description='test'

      or

      python train_single.py \
          ensemble_and_network_id=0045/000 \
          task_name=flow \
          train=true \
          resume=false \
          task.n_iters=1000 \
          description='test'


  High-level options:

    - train (bool): Whether to run the training process
    - resume (bool): Whether to resume training from the last checkpoint
    - overfit (bool): Whether to use overfitting mode in training
    - checkpoint_only (bool): Whether to create a checkpoint without training
    - save_environment (bool): Whether to save the source code and environment details

== Configuration groups ==
Compose your configuration from those groups (group=option)

network: network
network/connectome: connectome
network/dynamics: dynamics
network/edge_config: edge_config
network/edge_config/sign: sign
network/edge_config/syn_count: syn_count
network/edge_config/syn_strength: syn_strength
network/node_config: node_config
network/node_config/bias: bias
network/node_config/time_const: time_const
network/stimulus_config: stimulus_config
optim: optim
penalizer: penalizer
scheduler: scheduler
task: task


== Config ==
Override anything in the config (foo.bar=value)

ensemble_and_network_id: ???
task_name: ???
train: true
resume: false
checkpoint_only: false
network_name: ${task_name}/${ensemble_and_network_id}
description: ???
overfit: false
delete_if_exists: false
save_environment: false
network:
  connectome:
    type: ConnectomeFromAvgFilters
    file: fib25-fib19_v2.2.json
    extent: 15
    n_syn_fill: 1
  dynamics:
    type: PPNeuronIGRSynapses
    activation:
      type: relu
  edge_config:
    sign:
      type: SynapseSign
      form: value
      requires_grad: false
      initial_dist: Value
      groupby:
      - source_type
      - target_type
    syn_count:
      type: SynapseCount
      initial_dist: Lognormal
      mode: mean
      requires_grad: false
      std: 1.0
      groupby:
      - source_type
      - target_type
      - dv
      - du
      penalize:
        function: weight_decay
        kwargs:
          lambda: 0
    syn_strength:
      type: SynapseCountScaling
      initial_dist: Value
      requires_grad: true
      scale: 0.01
      clamp: non_negative
      groupby:
      - source_type
      - target_type
      penalize:
        function: weight_decay
        kwargs:
          lambda: 0
  node_config:
    bias:
      type: RestingPotential
      groupby:
      - type
      initial_dist: Normal
      mode: sample
      requires_grad: true
      seed: 0
      mean: 0.5
      std: 0.05
      symmetric: []
      penalize:
        activity: true
    time_const:
      type: TimeConstant
      groupby:
      - type
      initial_dist: Value
      value: 0.05
      requires_grad: true
  stimulus_config:
    type: Stimulus
    init_buffer: false
task:
  dataset:
    type: MultiTaskSintel
    tasks:
    - flow
    boxfilter:
      extent: 15
      kernel_size: 13
    vertical_splits: 3
    n_frames: 19
    center_crop_fraction: 0.7
    dt: 0.02
    augment: true
    random_temporal_crop: true
    all_frames: false
    resampling: true
    interpolate: true
    p_flip: 0.5
    p_rot: 0.5
    contrast_std: 0.2
    brightness_std: 0.1
    gaussian_white_noise: 0.08
    gamma_std: null
    _init_cache: true
    unittest: false
    flip_axes:
    - 0
    - 1
    - 2
    - 3
  decoder:
    flow:
      type: DecoderGAVP
      shape:
      - 8
      - 2
      kernel_size: 5
      const_weight: 0.001
      n_out_features: null
      p_dropout: 0.5
  loss:
    flow: l2norm
  task_weights: null
  batch_size: 4
  n_iters: 250000
  n_folds: 4
  fold: 1
  seed: 0
  original_split: true
optim:
  type: Adam
  optim_dec:
    lr: ${scheduler.lr_dec.start}
  optim_net:
    lr: ${scheduler.lr_net.start}
penalizer:
  activity_penalty:
    activity_baseline: 5.0
    activity_penalty: 0.1
    stop_iter: 150000
    below_baseline_penalty_weight: 1.0
    above_baseline_penalty_weight: 0.1
  optim: SGD
scheduler:
  lr_net:
    function: stepwise
    start: 5.0e-05
    stop: 5.0e-06
    steps: 10
  lr_dec:
    function: stepwise
    start: 5.0e-05
    stop: 5.0e-06
    steps: 10
  lr_pen:
    function: stepwise
    start: ${scheduler.lr_net.start}
    stop: ${scheduler.lr_net.stop}
    steps: 10
  dt:
    function: stepwise
    start: 0.02
    stop: 0.02
    steps: 10
  chkpt_every_epoch: 300


Powered by Hydra (https://hydra.cc)
Use --hydra-help to view Hydra specific help



```
