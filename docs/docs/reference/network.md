# Network

## Optional sparse training

`Network(backend="sparse_trainable")` selects CSR aggregation for the default
`PPNeuronIGRSynapses` dynamics. The default remains `backend="scatter"`.
For Hydra-based training, add `+network.backend=sparse_trainable` to the overrides.

The default scatter path expands activity to `[batch, edges]` before activation
and weighted summation. The sparse backend instead activates `[batch, nodes]`
and computes the same summed current with sparse matrix multiplication. Its
custom backward samples the weight gradient only at existing connections; it
does not construct a dense `[nodes, nodes]` gradient. Forward and transposed CSR
layouts are cached, so backward does not sort the graph at each time step.

Shared synaptic strengths, signs, counts, biases, time constants, Euler updates,
and node state hooks retain their existing meaning. Unsorted and parallel edges
are supported: parallel weights are summed once per forward and gradients flow
to every original edge. Per-edge current inspection still uses the original
connectome ordering and will allocate edge-sized tensors when requested.

Backend activation storage scales with `batch * nodes * steps` instead of
`batch * edges * steps`. Graph layouts, parameters, output trajectories and
temporary workspace still consume memory. This is not a guarantee of faster
execution: benchmark your graph and batch size before selecting a backend.

Requirements and limitations:

- CPU or CUDA, float32 or float64, with PyTorch CSR `sampled_addmm` support.
- First-order gradients only; higher-order derivatives are unsupported.
- The built-in `PPNeuronIGRSynapses` class and its out-of-place pointwise
  activations. Other dynamics continue to use scatter.
- Fixed topology. Construct a new network for a different connectome.
- No AMP, `torch.compile`, MPS or arbitrary custom-activation support is claimed.

CSR layouts are nonpersistent buffers. A sparse network and a scatter network
constructed from the same configuration have compatible parameter state dicts.
The selected backend is also retained in `network.config` for reconstruction.

Run `python benchmarks/sparse_current.py --out results.jsonl` for a bounded
forward/backward benchmark on the supplied visual-system connectome. It does
not require downloading the training dataset or running an optimization job.

::: flyvis.network.network.Network

## Stimulus

Stimuli must implement the `StimulusProtocol` to be compatible with
`flyvis.network.network.Network`.

::: flyvis.network.stimulus.StimulusProtocol

::: flyvis.network.stimulus.Stimulus

::: flyvis.network.stimulus.register_stimulus

::: flyvis.network.stimulus.init_stimulus

## Dynamics

::: flyvis.network.dynamics.NetworkDynamics

::: flyvis.network.dynamics.PPNeuronIGRSynapses

## Initialization

::: flyvis.network.initialization
