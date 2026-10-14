# Sparse training on the FlyVis visual-system graph

Measured 2026-09-26 on an RTX 5090, Windows 11, Python 3.12.11, PyTorch
2.14.0+cu130 (CUDA 13.0), float32, four CPU threads. Source commit:
`7a76dc3cf98c8c1d380d0d86275b094c22b081c8` (clean at benchmark start).

The supplied `fib25-fib19_v2.2.json` graph at extent 15 has **45,669 nodes,
1,513,231 edges and 734 trainable shared parameters**. This measures the
actual FlyVis `Network` forward/backward with its default dynamics, synthetic
input, `dt=0.02`, and a squared-activity loss. It excludes graph compilation,
data loading, a task decoder, and optimizer updates. It does not measure
optical-flow accuracy or learning speed to a target score.

Times are medians of 10 synchronized wall-clock repetitions after 3 warmups.
Peak is `torch.cuda.max_memory_allocated`, including live model tensors,
input, activations and gradients. It is not allocator-reserved or total
process VRAM. The JSONL includes every timing, the baseline allocation,
incremental peak, loss, and environment. Initialization uses seed 20260926;
input uses seed 20260927 for both backends.

| Batch | Steps | Backend | Forward + backward (ms) | Peak allocated (GiB) |
|---:|---:|---|---:|---:|
| 4 | 8 | `scatter` | 6.66 | 0.411 |
| 4 | 8 | `sparse_trainable` | 9.96 | 0.242 |
| 4 | 32 | `scatter` | 20.70 | 1.042 |
| 4 | 32 | `sparse_trainable` | 29.01 | 0.356 |
| 16 | 8 | `scatter` | 13.38 | 1.205 |
| 16 | 8 | `sparse_trainable` | 8.46 | 0.348 |
| 16 | 32 | `scatter` | 48.02 | 3.641 |
| 16 | 32 | `sparse_trainable` | 30.95 | 0.886 |

At batch 16 and 32 steps, the sparse backend used **4.11x less peak allocated memory**
(75.7% reduction) and was **1.55x faster** in this run.
At batch 4, it saved memory but was slower. This supports an opt-in
backend, not a universal speedup claim. Timing varies across runs;
the individual repetitions are retained for inspection. All gradients
were finite and paired final losses matched at the recorded precision.

Reproduce:

```bash
python benchmarks/sparse_current.py --extent 15 --batches 4 16 --steps 8 32 --repeats 10 --warmup 3 --out results.jsonl
```

Local Windows setup note: datamate 1.0.0 needed its HDF5 handle closed
before fallback file deletion when building the connectome cache.
That environment-only workaround does not affect timed tensor operations
and is not included in this FlyVis source patch. Additional environment
versions: NumPy 2.5.3, pandas 3.0.6, SciPy 1.18.1, torchvision 0.29.0+cu130.

Validation: 198 passed, 12 skipped across the new sparse tests and
existing network, dynamics and initialization suites. Ruff lint and
format checks passed for changed Python files. This is targeted
validation, not the full suite requiring pretrained assets.
