"""Benchmark actual FlyVis Network forward/backward on its visual connectome.

Run after installing FlyVis: python benchmarks/sparse_current.py --out results.jsonl
Model compilation and data loading are excluded. No training dataset is needed.
"""

import argparse
import gc
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch

from flyvis import Network


def measure(backend, extent, batch, steps, repeats, warmup):
    torch.manual_seed(20260926)
    model = Network(
        connectome=dict(
            type="ConnectomeFromAvgFilters",
            file="fib25-fib19_v2.2.json",
            extent=extent,
            n_syn_fill=1,
        ),
        backend=backend,
    )
    torch.manual_seed(20260927)
    stimulus = torch.randn(batch, steps, model.n_nodes) * 0.1

    def step():
        model.zero_grad(set_to_none=True)
        output = model(stimulus, dt=0.02)
        loss = output.square().mean()
        loss.backward()
        return loss

    for _ in range(warmup):
        step()
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    measurements = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss = step()
        torch.cuda.synchronize()
        measurements.append((time.perf_counter() - start) * 1000)
    peak = torch.cuda.max_memory_allocated()
    result = dict(
        backend=backend,
        extent=extent,
        nodes=model.n_nodes,
        edges=model.n_edges,
        trainable_parameters=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        batch=batch,
        steps=steps,
        dtype=str(stimulus.dtype),
        dt=0.02,
        repeats=repeats,
        warmup=warmup,
        forward_backward_ms=statistics.median(measurements),
        repetition_ms=measurements,
        baseline_bytes=baseline,
        peak_bytes=peak,
        incremental_peak_bytes=peak - baseline,
        loss=float(loss.detach()),
        gradients_finite=all(
            bool(torch.isfinite(p.grad).all())
            for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ),
    )
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extent", type=int, default=15)
    parser.add_argument("--batches", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--steps", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("This peak-memory benchmark requires CUDA")
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(4)
    environment = dict(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        platform=platform.platform(),
        python=platform.python_version(),
        threads=torch.get_num_threads(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"])),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for batch in args.batches:
            for steps in args.steps:
                for backend in ("scatter", "sparse_trainable"):
                    result = measure(
                        backend, args.extent, batch, steps, args.repeats, args.warmup
                    )
                    result["environment"] = environment
                    stream.write(json.dumps(result) + "\n")
                    stream.flush()
                    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
