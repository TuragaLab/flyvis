"""Opt-in CSR aggregation for instantaneous, graded-release synapses.

Only first-order differentiation is supported. The edge gradient is sampled at
existing coordinates, avoiding both dense adjacency gradients and edge-by-batch
messages. Layouts are derived from the connectome and are not checkpoint data.
"""

import torch
from torch import Tensor, nn
from torch.autograd.function import once_differentiable


class _CSRMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, activity, crow, col, transpose_crow, transpose_col, order):
        n = activity.shape[-1]
        adjacency = torch.sparse_csr_tensor(
            crow,
            col,
            values,
            size=(n, n),
            device=values.device,
            dtype=values.dtype,
            check_invariants=False,
        )
        ctx.save_for_backward(
            values, activity, crow, col, transpose_crow, transpose_col, order
        )
        return torch.sparse.mm(adjacency, activity.T).T

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        values, activity, crow, col, transpose_crow, transpose_col, order = (
            ctx.saved_tensors
        )
        n = activity.shape[-1]
        adjacency = torch.sparse_csr_tensor(
            crow,
            col,
            values,
            size=(n, n),
            device=values.device,
            dtype=values.dtype,
            check_invariants=False,
        )
        grad_values = grad_activity = None
        if ctx.needs_input_grad[0]:
            grad_values = torch.sparse.sampled_addmm(
                adjacency, grad_output.T, activity, beta=0
            ).values()
        if ctx.needs_input_grad[1]:
            # CSC @ dense would sort/convert to CSR at every backward step.
            # Cache the transposed topology and only gather the current values.
            transposed = torch.sparse_csr_tensor(
                transpose_crow,
                transpose_col,
                values.index_select(0, order),
                size=(n, n),
                device=values.device,
                dtype=values.dtype,
                check_invariants=False,
            )
            grad_activity = torch.sparse.mm(transposed, grad_output.T).T
        return grad_values, grad_activity, None, None, None, None, None


class SparseCurrent(nn.Module):
    """Fixed connectivity with dynamic weights and batch-major neuron activity.

    Unsorted and parallel edges are supported. Parallel weights are summed once per
    network forward, with gradients routed back to every original edge. Preparing
    weights once also avoids retaining a reordered edge vector per recurrent step.
    """

    def __init__(self, source: Tensor, target: Tensor, num_nodes: int):
        super().__init__()
        if num_nodes < 1 or source.ndim != 1 or source.shape != target.shape:
            raise ValueError(
                "sparse_trainable requires 1D source/target arrays and nodes"
            )
        if source.dtype not in (torch.int32, torch.int64) or target.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("sparse_trainable requires integer edge indices")
        if source.device != target.device:
            raise ValueError("sparse_trainable source/target indices must share a device")
        if source.numel() and bool(
            (
                (source < 0)
                | (source >= num_nodes)
                | (target < 0)
                | (target >= num_nodes)
            ).any()
        ):
            raise ValueError("sparse_trainable edge index is out of bounds")
        self.num_nodes = num_nodes
        self.num_edges = source.numel()
        coordinates, inverse = torch.unique(
            target.to(torch.int64) * num_nodes + source.to(torch.int64),
            sorted=True,
            return_inverse=True,
        )
        rows = coordinates // num_nodes
        counts = torch.bincount(rows, minlength=num_nodes)
        crow = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        self.register_buffer("crow", crow, persistent=False)
        self.register_buffer("col", coordinates % num_nodes, persistent=False)
        self.register_buffer("inverse", inverse, persistent=False)
        order = torch.argsort((coordinates % num_nodes) * num_nodes + rows)
        transpose_counts = torch.bincount(self.col, minlength=num_nodes)
        self.register_buffer(
            "transpose_crow",
            torch.cat((transpose_counts.new_zeros(1), transpose_counts.cumsum(0))),
            persistent=False,
        )
        self.register_buffer("transpose_col", rows[order], persistent=False)
        self.register_buffer("transpose_order", order, persistent=False)

    def prepare(self, edge_weight: Tensor) -> Tensor:
        """Map current edge weights to unique CSR coordinates, preserving autograd."""
        if edge_weight.shape != (self.num_edges,):
            raise ValueError("sparse_trainable requires one weight per original edge")
        if edge_weight.device != self.inverse.device:
            raise ValueError("sparse_trainable weights and topology must share a device")
        return edge_weight.new_zeros(self.col.numel()).index_add(
            0, self.inverse, edge_weight
        )

    def forward(self, activity: Tensor, values: Tensor) -> Tensor:
        """Sum weighted activated inputs, returning ``[batch, nodes]``."""
        if activity.ndim != 2 or activity.shape[1] != self.num_nodes:
            raise ValueError("sparse_trainable expects activity shaped [batch, nodes]")
        if activity.device.type not in ("cpu", "cuda") or activity.dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise ValueError("sparse_trainable supports float32/float64 on CPU and CUDA")
        if activity.device != values.device or activity.dtype != values.dtype:
            raise ValueError(
                "sparse_trainable activity and weights must share device/dtype"
            )
        if values.shape != self.col.shape or values.device != self.col.device:
            raise ValueError("sparse_trainable values must match the prepared CSR layout")
        return _CSRMM.apply(
            values,
            activity,
            self.crow,
            self.col,
            self.transpose_crow,
            self.transpose_col,
            self.transpose_order,
        )
