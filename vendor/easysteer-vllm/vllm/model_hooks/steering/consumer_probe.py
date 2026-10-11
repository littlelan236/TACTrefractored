"""Optional full-vector producer/consumer comparison in captured GPU code."""

import torch
from torch import nn


class ConsumerProbe(nn.Module):
    def __init__(self, width, capacity, device):
        super().__init__()
        self.capacity = capacity
        for name, shape, dtype in (
            ("expected_hidden", (capacity, width), torch.float32),
            ("expected_residual", (capacity, width), torch.float32),
            ("valid", (capacity,), torch.bool),
            ("stats", (capacity, 8), torch.float32),
            ("overflow", (1,), torch.int32),
        ):
            self.register_buffer(name, torch.zeros(shape, dtype=dtype, device=device),
                                 persistent=False)
        self.logits_indices = None

    def reset(self):
        self.valid.zero_()
        self.stats.zero_()
        self.overflow.zero_()
        self.logits_indices = None

    def produce(self, hidden, residual, selected, actual_delta):
        n = min(hidden.shape[0], self.capacity)
        self.expected_hidden[:n].copy_(hidden[:n].float())
        if residual is not None:
            self.expected_residual[:n].copy_(residual[:n].float())
        else:
            self.expected_residual[:n].zero_()
        self.valid[:n].copy_(selected[:n])
        self.stats[:n, 0].copy_(selected[:n])
        self.stats[:n, 6].copy_(torch.where(selected[:n], actual_delta[:n], 0))
        self.overflow[0] += selected[n:].sum().to(torch.int32)

    def consume(self, hidden, residual, indices=None):
        n = min(hidden.shape[0], self.capacity)
        if indices is not None:
            n = min(n, indices.numel())
        rows = (torch.arange(n, device=hidden.device) if indices is None
                else indices[:n].long())
        in_range = (rows >= 0) & (rows < self.capacity)
        safe_rows = rows.clamp(0, self.capacity - 1)
        active = in_range & self.valid[safe_rows]
        actual_hidden = hidden[:n].float()
        actual_residual = (torch.zeros_like(actual_hidden) if residual is None
                           else residual[:n].float())
        hdiff = (actual_hidden - self.expected_hidden[safe_rows]).abs()
        rdiff = (actual_residual - self.expected_residual[safe_rows]).abs()
        hbad = (actual_hidden != self.expected_hidden[safe_rows]).sum(-1)
        rbad = (actual_residual != self.expected_residual[safe_rows]).sum(-1)
        values = torch.stack((active.float(),
            torch.where(active, hdiff.amax(-1), 0),
            torch.where(active, rdiff.amax(-1), 0),
            torch.where(active, hbad, 0), torch.where(active, rbad, 0)), -1)
        for i in range(5):
            self.stats[:, i + 1].index_add_(0, safe_rows, values[:, i])
        # A duplicate consumer call is visible as count > 1; do not clear valid.

    def norm_hook(self, module, args, kwargs):
        hidden = kwargs.get("x", kwargs.get("hidden_states"))
        if hidden is None and args:
            hidden = args[0]
        residual = kwargs.get("residual", args[1] if len(args) > 1 else None)
        if hidden is None:
            raise RuntimeError("Consumer norm hook did not receive hidden tensor")
        self.consume(hidden, residual)

    def logits_hook(self, module, args, kwargs):
        # LogitsProcessor.forward(lm_head, hidden_states, ...).
        hidden = kwargs.get("hidden_states", args[1] if len(args) > 1 else None)
        if hidden is None:
            raise RuntimeError("Logits consumer hook did not receive hidden tensor")
        if self.logits_indices is not None:
            self.consume(hidden, None, self.logits_indices)
