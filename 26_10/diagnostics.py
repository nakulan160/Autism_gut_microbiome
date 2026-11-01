import json, math
from collections import deque
import torch


class DiagnosticStop(RuntimeError):
    pass


def tensor_summary(t):
    x = t.detach()
    finite = torch.isfinite(x)
    v = x[finite]
    return dict(
        shape=list(x.shape),
        nonfinite=int((~finite).sum()),
        max_abs=float(v.abs().max()) if v.numel() else None,
    )


def check_tensor(t, label):
    if not torch.isfinite(t).all():
        raise DiagnosticStop(f"Nonfinite {label}")


def gradient_summary(model):
    max_value = 0.0
    name = None
    norm_squared = 0.0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        check_tensor(p.grad, "gradient " + n)
        # Per-tensor double reduction only; parameters/gradient storage remain FP32.
        g = p.grad.detach()
        m = float(g.abs().max())
        norm_squared += float(g.double().square().sum())
        if m > max_value:
            max_value = m
            name = n
    return dict(max_abs=max_value, parameter=name, l2=math.sqrt(norm_squared))


def check_state(model, optimizer):
    for n, p in model.named_parameters():
        check_tensor(p, "parameter " + n)
    for p, state in optimizer.state.items():
        for key, t in state.items():
            if torch.is_tensor(t):
                check_tensor(t, "optimizer " + key)


class Monitor:
    """Bounded scalar history. Detailed tensor hooks are reserved for replay."""

    def __init__(self, path):
        self.path = path
        self.recent = deque(maxlen=32)

    def event(self, record):
        self.recent.append(record)
        with open(self.path, "a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")


class Trace:
    """Layer boundaries and autograd-node gradients, scalar summaries only. Fused internals stay opaque."""

    def __init__(self, model, path):
        self.model = model
        self.path = path
        self.handles = []
        self.seen = set()

    def record(self, name, t):
        if not torch.is_tensor(t):
            return
        record = dict(name=name, **tensor_summary(t))
        if t.numel() and torch.isfinite(t).all():
            flat = int(t.detach().abs().argmax())
            coords = []
            for dim in reversed(t.shape):
                coords.append(flat % dim)
                flat //= dim
            record["max_index"] = list(reversed(coords))
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def watch(self, name, out):
        tensors = (
            [out]
            if torch.is_tensor(out)
            else list(out) if isinstance(out, (list, tuple)) else []
        )
        for i, t in enumerate(tensors):
            if not torch.is_tensor(t):
                continue
            self.record("forward:" + name + ":" + str(i), t)
            if t.requires_grad:
                self.handles.append(
                    t.register_hook(lambda g, n=name: self.record("backward:" + n, g))
                )

    def graph(self, loss):
        pending = [loss.grad_fn]
        while pending:
            node = pending.pop()
            if node is None or node in self.seen:
                continue
            self.seen.add(node)
            name = node.name()

            def hook(gi, go, n=name):
                for j, g in enumerate(go):
                    self.record(n + ":grad_output:" + str(j), g)
                for j, g in enumerate(gi):
                    self.record(n + ":grad_input:" + str(j), g)

            self.handles.append(node.register_hook(hook))
            pending.extend(n for n, _ in node.next_functions)

    def __enter__(self):
        for name, module in self.model.named_modules():
            if name:
                self.handles.append(
                    module.register_forward_hook(
                        lambda m, i, o, n=name: self.watch(n, o)
                    )
                )
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.seen.clear()
