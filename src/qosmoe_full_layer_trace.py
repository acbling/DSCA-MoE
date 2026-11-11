import os
import csv
import time
import atexit
from pathlib import Path

try:
    import torch
except Exception:
    torch = None


_BUFFER = []
_JOB_ID = 0

HEADER = [
    "system",
    "rank",
    "iteration",
    "layer",
    "job_id",
    "token_id",
    "expert_id",
    "ready_ms",
    "expert_start_ms",
    "expert_end_ms",
    "wait_ms",
    "expert_ms",
    "note",
]


def _enabled():
    return os.environ.get("QOSMOE_FULL_LAYER_TRACE", "0") == "1"


def _sync_cuda():
    if os.environ.get("QOSMOE_TRACE_SYNC_CUDA", "1") == "1":
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()


def _now_ms():
    _sync_cuda()
    return time.perf_counter_ns() / 1e6


def _rank():
    return int(os.environ.get("QOSMOE_TRACE_RANK", "0"))


def _iteration():
    return int(os.environ.get("QOSMOE_TRACE_ITER", "-1"))


def _out_dir():
    return Path(
        os.environ.get(
            "QOSMOE_FULL_LAYER_TRACE_DIR",
            os.environ.get(
                "QOSMOE_REAL_TOKEN_TRACE_DIR",
                os.environ.get("QOSMOE_TRACE_DIR", "."),
            ),
        )
    )


def _flush():
    if not _BUFFER:
        return

    out_dir = _out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"real_token_trace_qosmoe_full_rank{_rank()}.csv"
    exists = path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if not exists:
            writer.writeheader()
        writer.writerows(_BUFFER)

    _BUFFER.clear()


atexit.register(_flush)


def _build_groups(module, x):
    max_tokens = int(os.environ.get("QOSMOE_TRACE_MAX_TOKENS", "64"))

    try:
        flat = x.reshape(-1, x.shape[-1])
    except Exception:
        flat = x

    try:
        n = min(int(flat.shape[0]), max_tokens)
    except Exception:
        n = max_tokens

    if torch is None:
        return [(0, list(range(n)))]

    try:
        token_idx = torch.arange(n, device=flat.device)
    except Exception:
        token_idx = torch.arange(n)

    expert_index = None

    # Try to group by router if available, otherwise use one pseudo expert.
    try:
        router = getattr(module, "router", None)
        if router is not None:
            with torch.no_grad():
                inp = flat[:n]
                try:
                    out = router(inp.float())
                except Exception:
                    out = router(inp)

            if isinstance(out, tuple):
                for item in out:
                    if torch.is_tensor(item) and item.shape[0] == n:
                        if item.dim() >= 2:
                            expert_index = torch.argmax(item, dim=-1).long()
                            break
                        if item.dim() == 1:
                            expert_index = item.long()
                            break
            elif torch.is_tensor(out):
                if out.dim() >= 2:
                    expert_index = torch.argmax(out, dim=-1).long()
                elif out.dim() == 1:
                    expert_index = out.long()
    except Exception:
        expert_index = None

    if expert_index is None:
        expert_index = torch.zeros(n, dtype=torch.long, device=token_idx.device)

    groups = []
    try:
        for e in torch.unique(expert_index).detach().cpu().tolist():
            idxs = torch.nonzero(expert_index == int(e), as_tuple=False).reshape(-1)
            idxs = idxs[idxs < max_tokens]
            if idxs.numel() > 0:
                groups.append((int(e), idxs))
    except Exception:
        groups = [(0, token_idx)]

    return groups


def _collect_layers(model, get_layers_fn=None):
    layers = []

    if callable(get_layers_fn):
        try:
            raw_layers = list(get_layers_fn(model))
            for i, m in enumerate(raw_layers):
                layer_idx = int(getattr(m, "layer_idx", i))
                layers.append((layer_idx, f"get_layers[{i}]", m))
        except Exception as e:
            print(f"[TRACE][WARN] get_layers_fn failed: {repr(e)}")

    if layers:
        return layers

    # fallback collection
    selected = []
    for name, module in model.named_modules():
        cls = module.__class__.__name__
        if (
            "QoSMoE" in cls
            or "SparseMLP" in cls
            or (hasattr(module, "router") and hasattr(module, "experts"))
        ):
            if not callable(getattr(module, "forward", None)):
                continue
            if "router" in name.lower():
                continue
            if "expert" in name.lower() and "SparseMLP" not in cls and "QoSMoE" not in cls:
                continue
            selected.append((name, module))

    for i, (name, m) in enumerate(selected):
        layer_idx = int(getattr(m, "layer_idx", i))
        layers.append((layer_idx, name, m))

    # deduplicate layer id
    out = {}
    for layer_idx, name, m in layers:
        out[layer_idx] = (name, m)

    return [(idx, name, m) for idx, (name, m) in sorted(out.items())]


def install_qosmoe_full_layer_boundary_trace(model, get_layers_fn=None):
    if not _enabled():
        print("[TRACE] QoSMoE full-layer boundary trace disabled")
        return

    layers = _collect_layers(model, get_layers_fn)
    print(f"[TRACE] QoSMoE full-layer boundary hooks installed={len(layers)}")
    for layer_idx, name, _ in layers:
        print(f"[TRACE] QoSMoE full-layer idx={layer_idx}, name={name}")

    for layer_idx, name, layer in layers:
        def pre_hook(module, inputs, layer_idx=layer_idx):
            if not inputs:
                return
            x = inputs[0]

            ready_ms = _now_ms()
            groups = _build_groups(module, x)

            global _JOB_ID
            _JOB_ID += 1
            job_id = _JOB_ID

            module._qosmoe_full_layer_trace_ctx = (
                job_id,
                ready_ms,
                groups,
                int(layer_idx),
            )

        def post_hook(module, inputs, output, layer_idx=layer_idx):
            ctx = getattr(module, "_qosmoe_full_layer_trace_ctx", None)
            if ctx is None:
                return

            end_ms = _now_ms()
            job_id, ready_ms, groups, layer_idx = ctx

            for expert_id, idxs in groups:
                if torch is not None and torch.is_tensor(idxs):
                    token_ids = idxs.detach().cpu().reshape(-1).tolist()
                else:
                    token_ids = list(idxs)

                for tid in token_ids:
                    _BUFFER.append({
                        "system": "qosmoe",
                        "rank": _rank(),
                        "iteration": _iteration(),
                        "layer": int(layer_idx),
                        "job_id": int(job_id),
                        "token_id": int(tid),
                        "expert_id": int(expert_id),
                        "ready_ms": float(ready_ms),
                        "expert_start_ms": float(ready_ms),
                        "expert_end_ms": float(end_ms),
                        "wait_ms": 0.0,
                        "expert_ms": float(end_ms - ready_ms),
                        "note": "qosmoe.full_layer_boundary",
                    })

            if len(_BUFFER) >= 2048:
                _flush()

        layer.register_forward_pre_hook(pre_hook)
        layer.register_forward_hook(post_hook)

    _flush()
