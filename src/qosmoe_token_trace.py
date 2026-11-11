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

HEADER = [
    "system",
    "rank",
    "iteration",
    "batch_id",
    "layer",
    "token_id",
    "event",
    "time_ns",
    "time_s",
]


def _enabled():
    return os.environ.get("QOSMOE_TRACE_ENABLE", "0") == "1"


def _sync_cuda():
    if os.environ.get("QOSMOE_TRACE_SYNC_CUDA", "1") == "1":
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()


def set_trace_context(system, out_dir, rank):
    os.environ["QOSMOE_TRACE_SYSTEM"] = str(system)
    os.environ["QOSMOE_TRACE_DIR"] = str(out_dir)
    os.environ["QOSMOE_TRACE_RANK"] = str(rank)


def set_iteration(iteration):
    os.environ["QOSMOE_TRACE_ITER"] = str(iteration)


def _token_ids_from_batch(batch):
    try:
        n = int(batch["input_ids"].numel())
    except Exception:
        n = int(os.environ.get("QOSMOE_TRACE_MAX_TOKENS", "64"))

    max_tokens = int(os.environ.get("QOSMOE_TRACE_MAX_TOKENS", "64"))
    return list(range(min(n, max_tokens)))


def _token_ids_from_tensor(x):
    try:
        if hasattr(x, "dim") and x.dim() == 3:
            n = int(x.shape[0] * x.shape[1])
        else:
            n = int(x.shape[0])
    except Exception:
        n = int(os.environ.get("QOSMOE_TRACE_MAX_TOKENS", "64"))

    max_tokens = int(os.environ.get("QOSMOE_TRACE_MAX_TOKENS", "64"))
    return list(range(min(n, max_tokens)))


def trace_event(event, token_ids=None, layer=-1, batch_id=0, extra=None):
    if not _enabled():
        return

    _sync_cuda()

    system = os.environ.get("QOSMOE_TRACE_SYSTEM", "unknown")
    out_dir = os.environ.get("QOSMOE_TRACE_DIR", ".")
    rank = int(os.environ.get("QOSMOE_TRACE_RANK", "0"))
    iteration = int(os.environ.get("QOSMOE_TRACE_ITER", "-1"))

    if token_ids is None:
        token_ids = [-1]
    elif isinstance(token_ids, int):
        token_ids = [token_ids]
    else:
        token_ids = list(token_ids)

    now_ns = time.perf_counter_ns()
    now_s = now_ns / 1e9

    for tid in token_ids:
        _BUFFER.append({
            "system": system,
            "rank": rank,
            "iteration": iteration,
            "batch_id": int(batch_id),
            "layer": int(layer),
            "token_id": int(tid),
            "event": str(event),
            "time_ns": int(now_ns),
            "time_s": float(now_s),
        })

    if len(_BUFFER) >= 4096:
        flush_trace()


def trace_iter_start(batch):
    token_ids = _token_ids_from_batch(batch)
    trace_event("iter_start", token_ids, layer=-1)


def trace_iter_end(batch):
    token_ids = _token_ids_from_batch(batch)
    trace_event("iter_end", token_ids, layer=-1)
    flush_trace()


def install_moe_hooks(model, get_layers_fn, system_name):
    layers = list(get_layers_fn(model))

    for fallback_idx, layer in enumerate(layers):
        layer_idx = int(getattr(layer, "layer_idx", fallback_idx))

        def pre_hook(module, inputs, layer_idx=layer_idx):
            if not inputs:
                token_ids = [-1]
            else:
                token_ids = _token_ids_from_tensor(inputs[0])

            trace_event("attn_end", token_ids, layer=layer_idx)
            trace_event("moe_ready_time", token_ids, layer=layer_idx)
            trace_event("moe_launch_time", token_ids, layer=layer_idx)
            trace_event("expert_compute_start", token_ids, layer=layer_idx)

        def post_hook(module, inputs, output, layer_idx=layer_idx):
            if inputs:
                token_ids = _token_ids_from_tensor(inputs[0])
            else:
                token_ids = [-1]

            trace_event("expert_compute_end", token_ids, layer=layer_idx)
            trace_event("combine_start", token_ids, layer=layer_idx)
            trace_event("combine_end", token_ids, layer=layer_idx)

        layer.register_forward_pre_hook(pre_hook)
        layer.register_forward_hook(post_hook)

    print(f"[TRACE] installed hooks for {system_name}, num_layers={len(layers)}")


def flush_trace():
    if not _BUFFER:
        return

    out_dir = Path(os.environ.get("QOSMOE_TRACE_DIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)

    system = os.environ.get("QOSMOE_TRACE_SYSTEM", "unknown")
    rank = int(os.environ.get("QOSMOE_TRACE_RANK", "0"))

    path = out_dir / f"trace_{system}_rank{rank}.csv"
    exists = path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if not exists:
            writer.writeheader()
        writer.writerows(_BUFFER)

    _BUFFER.clear()


atexit.register(flush_trace)
