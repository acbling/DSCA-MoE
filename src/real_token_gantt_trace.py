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
    return os.environ.get("QOSMOE_REAL_TOKEN_TRACE", "1") == "1"


def _sync_cuda():
    if os.environ.get("QOSMOE_TRACE_SYNC_CUDA", "1") == "1":
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()


def now_ms():
    _sync_cuda()
    return time.perf_counter_ns() / 1e6


def _to_token_list(token_idx):
    if token_idx is None:
        return []
    try:
        if torch is not None and torch.is_tensor(token_idx):
            return [int(x) for x in token_idx.detach().cpu().reshape(-1).tolist()]
    except Exception:
        pass
    try:
        return [int(x) for x in list(token_idx)]
    except Exception:
        return []


def job_start(tokens=None, expert_mask=None, tag=""):
    global _JOB_ID
    if not _enabled():
        return -1, now_ms()

    _JOB_ID += 1
    return _JOB_ID, now_ms()


def record_group(
    job_id,
    ready_ms,
    expert_idx,
    token_idx,
    moe_start_ms,
    moe_end_ms,
    tokens=None,
    note="",
    layer_idx=-1,
):
    if not _enabled():
        return

    system = os.environ.get("QOSMOE_TRACE_SYSTEM", "qosmoe")
    rank = int(os.environ.get("QOSMOE_TRACE_RANK", "0"))
    iteration = int(os.environ.get("QOSMOE_TRACE_ITER", "-1"))

    max_tokens = int(os.environ.get("QOSMOE_TRACE_MAX_TOKENS", "64"))
    token_ids = _to_token_list(token_idx)

    # IMPORTANT:
    # max_tokens means "record at most N tokens from this expert group",
    # not "only record token_id < N".
    # Later layers may dispatch flattened token ids much larger than 64.
    if max_tokens > 0:
        token_ids = token_ids[:max_tokens]

    for tid in token_ids:
        if tid < 0:
            continue

        _BUFFER.append({
            "system": system,
            "rank": rank,
            "iteration": iteration,
            "layer": int(layer_idx),
            "job_id": int(job_id),
            "token_id": int(tid),
            "expert_id": int(expert_idx),
            "ready_ms": float(ready_ms),
            "expert_start_ms": float(moe_start_ms),
            "expert_end_ms": float(moe_end_ms),
            "wait_ms": float(moe_start_ms - ready_ms),
            "expert_ms": float(moe_end_ms - moe_start_ms),
            "note": str(note),
        })

    if len(_BUFFER) >= 2048:
        flush()


def flush():
    if not _BUFFER:
        return

    out_dir = Path(
        os.environ.get(
            "QOSMOE_REAL_TOKEN_TRACE_DIR",
            os.environ.get("QOSMOE_TRACE_DIR", "."),
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    system = os.environ.get("QOSMOE_TRACE_SYSTEM", "qosmoe")
    rank = int(os.environ.get("QOSMOE_TRACE_RANK", "0"))

    path = out_dir / f"real_token_trace_{system}_rank{rank}.csv"
    exists = path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if not exists:
            writer.writeheader()
        writer.writerows(_BUFFER)

    _BUFFER.clear()


atexit.register(flush)
