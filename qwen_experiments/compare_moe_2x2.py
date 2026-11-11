#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实结果的 2x2 MoE 性能比较工具，无需额外依赖。

输入四个日志文件，日志内需存在 Summary 表格，或者四个实验结果目录，
目录内需包含按 rank 分组的 e2e.csv 和 data.json。
不推测或填补任何缺失的吞吐量。
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

@dataclass
class Result:
    model: str
    method: str
    source: str
    iters: int
    mean_ms: float
    p95_ms: float
    throughput: Optional[float]
    status: str
    note: str = ""
    num_samples: Optional[int] = None
    time_s: Optional[float] = None


def number(v: str) -> float:
    n = float(v)
    if not math.isfinite(n):
        raise ValueError(f"invalid number: {v}")
    return n


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def parse_log(path: Path, model: str, method: str) -> Result:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    required = {"system", "iters", "mean_s", "p95_s", "samples_per_s", "status"}
    matches = []
    for i, line in enumerate(lines):
        fields = line.split()
        if not required.issubset(fields):
            continue
        for following in lines[i + 1:i + 8]:
            row = following.split()
            if len(row) < len(fields) or set(row) >= required:
                continue
            record = dict(zip(fields, row))
            try:
                if record["system"].lower() != method.lower() and method.lower() not in record["system"].lower():
                    # A log may contain multiple rows/systems. For third-party logs, names can differ.
                    continue
                matches.append(record)
            except KeyError:
                continue
        if matches:
            break
    if not matches:
        # A single-system log may label DSCA as qosmoe, etc.
        aliases = {"DSCA-MoE": {"dsca", "qosmoe", "dsca-moe", "qos-moe"},
                   "DeepSpeed": {"deepspeed", "deepspeed-inference", "deepspeed-moe"}}
        for i, line in enumerate(lines):
            fields = line.split()
            if not required.issubset(fields):
                continue
            for following in lines[i + 1:i + 8]:
                row = following.split()
                if len(row) < len(fields):
                    continue
                record = dict(zip(fields, row))
                if record.get("system", "").lower() in aliases.get(method, set()):
                    matches.append(record)
            if matches:
                break
    if not matches:
        # Some third-party scripts use arbitrary `system` labels; a single data row is unambiguous.
        for i, line in enumerate(lines):
            cols = line.split()
            if required.issubset(cols) and i + 1 < len(lines):
                row = lines[i + 1].split()
                if len(row) >= len(cols):
                    matches.append(dict(zip(cols, row)))
                    break
    if not matches:
        raise ValueError(f"no valid Summary table: {path}")
    r = matches[-1]
    status = r["status"]
    if status.lower() not in {"ok", "success", "passed", "pass", "done", "complete", "completed"}:
        raise ValueError(f"benchmark status not successful: {path}: {status}")
    iters = int(number(r["iters"]))
    mean_ms = 1000 * number(r["mean_s"])
    p95_ms = 1000 * number(r["p95_s"])
    throughput = number(r["samples_per_s"])
    if iters < 1 or min(mean_ms, p95_ms, throughput) <= 0:
        raise ValueError(f"invalid benchmark metrics in {path}")
    return Result(model, method, str(path), iters, mean_ms, p95_ms, throughput, status,
                  "Summary metrics provided by benchmark; verify experiment comparability")


def parse_directory(path: Path, model: str, method: str) -> Result:
    data_path = path / "data.json"
    data = json.loads(data_path.read_text(encoding="utf-8")) if data_path.exists() else {}
    rank_files = sorted(path.glob("*/e2e.csv"))
    if not rank_files and (path / "e2e.csv").exists():
        rank_files = [path / "e2e.csv"]
    if not rank_files:
        raise ValueError(f"missing rank/e2e.csv: {path}")
    per_rank = []
    for f in rank_files:
        latencies = []
        with f.open(newline="", encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                if "latency (s)" not in row:
                    raise ValueError(f"missing 'latency (s)' column: {f}")
                v = number(row["latency (s)"])
                if v <= 0:
                    raise ValueError(f"invalid latency: {f}")
                latencies.append(v)
        if not latencies:
            raise ValueError(f"empty latency results: {f}")
        per_rank.append(latencies)
    if len({len(x) for x in per_rank}) != 1:
        raise ValueError("rank CSV files have different iteration counts; refusing to infer alignment")

    # The e2e.csv rows are per-rank *batches*, not individual user requests.
    # Max over ranks is the latency of a globally synchronized batch iteration.
    iter_s = [max(rank[i] for rank in per_rank) for i in range(len(per_rank[0]))]
    mean_ms = statistics.mean(iter_s) * 1000
    p95_ms = percentile(iter_s, 0.95) * 1000
    tp = None
    n_samples = None
    elapsed = None
    notes = ["latency = synchronized global-batch latency (max across ranks), not request latency"]
    start, end = data.get("start"), data.get("end")
    batch_per_gpu = data.get("batch_size")
    world_size = data.get("world_size")
    num_samples = data.get("num_samples")
    if all(isinstance(v, (float, int)) for v in (start, end)) and end > start:
        elapsed = float(end - start)
        if isinstance(batch_per_gpu, int) and batch_per_gpu > 0 and isinstance(world_size, int) and world_size == len(rank_files):
            # For a full DistributedSampler run (drop_last=False), *processed* samples can
            # exceed original dataset length because sampler pads with duplicate indices.
            n_samples = len(per_rank[0]) * batch_per_gpu * world_size
            if isinstance(num_samples, int) and num_samples > 0 and n_samples != num_samples:
                notes.append(f"sampler processed {n_samples} samples vs num_samples={num_samples} (padding/partial batch possible)")
                if n_samples > num_samples:
                    notes.append("throughput counts sampler-processed samples, including potential duplicates")
            tp = n_samples / elapsed
        else:
            notes.append("throughput unavailable: need batch_size, world_size and complete per-rank CSV")
    else:
        notes.append("throughput unavailable: data.json lacks valid start/end timestamps")
    if "warmup_rounds" in data:
        notes.append("verify e2e.csv includes measured iterations only; warmup is not subtracted by this script")
    return Result(model, method, str(path), len(iter_s), mean_ms, p95_ms, tp, "OK", "; ".join(notes), n_samples, elapsed)


def read_result(source: str, model: str, method: str) -> Result:
    p = Path(source).expanduser()
    if not p.exists():
        raise FileNotFoundError(p)
    return parse_directory(p, model, method) if p.is_dir() else parse_log(p, model, method)


def fmt(v: Optional[float], digits: int = 2) -> str:
    return "N/A" if v is None else f"{v:.{digits}f}"


def improvement(new: Optional[float], old: Optional[float], higher: bool) -> Optional[float]:
    if new is None or old is None or old <= 0 or new <= 0:
        return None
    return 100.0 * ((new / old - 1) if higher else (1 - new / old))


def main():
    ap = argparse.ArgumentParser(description="Compare 2 algorithms x 2 MoE models from real logs / CSV directories")
    ap.add_argument("--switch-deepspeed", required=True)
    ap.add_argument("--switch-dsca", required=True)
    ap.add_argument("--qwen-deepspeed", required=True)
    ap.add_argument("--qwen-dsca", required=True)
    ap.add_argument("--output", default="moe_comparison_report")
    args = ap.parse_args()
    specifications = [
        ("Switch-base-64", "DeepSpeed", args.switch_deepspeed),
        ("Switch-base-64", "DSCA-MoE", args.switch_dsca),
        ("Qwen1.5-MoE-A2.7B", "DeepSpeed", args.qwen_deepspeed),
        ("Qwen1.5-MoE-A2.7B", "DSCA-MoE", args.qwen_dsca),
    ]
    results = []
    for model, method, source in specifications:
        try:
            r = read_result(source, model, method)
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as e:
            print(f"[ERROR] {model} / {method}: {e}", file=sys.stderr)
            return 2
        results.append(r)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["model", "method", "iterations", "throughput_samples_s", "mean_latency_ms", "p95_latency_ms", "status", "source", "notes"])
        for r in results:
            writer.writerow([r.model, r.method, r.iters, "" if r.throughput is None else f"{r.throughput:.8f}",
                             f"{r.mean_ms:.8f}", f"{r.p95_ms:.8f}", r.status, r.source, r.note])
    grouped = {}
    for r in results:
        grouped.setdefault(r.model, {})[r.method] = r
    change_rows = []
    for model, group in grouped.items():
        b, d = group["DeepSpeed"], group["DSCA-MoE"]
        change_rows.append((model, improvement(d.throughput, b.throughput, True),
                            improvement(d.mean_ms, b.mean_ms, False), improvement(d.p95_ms, b.p95_ms, False)))
    with (out / "improvements.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["model", "throughput_improvement_pct", "mean_latency_reduction_pct", "p95_latency_reduction_pct"])
        for model, a, b, c in change_rows:
            writer.writerow([model, "" if a is None else f"{a:.6f}", fmt(b, 6) if b is not None else "", fmt(c, 6) if c is not None else ""])

    lines = ["# MoE 2×2 Performance Comparison", "", "Generated: " + datetime.now().isoformat(timespec="seconds"), "",
             "## Raw performance", "", "| Model | Method | Iters | Throughput (samples/s) | Mean latency (ms) | P95 latency (ms) |",
             "|---|---|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r.model} | {r.method} | {r.iters} | {fmt(r.throughput)} | {fmt(r.mean_ms)} | {fmt(r.p95_ms)} |")
    lines.extend(["", "## DSCA-MoE relative to DeepSpeed", "", "Positive means improvement; negative means regression.", "",
                  "| Model | Throughput increase (%) | Mean latency reduction (%) | P95 latency reduction (%) |",
                  "|---|---:|---:|---:|"])
    for model, a, b, c in change_rows:
        lines.append(f"| {model} | {fmt(a)} | {fmt(b)} | {fmt(c)} |")
    lines.extend(["", "## Sources and metric definitions", "",
                  "- Log mode copies metrics from the actual Summary table, with successful status required.",
                  "- Directory mode computes iteration latency as the **maximum rank batch latency** for each iteration, and P95 across those global iterations.",
                  "- Directory mode throughput requires measured run start/end timestamps, plus rank count and batch_size metadata. It counts processed sampler slots, potentially including padded duplicates.",
                  "- Comparing multiple models requires checking that workload, batch size, dtype, warmup and inference mode match. This tool does not verify numerical equivalence or actual algorithm implementation.",
                  "- A normal forward pass benchmark is not an online autoregressive serving benchmark.", ""])
    for r in results:
        lines.append(f"- **{r.model} / {r.method}**: `{r.source}`. {r.note}")
    report = "\n".join(lines) + "\n"
    (out / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved reports to: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
