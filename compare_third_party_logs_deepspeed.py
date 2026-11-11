#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
透明的第三方结果汇总工具（DeepSpeed 对比版）。

第三方分别完成 DeepSpeed 和 DSCA-MoE 两次测试并保存日志后，
本脚本直接从两个真实日志的 "===== Summary =====" 表格读取：
  - mean latency
  - P95 latency
  - throughput
并计算 DSCA-MoE 相比 DeepSpeed 的性能变化。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict


def parse_summary(path: Path) -> Dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [x.strip() for x in text.splitlines() if x.strip()]

    for i, line in enumerate(lines):
        cols = line.split()
        required = {"system", "iters", "mean_s", "p95_s", "samples_per_s", "status"}
        if required.issubset(set(cols)) and i + 1 < len(lines):
            row = lines[i + 1].split()
            if len(row) >= len(cols):
                d = dict(zip(cols, row[:len(cols)]))
                return {
                    "iters": int(float(d["iters"])),
                    "mean_ms": float(d["mean_s"]) * 1000.0,
                    "p95_ms": float(d["p95_s"]) * 1000.0,
                    "throughput": float(d["samples_per_s"]),
                    "status": d["status"],
                }

    raise RuntimeError(f"未在日志中找到 Summary 表格：{path}")


def throughput_sentence(new: float, base: float) -> str:
    change = (new / base - 1.0) * 100.0
    if change >= 0:
        return f"Throughput: increased by {change:.2f}%"
    return f"Throughput: decreased by {abs(change):.2f}%"


def latency_sentence(metric: str, new: float, base: float) -> str:
    change = (new / base - 1.0) * 100.0
    if change < 0:
        return f"{metric}: reduced by {abs(change):.2f}%"
    if change > 0:
        return f"{metric}: increased by {change:.2f}%"
    return f"{metric}: unchanged"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--deepspeed", type=Path, required=True)
    ap.add_argument("--dsca", type=Path, required=True)
    args = ap.parse_args()

    ds = parse_summary(args.deepspeed)
    ours = parse_summary(args.dsca)

    print("=" * 88)
    print(f"{args.model}  Third-Party Test Result")
    print("=" * 88)
    print(f"{'Method':<16}{'Iters':>8}{'Throughput(samples/s)':>24}{'P95(ms)':>14}{'Mean(ms)':>14}")
    print("-" * 76)

    for name, m in [
        ("DeepSpeed", ds),
        ("DSCA-MoE", ours),
    ]:
        print(
            f"{name:<16}{m['iters']:>8d}"
            f"{m['throughput']:>24.2f}"
            f"{m['p95_ms']:>14.2f}"
            f"{m['mean_ms']:>14.2f}"
        )

    print()
    print("DSCA-MoE vs DeepSpeed")
    print("  " + throughput_sentence(ours["throughput"], ds["throughput"]))
    print("  " + latency_sentence("P95 latency", ours["p95_ms"], ds["p95_ms"]))
    print("  " + latency_sentence("Mean latency", ours["mean_ms"], ds["mean_ms"]))

    print()
    print("All values are parsed directly from the two benchmark logs.")


if __name__ == "__main__":
    main()
