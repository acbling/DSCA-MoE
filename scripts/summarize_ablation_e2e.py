#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path


def percentile(xs, p):
    xs = sorted(float(x) for x in xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def read_latencies(run_dir):
    vals = []
    for f in Path(run_dir).glob("*/e2e.csv"):
        with f.open() as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                if "latency (s)" in row:
                    vals.append(float(row["latency (s)"]))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="ablation_results_qosmoe")
    ap.add_argument("--slo-ms", type=float, default=None, help="If set, compute violation rate against this latency threshold in ms.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for run_dir in sorted(root.iterdir() if root.exists() else []):
        if not run_dir.is_dir():
            continue
        lats_s = read_latencies(run_dir)
        if not lats_s:
            continue
        lats_ms = [x * 1000.0 for x in lats_s]
        total_time = sum(lats_s)
        throughput = len(lats_s) / total_time if total_time > 0 else float("nan")
        slo_vr = ""
        if args.slo_ms is not None:
            slo_vr = 100.0 * sum(1 for x in lats_ms if x > args.slo_ms) / len(lats_ms)
        rows.append({
            "method": run_dir.name,
            "num_samples": len(lats_ms),
            "mean_latency_ms": sum(lats_ms) / len(lats_ms),
            "p50_latency_ms": percentile(lats_ms, 50),
            "p95_latency_ms": percentile(lats_ms, 95),
            "p99_latency_ms": percentile(lats_ms, 99),
            "throughput_iter_per_s": throughput,
            "slo_violation_pct": slo_vr,
        })

    out = Path(args.out) if args.out else root / "ablation_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "num_samples", "mean_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms", "throughput_iter_per_s", "slo_violation_pct"]
    with out.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[OK] wrote {out}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
