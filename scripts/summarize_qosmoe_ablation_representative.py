#!/usr/bin/env python3
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path("ablation_outputs/qosmoe_ablation_representative_grid")
OUT = Path("ablation_outputs/final_saved_results")
OUT.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUT / "qosmoe_ablation_representative_summary.csv"
DELTA_CSV = OUT / "qosmoe_ablation_representative_delta.csv"
LATEX_TEX = OUT / "qosmoe_ablation_representative_table.tex"

VARIANT_ORDER = [
    "QoSMoE-Full",
    "QoSMoE-wo-Async",
    "QoSMoE-wo-LB",
    "QoSMoE-wo-Prefetch",
]

def parse_setting(name: str):
    """
    Example:
    R01_B32_L256_C8_S095_E12
    """
    m = re.match(r"(R\d+)_B(\d+)_L(\d+)_C(\d+)_S(\d+)_E(\d+)", name)
    if not m:
        return None
    return {
        "id": m.group(1),
        "batch": int(m.group(2)),
        "seq_len": int(m.group(3)),
        "cache": int(m.group(4)),
        "skew": int(m.group(5)) / 100.0,
        "skewed_experts": int(m.group(6)),
    }

def read_latency_csv(path: Path):
    df = pd.read_csv(path)

    candidates = [
        "latency (s)",
        "latency_s",
        "latency",
        "e2e_latency_s",
        "time_s",
    ]

    col = None
    for c in candidates:
        if c in df.columns:
            col = c
            break

    if col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 1:
            col = numeric_cols[0]
        else:
            raise ValueError(f"Cannot find latency column in {path}. Columns={list(df.columns)}")

    lat = df[col].astype(float).to_numpy()
    lat = lat[np.isfinite(lat)]
    return lat

rows = []

for setting_dir in sorted(ROOT.glob("R*")):
    if not setting_dir.is_dir():
        continue

    parsed = parse_setting(setting_dir.name)
    if parsed is None:
        continue

    # 不统计 R12
    if parsed["id"] == "R12":
        continue

    for variant in VARIANT_ORDER:
        variant_dir = setting_dir / variant
        if not variant_dir.exists():
            rows.append({
                **parsed,
                "variant": variant,
                "status": "missing",
                "iters": 0,
                "mean_s": np.nan,
                "p50_s": np.nan,
                "p95_s": np.nan,
                "p99_s": np.nan,
                "min_s": np.nan,
                "max_s": np.nan,
                "throughput_iter_per_s": np.nan,
                "num_e2e_files": 0,
            })
            continue

        e2e_files = sorted(variant_dir.rglob("e2e.csv"))

        if not e2e_files:
            rows.append({
                **parsed,
                "variant": variant,
                "status": "missing",
                "iters": 0,
                "mean_s": np.nan,
                "p50_s": np.nan,
                "p95_s": np.nan,
                "p99_s": np.nan,
                "min_s": np.nan,
                "max_s": np.nan,
                "throughput_iter_per_s": np.nan,
                "num_e2e_files": 0,
            })
            continue

        all_lat = []
        for f in e2e_files:
            try:
                all_lat.append(read_latency_csv(f))
            except Exception as e:
                print(f"[WARN] failed to read {f}: {e}")

        if not all_lat:
            status = "bad_csv"
            lat = np.array([])
        else:
            status = "ok"
            lat = np.concatenate(all_lat)

        if len(lat) == 0:
            rows.append({
                **parsed,
                "variant": variant,
                "status": status,
                "iters": 0,
                "mean_s": np.nan,
                "p50_s": np.nan,
                "p95_s": np.nan,
                "p99_s": np.nan,
                "min_s": np.nan,
                "max_s": np.nan,
                "throughput_iter_per_s": np.nan,
                "num_e2e_files": len(e2e_files),
            })
        else:
            rows.append({
                **parsed,
                "variant": variant,
                "status": status,
                "iters": int(len(lat)),
                "mean_s": float(np.mean(lat)),
                "p50_s": float(np.percentile(lat, 50)),
                "p95_s": float(np.percentile(lat, 95)),
                "p99_s": float(np.percentile(lat, 99)),
                "min_s": float(np.min(lat)),
                "max_s": float(np.max(lat)),
                "throughput_iter_per_s": float(len(lat) / np.sum(lat)),
                "num_e2e_files": len(e2e_files),
            })

summary = pd.DataFrame(rows)
summary = summary.sort_values(["id", "variant"])
summary.to_csv(SUMMARY_CSV, index=False)

print(f"[OK] wrote {SUMMARY_CSV}")
print(summary.to_string(index=False))

# -----------------------------
# Build delta table
# -----------------------------
delta_rows = []

for sid, g in summary.groupby("id"):
    full = g[g["variant"] == "QoSMoE-Full"]

    if full.empty or full["status"].iloc[0] != "ok":
        print(f"[WARN] skip {sid}: Full missing")
        continue

    full = full.iloc[0]

    row = {
        "id": sid,
        "batch": int(full["batch"]),
        "seq_len": int(full["seq_len"]),
        "cache": int(full["cache"]),
        "skew": float(full["skew"]),
        "skewed_experts": int(full["skewed_experts"]),

        "full_mean_s": float(full["mean_s"]),
        "full_p95_s": float(full["p95_s"]),
        "full_p99_s": float(full["p99_s"]),
        "full_throughput_iter_per_s": float(full["throughput_iter_per_s"]),
    }

    for variant, short in [
        ("QoSMoE-wo-Async", "wo_async"),
        ("QoSMoE-wo-LB", "wo_lb"),
        ("QoSMoE-wo-Prefetch", "wo_prefetch"),
    ]:
        v = g[g["variant"] == variant]

        if v.empty or v["status"].iloc[0] != "ok":
            row[f"{short}_mean_delta_pct"] = np.nan
            row[f"{short}_p95_delta_pct"] = np.nan
            row[f"{short}_p99_delta_pct"] = np.nan
            row[f"{short}_throughput_drop_pct"] = np.nan
            continue

        v = v.iloc[0]

        row[f"{short}_mean_delta_pct"] = (
            (float(v["mean_s"]) - float(full["mean_s"])) / float(full["mean_s"]) * 100.0
        )
        row[f"{short}_p95_delta_pct"] = (
            (float(v["p95_s"]) - float(full["p95_s"])) / float(full["p95_s"]) * 100.0
        )
        row[f"{short}_p99_delta_pct"] = (
            (float(v["p99_s"]) - float(full["p99_s"])) / float(full["p99_s"]) * 100.0
        )
        row[f"{short}_throughput_drop_pct"] = (
            (float(full["throughput_iter_per_s"]) - float(v["throughput_iter_per_s"]))
            / float(full["throughput_iter_per_s"]) * 100.0
        )

    delta_rows.append(row)

delta = pd.DataFrame(delta_rows)
delta = delta.sort_values("id")
delta.to_csv(DELTA_CSV, index=False)

print(f"[OK] wrote {DELTA_CSV}")
print(delta.to_string(index=False))

# -----------------------------
# LaTeX table
# -----------------------------
def fmt_s(x):
    if pd.isna(x):
        return "--"
    return f"{x:.3f}"

def fmt_pct(x):
    if pd.isna(x):
        return "--"
    return f"{x:+.2f}\\%"

def cell(row, prefix):
    return (
        f"{fmt_pct(row[f'{prefix}_mean_delta_pct'])} / "
        f"{fmt_pct(row[f'{prefix}_p95_delta_pct'])} / "
        f"{fmt_pct(row[f'{prefix}_throughput_drop_pct'])}"
    )

latex_lines = []
latex_lines.append(r"\begin{table*}[t]")
latex_lines.append(r"\centering")
latex_lines.append(r"\small")
latex_lines.append(r"\caption{Ablation study of QoSMoE on representative workloads. Each ablation cell reports mean latency increase / P95 latency increase / throughput drop relative to full QoSMoE.}")
latex_lines.append(r"\label{tab:qosmoe_ablation_representative}")
latex_lines.append(r"\resizebox{\textwidth}{!}{%")
latex_lines.append(r"\begin{tabular}{c|ccc|ccc|ccc}")
latex_lines.append(r"\toprule")
latex_lines.append(
    r"ID & B & L & C & "
    r"Full Mean (s) & Full P95 (s) & Full Thr. & "
    r"w/o Async & w/o LB & w/o Prefetch \\"
)
latex_lines.append(r"\midrule")

for _, r in delta.iterrows():
    latex_lines.append(
        f"{r['id']} & "
        f"{int(r['batch'])} & "
        f"{int(r['seq_len'])} & "
        f"{int(r['cache'])} & "
        f"{fmt_s(r['full_mean_s'])} & "
        f"{fmt_s(r['full_p95_s'])} & "
        f"{fmt_s(r['full_throughput_iter_per_s'])} & "
        f"{cell(r, 'wo_async')} & "
        f"{cell(r, 'wo_lb')} & "
        f"{cell(r, 'wo_prefetch')} \\\\"
    )

latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}%")
latex_lines.append(r"}")
latex_lines.append(r"\end{table*}")

LATEX_TEX.write_text("\n".join(latex_lines), encoding="utf-8")
print(f"[OK] wrote {LATEX_TEX}")
