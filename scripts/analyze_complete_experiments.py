#!/usr/bin/env python3

import os
import re
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

DEFAULT_ROOT = (
    "/home/lcf/MoE_Syn/HarMoEny-main/"
    "experiment_outputs/complete_20260923_095020"
)

METHOD_MAP = {
    "deepspeed": "DeepSpeed",
    "harmony": "HarMoEny",
    "qosmoe": "DSCA-MoE",
}

METHOD_ORDER = [
    "DeepSpeed",
    "HarMoEny",
    "DSCA-MoE",
]

COLORS = {
    "DeepSpeed": "#E69F00",
    "HarMoEny": "#009E73",
    "DSCA-MoE": "#0072B2",
}

MARKERS = {
    "DeepSpeed": "o",
    "HarMoEny": "s",
    "DSCA-MoE": "^",
}

LINESTYLES = {
    "DeepSpeed": "--",
    "HarMoEny": "-.",
    "DSCA-MoE": "-",
}


# ============================================================
# Matplotlib style
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Helpers
# ============================================================

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def ci95_half_width(values):
    """
    95% CI half width using Student-t critical values.
    Appropriate for the 3-repeat experiment.
    """
    vals = pd.Series(values).dropna().astype(float).values
    n = len(vals)

    if n <= 1:
        return 0.0

    std = np.std(vals, ddof=1)
    sem = std / math.sqrt(n)

    # df = n - 1
    t_critical = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
    }

    df = n - 1

    if df in t_critical:
        tval = t_critical[df]
    else:
        tval = 1.96

    return tval * sem


def save_figure(fig, fig_dir, name):
    pdf = fig_dir / f"{name}.pdf"
    png = fig_dir / f"{name}.png"

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        png,
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[FIG] {pdf}")


def style_axis(ax):
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.5,
        alpha=0.28,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_global_legend(fig, axes, ncol=3, y=1.02):
    handles = []
    labels = []

    for ax in np.ravel(axes):
        h, l = ax.get_legend_handles_labels()

        for hh, ll in zip(h, l):
            if ll not in labels and ll in METHOD_ORDER:
                handles.append(hh)
                labels.append(ll)

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, y),
            ncol=ncol,
            frameon=False,
        )


def plot_metric(
    ax,
    dataframe,
    x,
    y,
    yerr=None,
    xlabel="",
    ylabel="",
    title="",
    xticks=None,
    xticklabels=None,
):
    for method in METHOD_ORDER:

        tmp = dataframe[
            dataframe["method"] == method
        ].sort_values(x)

        if len(tmp) == 0:
            continue

        err = None

        if yerr is not None and yerr in tmp.columns:
            err = tmp[yerr].fillna(0).values

        ax.errorbar(
            tmp[x],
            tmp[y],
            yerr=err,
            marker=MARKERS[method],
            markersize=5.5,
            linewidth=1.8,
            capsize=2.5,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            label=method,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if xticks is not None:
        ax.set_xticks(xticks)

    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)

    style_axis(ax)


# ============================================================
# Parse benchmark summary from one log
# ============================================================

def parse_summary(log_path):

    text = Path(log_path).read_text(
        errors="ignore"
    )

    result = None

    # Robust parsing:
    #
    # system iters mean p50 p95 min max samples_per_s status
    #
    for line in text.splitlines():

        parts = line.strip().split()

        if len(parts) < 9:
            continue

        if parts[-1].lower() != "ok":
            continue

        try:
            iters = int(parts[-8])
            mean_s = float(parts[-7])
            p50_s = float(parts[-6])
            p95_s = float(parts[-5])
            min_s = float(parts[-4])
            max_s = float(parts[-3])
            samples_per_s = float(parts[-2])
        except Exception:
            continue

        result = {
            "summary_system": parts[0],
            "iters": iters,
            "mean_s": mean_s,
            "p50_s": p50_s,
            "p95_s": p95_s,
            "min_s": min_s,
            "max_s": max_s,
            "reported_samples_per_s": samples_per_s,
        }

    if result is None:
        return None

    # --------------------------------------------
    # Optional QoS metrics
    # --------------------------------------------

    miss_matches = re.findall(
        r"miss_rate_pct\s*=\s*([0-9.eE+\-]+)",
        text,
        flags=re.IGNORECASE,
    )

    if miss_matches:
        result["slo_violation_pct"] = float(
            miss_matches[-1]
        )
    else:
        result["slo_violation_pct"] = np.nan


    load_matches = re.findall(
        r"load_std\s*=\s*([0-9.eE+\-]+)",
        text,
        flags=re.IGNORECASE,
    )

    if load_matches:
        result["load_std"] = float(
            load_matches[-1]
        )
    else:
        result["load_std"] = np.nan


    return result


# ============================================================
# Load run_status.csv
# ============================================================

def load_all_runs(root):

    status_file = root / "run_status.csv"

    if not status_file.exists():
        raise FileNotFoundError(
            f"Missing: {status_file}"
        )

    status = pd.read_csv(status_file)

    print(
        f"[INFO] run_status rows: {len(status)}"
    )

    rows = []

    for _, r in status.iterrows():

        if str(r["status"]).upper() != "SUCCESS":
            print(
                "[WARN] Non-success run:",
                r["family"],
                r["system"],
                r["repeat"],
            )
            continue

        log_path = Path(str(r["log"]))

        if not log_path.exists():

            # Fallback if directory has moved.
            candidate = (
                root
                / "logs"
                / str(r["family"])
                / log_path.name
            )

            if candidate.exists():
                log_path = candidate
            else:
                print(
                    "[WARN] Missing log:",
                    log_path
                )
                continue

        parsed = parse_summary(log_path)

        if parsed is None:
            print(
                "[WARN] Cannot parse summary:",
                log_path
            )
            continue

        system_key = str(r["system"]).lower()

        method = METHOD_MAP.get(
            system_key,
            system_key
        )

        gpus = int(r["gpus"])

        # IMPORTANT:
        # run_all benchmark reports local/rank throughput.
        # Aggregate system throughput is rank throughput * world_size.
        system_throughput = (
            parsed["reported_samples_per_s"]
            * gpus
        )

        row = {
            "family": r["family"],
            "system": system_key,
            "method": method,
            "gpus": gpus,
            "batch": int(r["batch"]),
            "num_samples": int(r["num_samples"]),
            "seq_len": int(r["seq_len"]),
            "cache": int(r["cache"]),
            "skew": float(r["skew"]),
            "deadline_s": float(r["deadline"]),
            "repeat": int(r["repeat"]),
            "log": str(log_path),

            **parsed,

            "mean_ms":
                parsed["mean_s"] * 1000.0,

            "p50_ms":
                parsed["p50_s"] * 1000.0,

            "p95_ms":
                parsed["p95_s"] * 1000.0,

            "min_ms":
                parsed["min_s"] * 1000.0,

            "max_ms":
                parsed["max_s"] * 1000.0,

            "system_throughput":
                system_throughput,

            "per_gpu_throughput":
                system_throughput / gpus,

            "tail_amplification":
                parsed["p95_s"]
                / parsed["p50_s"]
                if parsed["p50_s"] > 0
                else np.nan,

            "latency_spread_pct":
                (
                    parsed["max_s"]
                    - parsed["min_s"]
                )
                / parsed["mean_s"]
                * 100.0
                if parsed["mean_s"] > 0
                else np.nan,

            "deadline_normalized_p95":
                parsed["p95_s"]
                / float(r["deadline"])
                if float(r["deadline"]) > 0
                else np.nan,
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError(
            "No valid benchmark results parsed."
        )

    return df


# ============================================================
# Aggregate repeats
# ============================================================

def aggregate(df):

    group_cols = [
        "family",
        "system",
        "method",
        "gpus",
        "batch",
        "num_samples",
        "seq_len",
        "cache",
        "skew",
        "deadline_s",
    ]

    metric_cols = [
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "min_ms",
        "max_ms",
        "reported_samples_per_s",
        "system_throughput",
        "per_gpu_throughput",
        "tail_amplification",
        "latency_spread_pct",
        "deadline_normalized_p95",
        "slo_violation_pct",
        "load_std",
    ]

    rows = []

    for keys, g in df.groupby(
        group_cols,
        dropna=False
    ):

        row = dict(zip(
            group_cols,
            keys
        ))

        row["runs"] = len(g)

        for col in metric_cols:

            vals = g[col].dropna()

            if len(vals) == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_std"] = np.nan
                row[f"{col}_ci95"] = np.nan

            else:
                row[f"{col}_mean"] = vals.mean()

                row[f"{col}_std"] = (
                    vals.std(ddof=1)
                    if len(vals) > 1
                    else 0.0
                )

                row[f"{col}_ci95"] = (
                    ci95_half_width(vals)
                )

        rows.append(row)

    result = pd.DataFrame(rows)

    return result


# ============================================================
# Strong scaling derived metrics
# ============================================================

def derive_strong(summary):

    strong = summary[
        summary["family"] == "strong"
    ].copy()

    strong["speedup"] = np.nan
    strong["parallel_efficiency_pct"] = np.nan

    for method in strong["method"].unique():

        tmp = strong[
            strong["method"] == method
        ]

        base = tmp[
            tmp["gpus"] == 1
        ]

        if len(base) == 0:
            continue

        base_thr = float(
            base.iloc[0][
                "system_throughput_mean"
            ]
        )

        idx = (
            strong["method"] == method
        )

        strong.loc[
            idx,
            "speedup"
        ] = (
            strong.loc[
                idx,
                "system_throughput_mean"
            ]
            / base_thr
        )

        strong.loc[
            idx,
            "parallel_efficiency_pct"
        ] = (
            strong.loc[
                idx,
                "speedup"
            ]
            / strong.loc[
                idx,
                "gpus"
            ]
            * 100.0
        )

    return strong


# ============================================================
# Weak scaling derived metrics
# ============================================================

def derive_weak(summary):

    weak = summary[
        summary["family"] == "weak"
    ].copy()

    weak["weak_efficiency_pct"] = np.nan

    for method in weak["method"].unique():

        tmp = weak[
            weak["method"] == method
        ]

        base = tmp[
            tmp["gpus"] == 1
        ]

        if len(base) == 0:
            continue

        base_per_gpu = float(
            base.iloc[0][
                "per_gpu_throughput_mean"
            ]
        )

        idx = (
            weak["method"] == method
        )

        weak.loc[
            idx,
            "weak_efficiency_pct"
        ] = (
            weak.loc[
                idx,
                "per_gpu_throughput_mean"
            ]
            / base_per_gpu
            * 100.0
        )

    return weak


# ============================================================
# Relative improvement tables
# ============================================================

def relative_improvements(summary):

    config_cols = [
        "family",
        "gpus",
        "batch",
        "num_samples",
        "seq_len",
        "cache",
        "skew",
        "deadline_s",
    ]

    rows = []

    for keys, g in summary.groupby(
        config_cols,
        dropna=False
    ):

        dsca = g[
            g["method"] == "DSCA-MoE"
        ]

        if len(dsca) == 0:
            continue

        dsca = dsca.iloc[0]

        for baseline in [
            "DeepSpeed",
            "HarMoEny",
        ]:

            base = g[
                g["method"] == baseline
            ]

            if len(base) == 0:
                continue

            base = base.iloc[0]

            row = dict(
                zip(
                    config_cols,
                    keys
                )
            )

            row["baseline"] = baseline

            dsca_thr = dsca[
                "system_throughput_mean"
            ]

            base_thr = base[
                "system_throughput_mean"
            ]

            dsca_p95 = dsca[
                "p95_ms_mean"
            ]

            base_p95 = base[
                "p95_ms_mean"
            ]

            row[
                "throughput_gain_pct"
            ] = (
                dsca_thr / base_thr - 1
            ) * 100.0

            row[
                "p95_reduction_pct"
            ] = (
                1
                - dsca_p95 / base_p95
            ) * 100.0

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Figure 1: Strong scaling
# ============================================================

def plot_strong(strong, fig_dir):

    if len(strong) == 0:
        return

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.1, 5.1),
    )

    # (a) Throughput
    plot_metric(
        axes[0, 0],
        strong,
        "gpus",
        "system_throughput_mean",
        "system_throughput_ci95",
        "Number of GPUs",
        "System Throughput (samples/s)",
        "(a) Aggregate Throughput",
        xticks=[1, 2, 4],
    )

    # (b) P95
    plot_metric(
        axes[0, 1],
        strong,
        "gpus",
        "p95_ms_mean",
        "p95_ms_ci95",
        "Number of GPUs",
        "P95 Latency (ms)",
        "(b) Tail Latency",
        xticks=[1, 2, 4],
    )

    # (c) speedup
    ax = axes[1, 0]

    for method in METHOD_ORDER:

        tmp = strong[
            strong["method"] == method
        ].sort_values("gpus")

        if len(tmp) == 0:
            continue

        ax.plot(
            tmp["gpus"],
            tmp["speedup"],
            marker=MARKERS[method],
            markersize=5.5,
            linewidth=1.8,
            linestyle=LINESTYLES[method],
            color=COLORS[method],
            label=method,
        )

    ax.plot(
        [1, 2, 4],
        [1, 2, 4],
        color="0.55",
        linestyle=":",
        linewidth=1.2,
        label="_nolegend_",
    )

    ax.set_xlabel(
        "Number of GPUs"
    )

    ax.set_ylabel(
        "Speedup (×)"
    )

    ax.set_title(
        "(c) Strong-Scaling Speedup"
    )

    ax.set_xticks(
        [1, 2, 4]
    )

    style_axis(ax)


    # (d) efficiency
    ax = axes[1, 1]

    for method in METHOD_ORDER:

        tmp = strong[
            strong["method"] == method
        ].sort_values("gpus")

        if len(tmp) == 0:
            continue

        ax.plot(
            tmp["gpus"],
            tmp[
                "parallel_efficiency_pct"
            ],
            marker=MARKERS[method],
            markersize=5.5,
            linewidth=1.8,
            linestyle=LINESTYLES[method],
            color=COLORS[method],
            label=method,
        )

    ax.axhline(
        100,
        color="0.55",
        linestyle=":",
        linewidth=1.2,
    )

    ax.set_xlabel(
        "Number of GPUs"
    )

    ax.set_ylabel(
        "Parallel Efficiency (%)"
    )

    ax.set_title(
        "(d) Parallel Efficiency"
    )

    ax.set_xticks(
        [1, 2, 4]
    )

    style_axis(ax)

    add_global_legend(
        fig,
        axes,
        y=1.01
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    save_figure(
        fig,
        fig_dir,
        "fig1_strong_scaling"
    )


# ============================================================
# Figure 2: Weak scaling
# ============================================================

def plot_weak(weak, fig_dir):

    if len(weak) == 0:
        return

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.0, 3.0),
    )

    plot_metric(
        axes[0],
        weak,
        "gpus",
        "system_throughput_mean",
        "system_throughput_ci95",
        "Number of GPUs",
        "System Throughput (samples/s)",
        "(a) Aggregate Throughput",
        xticks=[1, 2, 4],
    )

    # Weak scaling efficiency
    ax = axes[1]

    for method in METHOD_ORDER:

        tmp = weak[
            weak["method"] == method
        ].sort_values("gpus")

        if len(tmp) == 0:
            continue

        ax.plot(
            tmp["gpus"],
            tmp["weak_efficiency_pct"],
            marker=MARKERS[method],
            markersize=5.5,
            linewidth=1.8,
            linestyle=LINESTYLES[method],
            color=COLORS[method],
            label=method,
        )

    ax.axhline(
        100,
        color="0.55",
        linestyle=":",
        linewidth=1.2,
    )

    ax.set_xlabel(
        "Number of GPUs"
    )

    ax.set_ylabel(
        "Weak-Scaling Efficiency (%)"
    )

    ax.set_title(
        "(b) Scaling Efficiency"
    )

    ax.set_xticks(
        [1, 2, 4]
    )

    style_axis(ax)


    plot_metric(
        axes[2],
        weak,
        "gpus",
        "p95_ms_mean",
        "p95_ms_ci95",
        "Number of GPUs",
        "P95 Latency (ms)",
        "(c) Tail Latency",
        xticks=[1, 2, 4],
    )

    add_global_legend(
        fig,
        axes,
        y=1.02
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.91]
    )

    save_figure(
        fig,
        fig_dir,
        "fig2_weak_scaling"
    )


# ============================================================
# Generic 1x2 sensitivity figure
# ============================================================

def sensitivity_figure(
    data,
    x_col,
    x_label,
    title_prefix,
    fig_name,
    fig_dir,
    xticks=None,
    xticklabels=None,
):

    if len(data) == 0:
        return

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.9),
    )

    plot_metric(
        axes[0],
        data,
        x_col,
        "system_throughput_mean",
        "system_throughput_ci95",
        x_label,
        "System Throughput (samples/s)",
        f"(a) {title_prefix}: Throughput",
        xticks,
        xticklabels,
    )

    plot_metric(
        axes[1],
        data,
        x_col,
        "p95_ms_mean",
        "p95_ms_ci95",
        x_label,
        "P95 Latency (ms)",
        f"(b) {title_prefix}: P95 Latency",
        xticks,
        xticklabels,
    )

    add_global_legend(
        fig,
        axes,
        y=1.03
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.90]
    )

    save_figure(
        fig,
        fig_dir,
        fig_name
    )


# ============================================================
# Batch
# ============================================================

def plot_batch(summary, fig_dir):

    d = summary[
        summary["family"].isin([
            "batch_b4",
            "batch_b8",
            "batch_b16",
        ])
    ].copy()

    if len(d) == 0:
        return

    d["batch_x"] = d["batch"]

    sensitivity_figure(
        d,
        "batch_x",
        "Batch Size per GPU",
        "Batch Sensitivity",
        "fig3_batch_sensitivity",
        fig_dir,
        xticks=[4, 8, 16],
    )


# ============================================================
# Router skew
# ============================================================

def plot_skew(summary, fig_dir):

    mapping = {
        "skew_natural": 0,
        "skew_moderate": 1,
        "skew_severe": 2,
    }

    d = summary[
        summary["family"].isin(
            mapping.keys()
        )
    ].copy()

    if len(d) == 0:
        return

    d["skew_x"] = d["family"].map(
        mapping
    )

    sensitivity_figure(
        d,
        "skew_x",
        "Routing Distribution",
        "Router Skew",
        "fig4_router_skew",
        fig_dir,
        xticks=[0, 1, 2],
        xticklabels=[
            "Natural",
            "Moderate",
            "Severe",
        ],
    )


# ============================================================
# Cache
# ============================================================

def plot_cache(summary, fig_dir):

    d = summary[
        summary["family"].isin([
            "cache_c2",
            "cache_c4",
            "cache_c8",
        ])
    ].copy()

    if len(d) == 0:
        return

    d["cache_x"] = d["cache"]

    sensitivity_figure(
        d,
        "cache_x",
        "Expert Cache Size",
        "Cache Pressure",
        "fig5_cache_pressure",
        fig_dir,
        xticks=[2, 4, 8],
    )


# ============================================================
# SLO
# ============================================================

def plot_slo(summary, fig_dir):

    d = summary[
        summary["family"].str.startswith(
            "slo_d"
        )
    ].copy()

    if len(d) == 0:
        return

    d = d.sort_values(
        "deadline_s"
    )

    has_violation = (
        d["slo_violation_pct_mean"]
        .notna()
        .any()
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.0, 3.0),
    )


    # (a)
    if has_violation:

        ax = axes[0]

        tmp = d[
            d["method"] == "DSCA-MoE"
        ]

        ax.errorbar(
            tmp["deadline_s"] * 1000,
            tmp["slo_violation_pct_mean"],
            yerr=tmp[
                "slo_violation_pct_ci95"
            ].fillna(0),
            marker="^",
            markersize=5.5,
            linewidth=1.8,
            capsize=2.5,
            color=COLORS["DSCA-MoE"],
        )

        ax.set_ylabel(
            "SLO Violation Rate (%)"
        )

        ax.set_title(
            "(a) SLO Violations"
        )

    else:

        ax = axes[0]

        tmp = d[
            d["method"] == "DSCA-MoE"
        ]

        ax.plot(
            tmp["deadline_s"] * 1000,
            tmp[
                "deadline_normalized_p95_mean"
            ],
            marker="^",
            markersize=5.5,
            linewidth=1.8,
            color=COLORS["DSCA-MoE"],
        )

        ax.axhline(
            1.0,
            color="0.55",
            linestyle=":",
            linewidth=1.2,
        )

        ax.set_ylabel(
            "P95 / SLO Target"
        )

        ax.set_title(
            "(a) Normalized P95"
        )

        print(
            "[WARN] No miss_rate_pct found in SLO logs; "
            "plotting P95/deadline instead."
        )


    ax.set_xlabel(
        "SLO Target (ms)"
    )

    style_axis(ax)


    # (b) throughput
    tmp = d[
        d["method"] == "DSCA-MoE"
    ]

    axes[1].errorbar(
        tmp["deadline_s"] * 1000,
        tmp["system_throughput_mean"],
        yerr=tmp[
            "system_throughput_ci95"
        ].fillna(0),
        marker="^",
        markersize=5.5,
        linewidth=1.8,
        capsize=2.5,
        color=COLORS["DSCA-MoE"],
    )

    axes[1].set_xlabel(
        "SLO Target (ms)"
    )

    axes[1].set_ylabel(
        "System Throughput (samples/s)"
    )

    axes[1].set_title(
        "(b) Throughput"
    )

    style_axis(
        axes[1]
    )


    # (c) p95
    axes[2].errorbar(
        tmp["deadline_s"] * 1000,
        tmp["p95_ms_mean"],
        yerr=tmp[
            "p95_ms_ci95"
        ].fillna(0),
        marker="^",
        markersize=5.5,
        linewidth=1.8,
        capsize=2.5,
        color=COLORS["DSCA-MoE"],
    )

    axes[2].plot(
        tmp["deadline_s"] * 1000,
        tmp["deadline_s"] * 1000,
        color="0.55",
        linestyle=":",
        linewidth=1.2,
    )

    axes[2].set_xlabel(
        "SLO Target (ms)"
    )

    axes[2].set_ylabel(
        "P95 Latency (ms)"
    )

    axes[2].set_title(
        "(c) Tail Latency"
    )

    style_axis(
        axes[2]
    )


    fig.tight_layout()

    save_figure(
        fig,
        fig_dir,
        "fig6_slo_sensitivity"
    )


# ============================================================
# Sequence length
# ============================================================

def plot_sequence(summary, fig_dir):

    d = summary[
        summary["family"].isin([
            "seq_l64",
            "seq_l128",
            "seq_l256",
        ])
    ].copy()

    if len(d) == 0:
        return

    d["seq_x"] = d["seq_len"]

    sensitivity_figure(
        d,
        "seq_x",
        "Sequence Length",
        "Sequence-Length Sensitivity",
        "fig7_sequence_length",
        fig_dir,
        xticks=[64, 128, 256],
    )


# ============================================================
# Repeat stability
# ============================================================

def plot_repeat_stability(raw, fig_dir):

    d = raw[
        (raw["family"] == "strong")
        &
        (raw["gpus"] == 4)
    ].copy()

    if len(d) == 0:
        return

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.9),
    )


    for method in METHOD_ORDER:

        tmp = d[
            d["method"] == method
        ].sort_values("repeat")

        if len(tmp) == 0:
            continue

        axes[0].plot(
            tmp["repeat"],
            tmp["p95_ms"],
            marker=MARKERS[method],
            linewidth=1.6,
            markersize=5.5,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            label=method,
        )

        axes[1].plot(
            tmp["repeat"],
            tmp["system_throughput"],
            marker=MARKERS[method],
            linewidth=1.6,
            markersize=5.5,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            label=method,
        )


    axes[0].set_xlabel(
        "Repeat"
    )

    axes[0].set_ylabel(
        "P95 Latency (ms)"
    )

    axes[0].set_title(
        "(a) Tail-Latency Stability"
    )

    axes[0].set_xticks(
        [1, 2, 3]
    )

    style_axis(
        axes[0]
    )


    axes[1].set_xlabel(
        "Repeat"
    )

    axes[1].set_ylabel(
        "System Throughput (samples/s)"
    )

    axes[1].set_title(
        "(b) Throughput Stability"
    )

    axes[1].set_xticks(
        [1, 2, 3]
    )

    style_axis(
        axes[1]
    )


    add_global_legend(
        fig,
        axes,
        y=1.03
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.90]
    )

    save_figure(
        fig,
        fig_dir,
        "fig8_repeat_stability"
    )


# ============================================================
# Print concise paper tables
# ============================================================

def print_strong_table(strong):

    print()
    print("=" * 88)
    print("STRONG SCALING")
    print("=" * 88)

    cols = [
        "method",
        "gpus",
        "system_throughput_mean",
        "p95_ms_mean",
        "speedup",
        "parallel_efficiency_pct",
    ]

    print(
        strong[cols]
        .sort_values(
            ["gpus", "method"]
        )
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


def print_improvements(rel):

    if len(rel) == 0:
        return

    print()
    print("=" * 88)
    print("DSCA-MoE RELATIVE IMPROVEMENTS")
    print("=" * 88)

    show = rel[
        rel["family"] == "strong"
    ].copy()

    if len(show):

        cols = [
            "gpus",
            "baseline",
            "throughput_gain_pct",
            "p95_reduction_pct",
        ]

        print(
            show[cols]
            .sort_values(
                ["gpus", "baseline"]
            )
            .to_string(
                index=False,
                float_format=lambda x: f"{x:.2f}",
            )
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help="complete experiment output directory",
    )

    args = parser.parse_args()

    root = Path(args.root)

    if not root.exists():
        raise FileNotFoundError(root)

    analysis_dir = root / "analysis"
    fig_dir = root / "figures"

    analysis_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fig_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Parse
    # --------------------------------------------------------

    raw = load_all_runs(root)

    print(
        f"[INFO] Parsed valid runs: {len(raw)}"
    )

    raw.to_csv(
        analysis_dir / "raw_runs.csv",
        index=False
    )


    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    summary = aggregate(raw)

    summary.to_csv(
        analysis_dir / "summary_all.csv",
        index=False
    )


    # --------------------------------------------------------
    # Strong / weak
    # --------------------------------------------------------

    strong = derive_strong(
        summary
    )

    weak = derive_weak(
        summary
    )

    strong.to_csv(
        analysis_dir / "strong_scaling.csv",
        index=False
    )

    weak.to_csv(
        analysis_dir / "weak_scaling.csv",
        index=False
    )


    # --------------------------------------------------------
    # Individual experiment CSVs
    # --------------------------------------------------------

    summary[
        summary["family"].str.startswith(
            "batch_"
        )
    ].to_csv(
        analysis_dir / "batch_sensitivity.csv",
        index=False
    )

    summary[
        summary["family"].str.startswith(
            "skew_"
        )
    ].to_csv(
        analysis_dir / "router_skew.csv",
        index=False
    )

    summary[
        summary["family"].str.startswith(
            "cache_"
        )
    ].to_csv(
        analysis_dir / "cache_pressure.csv",
        index=False
    )

    summary[
        summary["family"].str.startswith(
            "slo_"
        )
    ].to_csv(
        analysis_dir / "slo_sensitivity.csv",
        index=False
    )

    summary[
        summary["family"].str.startswith(
            "seq_"
        )
    ].to_csv(
        analysis_dir / "sequence_length.csv",
        index=False
    )


    # --------------------------------------------------------
    # Relative improvements
    # --------------------------------------------------------

    rel = relative_improvements(
        summary
    )

    rel.to_csv(
        analysis_dir
        / "relative_improvements.csv",
        index=False
    )


    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    plot_strong(
        strong,
        fig_dir
    )

    plot_weak(
        weak,
        fig_dir
    )

    plot_batch(
        summary,
        fig_dir
    )

    plot_skew(
        summary,
        fig_dir
    )

    plot_cache(
        summary,
        fig_dir
    )

    plot_slo(
        summary,
        fig_dir
    )

    plot_sequence(
        summary,
        fig_dir
    )

    plot_repeat_stability(
        raw,
        fig_dir
    )


    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print_strong_table(
        strong
    )

    print_improvements(
        rel
    )


    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    print()
    print("=" * 88)
    print("SANITY CHECK")
    print("=" * 88)

    print(
        "Parsed runs:",
        len(raw)
    )

    print(
        "Expected complete run count: 165"
    )

    if len(raw) == 165:
        print(
            "[OK] All 165 successful runs parsed."
        )
    else:
        print(
            "[WARN] Parsed run count != 165."
        )

    print()

    print(
        "Families:"
    )

    print(
        raw.groupby("family")
        .size()
        .sort_index()
        .to_string()
    )


    print()
    print("=" * 88)
    print("OUTPUT")
    print("=" * 88)

    print(
        "CSV:",
        analysis_dir
    )

    print(
        "Figures:",
        fig_dir
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
