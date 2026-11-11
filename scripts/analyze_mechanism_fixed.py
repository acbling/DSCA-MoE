#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
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

LOAD_METHODS = [
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


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "DejaVu Serif",
    ],
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
# Plot helpers
# ============================================================

def style_axis(ax):

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.5,
        alpha=0.28,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, out_dir, name):

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"

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

    print("[FIG]", pdf)


def add_legend(
    fig,
    axes,
    methods,
):

    handles = []
    labels = []

    for ax in np.ravel(axes):

        h, l = (
            ax.get_legend_handles_labels()
        )

        for hh, ll in zip(h, l):

            if (
                ll in methods
                and ll not in labels
            ):

                handles.append(hh)
                labels.append(ll)

    if handles:

        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(methods),
            frameon=False,
        )


# ============================================================
# Status / path
# ============================================================

def find_run_dir(
    root,
    row,
):

    family = str(
        row["family"]
    )

    system = str(
        row["system"]
    )

    gpu = int(
        row["gpus"]
    )

    repeat = int(
        row["repeat"]
    )

    tag = (
        f"{family}_"
        f"{system}_"
        f"{gpu}gpu_"
        f"r{repeat}"
    )

    outer = (
        root
        /
        family
        /
        tag
    )

    if outer.exists():
        return outer

    hits = list(
        (
            root
            /
            family
        ).glob(
            f"{tag}*"
        )
    )

    hits = [
        p
        for p in hits
        if p.is_dir()
    ]

    if hits:
        return hits[0]

    return None


def load_status(root):

    f = (
        root
        /
        "run_status.csv"
    )

    df = pd.read_csv(f)

    df = df[
        df["status"]
        .str.upper()
        ==
        "SUCCESS"
    ].copy()

    df["system"] = (
        df["system"]
        .str.lower()
    )

    df["method"] = (
        df["system"]
        .map(METHOD_MAP)
    )

    run_dirs = []

    for _, row in df.iterrows():

        p = find_run_dir(
            root,
            row,
        )

        run_dirs.append(
            str(p)
            if p
            else ""
        )

    df["run_dir"] = (
        run_dirs
    )

    print(
        "[INFO] successful runs:",
        len(df),
    )

    return df


# ============================================================
# Log summary
# ============================================================

def parse_log(log_path):

    path = Path(
        str(log_path)
    )

    if not path.exists():
        return None

    text = path.read_text(
        errors="ignore"
    )

    result = None

    for line in text.splitlines():

        parts = (
            line.strip().split()
        )

        if (
            len(parts) < 9
            or
            parts[-1].lower()
            != "ok"
        ):
            continue

        try:

            result = {

                "mean_s":
                    float(parts[-7]),

                "p50_s":
                    float(parts[-6]),

                "p95_s":
                    float(parts[-5]),

                "throughput_rank":
                    float(parts[-2]),

            }

        except Exception:
            continue

    return result


def extract_performance(
    status,
):

    rows = []

    for _, row in status.iterrows():

        log = Path(
            str(row["log"])
        )

        if not log.exists():

            # fallback after directory move
            log = (
                Path(
                    row["run_dir"]
                )
                /
                log.name
            )

        result = parse_log(
            log
        )

        if result is None:
            continue

        gpu = int(
            row["gpus"]
        )

        rows.append({

            "family":
                row["family"],

            "method":
                row["method"],

            "system":
                row["system"],

            "gpus":
                gpu,

            "repeat":
                int(
                    row["repeat"]
                ),

            "throughput":
                result[
                    "throughput_rank"
                ]
                *
                gpu,

            "mean_ms":
                result[
                    "mean_s"
                ]
                *
                1000,

            "p50_ms":
                result[
                    "p50_s"
                ]
                *
                1000,

            "p95_ms":
                result[
                    "p95_s"
                ]
                *
                1000,

        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Exact layer CSV parsing
# ============================================================

def read_layer_csv(path):

    try:

        df = pd.read_csv(
            path
        )

    except Exception:

        return None

    # --------------------------------------------------------
    # IMPORTANT:
    # use exact column only
    # --------------------------------------------------------

    if (
        "latency (ms)"
        not in df.columns
    ):

        return None

    return df


def layer_id(path):

    name = path.name

    for prefix in [
        "moe_layer-",
        "layer-",
    ]:

        if (
            name.startswith(prefix)
            and
            name.endswith(".csv")
        ):

            return int(
                name[
                    len(prefix):-4
                ]
            )

    return None


# ============================================================
# Correct layer latency
# ============================================================

def extract_layer_latency(
    status,
):

    rows = []

    # mechanism analysis:
    # strong scaling, 4 GPU only
    selected = status[
        (status["family"] == "strong")
        &
        (status["gpus"] == 4)
    ]

    for _, row in selected.iterrows():

        run_dir = Path(
            row["run_dir"]
        )

        files = list(
            run_dir.rglob(
                "layer-*.csv"
            )
        )

        files += list(
            run_dir.rglob(
                "moe_layer-*.csv"
            )
        )

        for f in files:

            lid = layer_id(
                f
            )

            if lid is None:
                continue

            df = read_layer_csv(
                f
            )

            if df is None:
                continue

            latency = pd.to_numeric(
                df["latency (ms)"],
                errors="coerce",
            )

            latency = latency[
                latency.notna()
            ]

            if len(latency) == 0:
                continue

            rows.append({

                "method":
                    row["method"],

                "system":
                    row["system"],

                "repeat":
                    int(
                        row["repeat"]
                    ),

                "layer":
                    lid,

                "mean_ms":
                    latency.mean(),

                "p95_ms":
                    np.percentile(
                        latency,
                        95,
                    ),

            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Expert distribution
# ============================================================

def parse_distribution(x):

    if pd.isna(x):
        return None

    try:

        value = ast.literal_eval(
            str(x)
        )

    except Exception:

        return None

    if not isinstance(
        value,
        (
            list,
            tuple,
        )
    ):

        return None

    arr = np.asarray(
        value,
        dtype=float,
    )

    if (
        arr.ndim != 1
        or
        len(arr) != 64
    ):

        return None

    return arr


def distribution_stats(
    counts,
):

    counts = np.asarray(
        counts,
        dtype=float,
    )

    total = counts.sum()

    if total <= 0:

        return {
            "load_cv": np.nan,
            "max_mean": np.nan,
            "top8_share_pct": np.nan,
            "active_experts": 0,
            "entropy_norm": np.nan,
        }

    mean = counts.mean()

    cv = (
        counts.std()
        /
        mean
    )

    max_mean = (
        counts.max()
        /
        mean
    )

    top8 = (
        np.sort(counts)[-8:]
        .sum()
        /
        total
        *
        100
    )

    active = int(
        np.sum(
            counts > 0
        )
    )

    p = (
        counts[
            counts > 0
        ]
        /
        total
    )

    entropy = (
        -np.sum(
            p
            *
            np.log(p)
        )
    )

    entropy_norm = (
        entropy
        /
        np.log(
            len(counts)
        )
    )

    return {

        "load_cv":
            cv,

        "max_mean":
            max_mean,

        "top8_share_pct":
            top8,

        "active_experts":
            active,

        "entropy_norm":
            entropy_norm,

    }


# ============================================================
# Reliable expert load
#
# ONLY HarMoEny and DSCA because DeepSpeed layer CSV
# does not contain expert distribution.
# ============================================================

def extract_expert_load(
    status,
):

    families = [

        "skew_natural",
        "skew_moderate",
        "skew_severe",

    ]

    selected = status[
        status["family"].isin(
            families
        )
        &
        status["method"].isin(
            LOAD_METHODS
        )
    ]

    rows = []

    per_layer_rows = []

    for _, row in selected.iterrows():

        run_dir = Path(
            row["run_dir"]
        )

        files = list(
            run_dir.rglob(
                "moe_layer-*.csv"
            )
        )

        total_counts = np.zeros(
            64,
            dtype=float,
        )

        layer_counts = {}

        valid_rows = 0

        for f in files:

            lid = layer_id(
                f
            )

            if lid is None:
                continue

            try:

                df = pd.read_csv(
                    f
                )

            except Exception:

                continue

            if (
                "expert distribution"
                not in df.columns
            ):

                continue

            if lid not in layer_counts:

                layer_counts[
                    lid
                ] = np.zeros(
                    64,
                    dtype=float,
                )

            for x in df[
                "expert distribution"
            ]:

                arr = parse_distribution(
                    x
                )

                if arr is None:
                    continue

                total_counts += arr

                layer_counts[
                    lid
                ] += arr

                valid_rows += 1

        if valid_rows == 0:
            continue

        stats = distribution_stats(
            total_counts
        )

        rows.append({

            "family":
                row["family"],

            "method":
                row["method"],

            "system":
                row["system"],

            "repeat":
                int(
                    row["repeat"]
                ),

            **stats,

        })

        for lid, counts in (
            layer_counts.items()
        ):

            stats_l = (
                distribution_stats(
                    counts
                )
            )

            record = {

                "family":
                    row["family"],

                "method":
                    row["method"],

                "repeat":
                    int(
                        row["repeat"]
                    ),

                "layer":
                    lid,

                **stats_l,

            }

            # expert counts
            for eid in range(64):

                record[
                    f"expert_{eid}"
                ] = counts[eid]

            per_layer_rows.append(
                record
            )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(
            per_layer_rows
        ),
    )


# ============================================================
# Plot layer latency
# ============================================================

def plot_layer_latency(
    layer,
    fig_dir,
):

    summary = (
        layer
        .groupby(
            [
                "method",
                "layer",
            ],
            as_index=False,
        )
        .agg(

            mean_ms=(
                "mean_ms",
                "mean",
            ),

            mean_std=(
                "mean_ms",
                "std",
            ),

            p95_ms=(
                "p95_ms",
                "mean",
            ),

            p95_std=(
                "p95_ms",
                "std",
            ),

        )
        .fillna(0)
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.9),
    )

    for method in METHOD_ORDER:

        d = summary[
            summary["method"]
            ==
            method
        ].sort_values(
            "layer"
        )

        if len(d) == 0:
            continue

        axes[0].errorbar(

            d["layer"],
            d["mean_ms"],

            yerr=d[
                "mean_std"
            ],

            marker=MARKERS[
                method
            ],

            color=COLORS[
                method
            ],

            linestyle=LINESTYLES[
                method
            ],

            capsize=2,

            label=method,

        )

        axes[1].errorbar(

            d["layer"],
            d["p95_ms"],

            yerr=d[
                "p95_std"
            ],

            marker=MARKERS[
                method
            ],

            color=COLORS[
                method
            ],

            linestyle=LINESTYLES[
                method
            ],

            capsize=2,

            label=method,

        )

    axes[0].set_xlabel(
        "MoE Layer"
    )

    axes[0].set_ylabel(
        "Mean Layer Latency (ms)"
    )

    axes[0].set_title(
        "(a) Layer-wise Mean Latency"
    )

    axes[1].set_xlabel(
        "MoE Layer"
    )

    axes[1].set_ylabel(
        "P95 Layer Latency (ms)"
    )

    axes[1].set_title(
        "(b) Layer-wise Tail Latency"
    )

    for ax in axes:
        ax.set_xticks(
            range(
                0,
                12,
                2,
            )
        )

        style_axis(
            ax
        )

    add_legend(
        fig,
        axes,
        METHOD_ORDER,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.90,
        ]
    )

    save_fig(
        fig,
        fig_dir,
        "fig_layer_latency_fixed",
    )

    return summary


# ============================================================
# Plot measured load imbalance
# ============================================================

def plot_load_metrics(
    load,
    fig_dir,
):

    family_x = {

        "skew_natural":
            0,

        "skew_moderate":
            1,

        "skew_severe":
            2,

    }

    labels = [
        "Natural",
        "Moderate",
        "Severe",
    ]

    d = load.copy()

    d["x"] = (
        d["family"]
        .map(
            family_x
        )
    )

    summary = (
        d
        .groupby(
            [
                "method",
                "x",
            ],
            as_index=False,
        )
        .agg(

            cv_mean=(
                "load_cv",
                "mean",
            ),

            cv_std=(
                "load_cv",
                "std",
            ),

            max_mean=(
                "max_mean",
                "mean",
            ),

            max_std=(
                "max_mean",
                "std",
            ),

            active_mean=(
                "active_experts",
                "mean",
            ),

            active_std=(
                "active_experts",
                "std",
            ),

            entropy_mean=(
                "entropy_norm",
                "mean",
            ),

            entropy_std=(
                "entropy_norm",
                "std",
            ),

        )
        .fillna(0)
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.0, 3.0),
    )

    metrics = [

        (
            "cv_mean",
            "cv_std",
            "Expert Load CV",
            "(a) Load Imbalance",
        ),

        (
            "max_mean",
            "max_std",
            "Max / Mean Expert Load",
            "(b) Peak Expert Load",
        ),

        (
            "active_mean",
            "active_std",
            "Active Experts",
            "(c) Active Expert Coverage",
        ),

    ]

    for ax, (
        y,
        err,
        ylabel,
        title,
    ) in zip(
        axes,
        metrics,
    ):

        for method in LOAD_METHODS:

            m = summary[
                summary["method"]
                ==
                method
            ].sort_values(
                "x"
            )

            if len(m) == 0:
                continue

            ax.errorbar(

                m["x"],
                m[y],

                yerr=m[
                    err
                ],

                marker=MARKERS[
                    method
                ],

                color=COLORS[
                    method
                ],

                linestyle=LINESTYLES[
                    method
                ],

                capsize=2,

                label=method,

            )

        ax.set_xticks(
            [
                0,
                1,
                2,
            ]
        )

        ax.set_xticklabels(
            labels
        )

        ax.set_xlabel(
            "Routing Distribution"
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_title(
            title
        )

        style_axis(
            ax
        )

    add_legend(
        fig,
        axes,
        LOAD_METHODS,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.90,
        ]
    )

    save_fig(
        fig,
        fig_dir,
        "fig_expert_load_fixed",
    )

    return summary


# ============================================================
# Load CV vs performance
# ============================================================

def plot_load_vs_performance(
    load,
    performance,
    fig_dir,
):

    perf = performance[
        performance["family"].isin(
            [
                "skew_natural",
                "skew_moderate",
                "skew_severe",
            ]
        )
        &
        performance["method"].isin(
            LOAD_METHODS
        )
    ].copy()

    perf_summary = (
        perf
        .groupby(
            [
                "family",
                "method",
            ],
            as_index=False,
        )
        .agg(

            throughput_mean=(
                "throughput",
                "mean",
            ),

            throughput_std=(
                "throughput",
                "std",
            ),

            p95_mean=(
                "p95_ms",
                "mean",
            ),

            p95_std=(
                "p95_ms",
                "std",
            ),

        )
        .fillna(0)
    )

    load_summary = (
        load
        .groupby(
            [
                "family",
                "method",
            ],
            as_index=False,
        )
        .agg(

            cv_mean=(
                "load_cv",
                "mean",
            ),

            cv_std=(
                "load_cv",
                "std",
            ),

        )
        .fillna(0)
    )

    merged = pd.merge(

        load_summary,

        perf_summary,

        on=[
            "family",
            "method",
        ],

        how="inner",

    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.9),
    )

    for method in LOAD_METHODS:

        d = merged[
            merged["method"]
            ==
            method
        ].sort_values(
            "cv_mean"
        )

        axes[0].errorbar(

            d["cv_mean"],
            d["throughput_mean"],

            xerr=d[
                "cv_std"
            ],

            yerr=d[
                "throughput_std"
            ],

            marker=MARKERS[
                method
            ],

            color=COLORS[
                method
            ],

            linestyle=LINESTYLES[
                method
            ],

            capsize=2,

            label=method,

        )

        axes[1].errorbar(

            d["cv_mean"],
            d["p95_mean"],

            xerr=d[
                "cv_std"
            ],

            yerr=d[
                "p95_std"
            ],

            marker=MARKERS[
                method
            ],

            color=COLORS[
                method
            ],

            linestyle=LINESTYLES[
                method
            ],

            capsize=2,

            label=method,

        )

    axes[0].set_xlabel(
        "Measured Expert-Load CV"
    )

    axes[0].set_ylabel(
        "System Throughput (samples/s)"
    )

    axes[0].set_title(
        "(a) Throughput vs. Load Imbalance"
    )

    axes[1].set_xlabel(
        "Measured Expert-Load CV"
    )

    axes[1].set_ylabel(
        "P95 Latency (ms)"
    )

    axes[1].set_title(
        "(b) Tail Latency vs. Load Imbalance"
    )

    for ax in axes:
        style_axis(
            ax
        )

    add_legend(
        fig,
        axes,
        LOAD_METHODS,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.90,
        ]
    )

    save_fig(
        fig,
        fig_dir,
        "fig_load_vs_performance",
    )

    return merged


# ============================================================
# Expert heatmaps
# ============================================================

def plot_heatmaps(
    per_layer,
    fig_dir,
):

    # severe is most informative
    severe = per_layer[
        per_layer["family"]
        ==
        "skew_severe"
    ]

    for method in LOAD_METHODS:

        d = severe[
            severe["method"]
            ==
            method
        ]

        if len(d) == 0:
            continue

        rows = []

        for layer in range(12):

            x = d[
                d["layer"]
                ==
                layer
            ]

            if len(x) == 0:
                rows.append(
                    np.zeros(64)
                )

                continue

            counts = np.array(
                [
                    x[
                        f"expert_{i}"
                    ].sum()
                    for i in range(64)
                ]
            )

            if counts.sum() > 0:

                counts = (
                    counts
                    /
                    counts.sum()
                    *
                    100
                )

            rows.append(
                counts
            )

        matrix = np.vstack(
            rows
        )

        fig, ax = plt.subplots(
            figsize=(7.0, 2.7)
        )

        im = ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
        )

        ax.set_xlabel(
            "Expert ID"
        )

        ax.set_ylabel(
            "MoE Layer"
        )

        ax.set_title(
            f"Expert Routing Distribution — {method}"
        )

        ax.set_yticks(
            range(12)
        )

        ax.set_xticks(
            range(
                0,
                64,
                8,
            )
        )

        cbar = fig.colorbar(
            im,
            ax=ax,
        )

        cbar.set_label(
            "Routed Tokens (%)"
        )

        fig.tight_layout()

        safe = (
            method
            .lower()
            .replace(
                "-",
                "_",
            )
        )

        save_fig(
            fig,
            fig_dir,
            f"fig_expert_heatmap_{safe}",
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
    )

    args = parser.parse_args()

    root = Path(
        args.root
    )

    analysis_dir = (
        root
        /
        "analysis_fixed"
    )

    fig_dir = (
        root
        /
        "figures_fixed"
    )

    analysis_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    status = load_status(
        root
    )

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    performance = (
        extract_performance(
            status
        )
    )

    performance.to_csv(
        analysis_dir
        /
        "performance_runs.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Correct layer latency
    # --------------------------------------------------------

    print()
    print(
        "===== Correct layer latency ====="
    )

    layer = extract_layer_latency(
        status
    )

    layer.to_csv(
        analysis_dir
        /
        "layer_latency_runs.csv",
        index=False,
    )

    layer_summary = (
        plot_layer_latency(
            layer,
            fig_dir,
        )
    )

    layer_summary.to_csv(
        analysis_dir
        /
        "layer_latency_summary.csv",
        index=False,
    )

    print(
        layer_summary
        .groupby("method")[
            "mean_ms"
        ]
        .mean()
    )

    # --------------------------------------------------------
    # Expert load
    # --------------------------------------------------------

    print()
    print(
        "===== Reliable expert load ====="
    )

    load, per_layer = (
        extract_expert_load(
            status
        )
    )

    load.to_csv(
        analysis_dir
        /
        "expert_load_runs.csv",
        index=False,
    )

    per_layer.to_csv(
        analysis_dir
        /
        "expert_load_per_layer.csv",
        index=False,
    )

    load_summary = (
        plot_load_metrics(
            load,
            fig_dir,
        )
    )

    load_summary.to_csv(
        analysis_dir
        /
        "expert_load_summary.csv",
        index=False,
    )

    print(
        load_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Actual CV -> performance
    # --------------------------------------------------------

    print()
    print(
        "===== Load vs performance ====="
    )

    merged = (
        plot_load_vs_performance(
            load,
            performance,
            fig_dir,
        )
    )

    merged.to_csv(
        analysis_dir
        /
        "load_vs_performance.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Heatmaps
    # --------------------------------------------------------

    plot_heatmaps(
        per_layer,
        fig_dir,
    )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "DONE"
    )

    print(
        "=" * 70
    )

    print(
        "Analysis:",
        analysis_dir,
    )

    print(
        "Figures:",
        fig_dir,
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "DeepSpeed is intentionally excluded from "
        "expert-load figures because its stored layer CSV "
        "does not contain a comparable expert-distribution field."
    )


if __name__ == "__main__":
    main()
