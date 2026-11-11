#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final paper figures for DSCA-MoE.

Input:
    <root>/analysis_fixed/performance_runs.csv
    <root>/run_status.csv
    <root>/analysis_trace/slo_violation_summary.csv

Output:
    <root>/figures_final/
        fig_final_scalability.pdf/png
        fig_final_workload.pdf/png
        fig_final_slo.pdf/png
        fig_final_cache.pdf/png

All error bars:
    mean ± standard deviation over 3 runs.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Default path
# ============================================================

DEFAULT_ROOT = (
    "/home/lcf/MoE_Syn/HarMoEny-main/"
    "experiment_outputs/complete_20260923_095020"
)


# ============================================================
# Visual identity
# ============================================================

METHODS = [
    "DeepSpeed",
    "HarMoEny",
    "DSCA-MoE",
]

COLORS = {
    "DeepSpeed": "#D99200",
    "HarMoEny": "#079B7A",
    "DSCA-MoE": "#0875B5",
}

MARKERS = {
    "DeepSpeed": "o",
    "HarMoEny": "s",
    "DSCA-MoE": "^",
}

LINESTYLES = {
    "DeepSpeed": (0, (5, 2)),
    "HarMoEny": (0, (5, 2, 1.2, 2)),
    "DSCA-MoE": "-",
}

LINEWIDTHS = {
    "DeepSpeed": 1.8,
    "HarMoEny": 1.8,
    "DSCA-MoE": 2.35,
}

MARKERSIZES = {
    "DeepSpeed": 5.3,
    "HarMoEny": 5.3,
    "DSCA-MoE": 5.9,
}


# ============================================================
# Publication style
# ============================================================

plt.rcParams.update({

    "font.family": "serif",

    "font.serif": [
        "Times New Roman",
        "Times",
        "DejaVu Serif",
    ],

    "font.size": 8.5,

    "axes.labelsize": 9.0,

    "axes.titlesize": 9.2,

    "xtick.labelsize": 8.2,

    "ytick.labelsize": 8.2,

    "legend.fontsize": 8.3,

    "axes.linewidth": 0.85,

    "xtick.major.width": 0.8,

    "ytick.major.width": 0.8,

    "xtick.major.size": 3.5,

    "ytick.major.size": 3.5,

    "pdf.fonttype": 42,

    "ps.fonttype": 42,

    "savefig.dpi": 600,
})


# ============================================================
# General utilities
# ============================================================

def style_axis(
    ax,
    y_zero=False,
):

    ax.grid(
        axis="y",
        linestyle=(0, (3, 3)),
        linewidth=0.55,
        color="#D5D5D5",
        alpha=0.70,
        zorder=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)

    ax.tick_params(
        direction="out",
        length=3.5,
        width=0.8,
        pad=3,
    )

    if y_zero:

        ymin, ymax = ax.get_ylim()

        ax.set_ylim(
            0,
            ymax,
        )


def panel_label(
    ax,
    label,
):

    ax.text(
        -0.13,
        1.055,
        label,
        transform=ax.transAxes,
        fontsize=9.3,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save_figure(
    fig,
    out_dir,
    name,
):

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf = (
        out_dir
        /
        f"{name}.pdf"
    )

    png = (
        out_dir
        /
        f"{name}.png"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.035,
    )

    fig.savefig(
        png,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.035,
    )

    plt.close(fig)

    print(
        f"[FIG] {pdf}"
    )


def add_shared_legend(
    fig,
    axes,
    methods=None,
    y=0.995,
):

    if methods is None:
        methods = METHODS

    handles = []
    labels = []

    for ax in np.ravel(axes):

        h, l = (
            ax.get_legend_handles_labels()
        )

        for handle, label in zip(h, l):

            if (
                label in methods
                and
                label not in labels
            ):

                handles.append(handle)
                labels.append(label)

    ordered_handles = []
    ordered_labels = []

    for method in methods:

        if method in labels:

            idx = labels.index(method)

            ordered_handles.append(
                handles[idx]
            )

            ordered_labels.append(
                method
            )

    if ordered_handles:

        fig.legend(
            ordered_handles,
            ordered_labels,

            loc="upper center",

            bbox_to_anchor=(
                0.5,
                y,
            ),

            ncol=len(
                ordered_labels
            ),

            frameon=False,

            handlelength=2.8,

            columnspacing=2.2,

            handletextpad=0.7,

            borderaxespad=0,
        )


# ============================================================
# Aggregate three repeats
# ============================================================

def aggregate(
    df,
    group_cols,
):

    if len(df) == 0:

        raise RuntimeError(
            f"No rows available for group {group_cols}"
        )

    result = (
        df
        .groupby(
            group_cols,
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

            mean_latency_mean=(
                "mean_ms",
                "mean",
            ),

            mean_latency_std=(
                "mean_ms",
                "std",
            ),

            runs=(
                "repeat",
                "nunique",
            ),
        )
    )

    for col in [
        "throughput_std",
        "p95_std",
        "mean_latency_std",
    ]:

        result[col] = (
            result[col]
            .fillna(0)
        )

    return result


# ============================================================
# Method plotting
# ============================================================

def plot_method_lines(
    ax,
    data,
    x,
    y,
    yerr=None,
    methods=None,
):

    if methods is None:
        methods = METHODS

    for method in methods:

        d = (
            data[
                data["method"]
                ==
                method
            ]
            .sort_values(x)
        )

        if len(d) == 0:
            continue

        error = None

        if (
            yerr is not None
            and
            yerr in d.columns
        ):

            error = (
                d[yerr]
                .fillna(0)
                .to_numpy()
            )

        ax.errorbar(

            d[x],
            d[y],

            yerr=error,

            color=COLORS[method],

            marker=MARKERS[method],

            linestyle=LINESTYLES[method],

            linewidth=LINEWIDTHS[method],

            markersize=MARKERSIZES[method],

            markerfacecolor=COLORS[method],

            markeredgecolor=COLORS[method],

            markeredgewidth=0.5,

            capsize=2.3,

            capthick=1.0,

            elinewidth=1.05,

            alpha=1.0,

            label=method,

            zorder=4
            if method == "DSCA-MoE"
            else 3,
        )


# ============================================================
# Load performance + merge experimental configuration
# ============================================================

def load_performance(
    root,
):

    perf_file = (
        root
        /
        "analysis_fixed"
        /
        "performance_runs.csv"
    )

    status_file = (
        root
        /
        "run_status.csv"
    )

    if not perf_file.exists():

        raise FileNotFoundError(
            perf_file
        )

    if not status_file.exists():

        raise FileNotFoundError(
            status_file
        )


    # --------------------------------------------------------
    # Performance measurements
    # --------------------------------------------------------

    perf = pd.read_csv(
        perf_file
    )


    # --------------------------------------------------------
    # Experiment configuration
    # --------------------------------------------------------

    status = pd.read_csv(
        status_file
    )


    # --------------------------------------------------------
    # Normalize merge keys
    # --------------------------------------------------------

    for df in [
        perf,
        status,
    ]:

        df["family"] = (
            df["family"]
            .astype(str)
        )

        df["system"] = (
            df["system"]
            .astype(str)
            .str.lower()
        )

        df["gpus"] = (
            pd.to_numeric(
                df["gpus"]
            )
            .astype(int)
        )

        df["repeat"] = (
            pd.to_numeric(
                df["repeat"]
            )
            .astype(int)
        )


    merge_keys = [
        "family",
        "system",
        "gpus",
        "repeat",
    ]


    desired_config = [
        "batch",
        "num_samples",
        "seq_len",
        "cache",
        "skew",
        "deadline",
    ]


    available_config = [
        c
        for c in desired_config
        if c in status.columns
    ]


    config = (

        status[
            merge_keys
            +
            available_config
        ]

        .drop_duplicates(
            subset=merge_keys
        )
    )


    # --------------------------------------------------------
    # Remove stale duplicated config columns
    # --------------------------------------------------------

    for col in available_config:

        if col in perf.columns:

            perf = perf.drop(
                columns=[
                    col
                ]
            )


    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    perf = perf.merge(

        config,

        on=merge_keys,

        how="left",

        validate="many_to_one",
    )


    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    required = [
        "batch",
        "seq_len",
        "cache",
    ]


    for col in required:

        if col not in perf.columns:

            raise RuntimeError(
                f"Missing column after merge: {col}"
            )

        missing = (
            perf[col]
            .isna()
            .sum()
        )

        if missing > 0:

            raise RuntimeError(
                f"{col}: "
                f"{missing} rows missing "
                "after run_status merge."
            )


    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for col in [
        "batch",
        "seq_len",
        "cache",
        "num_samples",
    ]:

        if col in perf.columns:

            perf[col] = (
                pd.to_numeric(
                    perf[col]
                )
                .astype(int)
            )


    if "skew" in perf.columns:

        perf["skew"] = (
            pd.to_numeric(
                perf["skew"]
            )
        )


    if "deadline" in perf.columns:

        perf["deadline"] = (
            pd.to_numeric(
                perf["deadline"]
            )
        )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print(
        "[INFO] performance rows:",
        len(perf)
    )

    print(
        "[INFO] merged columns:",
        ", ".join(
            perf.columns
        )
    )

    print()

    print(
        "[INFO] experiment families:"
    )

    print(
        perf
        .groupby(
            "family"
        )
        .size()
        .sort_index()
        .to_string()
    )

    print()


    if len(perf) != 165:

        print(
            "[WARN] Expected 165 performance rows, "
            f"found {len(perf)}."
        )

    else:

        print(
            "[OK] All 165 runs loaded."
        )

    return perf


# ============================================================
# Figure 1
# Scalability
# ============================================================

def plot_scalability(
    perf,
    out_dir,
):

    strong = aggregate(

        perf[
            perf["family"]
            ==
            "strong"
        ],

        [
            "method",
            "gpus",
        ],
    )


    weak = aggregate(

        perf[
            perf["family"]
            ==
            "weak"
        ],

        [
            "method",
            "gpus",
        ],
    )


    fig, axes = plt.subplots(

        2,
        2,

        figsize=(
            7.15,
            4.72,
        ),
    )


    # ========================================================
    # (a) Strong throughput
    # ========================================================

    ax = axes[0, 0]

    plot_method_lines(
        ax,
        strong,
        "gpus",
        "throughput_mean",
        "throughput_std",
    )

    ax.set_title(
        "Strong Scaling — Throughput"
    )

    ax.set_xlabel(
        "Number of GPUs"
    )

    ax.set_ylabel(
        "Throughput (samples/s)"
    )

    ax.set_xticks(
        [1, 2, 4]
    )

    panel_label(
        ax,
        "(a)"
    )

    style_axis(
        ax,
        y_zero=True,
    )


    # ========================================================
    # (b) Strong P95
    # ========================================================

    ax = axes[0, 1]

    plot_method_lines(
        ax,
        strong,
        "gpus",
        "p95_mean",
        "p95_std",
    )

    ax.set_title(
        "Strong Scaling — Tail Latency"
    )

    ax.set_xlabel(
        "Number of GPUs"
    )

    ax.set_ylabel(
        "P95 Latency (ms)"
    )

    ax.set_xticks(
        [1, 2, 4]
    )

    panel_label(
        ax,
        "(b)"
    )

    style_axis(
        ax,
        y_zero=False,
    )


    # ========================================================
    # (c) Weak throughput
    # ========================================================

    ax = axes[1, 0]

    plot_method_lines(
        ax,
        weak,
        "gpus",
        "throughput_mean",
        "throughput_std",
    )

    ax.set_title(
        "Weak Scaling — Throughput"
    )

    ax.set_xlabel(
        "Number of GPUs"
    )

    ax.set_ylabel(
        "Throughput (samples/s)"
    )

    ax.set_xticks(
        [1, 2, 4]
    )

    panel_label(
        ax,
        "(c)"
    )

    style_axis(
        ax,
        y_zero=True,
    )


    # ========================================================
    # (d) Weak P95
    # ========================================================

    ax = axes[1, 1]

    plot_method_lines(
        ax,
        weak,
        "gpus",
        "p95_mean",
        "p95_std",
    )

    ax.set_title(
        "Weak Scaling — Tail Latency"
    )

    ax.set_xlabel(
        "Number of GPUs"
    )

    ax.set_ylabel(
        "P95 Latency (ms)"
    )

    ax.set_xticks(
        [1, 2, 4]
    )

    panel_label(
        ax,
        "(d)"
    )

    style_axis(
        ax,
        y_zero=False,
    )


    add_shared_legend(
        fig,
        axes,
        y=0.992,
    )


    fig.subplots_adjust(

        top=0.845,

        bottom=0.105,

        left=0.095,

        right=0.985,

        hspace=0.50,

        wspace=0.34,
    )


    save_figure(
        fig,
        out_dir,
        "fig_final_scalability",
    )


    strong.to_csv(
        out_dir
        /
        "data_final_scalability_strong.csv",
        index=False,
    )

    weak.to_csv(
        out_dir
        /
        "data_final_scalability_weak.csv",
        index=False,
    )


# ============================================================
# Figure 2
# Workload robustness
# ============================================================

def plot_workload(
    perf,
    out_dir,
):

    batch_raw = perf[
        perf["family"].isin(
            [
                "batch_b4",
                "batch_b8",
                "batch_b16",
            ]
        )
    ].copy()


    seq_raw = perf[
        perf["family"].isin(
            [
                "seq_l64",
                "seq_l128",
                "seq_l256",
            ]
        )
    ].copy()


    if len(batch_raw) == 0:

        raise RuntimeError(
            "No batch-sensitivity rows found."
        )


    if len(seq_raw) == 0:

        raise RuntimeError(
            "No sequence-sensitivity rows found."
        )


    batch = aggregate(

        batch_raw,

        [
            "method",
            "batch",
        ],
    )


    seq = aggregate(

        seq_raw,

        [
            "method",
            "seq_len",
        ],
    )


    fig, axes = plt.subplots(

        2,
        2,

        figsize=(
            7.15,
            4.72,
        ),
    )


    # ========================================================
    # Batch throughput
    # ========================================================

    ax = axes[0, 0]

    plot_method_lines(
        ax,
        batch,
        "batch",
        "throughput_mean",
        "throughput_std",
    )

    ax.set_title(
        "Batch Size — Throughput"
    )

    ax.set_xlabel(
        "Batch Size per GPU"
    )

    ax.set_ylabel(
        "Throughput (samples/s)"
    )

    ax.set_xticks(
        [4, 8, 16]
    )

    panel_label(
        ax,
        "(a)"
    )

    style_axis(
        ax,
        y_zero=True,
    )


    # ========================================================
    # Batch latency
    # ========================================================

    ax = axes[0, 1]

    plot_method_lines(
        ax,
        batch,
        "batch",
        "p95_mean",
        "p95_std",
    )

    ax.set_title(
        "Batch Size — Tail Latency"
    )

    ax.set_xlabel(
        "Batch Size per GPU"
    )

    ax.set_ylabel(
        "P95 Latency (ms)"
    )

    ax.set_xticks(
        [4, 8, 16]
    )

    panel_label(
        ax,
        "(b)"
    )

    style_axis(ax)


    # ========================================================
    # Sequence throughput
    # ========================================================

    ax = axes[1, 0]

    plot_method_lines(
        ax,
        seq,
        "seq_len",
        "throughput_mean",
        "throughput_std",
    )

    ax.set_title(
        "Sequence Length — Throughput"
    )

    ax.set_xlabel(
        "Sequence Length"
    )

    ax.set_ylabel(
        "Throughput (samples/s)"
    )

    ax.set_xticks(
        [64, 128, 256]
    )

    panel_label(
        ax,
        "(c)"
    )

    style_axis(
        ax,
        y_zero=True,
    )


    # ========================================================
    # Sequence latency
    # ========================================================

    ax = axes[1, 1]

    plot_method_lines(
        ax,
        seq,
        "seq_len",
        "p95_mean",
        "p95_std",
    )

    ax.set_title(
        "Sequence Length — Tail Latency"
    )

    ax.set_xlabel(
        "Sequence Length"
    )

    ax.set_ylabel(
        "P95 Latency (ms)"
    )

    ax.set_xticks(
        [64, 128, 256]
    )

    panel_label(
        ax,
        "(d)"
    )

    style_axis(ax)


    add_shared_legend(
        fig,
        axes,
        y=0.992,
    )


    fig.subplots_adjust(

        top=0.845,

        bottom=0.105,

        left=0.095,

        right=0.985,

        hspace=0.50,

        wspace=0.34,
    )


    save_figure(
        fig,
        out_dir,
        "fig_final_workload",
    )


    batch.to_csv(
        out_dir
        /
        "data_final_batch.csv",
        index=False,
    )

    seq.to_csv(
        out_dir
        /
        "data_final_sequence.csv",
        index=False,
    )


# ============================================================
# Figure 3
# SLO
# ============================================================

def load_slo(
    root,
):

    candidates = [

        root
        /
        "analysis_trace"
        /
        "slo_violation_summary.csv",

        root
        /
        "analysis_trace"
        /
        "slo_e2e_violation.csv",

    ]


    for path in candidates:

        if path.exists():

            print(
                "[INFO] SLO data:",
                path
            )

            return pd.read_csv(
                path
            )


    raise FileNotFoundError(
        "Cannot find SLO summary in analysis_trace/"
    )


def plot_slo(
    slo,
    out_dir,
):

    required = [

        "method",
        "deadline_ms",
        "violation_mean",
        "violation_std",
        "p95_mean",
        "p95_std",
    ]


    missing = [
        c
        for c in required
        if c not in slo.columns
    ]


    if missing:

        raise RuntimeError(
            f"SLO CSV missing columns: {missing}"
        )


    fig, axes = plt.subplots(

        1,
        2,

        figsize=(
            7.15,
            2.72,
        ),
    )


    # ========================================================
    # Violation
    # ========================================================

    ax = axes[0]

    plot_method_lines(
        ax,
        slo,
        "deadline_ms",
        "violation_mean",
        "violation_std",
    )

    ax.set_title(
        "SLO Compliance"
    )

    ax.set_xlabel(
        "SLO Target (ms)"
    )

    ax.set_ylabel(
        "E2E SLO Violation Rate (%)"
    )

    ax.set_xticks(
        [
            300,
            400,
            500,
            700,
        ]
    )

    ax.set_xlim(
        280,
        720,
    )

    ax.set_ylim(
        -4,
        105,
    )

    panel_label(
        ax,
        "(a)"
    )

    style_axis(
        ax,
        y_zero=False,
    )


    # ========================================================
    # P95
    # ========================================================

    ax = axes[1]

    plot_method_lines(
        ax,
        slo,
        "deadline_ms",
        "p95_mean",
        "p95_std",
    )


    # ideal SLO boundary
    ax.plot(

        [
            300,
            700,
        ],

        [
            300,
            700,
        ],

        linestyle=(
            0,
            (
                1.5,
                2.0,
            )
        ),

        linewidth=1.2,

        color="#888888",

        alpha=0.9,

        zorder=1,
    )


    ax.set_title(
        "Tail Latency under SLO Targets"
    )

    ax.set_xlabel(
        "SLO Target (ms)"
    )

    ax.set_ylabel(
        "E2E P95 Latency (ms)"
    )

    ax.set_xticks(
        [
            300,
            400,
            500,
            700,
        ]
    )

    ax.set_xlim(
        280,
        720,
    )

    panel_label(
        ax,
        "(b)"
    )

    style_axis(ax)


    add_shared_legend(
        fig,
        axes,
        y=0.997,
    )


    fig.subplots_adjust(

        top=0.755,

        bottom=0.20,

        left=0.095,

        right=0.985,

        wspace=0.33,
    )


    save_figure(
        fig,
        out_dir,
        "fig_final_slo",
    )


    slo.to_csv(
        out_dir
        /
        "data_final_slo.csv",
        index=False,
    )


# ============================================================
# Figure 4
# Cache robustness
# ============================================================

def plot_cache(
    perf,
    out_dir,
):

    raw = perf[
        perf["family"].isin(
            [
                "cache_c2",
                "cache_c4",
                "cache_c8",
            ]
        )
    ].copy()


    if len(raw) == 0:

        raise RuntimeError(
            "No cache experiment rows found."
        )


    cache = aggregate(

        raw,

        [
            "method",
            "cache",
        ],
    )


    methods = [
        "HarMoEny",
        "DSCA-MoE",
    ]


    fig, axes = plt.subplots(

        1,
        2,

        figsize=(
            7.15,
            2.72,
        ),
    )


    # ========================================================
    # Throughput
    # ========================================================

    ax = axes[0]

    plot_method_lines(
        ax,
        cache,
        "cache",
        "throughput_mean",
        "throughput_std",
        methods=methods,
    )

    ax.set_title(
        "Cache Capacity — Throughput"
    )

    ax.set_xlabel(
        "Expert Cache Size"
    )

    ax.set_ylabel(
        "Throughput (samples/s)"
    )

    ax.set_xticks(
        [2, 4, 8]
    )

    panel_label(
        ax,
        "(a)"
    )

    style_axis(
        ax,
        y_zero=False,
    )


    # ========================================================
    # P95
    # ========================================================

    ax = axes[1]

    plot_method_lines(
        ax,
        cache,
        "cache",
        "p95_mean",
        "p95_std",
        methods=methods,
    )

    ax.set_title(
        "Cache Capacity — Tail Latency"
    )

    ax.set_xlabel(
        "Expert Cache Size"
    )

    ax.set_ylabel(
        "P95 Latency (ms)"
    )

    ax.set_xticks(
        [2, 4, 8]
    )

    panel_label(
        ax,
        "(b)"
    )

    style_axis(ax)


    add_shared_legend(
        fig,
        axes,
        methods=methods,
        y=0.997,
    )


    fig.subplots_adjust(

        top=0.755,

        bottom=0.20,

        left=0.095,

        right=0.985,

        wspace=0.33,
    )


    save_figure(
        fig,
        out_dir,
        "fig_final_cache",
    )


    cache.to_csv(
        out_dir
        /
        "data_final_cache.csv",
        index=False,
    )


# ============================================================
# Print important numerical results
# ============================================================

def print_final_numbers(
    perf,
):

    strong = aggregate(

        perf[
            perf["family"]
            ==
            "strong"
        ],

        [
            "method",
            "gpus",
        ],
    )


    d4 = (
        strong[
            strong["gpus"]
            ==
            4
        ]
        .set_index(
            "method"
        )
    )


    print()
    print(
        "=" * 72
    )

    print(
        "4-GPU STRONG SCALING SUMMARY"
    )

    print(
        "=" * 72
    )


    cols = [

        "throughput_mean",
        "throughput_std",

        "p95_mean",
        "p95_std",
    ]


    print(
        d4[cols]
        .round(3)
        .to_string()
    )


    if (
        "DSCA-MoE"
        in d4.index
    ):

        dsca = d4.loc[
            "DSCA-MoE"
        ]


        for baseline in [
            "DeepSpeed",
            "HarMoEny",
        ]:

            if baseline not in d4.index:
                continue

            base = d4.loc[
                baseline
            ]


            throughput_gain = (

                dsca[
                    "throughput_mean"
                ]

                /

                base[
                    "throughput_mean"
                ]

                -
                1

            ) * 100


            latency_reduction = (

                1

                -

                dsca[
                    "p95_mean"
                ]

                /

                base[
                    "p95_mean"
                ]

            ) * 100


            print()

            print(
                f"DSCA-MoE vs {baseline}:"
            )

            print(
                f"  Throughput gain  = "
                f"{throughput_gain:.2f}%"
            )

            print(
                f"  P95 reduction    = "
                f"{latency_reduction:.2f}%"
            )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(

        "--root",

        default=DEFAULT_ROOT,

        help=(
            "Root directory of complete experiment."
        ),
    )


    args = parser.parse_args()


    root = Path(
        args.root
    )


    if not root.exists():

        raise FileNotFoundError(
            root
        )


    out_dir = (
        root
        /
        "figures_final"
    )


    # ========================================================
    # Load and repair performance table
    # ========================================================

    perf = load_performance(
        root
    )


    # ========================================================
    # Fig. 1
    # ========================================================

    print()
    print(
        "===== Figure 1: Scalability ====="
    )

    plot_scalability(
        perf,
        out_dir,
    )


    # ========================================================
    # Fig. 2
    # ========================================================

    print()
    print(
        "===== Figure 2: Workload Robustness ====="
    )

    plot_workload(
        perf,
        out_dir,
    )


    # ========================================================
    # Fig. 3
    # ========================================================

    print()
    print(
        "===== Figure 3: SLO Guarantee ====="
    )

    slo = load_slo(
        root
    )

    plot_slo(
        slo,
        out_dir,
    )


    # ========================================================
    # Fig. 4
    # ========================================================

    print()
    print(
        "===== Figure 4: Cache Robustness ====="
    )

    plot_cache(
        perf,
        out_dir,
    )


    # ========================================================
    # Numerical summary
    # ========================================================

    print_final_numbers(
        perf
    )


    print()
    print(
        "=" * 72
    )

    print(
        "ALL FINAL FIGURES GENERATED"
    )

    print(
        "=" * 72
    )

    print(
        out_dir
    )

    print()

    print(
        "Expected files:"
    )

    print(
        "  fig_final_scalability.pdf"
    )

    print(
        "  fig_final_workload.pdf"
    )

    print(
        "  fig_final_slo.pdf"
    )

    print(
        "  fig_final_cache.pdf"
    )


if __name__ == "__main__":
    main()