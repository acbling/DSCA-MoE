#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final paper figures for DSCA-MoE.

No "Strong Scaling" / "Weak Scaling" terminology is used in figures.

Figures
-------
Fig. 1:
    GPU count
    Fixed: Batch/GPU = 16, Sequence Length = 256

Fig. 2:
    Batch Size
    Fixed: 4 GPUs, Sequence Length = 256

    Sequence Length
    Fixed: 4 GPUs, Batch/GPU = 16

Fig. 3:
    Expert Cache Size
    Fixed: 4 GPUs, Batch/GPU = 16, Sequence Length = 256

Fig. 4:
    SLO Target
    Fixed: 4 GPUs, Batch/GPU = 16, Sequence Length = 256

Inputs
------
<root>/analysis_fixed/performance_runs.csv
<root>/run_status.csv
<root>/analysis_trace/slo_violation_summary.csv

Outputs
-------
<root>/figures_final_v2/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# Path
# ============================================================

DEFAULT_ROOT = (
    "/home/lcf/MoE_Syn/HarMoEny-main/"
    "experiment_outputs/complete_20260923_095020"
)


# ============================================================
# Methods
# ============================================================

METHODS = [
    "DeepSpeed",
    "HarMoEny",
    "DSCA-MoE",
]


COLORS = {
    "DeepSpeed": "#D98C00",
    "HarMoEny": "#109B79",
    "DSCA-MoE": "#0877B5",
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
    "DeepSpeed": 5.2,
    "HarMoEny": 5.2,
    "DSCA-MoE": 5.8,
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

    "legend.fontsize": 8.2,

    "xtick.labelsize": 8.1,

    "ytick.labelsize": 8.1,

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
# Plot helpers
# ============================================================

def style_axis(ax, zero_bottom=False):

    ax.grid(
        axis="y",
        linestyle=(0, (3, 3)),
        linewidth=0.55,
        color="#D7D7D7",
        alpha=0.72,
        zorder=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    ax.tick_params(
        direction="out",
        pad=3,
    )

    if zero_bottom:

        ymin, ymax = ax.get_ylim()

        ax.set_ylim(
            0,
            ymax,
        )


def panel_label(ax, text):

    ax.text(
        -0.125,
        1.055,
        text,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def configuration_text(
    fig,
    text,
    y=0.905,
):

    fig.text(
        0.5,
        y,
        text,
        ha="center",
        va="center",
        fontsize=8.4,
        color="#555555",
    )


def save_figure(
    fig,
    output_dir,
    name,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf = (
        output_dir
        /
        f"{name}.pdf"
    )

    png = (
        output_dir
        /
        f"{name}.png"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    fig.savefig(
        png,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    plt.close(fig)

    print(
        "[FIG]",
        pdf,
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

        hs, ls = (
            ax.get_legend_handles_labels()
        )

        for h, l in zip(hs, ls):

            if (
                l in methods
                and
                l not in labels
            ):

                handles.append(h)
                labels.append(l)

    ordered_h = []
    ordered_l = []

    for method in methods:

        if method in labels:

            i = labels.index(
                method
            )

            ordered_h.append(
                handles[i]
            )

            ordered_l.append(
                method
            )

    if ordered_h:

        fig.legend(
            ordered_h,
            ordered_l,

            loc="upper center",

            bbox_to_anchor=(
                0.5,
                y,
            ),

            ncol=len(
                ordered_l
            ),

            frameon=False,

            handlelength=2.8,

            columnspacing=2.2,

            handletextpad=0.65,

            borderaxespad=0,
        )


def plot_methods(
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

            capsize=2.2,

            capthick=1.0,

            elinewidth=1.0,

            label=method,

            zorder=4
            if method == "DSCA-MoE"
            else 3,
        )


# ============================================================
# Load data
# ============================================================

def load_performance(root):

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


    perf = pd.read_csv(
        perf_file
    )

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


    config_cols = [
        "batch",
        "num_samples",
        "seq_len",
        "cache",
        "skew",
        "deadline",
    ]


    available = [
        c
        for c in config_cols
        if c in status.columns
    ]


    config = (
        status[
            merge_keys
            +
            available
        ]
        .drop_duplicates(
            subset=merge_keys
        )
    )


    for c in available:

        if c in perf.columns:

            perf = perf.drop(
                columns=[c]
            )


    perf = perf.merge(

        config,

        on=merge_keys,

        how="left",

        validate="many_to_one",
    )


    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for c in [
        "batch",
        "num_samples",
        "seq_len",
        "cache",
    ]:

        if c in perf.columns:

            perf[c] = (
                pd.to_numeric(
                    perf[c]
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


    print(
        "[INFO] loaded performance rows:",
        len(perf)
    )


    if len(perf) == 165:

        print(
            "[OK] all 165 runs loaded"
        )

    else:

        print(
            "[WARN] expected 165 runs"
        )


    return perf


# ============================================================
# Aggregate repeats
# ============================================================

def aggregate(
    df,
    groups,
):

    if len(df) == 0:

        raise RuntimeError(
            "No experiment data found."
        )


    result = (
        df
        .groupby(
            groups,
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

            runs=(
                "repeat",
                "nunique",
            ),
        )
    )


    result[
        "throughput_std"
    ] = (
        result[
            "throughput_std"
        ]
        .fillna(0)
    )


    result[
        "p95_std"
    ] = (
        result[
            "p95_std"
        ]
        .fillna(0)
    )


    return result


# ============================================================
# FIGURE 1
#
# GPU count
# B/GPU = 16
# L = 256
# ============================================================

def plot_gpu_scaling(
    perf,
    out_dir,
):

    # Internally this was stored under family="strong".
    # The paper figure does NOT use that terminology.
    raw = perf[
        perf["family"]
        ==
        "strong"
    ].copy()


    data = aggregate(

        raw,

        [
            "method",
            "gpus",
        ],
    )


    fig, axes = plt.subplots(

        1,
        2,

        figsize=(
            7.15,
            2.72,
        ),
    )


    # --------------------------------------------------------
    # Throughput
    # --------------------------------------------------------

    ax = axes[0]

    plot_methods(
        ax,
        data,
        "gpus",
        "throughput_mean",
        "throughput_std",
    )

    ax.set_title(
        "Aggregate Throughput"
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
        zero_bottom=True,
    )


    # --------------------------------------------------------
    # P95
    # --------------------------------------------------------

    ax = axes[1]

    plot_methods(
        ax,
        data,
        "gpus",
        "p95_mean",
        "p95_std",
    )

    ax.set_title(
        "Tail Latency"
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

    style_axis(ax)


    add_shared_legend(
        fig,
        axes,
        y=0.995,
    )


    configuration_text(
        fig,
        "Batch/GPU = 16   |   Sequence Length = 256",
        y=0.855,
    )


    fig.subplots_adjust(

        top=0.74,

        bottom=0.20,

        left=0.095,

        right=0.985,

        wspace=0.34,
    )


    save_figure(
        fig,
        out_dir,
        "fig1_gpu_count",
    )


    data.to_csv(
        out_dir
        /
        "fig1_gpu_count_data.csv",
        index=False,
    )


# ============================================================
# FIGURE 2
#
# Batch size and Sequence Length
# ============================================================

def plot_workload(
    perf,
    out_dir,
):

    # --------------------------------------------------------
    # Batch
    # --------------------------------------------------------

    batch_raw = perf[
        perf["family"].isin(
            [
                "batch_b4",
                "batch_b8",
                "batch_b16",
            ]
        )
    ].copy()


    batch = aggregate(

        batch_raw,

        [
            "method",
            "batch",
        ],
    )


    # --------------------------------------------------------
    # Sequence
    # --------------------------------------------------------

    seq_raw = perf[
        perf["family"].isin(
            [
                "seq_l64",
                "seq_l128",
                "seq_l256",
            ]
        )
    ].copy()


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
            4.82,
        ),
    )


    # ========================================================
    # Batch throughput
    # ========================================================

    ax = axes[0, 0]

    plot_methods(
        ax,
        batch,
        "batch",
        "throughput_mean",
        "throughput_std",
    )

    ax.set_title(
        "Throughput"
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
        zero_bottom=True,
    )


    # ========================================================
    # Batch latency
    # ========================================================

    ax = axes[0, 1]

    plot_methods(
        ax,
        batch,
        "batch",
        "p95_mean",
        "p95_std",
    )

    ax.set_title(
        "Tail Latency"
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

    plot_methods(
        ax,
        seq,
        "seq_len",
        "throughput_mean",
        "throughput_std",
    )

    ax.set_title(
        "Throughput"
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
        zero_bottom=True,
    )


    # ========================================================
    # Sequence latency
    # ========================================================

    ax = axes[1, 1]

    plot_methods(
        ax,
        seq,
        "seq_len",
        "p95_mean",
        "p95_std",
    )

    ax.set_title(
        "Tail Latency"
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
        y=0.995,
    )


    # --------------------------------------------------------
    # Real configuration labels
    # --------------------------------------------------------

    fig.text(
        0.5,
        0.885,
        "Batch Size Experiment: 4 GPUs   |   Sequence Length = 256",
        ha="center",
        va="center",
        fontsize=8.3,
        color="#555555",
    )


    fig.text(
        0.5,
        0.455,
        "Sequence Length Experiment: 4 GPUs   |   Batch/GPU = 16",
        ha="center",
        va="center",
        fontsize=8.3,
        color="#555555",
    )


    fig.subplots_adjust(

        top=0.82,

        bottom=0.10,

        left=0.095,

        right=0.985,

        hspace=0.69,

        wspace=0.34,
    )


    save_figure(
        fig,
        out_dir,
        "fig2_batch_sequence",
    )


    batch.to_csv(
        out_dir
        /
        "fig2_batch_data.csv",
        index=False,
    )


    seq.to_csv(
        out_dir
        /
        "fig2_sequence_data.csv",
        index=False,
    )


# ============================================================
# FIGURE 3
#
# Cache size
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


    data = aggregate(

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


    # --------------------------------------------------------
    # Throughput
    # --------------------------------------------------------

    ax = axes[0]

    plot_methods(
        ax,
        data,
        "cache",
        "throughput_mean",
        "throughput_std",
        methods=methods,
    )

    ax.set_title(
        "Throughput"
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

    style_axis(ax)


    # --------------------------------------------------------
    # P95
    # --------------------------------------------------------

    ax = axes[1]

    plot_methods(
        ax,
        data,
        "cache",
        "p95_mean",
        "p95_std",
        methods=methods,
    )

    ax.set_title(
        "Tail Latency"
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
        y=0.995,
    )


    configuration_text(
        fig,
        "4 GPUs   |   Batch/GPU = 16   |   Sequence Length = 256",
        y=0.855,
    )


    fig.subplots_adjust(

        top=0.74,

        bottom=0.20,

        left=0.095,

        right=0.985,

        wspace=0.34,
    )


    save_figure(
        fig,
        out_dir,
        "fig3_cache_size",
    )


    data.to_csv(
        out_dir
        /
        "fig3_cache_data.csv",
        index=False,
    )


# ============================================================
# SLO
# ============================================================

def load_slo(root):

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


    for f in candidates:

        if f.exists():

            print(
                "[INFO] SLO file:",
                f,
            )

            return pd.read_csv(
                f
            )


    raise FileNotFoundError(
        "SLO summary not found."
    )


# ============================================================
# FIGURE 4
#
# SLO Target
# ============================================================

def plot_slo(
    slo,
    out_dir,
):

    fig, axes = plt.subplots(

        1,
        2,

        figsize=(
            7.15,
            2.72,
        ),
    )


    # --------------------------------------------------------
    # Violation
    # --------------------------------------------------------

    ax = axes[0]

    plot_methods(
        ax,
        slo,
        "deadline_ms",
        "violation_mean",
        "violation_std",
    )

    ax.set_title(
        "SLO Violation Rate"
    )

    ax.set_xlabel(
        "SLO Target (ms)"
    )

    ax.set_ylabel(
        "E2E Violation Rate (%)"
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

    style_axis(ax)


    # --------------------------------------------------------
    # P95 latency
    # --------------------------------------------------------

    ax = axes[1]

    plot_methods(
        ax,
        slo,
        "deadline_ms",
        "p95_mean",
        "p95_std",
    )


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
                2,
            )
        ),

        color="#8D8D8D",

        linewidth=1.15,

        zorder=1,
    )


    ax.set_title(
        "Tail Latency"
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
        y=0.995,
    )


    configuration_text(
        fig,
        "4 GPUs   |   Batch/GPU = 16   |   Sequence Length = 256",
        y=0.855,
    )


    fig.subplots_adjust(

        top=0.74,

        bottom=0.20,

        left=0.095,

        right=0.985,

        wspace=0.34,
    )


    save_figure(
        fig,
        out_dir,
        "fig4_slo_target",
    )


    slo.to_csv(
        out_dir
        /
        "fig4_slo_data.csv",
        index=False,
    )


# ============================================================
# Print useful numerical summary
# ============================================================

def print_summary(
    perf,
):

    gpu_raw = perf[
        perf["family"]
        ==
        "strong"
    ]


    gpu_data = aggregate(

        gpu_raw,

        [
            "method",
            "gpus",
        ],
    )


    four = (
        gpu_data[
            gpu_data["gpus"]
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
        "GPU=4 RESULTS"
    )

    print(
        "=" * 72
    )


    print(
        four[
            [
                "throughput_mean",
                "throughput_std",
                "p95_mean",
                "p95_std",
            ]
        ]
        .round(3)
        .to_string()
    )


    if (
        "DSCA-MoE"
        in four.index
    ):

        dsca = four.loc[
            "DSCA-MoE"
        ]


        for baseline in [
            "DeepSpeed",
            "HarMoEny",
        ]:

            if baseline not in four.index:
                continue

            b = four.loc[
                baseline
            ]


            thr_gain = (

                dsca[
                    "throughput_mean"
                ]

                /
                b[
                    "throughput_mean"
                ]

                -
                1

            ) * 100


            p95_reduction = (

                1

                -

                dsca[
                    "p95_mean"
                ]

                /
                b[
                    "p95_mean"
                ]

            ) * 100


            print()

            print(
                f"DSCA-MoE vs {baseline}:"
            )

            print(
                f"  throughput gain = "
                f"{thr_gain:.2f}%"
            )

            print(
                f"  P95 reduction   = "
                f"{p95_reduction:.2f}%"
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


    if not root.exists():

        raise FileNotFoundError(
            root
        )


    output_dir = (
        root
        /
        "figures_final_v2"
    )


    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    perf = load_performance(
        root
    )


    # --------------------------------------------------------
    # Fig.1 GPU
    # --------------------------------------------------------

    print()
    print(
        "===== Figure 1: GPU Count ====="
    )

    plot_gpu_scaling(
        perf,
        output_dir,
    )


    # --------------------------------------------------------
    # Fig.2 Batch + Sequence
    # --------------------------------------------------------

    print()
    print(
        "===== Figure 2: Batch / Sequence Length ====="
    )

    plot_workload(
        perf,
        output_dir,
    )


    # --------------------------------------------------------
    # Fig.3 Cache
    # --------------------------------------------------------

    print()
    print(
        "===== Figure 3: Expert Cache Size ====="
    )

    plot_cache(
        perf,
        output_dir,
    )


    # --------------------------------------------------------
    # Fig.4 SLO
    # --------------------------------------------------------

    print()
    print(
        "===== Figure 4: SLO Target ====="
    )

    slo = load_slo(
        root
    )

    plot_slo(
        slo,
        output_dir,
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        perf
    )


    print()
    print(
        "=" * 72
    )

    print(
        "DONE"
    )

    print(
        "=" * 72
    )

    print(
        "Output:",
        output_dir
    )

    print()

    print(
        "Generated:"
    )

    print(
        "  fig1_gpu_count.pdf"
    )

    print(
        "  fig2_batch_sequence.pdf"
    )

    print(
        "  fig3_cache_size.pdf"
    )

    print(
        "  fig4_slo_target.pdf"
    )


if __name__ == "__main__":
    main()
