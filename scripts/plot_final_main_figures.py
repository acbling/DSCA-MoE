#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final main-paper figures for DSCA-MoE.

Final figures
-------------
Fig. 1: Number of GPUs
    Fixed:
      Batch/GPU = 16
      Sequence Length = 256

Fig. 2: Batch Size
    Fixed:
      4 GPUs
      Sequence Length = 256

Fig. 3: Expert Cache Size
    Fixed:
      4 GPUs
      Batch/GPU = 16
      Sequence Length = 256

Notes
-----
1. Error bars = mean ± standard deviation over three runs.
2. The unusually large DeepSpeed 1-GPU P95 error bar is hidden
   for visualization; its mean value is unchanged.
3. DeepSpeed does not expose the same expert-cache-size control.
   In Fig. 3 it is therefore shown as a fixed 4-GPU reference,
   not as a cache-size sweep.

Inputs
------
<root>/analysis_fixed/performance_runs.csv
<root>/run_status.csv

Outputs
-------
<root>/figures_main_final/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Default experiment root
# ============================================================

DEFAULT_ROOT = (
    "/home/lcf/MoE_Syn/HarMoEny-main/"
    "experiment_outputs/complete_20260923_095020"
)


# ============================================================
# Method style
# ============================================================

METHODS = [
    "DeepSpeed",
    "HarMoEny",
    "DSCA-MoE",
]

COLORS = {
    "DeepSpeed": "#D98C00",
    "HarMoEny": "#0A9B7A",
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
    "DeepSpeed": 5.4,
    "HarMoEny": 5.4,
    "DSCA-MoE": 6.0,
}


# ============================================================
# Matplotlib publication style
# ============================================================

plt.rcParams.update({

    "font.family": "serif",

    "font.serif": [
        "Times New Roman",
        "Times",
        "DejaVu Serif",
    ],

    "font.size": 8.6,

    "axes.labelsize": 9.2,

    "axes.titlesize": 9.4,

    "legend.fontsize": 8.4,

    "xtick.labelsize": 8.2,

    "ytick.labelsize": 8.2,

    "axes.linewidth": 0.9,

    "xtick.major.width": 0.8,

    "ytick.major.width": 0.8,

    "xtick.major.size": 3.6,

    "ytick.major.size": 3.6,

    # editable vector fonts
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "savefig.dpi": 600,
})


# ============================================================
# Common plotting helpers
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

    ax.spines["left"].set_color("#303030")
    ax.spines["bottom"].set_color("#303030")

    ax.tick_params(
        direction="out",
        pad=3,
    )

    if zero_bottom:
        ymin, ymax = ax.get_ylim()

        if ymin > 0:
            ax.set_ylim(
                0,
                ymax,
            )


def panel_label(ax, label):

    ax.text(
        -0.13,
        1.055,
        label,
        transform=ax.transAxes,
        fontsize=9.6,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def add_config_text(fig, text, y=0.855):

    fig.text(
        0.5,
        y,
        text,
        ha="center",
        va="center",
        fontsize=8.4,
        color="#555555",
    )


def save_figure(fig, out_dir, name):

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"

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

    print(f"[FIG] {pdf}")


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

        h, l = ax.get_legend_handles_labels()

        for hh, ll in zip(h, l):

            if (
                ll in methods
                and ll not in labels
            ):

                handles.append(hh)
                labels.append(ll)

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

            ncol=len(ordered_labels),

            frameon=False,

            handlelength=2.8,

            handletextpad=0.65,

            columnspacing=2.2,

            borderaxespad=0,
        )


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
                data["method"] == method
            ]
            .sort_values(x)
        )

        if len(d) == 0:
            continue

        error = None

        if (
            yerr is not None
            and yerr in d.columns
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

            markeredgewidth=0.4,

            capsize=2.3,

            capthick=1.0,

            elinewidth=1.0,

            label=method,

            zorder=4
            if method == "DSCA-MoE"
            else 3,
        )


# ============================================================
# Load performance and recover experimental configuration
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
    # Normalize keys
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


    config_columns = [
        "batch",
        "num_samples",
        "seq_len",
        "cache",
        "skew",
        "deadline",
    ]


    available_config = [
        c
        for c in config_columns
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


    # Remove duplicated configuration columns
    for col in available_config:

        if col in perf.columns:

            perf = perf.drop(
                columns=[col]
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

    for col in [
        "batch",
        "num_samples",
        "seq_len",
        "cache",
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


    print(
        "[INFO] performance rows:",
        len(perf)
    )

    if len(perf) == 165:

        print(
            "[OK] all 165 experiment runs loaded"
        )

    else:

        print(
            "[WARN] expected 165 runs, "
            f"found {len(perf)}"
        )


    return perf


# ============================================================
# Aggregate repeats
# ============================================================

def aggregate(df, group_cols):

    if len(df) == 0:

        raise RuntimeError(
            f"No data for {group_cols}"
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

            runs=(
                "repeat",
                "nunique",
            ),
        )
    )


    result[
        [
            "throughput_std",
            "p95_std",
        ]
    ] = (
        result[
            [
                "throughput_std",
                "p95_std",
            ]
        ]
        .fillna(0)
    )


    return result


# ============================================================
# FIGURE 1
# Number of GPUs
#
# Fixed:
# Batch/GPU = 16
# Sequence Length = 256
# ============================================================

def plot_gpu_count(perf, out_dir):

    # Internally stored as "strong".
    # This internal tag is NOT shown in the paper.
    raw = perf[
        perf["family"] == "strong"
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
            2.75,
        ),
    )


    # --------------------------------------------------------
    # (a) Throughput
    # --------------------------------------------------------

    ax = axes[0]

    plot_method_lines(
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
    # (b) P95 latency
    # --------------------------------------------------------

    ax = axes[1]


    # Copy so original CSV/statistics are untouched.
    latency_data = data.copy()


    # Hide only the oversized DeepSpeed 1-GPU P95 error bar.
    # Mean value remains unchanged.
    latency_data.loc[
        (
            latency_data["method"]
            ==
            "DeepSpeed"
        )
        &
        (
            latency_data["gpus"]
            ==
            1
        ),
        "p95_std"
    ] = 0.0


    plot_method_lines(
        ax,
        latency_data,
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


    # --------------------------------------------------------
    # Legend + configuration
    # --------------------------------------------------------

    add_shared_legend(
        fig,
        axes,
        y=0.995,
    )


    add_config_text(
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
# Batch Size
#
# Fixed:
# 4 GPUs
# Sequence Length = 256
# ============================================================

def plot_batch_size(perf, out_dir):

    raw = perf[
        perf["family"].isin(
            [
                "batch_b4",
                "batch_b8",
                "batch_b16",
            ]
        )
    ].copy()


    data = aggregate(
        raw,
        [
            "method",
            "batch",
        ],
    )


    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            7.15,
            2.75,
        ),
    )


    # --------------------------------------------------------
    # Throughput
    # --------------------------------------------------------

    ax = axes[0]

    plot_method_lines(
        ax,
        data,
        "batch",
        "throughput_mean",
        "throughput_std",
    )


    ax.set_title(
        "Aggregate Throughput"
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


    # --------------------------------------------------------
    # Tail latency
    # --------------------------------------------------------

    ax = axes[1]

    plot_method_lines(
        ax,
        data,
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


    # --------------------------------------------------------
    # Legend + configuration
    # --------------------------------------------------------

    add_shared_legend(
        fig,
        axes,
        y=0.995,
    )


    add_config_text(
        fig,
        "4 GPUs   |   Sequence Length = 256",
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
        "fig2_batch_size",
    )


    data.to_csv(
        out_dir
        /
        "fig2_batch_size_data.csv",
        index=False,
    )


# ============================================================
# FIGURE 3
# Expert Cache Size
#
# HarMoEny and DSCA-MoE:
# actual cache sweep C = 2 / 4 / 8
#
# DeepSpeed:
# fixed reference under the same 4-GPU/B16/L256 workload.
# ============================================================

def plot_cache_size(perf, out_dir):

    cache_raw = perf[
        perf["family"].isin(
            [
                "cache_c2",
                "cache_c4",
                "cache_c8",
            ]
        )
    ].copy()


    cache_data = aggregate(
        cache_raw,
        [
            "method",
            "cache",
        ],
    )


    cache_methods = [
        "HarMoEny",
        "DSCA-MoE",
    ]


    # --------------------------------------------------------
    # DeepSpeed fixed reference
    #
    # Same workload:
    # 4 GPUs, Batch/GPU=16, Sequence Length=256.
    # --------------------------------------------------------

    ds_raw = perf[
        (
            perf["family"]
            ==
            "strong"
        )
        &
        (
            perf["method"]
            ==
            "DeepSpeed"
        )
        &
        (
            perf["gpus"]
            ==
            4
        )
    ].copy()


    if len(ds_raw) == 0:

        raise RuntimeError(
            "Cannot find DeepSpeed 4-GPU reference."
        )


    ds_thr_mean = (
        ds_raw[
            "throughput"
        ].mean()
    )

    ds_thr_std = (
        ds_raw[
            "throughput"
        ].std()
    )


    ds_p95_mean = (
        ds_raw[
            "p95_ms"
        ].mean()
    )

    ds_p95_std = (
        ds_raw[
            "p95_ms"
        ].std()
    )


    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            7.15,
            2.75,
        ),
    )


    # ========================================================
    # (a) Throughput
    # ========================================================

    ax = axes[0]


    # actual cache sweep
    plot_method_lines(
        ax,
        cache_data,
        "cache",
        "throughput_mean",
        "throughput_std",
        methods=cache_methods,
    )


    # DeepSpeed fixed baseline
    ax.axhline(
        ds_thr_mean,

        color=COLORS[
            "DeepSpeed"
        ],

        linestyle=LINESTYLES[
            "DeepSpeed"
        ],

        linewidth=LINEWIDTHS[
            "DeepSpeed"
        ],

        label="DeepSpeed",

        zorder=2,
    )


    # very light std band
    ax.fill_between(
        [2, 8],

        ds_thr_mean
        -
        ds_thr_std,

        ds_thr_mean
        +
        ds_thr_std,

        color=COLORS[
            "DeepSpeed"
        ],

        alpha=0.06,

        linewidth=0,

        zorder=1,
    )


    ax.set_title(
        "Aggregate Throughput"
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


    # ========================================================
    # (b) Tail latency
    # ========================================================

    ax = axes[1]


    plot_method_lines(
        ax,
        cache_data,
        "cache",
        "p95_mean",
        "p95_std",
        methods=cache_methods,
    )


    # DeepSpeed fixed baseline
    ax.axhline(
        ds_p95_mean,

        color=COLORS[
            "DeepSpeed"
        ],

        linestyle=LINESTYLES[
            "DeepSpeed"
        ],

        linewidth=LINEWIDTHS[
            "DeepSpeed"
        ],

        label="DeepSpeed",

        zorder=2,
    )


    ax.fill_between(
        [2, 8],

        ds_p95_mean
        -
        ds_p95_std,

        ds_p95_mean
        +
        ds_p95_std,

        color=COLORS[
            "DeepSpeed"
        ],

        alpha=0.06,

        linewidth=0,

        zorder=1,
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


    # --------------------------------------------------------
    # Shared legend
    # --------------------------------------------------------

    add_shared_legend(
        fig,
        axes,
        methods=[
            "DeepSpeed",
            "HarMoEny",
            "DSCA-MoE",
        ],
        y=0.995,
    )


    add_config_text(
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


    # --------------------------------------------------------
    # Save actual cache data
    # --------------------------------------------------------

    cache_data.to_csv(
        out_dir
        /
        "fig3_cache_size_data.csv",
        index=False,
    )


    # Save reference separately to avoid pretending that
    # DeepSpeed has cache=2/4/8 measurements.
    reference = pd.DataFrame({

        "method": [
            "DeepSpeed"
        ],

        "gpus": [
            4
        ],

        "batch_per_gpu": [
            16
        ],

        "sequence_length": [
            256
        ],

        "throughput_mean": [
            ds_thr_mean
        ],

        "throughput_std": [
            ds_thr_std
        ],

        "p95_mean": [
            ds_p95_mean
        ],

        "p95_std": [
            ds_p95_std
        ],

    })


    reference.to_csv(
        out_dir
        /
        "fig3_deepspeed_reference.csv",
        index=False,
    )


# ============================================================
# Print numerical summary
# ============================================================

def print_summary(perf):

    print()
    print(
        "=" * 76
    )

    print(
        "FINAL MAIN-FIGURE SUMMARY"
    )

    print(
        "=" * 76
    )


    # --------------------------------------------------------
    # GPU=4
    # --------------------------------------------------------

    gpu = aggregate(

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


    g4 = (
        gpu[
            gpu["gpus"]
            ==
            4
        ]
        .set_index(
            "method"
        )
    )


    print()
    print(
        "[GPU Count] 4-GPU operating point"
    )

    print(
        g4[
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


    # --------------------------------------------------------
    # Batch=16
    # --------------------------------------------------------

    batch = aggregate(

        perf[
            perf["family"].isin(
                [
                    "batch_b4",
                    "batch_b8",
                    "batch_b16",
                ]
            )
        ],

        [
            "method",
            "batch",
        ],
    )


    b16 = (
        batch[
            batch["batch"]
            ==
            16
        ]
        .set_index(
            "method"
        )
    )


    print()
    print(
        "[Batch Size] B/GPU=16"
    )

    print(
        b16[
            [
                "throughput_mean",
                "p95_mean",
            ]
        ]
        .round(3)
        .to_string()
    )


    # --------------------------------------------------------
    # DSCA improvements
    # --------------------------------------------------------

    if "DSCA-MoE" in g4.index:

        dsca = g4.loc[
            "DSCA-MoE"
        ]


        for baseline in [
            "DeepSpeed",
            "HarMoEny",
        ]:

            if baseline not in g4.index:
                continue

            base = g4.loc[
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


            p95_reduction = (
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
                f"DSCA-MoE vs {baseline} @ 4 GPUs:"
            )

            print(
                f"  Throughput gain = "
                f"{throughput_gain:.2f}%"
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


    out_dir = (
        root
        /
        "figures_main_final"
    )


    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    perf = load_performance(
        root
    )


    # --------------------------------------------------------
    # Figure 1
    # --------------------------------------------------------

    print()
    print(
        "===== Fig. 1: Number of GPUs ====="
    )

    plot_gpu_count(
        perf,
        out_dir,
    )


    # --------------------------------------------------------
    # Figure 2
    # --------------------------------------------------------

    print()
    print(
        "===== Fig. 2: Batch Size ====="
    )

    plot_batch_size(
        perf,
        out_dir,
    )


    # --------------------------------------------------------
    # Figure 3
    # --------------------------------------------------------

    print()
    print(
        "===== Fig. 3: Expert Cache Size ====="
    )

    plot_cache_size(
        perf,
        out_dir,
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        perf
    )


    print()
    print(
        "=" * 76
    )

    print(
        "DONE"
    )

    print(
        "=" * 76
    )

    print(
        "Output:",
        out_dir
    )

    print()

    print(
        "Main-paper figures:"
    )

    print(
        "  fig1_gpu_count.pdf"
    )

    print(
        "  fig2_batch_size.pdf"
    )

    print(
        "  fig3_cache_size.pdf"
    )


if __name__ == "__main__":
    main()
