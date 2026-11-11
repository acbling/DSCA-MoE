#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import re
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


def save_fig(fig, outdir, name):

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf = outdir / f"{name}.pdf"
    png = outdir / f"{name}.png"

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


def add_legend(fig, axes):

    handles = []
    labels = []

    for ax in np.ravel(axes):

        h, l = ax.get_legend_handles_labels()

        for hh, ll in zip(h, l):

            if (
                ll in METHOD_ORDER
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
            ncol=3,
            frameon=False,
        )


def plot_metric(
    ax,
    df,
    x,
    y,
    yerr,
    xlabel,
    ylabel,
    title,
    xticks=None,
    xticklabels=None,
):

    for method in METHOD_ORDER:

        d = df[
            df["method"] == method
        ].sort_values(x)

        if len(d) == 0:
            continue

        err = None

        if (
            yerr is not None
            and yerr in d.columns
        ):

            err = d[yerr].fillna(0).values

        ax.errorbar(
            d[x],
            d[y],
            yerr=err,

            marker=MARKERS[method],
            markersize=5.5,

            linewidth=1.8,
            capsize=2.5,

            linestyle=LINESTYLES[method],
            color=COLORS[method],

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
# Find run directories
# ============================================================

def find_run_dir(root, row):

    family = str(row["family"])
    system = str(row["system"])
    gpus = int(row["gpus"])
    repeat = int(row["repeat"])

    tag = (
        f"{family}_"
        f"{system}_"
        f"{gpus}gpu_"
        f"r{repeat}"
    )

    p = root / family / tag

    if p.exists():
        return p

    candidates = list(
        (root / family).glob(
            f"{tag}*"
        )
    )

    candidates = [
        x for x in candidates
        if x.is_dir()
    ]

    if candidates:
        return candidates[0]

    return None


# ============================================================
# Detect latency column
# ============================================================

def to_ms(series, name):

    s = pd.to_numeric(
        series,
        errors="coerce",
    )

    s = s[
        np.isfinite(s)
    ].astype(float)

    name = name.lower()

    if len(s) == 0:
        return s

    if (
        "ns" in name
        or "nanosec" in name
    ):

        return s / 1e6

    if (
        "us" in name
        or "micro" in name
    ):

        return s / 1e3

    if (
        "ms" in name
        or "millisec" in name
    ):

        return s

    if (
        "second" in name
        or name.endswith("_s")
        or name.endswith("_sec")
    ):

        return s * 1000.0

    # Fallback:
    # inference latency below 20 is almost certainly seconds.
    median = np.nanmedian(
        np.abs(s)
    )

    if median < 20:

        return s * 1000.0

    return s


def detect_latency(df, source=""):

    candidates = []

    for col in df.columns:

        name = col.lower()

        if any(
            key in name
            for key in [
                "latency",
                "duration",
                "elapsed",
                "e2e",
                "iter_time",
                "batch_time",
                "total_time",
                "cost_time",
            ]
        ):

            if any(
                key in name
                for key in [
                    "timestamp",
                    "start",
                    "end",
                ]
            ):
                continue

            values = pd.to_numeric(
                df[col],
                errors="coerce",
            )

            if values.notna().sum() == 0:
                continue

            score = 0

            if "latency" in name:
                score += 10

            if "duration" in name:
                score += 8

            if "e2e" in name:
                score += 8

            if "elapsed" in name:
                score += 6

            if "ms" in name:
                score += 3

            candidates.append(
                (score, col)
            )

    if candidates:

        candidates.sort(
            reverse=True
        )

        col = candidates[0][1]

        values = to_ms(
            df[col],
            col,
        )

        print(
            f"[SCHEMA] {source}: "
            f"latency={col}"
        )

        return (
            values.reset_index(drop=True),
            col,
        )

    # --------------------------------------------------------
    # start/end fallback
    # --------------------------------------------------------

    starts = [
        c for c in df.columns
        if "start" in c.lower()
    ]

    ends = [
        c for c in df.columns
        if (
            "end" in c.lower()
            or "finish" in c.lower()
        )
    ]

    for sc in starts:

        for ec in ends:

            start = pd.to_numeric(
                df[sc],
                errors="coerce",
            )

            end = pd.to_numeric(
                df[ec],
                errors="coerce",
            )

            delta = end - start

            delta = delta[
                np.isfinite(delta)
                &
                (delta >= 0)
            ]

            if len(delta) == 0:
                continue

            median = delta.median()

            if median < 20:
                delta *= 1000.0

            print(
                f"[SCHEMA] {source}: "
                f"latency={ec}-{sc}"
            )

            return (
                delta.reset_index(drop=True),
                f"{ec}-{sc}",
            )

    return None, None


# ============================================================
# Load status
# ============================================================

def load_status(root):

    path = root / "run_status.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    df = df[
        df["status"].str.upper()
        == "SUCCESS"
    ].copy()

    df["method"] = (
        df["system"]
        .str.lower()
        .map(METHOD_MAP)
    )

    df["run_dir"] = [
        str(
            find_run_dir(
                root,
                row,
            )
            or ""
        )
        for _, row in df.iterrows()
    ]

    print(
        "[INFO] successful runs:",
        len(df),
    )

    return df


# ============================================================
# Read E2E matrix
#
# rows = iterations
# cols = GPU ranks
# ============================================================

def get_rank(path):

    try:
        return int(
            path.parent.name
        )

    except Exception:

        m = re.search(
            r"rank[_-]?(\d+)",
            str(path),
            re.I,
        )

        if m:
            return int(m.group(1))

    return -1


def load_e2e_matrix(run_dir):

    run_dir = Path(run_dir)

    if not run_dir.exists():
        return None

    files = sorted(
        run_dir.rglob(
            "e2e.csv"
        )
    )

    if not files:
        return None

    values = []

    for f in files:

        try:

            df = pd.read_csv(f)

        except Exception as e:

            print(
                "[WARN] read failed:",
                f,
                e,
            )

            continue

        latency, col = detect_latency(
            df,
            str(f),
        )

        if latency is None:
            continue

        if len(latency) == 0:
            continue

        values.append(
            (
                get_rank(f),
                latency.values,
            )
        )

    if not values:
        return None

    values.sort(
        key=lambda x: x[0]
    )

    n = min(
        len(x[1])
        for x in values
    )

    if n == 0:
        return None

    matrix = np.stack(
        [
            x[1][:n]
            for x in values
        ],
        axis=1,
    )

    return matrix


# ============================================================
# E2E metrics
# ============================================================

def extract_e2e(status):

    rows = []

    arrays = {}

    for _, r in status.iterrows():

        matrix = load_e2e_matrix(
            r["run_dir"]
        )

        if matrix is None:
            continue

        # System latency = slowest rank
        system_latency = np.max(
            matrix,
            axis=1,
        )

        rank_mean = np.mean(
            matrix,
            axis=1,
        )

        rank_min = np.min(
            matrix,
            axis=1,
        )

        rank_max = np.max(
            matrix,
            axis=1,
        )

        spread = (
            rank_max
            -
            rank_min
        )

        straggler = np.where(
            rank_max > 0,

            (
                rank_max
                -
                rank_mean
            )
            /
            rank_max
            *
            100,

            np.nan,
        )

        key = (
            r["family"],
            r["system"],
            int(r["gpus"]),
            int(r["repeat"]),
        )

        arrays[key] = system_latency

        rows.append({

            "family":
                r["family"],

            "system":
                r["system"],

            "method":
                r["method"],

            "gpus":
                int(r["gpus"]),

            "repeat":
                int(r["repeat"]),

            "deadline_s":
                float(r["deadline"]),

            "num_iterations":
                len(system_latency),

            "mean_ms":
                np.mean(system_latency),

            "p50_ms":
                np.percentile(
                    system_latency,
                    50,
                ),

            "p95_ms":
                np.percentile(
                    system_latency,
                    95,
                ),

            "p99_ms":
                np.percentile(
                    system_latency,
                    99,
                ),

            "rank_spread_ms":
                np.mean(spread),

            "straggler_pct":
                np.mean(straggler),

        })

    return (
        pd.DataFrame(rows),
        arrays,
    )


# ============================================================
# TRUE SLO violation
# ============================================================

def build_slo(
    e2e,
    arrays,
):

    deadlines = [
        300,
        400,
        500,
        700,
    ]

    rows = []

    # --------------------------------------------------------
    # Baselines:
    # use same strong-4GPU runs;
    # scheduler is not deadline-aware.
    # --------------------------------------------------------

    for method, system in [
        ("DeepSpeed", "deepspeed"),
        ("HarMoEny", "harmony"),
    ]:

        base = e2e[
            (e2e["family"] == "strong")
            &
            (e2e["system"] == system)
            &
            (e2e["gpus"] == 4)
        ]

        for deadline in deadlines:

            for _, r in base.iterrows():

                key = (
                    r["family"],
                    r["system"],
                    int(r["gpus"]),
                    int(r["repeat"]),
                )

                lat = arrays[key]

                rows.append({

                    "method":
                        method,

                    "deadline_ms":
                        deadline,

                    "repeat":
                        int(r["repeat"]),

                    "violation_pct":
                        np.mean(
                            lat > deadline
                        )
                        *
                        100,

                    "p95_ms":
                        np.percentile(
                            lat,
                            95,
                        ),

                })

    # --------------------------------------------------------
    # DSCA:
    # use deadline-specific reruns
    # --------------------------------------------------------

    for deadline in deadlines:

        family = (
            f"slo_d"
            f"{deadline/1000:.2f}"
        )

        d = e2e[
            (e2e["family"] == family)
            &
            (e2e["system"] == "qosmoe")
        ]

        for _, r in d.iterrows():

            key = (
                r["family"],
                r["system"],
                int(r["gpus"]),
                int(r["repeat"]),
            )

            lat = arrays[key]

            rows.append({

                "method":
                    "DSCA-MoE",

                "deadline_ms":
                    deadline,

                "repeat":
                    int(r["repeat"]),

                "violation_pct":
                    np.mean(
                        lat > deadline
                    )
                    *
                    100,

                "p95_ms":
                    np.percentile(
                        lat,
                        95,
                    ),

            })

    raw = pd.DataFrame(rows)

    summary = (
        raw
        .groupby(
            [
                "method",
                "deadline_ms",
            ],
            as_index=False,
        )
        .agg(

            violation_mean=(
                "violation_pct",
                "mean",
            ),

            violation_std=(
                "violation_pct",
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

    return raw, summary


# ============================================================
# Expert load parsing
# ============================================================

def parse_cell_ids(x):

    if pd.isna(x):
        return []

    if isinstance(
        x,
        (
            int,
            np.integer,
        )
    ):

        return [int(x)]

    if isinstance(
        x,
        (
            float,
            np.floating,
        )
    ):

        if np.isfinite(x):
            return [int(x)]

        return []

    text = str(x).strip()

    if not text:
        return []

    if (
        text.startswith("[")
        or text.startswith("(")
    ):

        try:

            obj = ast.literal_eval(
                text
            )

            if isinstance(
                obj,
                (
                    list,
                    tuple,
                )
            ):

                return [
                    int(v)
                    for v in obj
                ]

        except Exception:
            pass

    return [
        int(v)
        for v in re.findall(
            r"-?\d+",
            text,
        )
    ]


def get_layer_col(df):

    candidates = {
        "layer",
        "layer_id",
        "layer_idx",
        "layer_index",
        "moe_layer",
    }

    for c in df.columns:

        if c.lower() in candidates:
            return c

    return None


def expert_counts(
    df,
    num_experts=64,
):

    # --------------------------------------------------------
    # Case 1:
    # expert_0 ... expert_63
    # --------------------------------------------------------

    wide = {}

    for c in df.columns:

        m = re.fullmatch(
            r"(?:expert|e)[_-]?(\d+)",
            c.lower(),
        )

        if not m:
            continue

        expert = int(
            m.group(1)
        )

        if (
            0
            <= expert
            < num_experts
        ):

            wide[expert] = c

    if len(wide) >= 4:

        counts = np.zeros(
            num_experts
        )

        for expert, col in wide.items():

            values = pd.to_numeric(
                df[col],
                errors="coerce",
            ).fillna(0)

            counts[expert] += (
                values.sum()
            )

        return counts

    # --------------------------------------------------------
    # Case 2:
    # expert_id / selected_expert / ...
    # --------------------------------------------------------

    candidates = []

    for c in df.columns:

        name = c.lower()

        if "expert" not in name:
            continue

        if any(
            x in name
            for x in [
                "num_expert",
                "expert_count",
                "expert_cache",
                "expert_size",
            ]
        ):
            continue

        score = 0

        if name in {
            "expert",
            "expert_id",
            "expert_idx",
            "expert_index",
        }:
            score += 10

        if any(
            x in name
            for x in [
                "selected",
                "route",
                "routed",
            ]
        ):
            score += 5

        if any(
            x in name
            for x in [
                "id",
                "idx",
                "index",
            ]
        ):
            score += 3

        candidates.append(
            (score, c)
        )

    candidates.sort(
        reverse=True
    )

    for _, col in candidates:

        ids = []

        for x in df[col]:

            values = parse_cell_ids(x)

            values = [
                v for v in values
                if (
                    0
                    <= v
                    < num_experts
                )
            ]

            ids.extend(values)

        if len(ids) < 10:
            continue

        return np.bincount(
            ids,
            minlength=num_experts,
        ).astype(float)

    return None


def compute_load_stats(counts):

    mean = np.mean(counts)

    if mean <= 0:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    cv = (
        np.std(counts)
        /
        mean
    )

    max_mean = (
        np.max(counts)
        /
        mean
    )

    top8 = (
        np.sort(counts)[-8:]
        .sum()
        /
        counts.sum()
        *
        100
    )

    return (
        cv,
        max_mean,
        top8,
    )


def extract_expert_load(
    status,
    num_experts=64,
):

    rows = []

    for _, r in status.iterrows():

        run_dir = Path(
            r["run_dir"]
        )

        if not run_dir.exists():
            continue

        files = sorted(
            run_dir.rglob(
                "real_token_trace*.csv"
            )
        )

        if not files:
            continue

        total = np.zeros(
            num_experts
        )

        success = False

        for f in files:

            try:

                df = pd.read_csv(f)

            except Exception:

                continue

            counts = expert_counts(
                df,
                num_experts,
            )

            if counts is None:
                continue

            total += counts

            success = True

        if not success:
            continue

        cv, max_mean, top8 = (
            compute_load_stats(
                total
            )
        )

        rows.append({

            "family":
                r["family"],

            "system":
                r["system"],

            "method":
                r["method"],

            "gpus":
                int(r["gpus"]),

            "repeat":
                int(r["repeat"]),

            "load_cv":
                cv,

            "max_mean_load":
                max_mean,

            "top8_share_pct":
                top8,

        })

    return pd.DataFrame(rows)


# ============================================================
# Layer latency
# ============================================================

def get_layer_id(path):

    m = re.search(
        r"(?:moe_)?layer-(\d+)\.csv$",
        path.name,
    )

    if m:
        return int(
            m.group(1)
        )

    return None


def extract_layer_latency(status):

    rows = []

    strong = status[
        (status["family"] == "strong")
        &
        (status["gpus"] == 4)
    ]

    for _, r in strong.iterrows():

        run_dir = Path(
            r["run_dir"]
        )

        files = (
            list(
                run_dir.rglob(
                    "moe_layer-*.csv"
                )
            )
            +
            list(
                run_dir.rglob(
                    "layer-*.csv"
                )
            )
        )

        layer_values = {}

        for f in files:

            layer = get_layer_id(f)

            if layer is None:
                continue

            try:

                df = pd.read_csv(f)

            except Exception:

                continue

            values, col = detect_latency(
                df,
                str(f),
            )

            if values is None:
                continue

            if layer not in layer_values:
                layer_values[layer] = []

            layer_values[layer].extend(
                values.tolist()
            )

        for layer, values in (
            layer_values.items()
        ):

            values = np.asarray(
                values
            )

            values = values[
                np.isfinite(values)
            ]

            if len(values) == 0:
                continue

            rows.append({

                "method":
                    r["method"],

                "repeat":
                    int(r["repeat"]),

                "layer":
                    layer,

                "mean_ms":
                    np.mean(values),

                "p95_ms":
                    np.percentile(
                        values,
                        95,
                    ),

            })

    return pd.DataFrame(rows)


# ============================================================
# Figures
# ============================================================

def plot_slo(
    summary,
    fig_dir,
):

    if len(summary) == 0:
        return

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.9),
    )

    plot_metric(

        axes[0],

        summary,

        "deadline_ms",

        "violation_mean",

        "violation_std",

        "SLO Target (ms)",

        "E2E SLO Violation Rate (%)",

        "(a) E2E SLO Violations",

        [
            300,
            400,
            500,
            700,
        ],

    )

    plot_metric(

        axes[1],

        summary,

        "deadline_ms",

        "p95_mean",

        "p95_std",

        "SLO Target (ms)",

        "E2E P95 Latency (ms)",

        "(b) Tail Latency",

        [
            300,
            400,
            500,
            700,
        ],

    )

    axes[1].plot(
        [300, 700],
        [300, 700],

        linestyle=":",
        color="0.5",
        linewidth=1.2,
    )

    add_legend(
        fig,
        axes,
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
        "fig_slo_e2e",
    )


def plot_straggler(
    e2e,
    fig_dir,
):

    d = e2e[
        e2e["family"]
        == "strong"
    ]

    if len(d) == 0:
        return

    summary = (
        d
        .groupby(
            [
                "method",
                "gpus",
            ],
            as_index=False,
        )
        .agg(

            spread_mean=(
                "rank_spread_ms",
                "mean",
            ),

            spread_std=(
                "rank_spread_ms",
                "std",
            ),

            straggler_mean=(
                "straggler_pct",
                "mean",
            ),

            straggler_std=(
                "straggler_pct",
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

    plot_metric(

        axes[0],

        summary,

        "gpus",

        "spread_mean",

        "spread_std",

        "Number of GPUs",

        "Rank-Latency Spread (ms)",

        "(a) Cross-GPU Latency Imbalance",

        [
            1,
            2,
            4,
        ],

    )

    plot_metric(

        axes[1],

        summary,

        "gpus",

        "straggler_mean",

        "straggler_std",

        "Number of GPUs",

        "Straggler Overhead (%)",

        "(b) Straggler Overhead",

        [
            1,
            2,
            4,
        ],

    )

    add_legend(
        fig,
        axes,
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
        "fig_cross_gpu_straggler",
    )

    return summary


def plot_expert_load(
    expert,
    fig_dir,
):

    families = {

        "skew_natural":
            0,

        "skew_moderate":
            1,

        "skew_severe":
            2,

    }

    d = expert[
        expert["family"].isin(
            families.keys()
        )
    ].copy()

    if len(d) == 0:
        return

    d["x"] = (
        d["family"]
        .map(families)
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
                "max_mean_load",
                "mean",
            ),

            max_std=(
                "max_mean_load",
                "std",
            ),

            top8_mean=(
                "top8_share_pct",
                "mean",
            ),

            top8_std=(
                "top8_share_pct",
                "std",
            ),

        )
        .fillna(0)
    )

    labels = [
        "Natural",
        "Moderate",
        "Severe",
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.0, 3.0),
    )

    plot_metric(

        axes[0],

        summary,

        "x",

        "cv_mean",

        "cv_std",

        "Routing Distribution",

        "Expert Load CV",

        "(a) Load Imbalance",

        [
            0,
            1,
            2,
        ],

        labels,

    )

    plot_metric(

        axes[1],

        summary,

        "x",

        "max_mean",

        "max_std",

        "Routing Distribution",

        "Max / Mean Expert Load",

        "(b) Peak Expert Load",

        [
            0,
            1,
            2,
        ],

        labels,

    )

    plot_metric(

        axes[2],

        summary,

        "x",

        "top8_mean",

        "top8_std",

        "Routing Distribution",

        "Top-8 Expert Share (%)",

        "(c) Hot-Expert Concentration",

        [
            0,
            1,
            2,
        ],

        labels,

    )

    add_legend(
        fig,
        axes,
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
        "fig_expert_load_imbalance",
    )

    return summary


def plot_layer(
    layer,
    fig_dir,
):

    if len(layer) == 0:
        return

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

            mean_mean=(
                "mean_ms",
                "mean",
            ),

            mean_std=(
                "mean_ms",
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

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.9),
    )

    plot_metric(

        axes[0],

        summary,

        "layer",

        "mean_mean",

        "mean_std",

        "MoE Layer",

        "Mean Layer Latency (ms)",

        "(a) Layer-wise Mean Latency",

    )

    plot_metric(

        axes[1],

        summary,

        "layer",

        "p95_mean",

        "p95_std",

        "MoE Layer",

        "P95 Layer Latency (ms)",

        "(b) Layer-wise Tail Latency",

    )

    add_legend(
        fig,
        axes,
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
        "fig_layer_latency",
    )

    return summary


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
    )

    parser.add_argument(
        "--num_experts",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    root = Path(
        args.root
    )

    if not root.exists():

        raise FileNotFoundError(
            root
        )

    analysis_dir = (
        root
        /
        "analysis_trace"
    )

    fig_dir = (
        root
        /
        "figures_trace"
    )

    analysis_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Status
    # ========================================================

    status = load_status(
        root
    )

    status.to_csv(
        analysis_dir
        /
        "runs.csv",
        index=False,
    )

    # ========================================================
    # E2E
    # ========================================================

    print()
    print(
        "===== E2E latency ====="
    )

    e2e, arrays = extract_e2e(
        status
    )

    e2e.to_csv(
        analysis_dir
        /
        "e2e_metrics.csv",
        index=False,
    )

    print(
        "[INFO] E2E runs:",
        len(e2e),
    )

    # ========================================================
    # SLO
    # ========================================================

    print()
    print(
        "===== TRUE SLO violation ====="
    )

    slo_raw, slo_summary = (
        build_slo(
            e2e,
            arrays,
        )
    )

    slo_raw.to_csv(
        analysis_dir
        /
        "slo_violation_raw.csv",
        index=False,
    )

    slo_summary.to_csv(
        analysis_dir
        /
        "slo_violation_summary.csv",
        index=False,
    )

    print(
        slo_summary.to_string(
            index=False
        )
    )

    plot_slo(
        slo_summary,
        fig_dir,
    )

    # ========================================================
    # Cross GPU
    # ========================================================

    print()
    print(
        "===== Cross-GPU straggler ====="
    )

    straggler = plot_straggler(
        e2e,
        fig_dir,
    )

    if straggler is not None:

        straggler.to_csv(
            analysis_dir
            /
            "cross_gpu_straggler.csv",
            index=False,
        )

    # ========================================================
    # Expert load
    # ========================================================

    print()
    print(
        "===== Expert load ====="
    )

    expert = extract_expert_load(
        status,
        args.num_experts,
    )

    expert.to_csv(
        analysis_dir
        /
        "expert_load_runs.csv",
        index=False,
    )

    print(
        "[INFO] expert-load runs:",
        len(expert),
    )

    if len(expert):

        expert_summary = (
            plot_expert_load(
                expert,
                fig_dir,
            )
        )

        if expert_summary is not None:

            expert_summary.to_csv(
                analysis_dir
                /
                "expert_load_summary.csv",
                index=False,
            )

    else:

        print(
            "[WARN] No expert IDs/counts "
            "could be inferred from trace."
        )

    # ========================================================
    # Layer
    # ========================================================

    print()
    print(
        "===== Layer latency ====="
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

    print(
        "[INFO] layer records:",
        len(layer),
    )

    if len(layer):

        layer_summary = plot_layer(
            layer,
            fig_dir,
        )

        if layer_summary is not None:

            layer_summary.to_csv(
                analysis_dir
                /
                "layer_latency_summary.csv",
                index=False,
            )

    # ========================================================
    # Finish
    # ========================================================

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


if __name__ == "__main__":
    main()
