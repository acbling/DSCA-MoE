#!/usr/bin/env python3

import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt


ROOT = "/home/lcf/MoE_Syn/HarMoEny-main/scalability_outputs"

FIG_DIR = os.path.join(
    ROOT,
    "figures"
)

os.makedirs(
    FIG_DIR,
    exist_ok=True
)



METHOD_MAP = {
    "deepspeed": "DeepSpeed",
    "harmony": "HarMoEny",
    "qosmoe": "DSCA-MoE"
}



records = []



# =====================================================
# Parse log files
# =====================================================

logs = glob.glob(
    ROOT + "/logs/*.log"
)


print(
    "Found logs:",
    len(logs)
)



for log in logs:


    fname = os.path.basename(log).lower()


    # -------------------------
    # only r1/r2/r3
    # -------------------------

    if "_r" not in fname:
        continue



    # -------------------------
    # method
    # -------------------------

    method = None

    for k,v in METHOD_MAP.items():

        if k in fname:

            method = v
            break


    if method is None:

        continue



    # -------------------------
    # gpu number
    # -------------------------

    gpu_match = re.search(
        r"_(\d+)gpu",
        fname
    )


    if gpu_match is None:

        continue


    gpu = int(
        gpu_match.group(1)
    )



    # -------------------------
    # read log
    # -------------------------

    with open(
        log,
        "r",
        errors="ignore"
    ) as f:

        text = f.read()



    # =================================================
    # parse summary row
    #
    # example:
    #
    # scalability_qosmoe_4gpu_r3_qosmoe
    # 64 0.415004 0.416205 0.422117 ...
    #
    # =================================================


    match = re.search(
        r"""
        scalability_\S+
        \s+
        \d+
        \s+
        ([0-9.]+)
        \s+
        ([0-9.]+)
        \s+
        ([0-9.]+)
        \s+
        ([0-9.]+)
        \s+
        ([0-9.]+)
        \s+
        ([0-9.]+)
        """,
        text,
        re.MULTILINE | re.VERBOSE
    )



    if match is None:

        print(
            "Cannot parse:",
            fname
        )

        continue



    mean_s = float(
        match.group(1)
    )


    p50_s = float(
        match.group(2)
    )


    p95_s = float(
        match.group(3)
    )


    min_s = float(
        match.group(4)
    )


    max_s = float(
        match.group(5)
    )


    throughput = float(
        match.group(6)
    )



    records.append({

        "method": method,

        "gpu": gpu,

        "mean_s": mean_s,

        "p50_s": p50_s,

        "p95_s": p95_s,

        "min_s": min_s,

        "max_s": max_s,

        "throughput": throughput,

        "log": fname

    })





# =====================================================
# DataFrame
# =====================================================


df = pd.DataFrame(records)



if len(df)==0:

    raise RuntimeError(
        "No valid scalability results parsed"
    )



print("\n========== Raw ==========")

print(df)



# =====================================================
# Average 3 repeats
# =====================================================


summary = (

    df.groupby(
        [
            "method",
            "gpu"
        ]
    )
    [
        [
            "mean_s",
            "p50_s",
            "p95_s",
            "throughput"
        ]
    ]
    .mean()
    .reset_index()

)



print("\n========== Average ==========")

print(summary)



summary.to_csv(

    ROOT + "/scalability_summary.csv",

    index=False

)



# =====================================================
# Plot throughput
# =====================================================


plt.figure(
    figsize=(6,4)
)


for method, group in summary.groupby(
    "method"
):

    group = group.sort_values(
        "gpu"
    )

    plt.plot(

        group["gpu"],

        group["throughput"],

        marker="o",

        linewidth=2,

        label=method

    )



plt.xlabel(
    "Number of GPUs"
)


plt.ylabel(
    "Throughput (samples/s)"
)


plt.xticks(
    [1,2,4]
)


plt.grid(
    linestyle="--",
    alpha=0.3
)


plt.legend()


plt.tight_layout()


plt.savefig(

    FIG_DIR + "/scalability_throughput.pdf",

    bbox_inches="tight"

)



# =====================================================
# Plot P95 latency
# =====================================================


plt.figure(
    figsize=(6,4)
)



for method, group in summary.groupby(
    "method"
):

    group = group.sort_values(
        "gpu"
    )


    plt.plot(

        group["gpu"],

        group["p95_s"],

        marker="o",

        linewidth=2,

        label=method

    )



plt.xlabel(
    "Number of GPUs"
)


plt.ylabel(
    "P95 Latency (s)"
)


plt.xticks(
    [1,2,4]
)


plt.grid(
    linestyle="--",
    alpha=0.3
)


plt.legend()


plt.tight_layout()


plt.savefig(

    FIG_DIR + "/scalability_p95_latency.pdf",

    bbox_inches="tight"

)



print("\n================================")

print(
    "Saved:"
)

print(
    ROOT + "/scalability_summary.csv"
)

print(
    FIG_DIR
)

print(
    "================================"
)