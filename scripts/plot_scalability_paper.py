#!/usr/bin/env python3

import os
import json
import glob
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


ROOT = "/home/lcf/MoE_Syn/HarMoEny-main/scalability_outputs"

OUT_DIR = os.path.join(ROOT, "figures")

os.makedirs(OUT_DIR, exist_ok=True)


# ===============================
# style
# ===============================

METHOD_STYLE = {
    "DSCA-MoE": {
        "color": "#1f77b4",
        "marker": "o"
    },
    "HarMoEny": {
        "color": "#2ca02c",
        "marker": "s"
    },
    "DeepSpeed": {
        "color": "#ff7f0e",
        "marker": "^"
    }
}



# ===============================
# parse
# ===============================

def parse_method_gpu(name):

    name=name.lower()

    if "qosmoe" in name:
        method="DSCA-MoE"

    elif "harmony" in name:
        method="HarMoEny"

    elif "deepspeed" in name:
        method="DeepSpeed"

    else:
        return None,None


    m=re.search(r"(\d+)gpu",name)

    if m:
        gpu=int(m.group(1))
    else:
        gpu=None


    return method,gpu



records=[]


dirs=glob.glob(
    os.path.join(ROOT,"scalability_*")
)


print("Found:",len(dirs))


for d in dirs:

    name=os.path.basename(d)

    method,gpu=parse_method_gpu(name)

    if method is None:
        continue


    data={}

    # -----------------------
    # data.json
    # -----------------------

    json_file=os.path.join(d,"data.json")

    if os.path.exists(json_file):

        try:
            with open(json_file) as f:
                js=json.load(f)

            if isinstance(js,dict):

                for k,v in js.items():

                    if isinstance(v,(int,float)):
                        data[k]=v

        except:
            pass


    # -----------------------
    # gpu.csv
    # -----------------------

    gpu_file=os.path.join(d,"gpu.csv")

    gpu_util=[]

    if os.path.exists(gpu_file):

        try:

            df=pd.read_csv(gpu_file)

            for c in df.columns:

                if "util" in c.lower():

                    gpu_util.extend(
                        df[c].dropna().tolist()
                    )


        except:
            pass


    if gpu_util:

        data["gpu_util"] = np.mean(gpu_util)



    # -----------------------
    # log summary
    # -----------------------

    logs=glob.glob(
        os.path.join(d,"*.log")
    )


    mean_s=None
    p95_s=None
    throughput=None


    for log in logs:

        try:

            text=open(log).read()


            m=re.search(
                r"mean_s\s+([0-9.]+)",
                text
            )

            if m:
                mean_s=float(m.group(1))


            m=re.search(
                r"p95_s\s+([0-9.]+)",
                text
            )

            if m:
                p95_s=float(m.group(1))


            m=re.search(
                r"samples_per_s\s+([0-9.]+)",
                text
            )

            if m:
                throughput=float(m.group(1))


        except:
            pass



    if throughput is None:
        continue


    records.append({

        "method":method,
        "gpu":gpu,
        "throughput":throughput,
        "latency":p95_s,
        "gpu_util":data.get("gpu_util",np.nan)

    })


df=pd.DataFrame(records)


if len(df)==0:
    raise RuntimeError(
        "No valid results found"
    )


print(df)


# ===============================
# average repeat
# ===============================

df=df.groupby(
    [
        "method",
        "gpu"
    ],
    as_index=False
).mean()



# save

csv_out=os.path.join(
    ROOT,
    "scalability_summary_paper.csv"
)

df.to_csv(csv_out,index=False)


print("Saved:",csv_out)



# ===============================
# derive metrics
# ===============================


df["speedup"]=0.0
df["efficiency"]=0.0


for m in df.method.unique():

    base=df[
        (df.method==m)
        &
        (df.gpu==1)
    ]["throughput"].values


    if len(base)==0:
        continue


    base=base[0]


    idx=df.method==m

    df.loc[idx,"speedup"] = (
        df.loc[idx,"throughput"]
        /
        base
    )

    df.loc[idx,"efficiency"] = (
        df.loc[idx,"speedup"]
        /
        df.loc[idx,"gpu"]
    )




# ===============================
# plotting
# ===============================


metrics=[

    (
        "throughput",
        "Throughput\n(samples/s)",
        True
    ),

    (
        "latency",
        "P95 Latency (ms)",
        False
    ),

    (
        "speedup",
        "Speedup",
        True
    ),

    (
        "efficiency",
        "Parallel Efficiency",
        True
    ),

    (
        "gpu_util",
        "GPU Utilization (%)",
        True
    )

]


fig,axs=plt.subplots(
    2,
    3,
    figsize=(10,6)
)


axs=axs.flatten()



for ax,(metric,title,up) in zip(
        axs,
        metrics
):


    for method in [
        "DSCA-MoE",
        "HarMoEny",
        "DeepSpeed"
    ]:


        sub=df[
            df.method==method
        ].sort_values("gpu")


        if metric not in sub:
            continue


        if sub[metric].isna().all():
            continue


        y=sub[metric]


        if metric=="latency":

            y=y*1000


        ax.plot(

            sub.gpu,
            y,

            linewidth=2,

            marker=
            METHOD_STYLE[method]["marker"],

            markersize=6,

            label=method

        )


    ax.set_xlabel(
        "Number of GPUs"
    )

    ax.set_ylabel(title)

    ax.grid(
        linestyle="--",
        alpha=0.35
    )



axs[0].legend(
    fontsize=8
)



for ax in axs:
    ax.tick_params(
        labelsize=9
    )


plt.tight_layout()


pdf=os.path.join(
    OUT_DIR,
    "fig_scalability.pdf"
)

png=os.path.join(
    OUT_DIR,
    "fig_scalability.png"
)


plt.savefig(
    pdf,
    bbox_inches="tight"
)

plt.savefig(
    png,
    dpi=300,
    bbox_inches="tight"
)


print("Saved:")
print(pdf)
print(png)
