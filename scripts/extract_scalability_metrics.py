#!/usr/bin/env python3

import os
import re
import glob
import json

import numpy as np
import pandas as pd



ROOT = "/home/lcf/MoE_Syn/HarMoEny-main/scalability_outputs"



METHOD_MAP = {
    "deepspeed": "DeepSpeed",
    "harmony": "HarMoEny",
    "qosmoe": "DSCA-MoE"
}



records = []



# =====================================================
# Parse log
# =====================================================


logs = glob.glob(
    ROOT + "/logs/*.log"
)


print("logs:", len(logs))



for log in logs:


    fname=os.path.basename(log).lower()


    if "_r" not in fname:
        continue



    method=None


    for k,v in METHOD_MAP.items():

        if k in fname:
            method=v
            break


    if method is None:
        continue



    m=re.search(
        r"_(\d+)gpu",
        fname
    )


    if m is None:
        continue


    gpu=int(m.group(1))



    text=open(
        log,
        errors="ignore"
    ).read()



    # summary format:
    #
    # iters mean p50 p95 min max throughput


    match=re.search(

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
            "skip:",
            fname
        )

        continue



    mean_s=float(match.group(1))

    p95_s=float(match.group(3))

    throughput=float(match.group(6))



    records.append({

        "method":method,

        "gpu":gpu,

        "mean_s":mean_s,

        "p95_ms":p95_s*1000,

        "throughput":throughput,

        "source":fname

    })




df=pd.DataFrame(records)



if len(df)==0:

    raise RuntimeError(
        "No results"
    )



print(df)



# =====================================================
# Average repeat
# =====================================================


summary=(

df.groupby(
    [
        "method",
        "gpu"
    ]
)

.agg(

    {

    "mean_s":"mean",

    "p95_ms":"mean",

    "throughput":"mean"

    }

)

.reset_index()

)



# =====================================================
# Speedup
# =====================================================


summary["speedup"]=0.0

summary["efficiency"]=0.0



for method in summary.method.unique():


    base=summary[

        (summary.method==method)

        &

        (summary.gpu==1)

    ]


    if len(base)==0:
        continue



    base=base.iloc[0]["throughput"]



    idx=summary.method==method



    summary.loc[idx,"speedup"]=(

        summary.loc[idx,"throughput"]

        /

        base

    )



    summary.loc[idx,"efficiency"]=(

        summary.loc[idx,"speedup"]

        /

        summary.loc[idx,"gpu"]

    )





# =====================================================
# GPU utilization
# =====================================================


gpu_utils=[]


for exp in glob.glob(
    ROOT+"/scalability_*"
):


    name=os.path.basename(exp)


    method,gpu=None,None


    for k,v in METHOD_MAP.items():

        if k in name.lower():

            method=v



    m=re.search(
        r"_(\d+)gpu",
        name
    )


    if m:

        gpu=int(m.group(1))


    csv=os.path.join(
        exp,
        "gpu.csv"
    )


    if os.path.exists(csv):

        try:

            gdf=pd.read_csv(csv)


            for c in gdf.columns:

                if "util" in c.lower():

                    val=gdf[c].mean()


                    gpu_utils.append({

                    "method":method,

                    "gpu":gpu,

                    "gpu_util":val

                    })


        except Exception:

            pass



if gpu_utils:


    util=pd.DataFrame(gpu_utils)


    util=(

    util.groupby(
        [
            "method",
            "gpu"
        ]
    )

    .mean()

    .reset_index()

    )


    summary=summary.merge(

        util,

        on=[
            "method",
            "gpu"
        ],

        how="left"

    )

else:

    summary["gpu_util"]=np.nan




# =====================================================
# Expert imbalance CV
# =====================================================


expert_records=[]



traces=glob.glob(

    ROOT+
    "/scalability_*/real_token_trace*.csv"

)



for trace in traces:


    try:

        t=pd.read_csv(trace)


        exp_cols=[

            c for c in t.columns

            if "expert" in c.lower()

        ]


        if len(exp_cols)==0:
            continue



        counts=t[exp_cols].values.flatten()



        cv=np.std(counts)/(
            np.mean(counts)+1e-8
        )


        name=os.path.basename(
            os.path.dirname(trace)
        )


        method=None


        for k,v in METHOD_MAP.items():

            if k in name.lower():

                method=v



        m=re.search(
            r"_(\d+)gpu",
            name
        )


        if method and m:

            expert_records.append({

                "method":method,

                "gpu":int(m.group(1)),

                "expert_cv":cv

            })


    except:

        pass




if expert_records:


    expdf=pd.DataFrame(
        expert_records
    )


    expdf=(

    expdf.groupby(
        [
            "method",
            "gpu"
        ]
    )

    .mean()

    .reset_index()

    )


    summary=summary.merge(

        expdf,

        on=[
            "method",
            "gpu"
        ],

        how="left"

    )

else:

    summary["expert_cv"]=np.nan





# =====================================================
# Save
# =====================================================


out=os.path.join(

    ROOT,

    "scalability_metrics.csv"

)


summary.to_csv(

    out,

    index=False

)



print("\n========== Metrics ==========")

print(summary)


print("\nSaved:")

print(out)
