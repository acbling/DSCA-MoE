#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def build_common_flags(args):
    return [
        "--dataset", args.dataset,
        "--num_samples", str(args.num_samples),
        "--batch_size", str(args.batch_size),
        "--seq_len", str(args.seq_len),
        "--model_name", args.model_name,
        "--d_model", str(args.d_model),
        "--num_experts", str(args.num_experts),
        "--type_moe_parent", args.type_moe_parent,
        "--type_moe", args.type_moe,
        "--router_tensor_path", args.router_tensor_path,
        "--name_experts", args.name_experts,
        "--world_size", str(args.world_size),
        "--warmup_rounds", str(args.warmup_rounds),
        "--enable_router_skew", str(args.enable_router_skew),
        "--enable_router_random", str(args.enable_router_random),
        "--enable_router_uniform", str(args.enable_router_uniform),
        "--router_skew", str(args.router_skew),
        "--router_num_experts_skewed", str(args.router_num_experts_skewed),
        "--random_router_skew", str(args.random_router_skew),
    ]


def build_env(args):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{args.project_root / 'src'}:" + env.get("PYTHONPATH", "")
    env["PYTHONNOUSERSITE"] = "1"
    env["HF_HOME"] = env.get("HF_HOME", str(Path.home() / ".cache/harmoeny/hf"))
    env["HF_DATASETS_CACHE"] = env.get(
        "HF_DATASETS_CACHE", str(Path.home() / ".cache/harmoeny/datasets")
    )
    env["TRITON_HOME"] = env.get("TRITON_HOME", str(Path.home() / ".cache/harmoeny/triton"))
    env["HF_HUB_OFFLINE"] = env.get("HF_HUB_OFFLINE", "1")
    env["TRANSFORMERS_OFFLINE"] = env.get("TRANSFORMERS_OFFLINE", "1")
    env["HF_DATASETS_OFFLINE"] = env.get("HF_DATASETS_OFFLINE", "1")
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    return env


def run_cmd(cmd, cwd, env):
    print("\n>>> Running:")
    print(" ".join(cmd))
    sys.stdout.flush()
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def run_qosmoe(args, env, port, out_dir):
    cmd = [
        sys.executable, "src/start_qosmoe.py",
        *build_common_flags(args),
        "--loader", args.loader,
        "--expert_cache_size", str(args.expert_cache_size),
        "--eq_tokens", str(args.eq_tokens),
        "--port", str(port),
        "--path", str(out_dir),
        "--qos_w1", str(args.qos_w1),
        "--qos_w2", str(args.qos_w2),
        "--qos_w3", str(args.qos_w3),
        "--qos_alpha", str(args.qos_alpha),
        "--qos_beta", str(args.qos_beta),
        "--qos_delta", str(args.qos_delta),
        "--qos_enable_prefetch", str(args.qos_enable_prefetch),
        "--qos_prefetch_topk", str(args.qos_prefetch_topk),
        "--qos_prefetch_budget_experts", str(args.qos_prefetch_budget_experts),
        "--qos_prefetch_score_power", str(args.qos_prefetch_score_power),
        "--qos_target_batch", str(args.qos_target_batch),
        "--qos_default_deadline_s", str(args.qos_default_deadline_s),
        "--qos_avg_token_cost", str(args.qos_avg_token_cost),
        "--qos_attention_mode", args.qos_attention_mode,
    ]
    run_cmd(cmd, args.project_root, env)


def run_qosmoe_impl(args, env, port, out_dir):
    cmd = [
        sys.executable, "src/start_qosmoe_impl.py",
        *build_common_flags(args),
        "--loader", args.loader,
        "--expert_cache_size", str(args.expert_cache_size),
        "--eq_tokens", str(args.eq_tokens),
        "--expert_fetching_strategy", args.expert_fetching_strategy,
        "--port", str(port),
        "--path", str(out_dir),
        "--qos_w1", str(args.qos_w1),
        "--qos_w2", str(args.qos_w2),
        "--qos_w3", str(args.qos_w3),
        "--qos_alpha", str(args.qos_alpha),
        "--qos_beta", str(args.qos_beta),
        "--qos_delta", str(args.qos_delta),
        "--qos_enable_prefetch", str(args.qos_enable_prefetch),
        "--qos_prefetch_topk", str(args.qos_prefetch_topk),
        "--qos_prefetch_budget_experts", str(args.qos_prefetch_budget_experts),
        "--qos_prefetch_score_power", str(args.qos_prefetch_score_power),
        "--qos_target_batch", str(args.qos_target_batch),
        "--qos_default_deadline_s", str(args.qos_default_deadline_s),
        "--qos_avg_token_cost", str(args.qos_avg_token_cost),
        "--qos_attention_mode", args.qos_attention_mode,
        # Keep impl runner compatible with existing Args().qosmoe() parser.
        # Only pass flags that parser already understands.
    ]
    run_cmd(cmd, args.project_root, env)


def run_harmony(args, env, port, out_dir):
    cmd = [
        sys.executable, "src/start_harmony.py",
        *build_common_flags(args),
        "--loader", args.loader,
        "--scheduling_policy", "harmony",
        "--expert_cache_size", str(args.expert_cache_size),
        "--eq_tokens", str(args.eq_tokens),
        "--expert_fetching_strategy", args.expert_fetching_strategy,
        "--port", str(port),
        "--path", str(out_dir),
    ]
    run_cmd(cmd, args.project_root, env)


def run_deepspeed(args, env, port, out_dir):
    cmd = [
        "deepspeed", "--master_port", str(port), "src/start_deepspeed.py",
        *build_common_flags(args),
        "--capacity_factor", str(args.capacity_factor),
        "--path", str(out_dir),
    ]
    run_cmd(cmd, args.project_root, env)


def run_fastmoe(args, env, port, out_dir):
    cmd = [
        sys.executable, "src/start_fastmoe.py",
        *build_common_flags(args),
        "--system_name", "fastmoe",
        "--port", str(port),
        "--path", str(out_dir),
    ]
    run_cmd(cmd, args.project_root, env)


def summarize(outputs_root, systems, batch_size):
    rows = []
    for system in systems:
        path = outputs_root / system / "0" / "e2e.csv"
        if not path.exists():
            rows.append({"system": system, "status": "missing"})
            continue

        df = pd.read_csv(path)
        col = "latency (s)"
        if len(df) == 0:
            rows.append({"system": system, "status": "empty"})
            continue

        rows.append({
            "system": system,
            "iters": len(df),
            "mean_s": df[col].mean(),
            "p50_s": df[col].quantile(0.50),
            "p95_s": df[col].quantile(0.95),
            "min_s": df[col].min(),
            "max_s": df[col].max(),
            "samples_per_s": batch_size / df[col].mean(),
            "status": "ok",
        })

    out = pd.DataFrame(rows)
    if "mean_s" in out.columns:
        out = out.sort_values(["status", "mean_s"], ascending=[True, True])
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Run QoSMoE-Impl / QoSMoE / Harmony / DeepSpeed / FastMoE benchmarks sequentially."
    )
    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--outputs_root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs",
    )
    parser.add_argument(
        "--systems",
        type=str,
        default="qosmoe_impl,qosmoe,harmony,deepspeed,fastmoe",
        help="Comma-separated: qosmoe_impl,qosmoe,harmony,deepspeed,fastmoe",
    )
    parser.add_argument("--cuda_visible_devices", type=str, default="2")
    parser.add_argument("--dataset", type=str, default="random")
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=120)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--loader", type=str, default="transformers")
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_experts", type=int, default=64)
    parser.add_argument("--type_moe_parent", type=str, default="SwitchTransformersLayerFF")
    parser.add_argument("--type_moe", type=str, default="SwitchTransformersSparseMLP")
    parser.add_argument("--router_tensor_path", type=str, default="router.classifier")
    parser.add_argument("--name_experts", type=str, default="experts")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--warmup_rounds", type=int, default=3)
    parser.add_argument("--expert_cache_size", type=int, default=2)
    parser.add_argument("--eq_tokens", type=int, default=150)
    parser.add_argument("--expert_fetching_strategy", type=str, default="async-cpu")
    parser.add_argument("--capacity_factor", type=float, default=10.0)

    parser.add_argument("--qos_w1", type=float, default=1.0)
    parser.add_argument("--qos_w2", type=float, default=0.1)
    parser.add_argument("--qos_w3", type=float, default=0.5)
    parser.add_argument("--qos_alpha", type=float, default=1.0)
    parser.add_argument("--qos_beta", type=float, default=0.1)
    parser.add_argument("--qos_delta", type=float, default=0.5)
    parser.add_argument("--qos_enable_prefetch", type=int, default=0)
    parser.add_argument("--qos_prefetch_topk", type=int, default=2)
    parser.add_argument("--qos_prefetch_budget_experts", type=int, default=1)
    parser.add_argument("--qos_prefetch_score_power", type=float, default=1.0)
    parser.add_argument("--qos_target_batch", type=int, default=64)
    parser.add_argument("--qos_default_deadline_s", default=2.0, type=float)
    parser.add_argument("--qos_avg_token_cost", type=float, default=0.005)
    parser.add_argument("--qos_attention_mode", type=str, default="router_var")

    parser.add_argument("--enable_router_skew", type=int, default=0)
    parser.add_argument("--enable_router_random", type=int, default=0)
    parser.add_argument("--enable_router_uniform", type=int, default=0)
    parser.add_argument("--router_skew", type=float, default=0.7)
    parser.add_argument("--router_num_experts_skewed", type=int, default=4)
    parser.add_argument("--random_router_skew", type=float, default=0.0)

    parser.add_argument("--base_port", type=int, default=29631)
    parser.add_argument("--tag", type=str, default="switch64_run")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    env = build_env(args)

    port = args.base_port
    for system in systems:
        out_dir = args.outputs_root / f"{args.tag}_{system}"
        if args.clean and out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if system == "qosmoe_impl":
            run_qosmoe_impl(args, env, port, out_dir)
            port += 1
        elif system == "qosmoe":
            run_qosmoe(args, env, port, out_dir)
            port += 1
        elif system == "harmony":
            run_harmony(args, env, port, out_dir)
            port += 1
        elif system == "deepspeed":
            run_deepspeed(args, env, port, out_dir)
            port += 1
        elif system == "fastmoe":
            run_fastmoe(args, env, port, out_dir)
            port += 1
        else:
            raise ValueError(f"Unknown system: {system}")

    df = summarize(args.outputs_root, [f"{args.tag}_{s}" for s in systems], args.batch_size)
    print("\n===== Summary =====")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
