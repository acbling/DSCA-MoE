from qosmoe_full_layer_trace import install_qosmoe_full_layer_boundary_trace
import csv
import json
import logging
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from qosmoe.config import QoSMoEConfig
from qosmoe.runtime import RequestStateTable
from qosmoe.utils import replace_qosmoe_layer, get_qosmoe_layers
from qosmoe_ablation_flags import QoSMoEAblationFlags
try:
    from real_token_gantt_trace import (
        job_start as _rt_job_start,
        now_ms as _rt_now_ms,
        record_group as _rt_record_group,
        flush as _rt_flush,
    )
except Exception:
    def _rt_now_ms():
        import time
        return time.perf_counter_ns() / 1e6
    def _rt_job_start(tokens=None, expert_mask=None, tag=""):
        return -1, _rt_now_ms()
    def _rt_record_group(*args, **kwargs):
        return
    def _rt_flush():
        return


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from args import Args
from flexible_dataset import FlexibleDataset
from stats import Stats

from qosmoe.config import QoSMoEConfig
from qosmoe.runtime import RequestStateTable
from qosmoe.utils import replace_qosmoe_layer, get_qosmoe_layers

from qosmoe_token_trace import (
    set_trace_context,
    set_iteration,
    trace_iter_start,
    trace_iter_end,
    install_moe_hooks,
)

import os
print("[TRACE_CHECK]", os.environ.get("QOSMOE_TRACE_TOKEN_TIMELINE"), os.environ.get("QOSMOE_TRACE_DIR"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

args = Args().qosmoe()


def setup(rank, timeout=timedelta(minutes=30)):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = args.port

    os.environ["HF_HOME"] = os.environ.get(
        "HF_HOME", os.path.expanduser("~/.cache/harmoeny/hf")
    )
    os.environ["HF_DATASETS_CACHE"] = os.environ.get(
        "HF_DATASETS_CACHE", os.path.expanduser("~/.cache/harmoeny/datasets")
    )
    os.environ["TRITON_HOME"] = os.environ.get(
        "TRITON_HOME", os.path.expanduser("~/.cache/harmoeny/triton")
    )

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=args.world_size,
        timeout=timeout,
    )


def generate_model(rank: int):
    ablation_flags = QoSMoEAblationFlags.from_env()

    if rank == 0:
        print(
            "[QoSMoE_Ablation]",
            f"async={ablation_flags.enable_async}",
            f"load_balance={ablation_flags.enable_load_balance}",
            f"prefetch={ablation_flags.enable_prefetch}",
            flush=True,
        )

    dtype = "auto"
    if args.model_dtype == "float16":
        dtype = torch.float16
    elif args.model_dtype == "float":
        dtype = torch.float32

    if args.loader != "transformers":
        raise ValueError(
            f"QoSMoE V2 only supports --loader transformers for now, got: {args.loader}"
        )

    model = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
    )

    config = QoSMoEConfig(
        rank=rank,
        world_size=args.world_size,
        experts=None,
        expert_example=None,
        router_weights=None,
        layer_idx=0,
        d_model=args.d_model,
        num_experts=args.num_experts,
        model_name=args.model_name,
        cache_policy=args.cache_policy,
        expert_cache_size=args.expert_cache_size,
        eq_tokens=args.eq_tokens,
        expert_placement=args.expert_placement,
        fetching_strategy=args.expert_fetching_strategy,
        enable_skew=args.enable_router_skew,
        enable_random=args.enable_router_random,
        enable_uniform=args.enable_router_uniform,
        skew=args.router_skew,
        num_experts_skewed=args.router_num_experts_skewed,
        qos_w1=(args.qos_w1 if ablation_flags.enable_load_balance else 0.0),
        qos_w2=(args.qos_w2 if ablation_flags.enable_load_balance else 0.0),
        qos_w3=(args.qos_w3 if ablation_flags.enable_load_balance else 0.0),
        # QoSMoE ablation: neutralize QoS/load-balance scoring when
        # QOSMOE_ENABLE_LOAD_BALANCE=0. This removes QoSMoE's extra
        # load/QoS-aware prioritization pressure while keeping routing valid.
        qos_alpha=args.qos_alpha,
        qos_beta=args.qos_beta,
        qos_delta=args.qos_delta,
        qos_epsilon=args.qos_epsilon,
        qos_avg_token_cost=args.qos_avg_token_cost,
        qos_default_deadline_s=args.qos_default_deadline_s,
        qos_phase_prefill=args.qos_phase_prefill,
        qos_phase_decode=args.qos_phase_decode,
        qos_target_batch=args.qos_target_batch,
        qos_attention_mode=(args.qos_attention_mode if ablation_flags.enable_load_balance else "none"),
        qos_budget_tokens_per_gpu=(args.qos_budget_tokens_per_gpu if ablation_flags.enable_load_balance else 0),
        qos_enable_prefetch=(
            bool(getattr(args, "qos_enable_prefetch", 0))
            and ablation_flags.enable_prefetch
        ),
        qos_prefetch_topk=getattr(args, "qos_prefetch_topk", 2),
        qos_prefetch_budget_experts=getattr(args, "qos_prefetch_budget_experts", 1),
        qos_prefetch_score_power=getattr(args, "qos_prefetch_score_power", 1.0),
    )

    # Ablation flags are attached after construction to avoid requiring
    # QoSMoEConfig.__init__ signature changes for new fields.
    config.qos_enable_async = ablation_flags.enable_async
    config.qos_enable_load_balance = ablation_flags.enable_load_balance
    config.qos_enable_prefetch = (
        bool(getattr(args, "qos_enable_prefetch", 0))
        and ablation_flags.enable_prefetch
    )

    replace_qosmoe_layer(
        model,
        args.type_moe_parent,
        args.type_moe,
        args.name_experts,
        args.router_tensor_path,
        config,
    )

    qos_layers = get_qosmoe_layers(model)
    for i in range(len(qos_layers) - 1):
        qos_layers[i].set_next_layer(qos_layers[i + 1])
    if len(qos_layers) > 0:
        qos_layers[-1].set_next_layer(None)

    logger.info("Finished loading QoSMoE model")
    return model


def apply_runtime_context(model, ctx):
    for layer in get_qosmoe_layers(model):
        layer.set_runtime_context(ctx)


def run_standard_experiment(model, loader, qos_state):
    latencies = []

    with torch.no_grad():
        itr = 0
        for batch in loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            ctx = qos_state.build_context(batch, phase="prefill")
            apply_runtime_context(model, ctx)
            model(**batch)

            itr += 1
            if itr == args.warmup_rounds:
                break

        run_start = time.time()

        for iter_idx, batch in enumerate(tqdm(loader)):
            set_iteration(iter_idx)

            batch = {k: v.cuda() for k, v in batch.items()}
            ctx = qos_state.build_context(batch, phase="prefill")
            apply_runtime_context(model, ctx)

            trace_iter_start(batch)

            start = time.time()
            model(**batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.time()

            trace_iter_end(batch)

            latencies.append(end - start)

        run_end = time.time()

    return latencies, run_start, run_end


def run_inference_workload(rank):
    try:
        logger.info(f"Starting process {rank}")
        setup(rank)

        os.environ["QOSMOE_TRACE_SYSTEM"] = "qosmoe"
        os.environ["QOSMOE_TRACE_RANK"] = str(rank)
        os.environ["QOSMOE_TRACE_DIR"] = args.path
        os.environ["QOSMOE_REAL_TOKEN_TRACE_DIR"] = args.path
        os.environ["QOSMOE_FULL_LAYER_TRACE_DIR"] = args.path

        os.environ["QOSMOE_TRACE_SYSTEM"] = "qosmoe"
        os.environ["QOSMOE_TRACE_RANK"] = str(rank)
        os.environ["QOSMOE_TRACE_DIR"] = args.path
        os.environ["QOSMOE_REAL_TOKEN_TRACE_DIR"] = args.path
        os.environ["QOSMOE_TRACE_TYPE_MOE"] = getattr(args, "type_moe", "SwitchTransformersSparseMLP")

        model = generate_model(rank)
        model.to(f"cuda:{rank}")
        model.eval()

        try:
            install_qosmoe_full_layer_boundary_trace(model, get_qosmoe_layers)
        except NameError:
            install_qosmoe_full_layer_boundary_trace(model, None)

        install_qosmoe_full_layer_trace_hooks(model)

        set_trace_context("qosmoe", args.path, rank)
        install_moe_hooks(model, get_qosmoe_layers, "qosmoe")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        dataset = FlexibleDataset(
            args.dataset,
            tokenizer,
            model,
            seq_len=args.seq_len,
            num_samples=args.num_samples,
            model_name=args.model_name,
        )

        sampler = DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=rank,
            shuffle=True,
            seed=49,
        )

        loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batch_size,
        )

        qos_state = RequestStateTable(args)

        path = f"{args.path}/{rank}"
        os.makedirs(path, exist_ok=True)

        latencies, run_start, run_end = run_standard_experiment(
            model, loader, qos_state
        )

        with open(f"{path}/e2e.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=["iteration", "latency (s)"])
            writer.writeheader()
            for idx, latency in enumerate(latencies):
                writer.writerow({"iteration": idx, "latency (s)": latency})

        for layer in get_qosmoe_layers(model):
            stats = layer.get_statistics()[args.warmup_rounds:]
            if len(stats) == 0:
                continue

            with open(f"{path}/moe_layer-{layer.layer_idx}.csv", "w") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["iteration"] + list(stats[0].keys())
                )
                writer.writeheader()
                for i, row in enumerate(stats):
                    writer.writerow({"iteration": i, **row})

# 原有代码：保存 data.json
        if rank == 0:
            run_info = vars(args).copy()
            run_info["system_name"] = "qosmoe"
            with open(f"{args.path}/data.json", "w") as f:
                json.dump(
                    {"start": run_start, "end": run_end, **run_info},
                    f,
                    indent=4,
                )
            
            # =================================================================
            # 新增代码：聚合微观指标 (违约率 & 负载均衡) 并输出给外层解析
            # =================================================================
            try:
                total_tokens = 0
                missed_tokens = 0
                load_stds = []
                
                for layer in get_qosmoe_layers(model):
                    stats = layer.get_statistics()[args.warmup_rounds:]
                    for row in stats:
                        # 1. 统计违约率
                        # 注意：请根据你字典中实际的 key 名称修改 ("missed_tokens", "dropped" 等)
                        t = row.get("total_tokens", row.get("num_tokens", 0))
                        m = row.get("missed_tokens", row.get("deadline_misses", row.get("dropped_tokens", 0)))
                        total_tokens += float(t)
                        missed_tokens += float(m)

                        # 2. 统计负载均衡 (计算所有专家负载的标准差)
                        # 假设字典中专家负载的 key 包含 "expert" 或 "load" 
                        # (如果你的 key 是如 "exp_0", "exp_1"，请自行调整匹配逻辑)
                        loads = [float(v) for k, v in row.items() if "expert" in str(k).lower() and "load" in str(k).lower()]
                        if not loads: # 备用匹配逻辑
                            loads = [float(v) for k, v in row.items() if str(k).startswith("exp_")]
                        
                        if loads:
                            import numpy as np
                            load_stds.append(np.std(loads))
                
                miss_rate_pct = (missed_tokens / total_tokens * 100) if total_tokens > 0 else 0.0
                avg_load_std = sum(load_stds) / len(load_stds) if load_stds else 0.0
                
                # 以特定格式打印，供外层正则捕获
                print(f"[QoS_Metrics] miss_rate_pct={miss_rate_pct:.4f} load_std={avg_load_std:.4f}")
            except Exception as e:
                print(f"[QoS_Metrics] miss_rate_pct=0.0 load_std=0.0 error={str(e)}")
            # =================================================================

        logger.info("All done :)")

    except KeyboardInterrupt:
        logger.info(f"Worker {rank} received KeyboardInterrupt, shutting down...")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def signal_handler(sig, frame):
    logger.info("Main process received Ctrl+C! Terminating all child processes...")
    for child in mp.active_children():
        logger.info(f"Terminating child process PID: {child.pid}")
        child.terminate()
    sys.exit(0)


def launch_processes_and_wait():
    processes = [None] * args.world_size

    def start_process(i):
        process = mp.Process(
            target=run_inference_workload,
            args=(i,),
        )
        process.start()
        return i, process

    mp.set_start_method("spawn", force=True)

    with ThreadPoolExecutor(max_workers=args.world_size) as executor:
        futures = [executor.submit(start_process, i) for i in range(args.world_size)]
        for fut in as_completed(futures):
            idx, process = fut.result()
            processes[idx] = process

    for process in processes:
        process.join()

    logger.info("All processes complete")



def _qosmoe_layer_trace_sync_cuda():
    try:
        import os
        import torch
        if os.environ.get("QOSMOE_TRACE_SYNC_CUDA", "0") == "1" and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _qosmoe_guess_layer_idx(name, fallback_idx):
    import re
    # Common HF SwitchTransformer names: encoder.block.0..., block.0..., layer.0...
    pats = [
        r"(?:encoder|decoder)\.block\.(\d+)",
        r"\bblock\.(\d+)",
        r"\blayers?\.(\d+)",
        r"\blayer\.(\d+)",
    ]
    for pat in pats:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return int(fallback_idx)


def _qosmoe_collect_moe_layers_for_trace(model):
    import os

    type_moe = os.environ.get("QOSMOE_TRACE_TYPE_MOE", "SwitchTransformersSparseMLP")
    candidates = []

    for name, module in model.named_modules():
        cls = module.__class__.__name__

        # Conservative matching:
        # 1) original Switch sparse MLP
        # 2) replaced QoSMoE wrapper/module
        # 3) module with router + experts
        is_moe = (
            cls == type_moe
            or "SparseMLP" in cls
            or "QoSMoE" in cls
            or (hasattr(module, "router") and hasattr(module, "experts"))
        )

        if not is_moe:
            continue

        # Avoid tiny submodules that are not real MoE forward units.
        if not callable(getattr(module, "forward", None)):
            continue

        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None:
            layer_idx = _qosmoe_guess_layer_idx(name, len(candidates))

        candidates.append((int(layer_idx), name, module))

    # Deduplicate by layer id first, then module id.
    # Prefer modules whose name looks closer to a full sparse MLP / QoSMoE unit.
    selected = {}
    for layer_idx, name, module in candidates:
        score = 0
        cls = module.__class__.__name__
        if "QoSMoE" in cls:
            score += 3
        if "SparseMLP" in cls:
            score += 2
        if hasattr(module, "router") and hasattr(module, "experts"):
            score += 2
        if "router" in name.lower():
            score -= 5
        if "expert" in name.lower() and "SparseMLP" not in cls and "QoSMoE" not in cls:
            score -= 5

        old = selected.get(layer_idx)
        if old is None or score > old[0]:
            selected[layer_idx] = (score, name, module)

    layers = [(idx, item[1], item[2]) for idx, item in sorted(selected.items(), key=lambda x: x[0])]
    return layers


def _qosmoe_build_fallback_groups(module, x):
    import os
    import torch

    max_tokens = int(os.environ.get("QOSMOE_TRACE_MAX_TOKENS", "64"))

    try:
        flat = x.reshape(-1, x.shape[-1])
    except Exception:
        flat = x

    try:
        n = min(int(flat.shape[0]), max_tokens)
    except Exception:
        n = max_tokens

    # Only keep small prefix tokens for trace.
    tokens = flat[:n].detach() if hasattr(flat, "detach") else flat[:n]

    expert_index = None

    # Try router-based grouping, but never depend on it.
    try:
        router = getattr(module, "router", None)
        if router is not None and hasattr(tokens, "shape"):
            with torch.no_grad():
                out = router(tokens.float() if hasattr(tokens, "float") else tokens)

            if isinstance(out, tuple):
                tensors = [z for z in out if torch.is_tensor(z)]
                for z in tensors:
                    if z.shape[0] == tokens.shape[0]:
                        if z.dim() >= 2:
                            expert_index = torch.argmax(z, dim=-1).long()
                            break
                        if z.dim() == 1:
                            expert_index = z.long()
                            break
            elif torch.is_tensor(out):
                if out.dim() >= 2:
                    expert_index = torch.argmax(out, dim=-1).long()
                elif out.dim() == 1:
                    expert_index = out.long()
    except Exception:
        expert_index = None

    if expert_index is None:
        try:
            expert_index = torch.zeros(n, dtype=torch.long, device=tokens.device)
        except Exception:
            expert_index = torch.zeros(n, dtype=torch.long)

    groups = []
    try:
        for e in torch.unique(expert_index).detach().cpu().tolist():
            idxs = torch.nonzero(expert_index == int(e), as_tuple=False).reshape(-1)
            if idxs.numel() > 0:
                groups.append((int(e), idxs))
    except Exception:
        idxs = torch.arange(n)
        groups = [(0, idxs)]

    return tokens, groups


def install_qosmoe_full_layer_trace_hooks(model):
    import os

    if os.environ.get("QOSMOE_LAYER_FALLBACK_TRACE", "0") != "1":
        print("[TRACE] QoSMoE layer fallback trace disabled")
        return

    layers = _qosmoe_collect_moe_layers_for_trace(model)
    print(f"[TRACE] QoSMoE layer-fallback hooks installed={len(layers)}")
    for idx, name, _ in layers:
        print(f"[TRACE] QoSMoE fallback layer idx={idx}, name={name}")

    for layer_idx, name, layer in layers:
        def pre_hook(module, inputs, layer_idx=layer_idx, name=name):
            if not inputs:
                return
            x = inputs[0]

            _qosmoe_layer_trace_sync_cuda()
            rt_ready_ms = _rt_now_ms()

            rt_tokens, rt_groups = _qosmoe_build_fallback_groups(module, x)
            rt_job_id, rt_job_ready_ms = _rt_job_start(
                rt_tokens,
                None,
                tag="qosmoe.layer_fallback",
            )

            # Use the hook ready time as the fallback ready time.
            module._qosmoe_layer_fallback_ctx = (
                rt_tokens,
                rt_groups,
                rt_job_id,
                rt_ready_ms,
                rt_ready_ms,
                int(layer_idx),
            )

        def post_hook(module, inputs, output, layer_idx=layer_idx, name=name):
            ctx = getattr(module, "_qosmoe_layer_fallback_ctx", None)
            if ctx is None:
                return

            _qosmoe_layer_trace_sync_cuda()
            rt_end_ms = _rt_now_ms()

            rt_tokens, rt_groups, rt_job_id, rt_ready_ms, rt_start_ms, layer_idx = ctx

            for expert_idx, token_idx in rt_groups:
                _rt_record_group(
                    rt_job_id,
                    rt_ready_ms,
                    expert_idx,
                    token_idx,
                    rt_start_ms,
                    rt_end_ms,
                    tokens=rt_tokens,
                    note="qosmoe.layer_fallback",
                    layer_idx=layer_idx,
                )

            _rt_flush()

        layer.register_forward_pre_hook(pre_hook)
        layer.register_forward_hook(post_hook)




def _qosmoe_boundary_build_groups(module, x):
    import torch
    max_tokens = int(os.environ.get("QOSMOE_TRACE_MAX_TOKENS", "64"))

    try:
        flat = x.reshape(-1, x.shape[-1])
    except Exception:
        flat = x

    n = min(int(flat.shape[0]), max_tokens)
    token_idx = torch.arange(n, device=flat.device)

    expert_index = None
    try:
        router = getattr(module, "router", None)
        if router is not None:
            with torch.no_grad():
                out = router(flat[:n].float())
            if isinstance(out, tuple):
                for item in out:
                    if torch.is_tensor(item) and item.shape[0] == n:
                        if item.dim() >= 2:
                            expert_index = torch.argmax(item, dim=-1).long()
                            break
                        elif item.dim() == 1:
                            expert_index = item.long()
                            break
            elif torch.is_tensor(out):
                if out.dim() >= 2:
                    expert_index = torch.argmax(out, dim=-1).long()
                elif out.dim() == 1:
                    expert_index = out.long()
    except Exception:
        expert_index = None

    if expert_index is None:
        expert_index = torch.zeros(n, dtype=torch.long, device=flat.device)

    groups = []
    for e in torch.unique(expert_index).detach().cpu().tolist():
        idxs = torch.nonzero(expert_index == int(e), as_tuple=False).reshape(-1)
        if idxs.numel() > 0:
            groups.append((int(e), idxs))

    return flat[:n], groups


def _qosmoe_collect_moe_layers_for_boundary(model):
    layers = []
    for name, module in model.named_modules():
        cls = module.__class__.__name__
        is_moe = (
            "QoSMoE" in cls
            or "SparseMLP" in cls
            or (hasattr(module, "router") and hasattr(module, "experts"))
        )
        if not is_moe:
            continue
        if "router" in name.lower():
            continue
        if "expert" in name.lower() and "SparseMLP" not in cls and "QoSMoE" not in cls:
            continue
        if not callable(getattr(module, "forward", None)):
            continue

        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None:
            m = re.search(r"(?:block|layer|layers)\.(\d+)", name)
            layer_idx = int(m.group(1)) if m else len(layers)

        layers.append((int(layer_idx), name, module))

    # dedup
    out = {}
    for idx, name, module in layers:
        out[idx] = (name, module)

    return [(idx, name, module) for idx, (name, module) in sorted(out.items())]


def install_qosmoe_layer_boundary_trace(model):
    if os.environ.get("QOSMOE_LAYER_BOUNDARY_TRACE", "0") != "1":
        print("[TRACE] QoSMoE layer-boundary trace disabled")
        return

    layers = _qosmoe_collect_moe_layers_for_boundary(model)
    print(f"[TRACE] QoSMoE layer-boundary hooks installed={len(layers)}")
    for idx, name, _ in layers:
        print(f"[TRACE] QoSMoE boundary layer={idx}, name={name}")

    for layer_idx, name, layer in layers:
        def pre_hook(module, inputs, layer_idx=layer_idx):
            if not inputs:
                return
            x = inputs[0]
            rt_tokens, rt_groups = _qosmoe_boundary_build_groups(module, x)
            rt_job_id, rt_ready_ms = _rt_job_start(rt_tokens, None, tag="qosmoe.layer_boundary")
            rt_start_ms = _rt_now_ms()
            module._qosmoe_layer_boundary_ctx = (
                rt_tokens, rt_groups, rt_job_id, rt_ready_ms, rt_start_ms, int(layer_idx)
            )

        def post_hook(module, inputs, output, layer_idx=layer_idx):
            ctx = getattr(module, "_qosmoe_layer_boundary_ctx", None)
            if ctx is None:
                return

            rt_tokens, rt_groups, rt_job_id, rt_ready_ms, rt_start_ms, layer_idx = ctx
            rt_end_ms = _rt_now_ms()

            for expert_idx, token_idx in rt_groups:
                _rt_record_group(
                    rt_job_id,
                    rt_ready_ms,
                    expert_idx,
                    token_idx,
                    rt_start_ms,
                    rt_end_ms,
                    tokens=rt_tokens,
                    note="qosmoe.layer.hook_sync_batch",
                    layer_idx=layer_idx,
                )

            _rt_flush()

        layer.register_forward_pre_hook(pre_hook)
        layer.register_forward_hook(post_hook)



if __name__ == "__main__":
    logger.info("Starting Main Process")

    signal.signal(signal.SIGINT, signal_handler)

    stats = Stats(gpu=True, cpu=True, num_gpus=args.world_size)
    stats.start()

    launch_processes_and_wait()

    stats.stop()