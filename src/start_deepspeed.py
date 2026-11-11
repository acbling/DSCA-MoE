# https://proceedings.mlr.press/v162/rajbhandari22a/rajbhandari22a.pdf
# https://github.com/microsoft/DeepSpeed

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from real_token_gantt_trace import (
        job_start as _rt_job_start,
        now_ms as _rt_now_ms,
        record_group as _rt_record_group,
    )
except Exception:
    def _rt_now_ms():
        import time
        return time.perf_counter_ns() / 1e6
    def _rt_job_start(tokens=None, expert_mask=None, tag=""):
        return -1, _rt_now_ms()
    def _rt_record_group(*args, **kwargs):
        return
from torch import Tensor
import torch.distributed as dist
import torch.multiprocessing as mp 
import sys
import os 
import time
import csv
import json
from tqdm import tqdm

from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel

from flexible_dataset import FlexibleDataset
from stats import Stats

from utils import TimedModule, get_timing_modules

from harmonymoe.router import Router, RouterConfig
from args import Args

import deepspeed
from deepspeed.utils import groups
from deepspeed.moe.sharded_moe import top2gating
try:
    from deepspeed.moe.sharded_moe import topkgating
except ImportError:
    def topkgating(*args, **kwargs):
        raise NotImplementedError("topkgating is not available in this deepspeed version")

from deepspeed.utils.timer import SynchronizedWallClockTimer
from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple, Union

try:
    # To enable Tutel MoE optimizations:
    #   python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.1.x
    from tutel import moe as tutel_moe
    TUTEL_INSTALLED = True
except:
    # Fail silently so we don't spam logs unnecessarily if user isn't using tutel
    TUTEL_INSTALLED = False
    pass

args = Args().deepspeed()


try:
    from qosmoe_token_trace import (
    trace_event as _ds_trace_event,
    set_trace_context as _ds_set_trace_context,
    set_iteration as _ds_set_iteration,
    trace_iter_start as _ds_trace_iter_start,
    trace_iter_end as _ds_trace_iter_end,
)
except Exception:
    _ds_trace_event = None


def _ds_trace_token_ids(num_tokens):
    try:
        max_tokens = int(os.environ.get("QOSMOE_TRACE_MAX_TOKENS", "32"))
    except Exception:
        max_tokens = 32
    return list(range(min(int(num_tokens), max_tokens)))


def _ds_trace(event_name, token_ids, layer_idx=0, batch_id=0, extra=None):
    if _ds_trace_event is None:
        return
    try:
        _ds_trace_event(
            event_name,
            token_ids=token_ids,
            layer=int(layer_idx) if layer_idx is not None else 0,
            batch_id=int(batch_id),
            extra=extra or {},
        )
    except Exception:
        pass


def _ds_build_expert_groups(module, x):
    try:
        flat = x.reshape(-1, x.shape[-1]).float()
        gate = module.child.deepspeed_moe.gate

        if hasattr(gate, "router"):
            router_mask, router_probs, router_logits = gate.router(flat)
            expert_index = torch.argmax(router_mask, dim=-1)
        elif hasattr(gate, "wg"):
            logits = torch.nn.functional.linear(flat, weight=gate.wg.weight.float(), bias=None)
            expert_index = torch.argmax(logits, dim=-1)
        else:
            expert_index = torch.zeros(flat.shape[0], dtype=torch.long, device=flat.device)

        groups = []
        max_tokens = int(os.environ.get("QOSMOE_TRACE_MAX_TOKENS", "32"))
        for expert_idx in torch.unique(expert_index).detach().cpu().tolist():
            idxs = torch.nonzero(expert_index == int(expert_idx), as_tuple=False).reshape(-1)
            idxs = idxs[idxs < max_tokens]
            if idxs.numel() > 0:
                groups.append((int(expert_idx), idxs))
        return flat, groups
    except Exception:
        flat = x.reshape(-1, x.shape[-1])
        max_tokens = int(os.environ.get("QOSMOE_TRACE_MAX_TOKENS", "32"))
        idxs = torch.arange(min(flat.shape[0], max_tokens), device=flat.device)
        return flat, [(0, idxs)]


def setup():
    os.environ["HF_HOME"] = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/harmoeny/hf"))
    os.environ["HF_DATASETS_CACHE"] = os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/harmoeny/datasets"))
    os.environ["TRITON_HOME"] = os.environ.get("TRITON_HOME", os.path.expanduser("~/.cache/harmoeny/triton"))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    deepspeed.init_distributed()

    torch.cuda.set_device(args.local_rank)

def run_inference_workload():
    setup()
    _ds_set_trace_context("deepspeed", args.path, args.local_rank)

    # [QWEN_DS_ADAPTER_V1]
    # Qwen1.5-MoE is substantially larger than Switch-base-64.
    # Keep Switch on the existing path, but load Qwen in FP16.
    is_qwen_moe = (
        str(getattr(args, "type_moe", "")) == "Qwen2MoeSparseMoeBlock"
        or "qwen" in str(args.model_name).lower()
    )
    model_load_dtype = torch.float16 if is_qwen_moe else None

    if is_qwen_moe:
        model = AutoModel.from_pretrained(
            args.model_name,
            cache_dir=os.environ["HF_HOME"],
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModel.from_pretrained(
            args.model_name,
            cache_dir=os.environ["HF_HOME"],
        )

    if args.local_rank == 0:
        print(
            f"[DeepSpeedAdapter] model_type={getattr(model.config, 'model_type', None)} "
            f"dtype={'float16' if is_qwen_moe else 'original'} "
            f"num_experts={getattr(model.config, 'num_experts', args.num_experts)} "
            f"top_k={getattr(model.config, 'num_experts_per_tok', 1)}",
            flush=True,
        )

    class MLPWrapper(nn.Module):
        def __init__(self, child, layer_idx=0):
            super().__init__()
            self.child = child
            self.layer_idx = layer_idx
        
        def forward(self, x):
            try:
                _num_trace_tokens = x.reshape(-1, x.shape[-1]).shape[0]
            except Exception:
                _num_trace_tokens = x.shape[0]
            token_ids = _ds_trace_token_ids(_num_trace_tokens)
            _ds_trace("attn_end", token_ids, self.layer_idx, batch_id=0)
            _ds_trace("moe_ready_time", token_ids, self.layer_idx, batch_id=0)

            # Fallback launch marker. TopKGate.forward will overwrite this
            # with a more meaningful timestamp after gating/dispatch-mask creation.
            _ds_trace("moe_launch_time", token_ids, self.layer_idx, batch_id=0)

            rt_tokens, rt_groups = _ds_build_expert_groups(self, x)
            rt_job_id, rt_ready_ms = _rt_job_start(rt_tokens, None, tag="deepspeed.MLPWrapper")
            rt_start_ms = _rt_now_ms()

            x = self.child(x)

            rt_end_ms = _rt_now_ms()

            for _expert_idx, _token_idx in rt_groups:
                _rt_record_group(
                    rt_job_id,
                    rt_ready_ms,
                    _expert_idx,
                    _token_idx,
                    rt_start_ms,
                    rt_end_ms,
                    tokens=rt_tokens,
                    note="deepspeed.MLPWrapper.sync_batch",
                    layer_idx=self.layer_idx,
                )

            _ds_trace("expert_compute_end", token_ids, self.layer_idx, batch_id=0)
            _ds_trace("combine_start", token_ids, self.layer_idx, batch_id=0)
            _ds_trace("combine_end", token_ids, self.layer_idx, batch_id=0)

            return x[0], (x[1], x[2])

    def _qwen_dense_topkgating(
        logits: torch.Tensor,
        k: int,
        ep_group=None,
        norm_topk_prob: bool = False,
    ):
        """Top-k routing compatible with DeepSpeed MOELayer's dense dispatch API.

        Qwen1.5-MoE-A2.7B uses top-4 routing. DeepSpeed 0.14.x installations
        are not uniform about exposing topkgating(), so keep the implementation
        local and explicit.
        """
        gates = F.softmax(logits, dim=1, dtype=torch.float32)
        num_tokens, num_experts = gates.shape
        k = int(min(max(1, k), num_experts))

        topk_weights, topk_idx = torch.topk(gates, k=k, dim=1)

        if norm_topk_prob:
            denom = topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
            topk_weights = topk_weights / denom

        mask = torch.zeros(
            (num_tokens, num_experts),
            dtype=torch.bool,
            device=logits.device,
        )
        mask.scatter_(1, topk_idx, True)

        exp_counts = mask.sum(dim=0).detach()

        # Infinite-capacity behavior: no token dropping.
        capacity_t = exp_counts.max().to(dtype=torch.int64)
        if ep_group is not None and dist.get_world_size(group=ep_group) > 1:
            # Scalar MAX only; use the shared/default NCCL communicator supplied
            # above instead of allocating a fresh per-layer communicator.
            dist.all_reduce(capacity_t, op=dist.ReduceOp.MAX, group=ep_group)
        capacity = max(1, int(capacity_t.item()))

        locations = torch.cumsum(mask.to(torch.int32), dim=0) - 1
        route_locations = locations.gather(1, topk_idx).to(torch.long)

        # Build [S,E,C] directly to avoid a much larger transient one-hot tensor.
        combine_weights = torch.zeros(
            (num_tokens, num_experts, capacity),
            dtype=gates.dtype,
            device=logits.device,
        )

        tok = (
            torch.arange(num_tokens, device=logits.device)
            .unsqueeze(1)
            .expand(num_tokens, k)
            .reshape(-1)
        )
        exp = topk_idx.reshape(-1)
        loc = route_locations.reshape(-1)
        val = topk_weights.reshape(-1)

        combine_weights[tok, exp, loc] = val
        dispatch_mask = combine_weights.bool()

        me = torch.mean(gates, dim=0)
        ce = torch.mean(mask.float(), dim=0)
        l_aux = torch.mean(me * ce) * num_experts * num_experts / float(k)

        return l_aux, combine_weights, dispatch_mask, exp_counts


    class QwenDeepSpeedMoEAdapter(nn.Module):
        """Preserve Qwen routed-MoE + shared-expert semantics."""

        def __init__(
            self,
            routed_moe,
            shared_expert,
            shared_expert_gate,
            layer_idx=0,
        ):
            super().__init__()
            self.routed_moe = routed_moe
            self.shared_expert = shared_expert
            self.shared_expert_gate = shared_expert_gate
            self.layer_idx = int(layer_idx)

        def forward(self, hidden_states):
            original_shape = hidden_states.shape
            flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])

            routed_result = self.routed_moe(hidden_states)
            routed_hidden = routed_result[0] if isinstance(routed_result, tuple) else routed_result

            # Original Qwen shared expert branch.
            shared_hidden = self.shared_expert(flat_hidden)
            shared_gate = torch.sigmoid(self.shared_expert_gate(flat_hidden))
            shared_hidden = (shared_hidden * shared_gate).view(original_shape)
            final_hidden = routed_hidden + shared_hidden

            # Reuse the exact logits used by the DeepSpeed gate.
            gate = self.routed_moe.deepspeed_moe.gate
            router_logits = getattr(gate, "last_router_logits", None)
            if router_logits is None:
                router_logits = F.linear(
                    flat_hidden.float(),
                    gate.wg.weight.float(),
                    bias=None,
                )
            router_logits = router_logits.to(dtype=hidden_states.dtype)

            # Qwen2MoeSparseMoeBlock v4.40.x returns (hidden, router_logits).
            return final_hidden, router_logits


    class TopKGate(nn.Module):
        """Gate module which implements Top2Gating as described in Gshard_.
        ::

            gate = TopKGate(model_dim, num_experts)
            l_aux, combine_weights, dispatch_mask = gate(input)

        .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

        Args:
            model_dim (int):
                size of model embedding dimension
            num_experts (int):
                number of experts in model
        """

        def __init__(self,
                    model_dim: int,
                    num_experts: int,
                    weight: any = None,
                    k: int = 1,
                    capacity_factor: float = 1.0,
                    eval_capacity_factor: float = 1.0,
                    min_capacity: int = 8,
                    noisy_gate_policy: Optional[str] = None,
                    drop_tokens: bool = True,
                    use_rts: bool = True,
                    ep_group: Union[torch.distributed.ProcessGroup, None] = None,
                    top2_2nd_expert_sampling: bool = True) -> None:
            super().__init__()
            self.layer_idx = 0

            self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
            self.ep_group = ep_group
            self.k = k
            self.capacity_factor = capacity_factor
            self.eval_capacity_factor = eval_capacity_factor
            self.min_capacity = min_capacity
            self.noisy_gate_policy = noisy_gate_policy
            self.timers = SynchronizedWallClockTimer()
            self.wall_clock_breakdown = False
            self.gate_time = 0.0
            self.drop_tokens = drop_tokens
            self.use_rts = use_rts
            self.top2_2nd_expert_sampling = top2_2nd_expert_sampling
            self.norm_topk_prob = False
            self.last_router_logits = None
            # Here is the update by adding my own router I can change the logits
            # I want to do this before the chaos that comes after which I cannot rewrite
            routerConfig = RouterConfig(
                d_model=args.d_model,
                num_experts=args.num_experts,
                weights=weight,
                enable_skew=args.enable_router_skew,
                enable_random=args.enable_router_random,
                enable_uniform=args.enable_router_uniform,
                skew=args.router_skew,
                num_experts_skewed=args.router_num_experts_skewed,
            )
            self.router = Router(routerConfig) 
            self.enable_router_skew = args.enable_router_skew

        def _set_ep_group(self, ep_group):
            assert self.ep_group is None, f'Attempting to override an existing ep_group'
            self.ep_group = ep_group

        def forward(self,
                    input: torch.Tensor,
                    used_token: torch.Tensor = None,
                    use_tutel: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore

            if self.wall_clock_breakdown:
                self.timers(TOPK_GATE_TIMER).start()

            input_fp32 = input.float()
            # input jittering
            if self.noisy_gate_policy == 'Jitter' and self.training:
                input_fp32 = multiplicative_jitter(input_fp32, device=input.device)
            
            if self.enable_router_skew:
                 _, _, logits = self.router(input_fp32)
            else:
                logits = torch.nn.functional.linear(input_fp32, weight=self.wg.weight.float(), bias=None)
           

            # Preserve the exact logits used for routing.
            self.last_router_logits = logits

            if self.k == 1:
                gate_output = TopKGate.top1gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                        self.min_capacity, used_token, self.noisy_gate_policy if self.training else None,
                                        self.drop_tokens, self.use_rts, self.ep_group, use_tutel)

            elif self.k == 2:
                gate_output = top2gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                        self.min_capacity, self.drop_tokens, self.ep_group, self.top2_2nd_expert_sampling)
            else:
                gate_output = _qwen_dense_topkgating(
                    logits,
                    self.k,
                    ep_group=self.ep_group,
                    norm_topk_prob=bool(getattr(self, "norm_topk_prob", False)),
                )

            if self.wall_clock_breakdown:
                self.timers(TOPK_GATE_TIMER).stop()
                self.gate_time = self.timers(TOPK_GATE_TIMER).elapsed(reset=False)

            token_ids = _ds_trace_token_ids(input.shape[0])
            _ds_trace("moe_launch_time", token_ids, self.layer_idx, batch_id=0)
            _ds_trace("expert_fetch_start", token_ids, self.layer_idx, batch_id=0)
            _ds_trace("expert_fetch_end", token_ids, self.layer_idx, batch_id=0)
            _ds_trace("expert_compute_start", token_ids, self.layer_idx, batch_id=0)

            return gate_output

        @torch.jit.script
        def _capacity(gates: Tensor, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
            # gates has shape of SE
            num_tokens = gates.shape[0]
            num_experts = gates.shape[1]
            # to(torch.int64) works around a bug in torch.onnx.export:
            # it should cast k to int64 when converting torch.topk but it doesn't.
            capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
            if capacity < min_capacity:
                capacity = min_capacity.to(torch.int64)
            return capacity
        

        @torch.jit.script
        def _top_idx(source, k):
            return torch.topk(source, k=k, dim=0)[1]
        
        @torch.jit.script
        def _one_hot_to_float(x, num_classes):
            return F.one_hot(x, num_classes=num_classes).float()
        
        def einsum(rule, a, b):
            USE_EINSUM = True

            if USE_EINSUM:
                return torch.einsum(rule, a, b)
            elif rule == 's,se->se':
                return a.reshape(a.shape[0], -1) * b
            elif rule == 'se,sc->sec':
                return a.unsqueeze(2) * b.unsqueeze(1)
            elif rule == 'se,se->s':
                return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).reshape(-1)
            elif rule == 'se,sec->sec':
                return a.unsqueeze(2) * b
            elif rule == 'sec,sm->ecm':
                s = a.shape[0]
                e = a.shape[1]
                c = a.shape[2]
                m = b.shape[1]
                return torch.matmul(a.reshape(s, -1).t(), b).reshape(e, c, m)
            elif rule == 'sec,ecm->sm':
                return torch.matmul(a.reshape(a.shape[0], -1), b.reshape(-1, b.shape[-1]))
            elif rule == 'ks,ksm->sm':
                k = b.shape[0]
                s = b.shape[1]
                m = b.shape[2]
                # [k, s] -> [s, k] -> [s, 1, k]
                a = a.t().unsqueeze(1)
                # [k,s,m] -> [k, sm] -> [sm, k] -> [s, m, k]
                b = b.reshape(k, -1).t().reshape(s, m, k)
                # bmm([s, 1, k], [s, m, k]^t) -> [s, m, 1]
                return torch.bmm(a, b.transpose(1, 2)).squeeze(2)
            else:
                return torch.einsum(rule, a, b)
        
        def top1gating(logits: Tensor,
               capacity_factor: float,
               min_capacity: int,
               used_token: Tensor = None,
               noisy_gate_policy: Optional[str] = None,
               drop_tokens: bool = True,
               use_rts: bool = True,
               ep_group: Union[torch.distributed.ProcessGroup, None] = None,
               use_tutel: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
            
            # TODO implement skew
            
            """Implements Top1Gating on logits."""
            if noisy_gate_policy == 'RSample':
                logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
            # everything is in fp32 in this function

            gates = F.softmax(logits, dim=1)
            capacity = TopKGate._capacity(gates, torch.tensor(capacity_factor), torch.tensor(min_capacity))

            # Create a mask for 1st's expert per token
            # noisy gating
            indices1_s = torch.argmax(logits_w_noise if noisy_gate_policy == 'RSample' else gates, dim=1)
            num_experts = int(gates.shape[1])
            mask1 = F.one_hot(indices1_s, num_classes=num_experts)

            # mask only used tokens
            if used_token is not None:
                mask1 = TopKGate.einsum("s,se->se", used_token, mask1)

            # gating decisions
            exp_counts = torch.sum(mask1, dim=0).detach().to(logits.device)

            # if we don't want to drop any tokens
            if not drop_tokens:
                new_capacity = torch.max(exp_counts).to(logits.device)
                # Communicate across expert processes to pick the maximum capacity.
                if ep_group is not None:
                    # print("Here WOAH")
                    dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=ep_group)
                if groups._get_expert_model_parallel_world_size() == 1:
                    # If the non-expert is tensor-parallel, we need to pad the capacity to 'tp'.
                    # This is since we are going to activate drop_tokens() to drop duplicate tokens.
                    tp = 1 if groups.mpu is None else bwc_tensor_model_parallel_world_size(mpu=groups.mpu)
                    new_capacity = torch.ceil(new_capacity / tp).mul(tp).to(new_capacity.dtype)
                # Make sure the capacity value does not exceed the number of tokens.
                capacity = min(new_capacity, torch.tensor(mask1.size(0)).to(new_capacity.device))
                capacity = min(capacity, torch.tensor(1000).to(capacity.device)) # Set a 1000 artificial max

            # Compute l_aux
            me = torch.mean(gates, dim=0)
            ce = torch.mean(mask1.float(), dim=0)
            l_aux = torch.sum(me * ce) * num_experts

            # Random Token Selection
            if use_rts:
                uniform = exp_selection_uniform_map.get(logits.device)
                if uniform is None:
                    uniform = torch.distributions.uniform.Uniform(low=torch.tensor(0.0, device=logits.device),
                                                                high=torch.tensor(1.0, device=logits.device)).rsample
                    exp_selection_uniform_map[logits.device] = uniform

                mask1_rand = mask1 * uniform(mask1.shape)
            else:
                mask1_rand = mask1

            assert logits.shape[
                0] >= min_capacity, "No. of tokens (batch-size) should be greater than min_capacity. Either set min_capacity to 0 or increase your batch size."

            top_idx = TopKGate._top_idx(mask1_rand, capacity)

            new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)
            mask1 = new_mask1

            if use_tutel:
                # Tutel doesn't support index values masked with zero
                # so we need to replace masked indices with -1
                indices_mask = mask1.sum(dim=1) * num_experts - 1
                indices1_s = torch.min(indices1_s, indices_mask)

            # Compute locations in capacity buffer
            if use_tutel:
                locations1 = tutel_moe.fast_cumsum_sub_one(mask1)
            else:
                locations1 = torch.cumsum(mask1, dim=0) - 1

            if use_tutel:
                gates1_s = (gates * mask1).sum(dim=1)
                locations1_s = torch.sum(locations1 * mask1, dim=1)
                return l_aux, capacity, num_experts, [
                    indices1_s,
                ], [
                    locations1_s,
                ], [
                    gates1_s,
                ], exp_counts

            # Store the capacity location for each token
            locations1_s = torch.sum(locations1 * mask1, dim=1)

            # Normalize gate probabilities
            mask1_float = mask1.float()
            gates = gates * mask1_float

            locations1_sc = TopKGate._one_hot_to_float(locations1_s, capacity)
            combine_weights = TopKGate.einsum("se,sc->sec", gates, locations1_sc)

            dispatch_mask = combine_weights.bool()

            return l_aux, combine_weights, dispatch_mask, exp_counts

    def get_tensor_by_path(module, path):
        parts = path.split(".")
        current = module
        for part in parts:
            current = getattr(current, part)
        return current

    # Update to add DeepspeedMoE to it
    def add_deepspeed_moe_model(module, idx):
        if type(module).__name__ == args.type_moe_parent:
            for child_name, child in module.named_children():
                if type(child).__name__ != args.type_moe:
                    continue

                router = get_tensor_by_path(child, args.router_tensor_path)
                experts_obj = get_tensor_by_path(child, args.name_experts)

                if isinstance(experts_obj, nn.ModuleDict):
                    experts_all = list(experts_obj.values())
                else:
                    experts_all = list(experts_obj)

                if args.num_experts % args.world_size != 0:
                    raise RuntimeError(
                        f"num_experts={args.num_experts} must be divisible by "
                        f"world_size={args.world_size}"
                    )

                num_experts_per_gpu = args.num_experts // args.world_size
                lo = args.local_rank * num_experts_per_gpu
                hi = (args.local_rank + 1) * num_experts_per_gpu
                experts = experts_all[lo:hi]

                if len(experts) != num_experts_per_gpu:
                    raise RuntimeError(
                        f"Expected {num_experts_per_gpu} local experts, got {len(experts)} "
                        f"for rank={args.local_rank}"
                    )

                is_qwen_block = type(child).__name__ == "Qwen2MoeSparseMoeBlock"

                if is_qwen_block:
                    top_k = int(getattr(model.config, "num_experts_per_tok", 4))
                    norm_topk_prob = bool(getattr(model.config, "norm_topk_prob", False))
                    shared_expert = child.shared_expert
                    shared_expert_gate = child.shared_expert_gate
                else:
                    top_k = 1
                    norm_topk_prob = False
                    shared_expert = None
                    shared_expert_gate = None

                # [QWEN_DS_CONSTRUCTOR_K_FIX_V1]
                # DeepSpeed 0.14.0 only accepts k=1/2 in its stock gate
                # constructor. We immediately replace that gate below with
                # our custom TopKGate(k=top_k), so constructor_k is only a
                # compatibility placeholder and does not change Qwen routing.
                constructor_k = top_k if top_k <= 2 else 1

                new = deepspeed.moe.layer.MoE(
                    hidden_size=args.d_model,
                    expert=experts[0],
                    num_experts=args.num_experts,
                    ep_size=args.world_size,
                    k=constructor_k,
                    eval_capacity_factor=args.capacity_factor,
                    drop_tokens=False,
                    use_tutel=False,
                    top2_2nd_expert_sampling=False,
                    use_rts=False,
                )

                custom_gate = TopKGate(
                    args.d_model,
                    args.num_experts,
                    weight=router.weight,
                    k=top_k,
                    capacity_factor=1.0,
                    eval_capacity_factor=args.capacity_factor,
                    min_capacity=0,
                    noisy_gate_policy=None,
                    drop_tokens=False,
                    use_rts=False,
                    # [QWEN_DS_SHARED_GROUP_FIX_V1]
                    # Reuse the already-initialized WORLD NCCL communicator.
                    # Creating one identical NCCL group per MoE layer can reserve
                    # substantial extra device memory and makes the first collective
                    # fail with ncclUnhandledCudaError / CUDA OOM on Qwen.
                    ep_group=dist.group.WORLD,
                    top2_2nd_expert_sampling=False,
                )
                custom_gate.norm_topk_prob = norm_topk_prob

                with torch.no_grad():
                    # Critical: TopKGate.forward uses wg.weight when router skew
                    # is disabled, so copy the original HF router here.
                    custom_gate.wg.weight.copy_(
                        router.weight.detach().to(
                            device=custom_gate.wg.weight.device,
                            dtype=custom_gate.wg.weight.dtype,
                        )
                    )

                setattr(new.deepspeed_moe, "gate", custom_gate)

                # Copy each local original expert into DeepSpeed's expert slots.
                with torch.no_grad():
                    for i in range(len(experts)):
                        dst_expert = new.deepspeed_moe.experts.deepspeed_experts[i]
                        src_expert = experts[i]
                        for name, param in dst_expert.named_parameters():
                            current_attr = src_expert
                            for part in name.split("."):
                                current_attr = getattr(current_attr, part)
                            param.data.copy_(
                                current_attr.data.to(
                                    device=param.device,
                                    dtype=param.dtype,
                                )
                            )

                layer_idx = idx[0]
                try:
                    new.deepspeed_moe.gate.layer_idx = layer_idx
                except Exception:
                    pass

                if is_qwen_block:
                    adapted = QwenDeepSpeedMoEAdapter(
                        routed_moe=new,
                        shared_expert=shared_expert,
                        shared_expert_gate=shared_expert_gate,
                        layer_idx=layer_idx,
                    )
                    setattr(module, child_name, TimedModule(adapted, idx=layer_idx))
                else:
                    setattr(
                        module,
                        child_name,
                        TimedModule(
                            MLPWrapper(new, layer_idx=layer_idx),
                            idx=layer_idx,
                        ),
                    )

                if args.local_rank == 0:
                    print(
                        f"[DeepSpeedAdapter] layer={layer_idx} "
                        f"type={type(child).__name__} "
                        f"top_k={top_k} "
                        f"constructor_k={constructor_k} "
                        f"norm_topk_prob={norm_topk_prob} "
                        f"shared_expert={is_qwen_block}",
                        flush=True,
                    )

                idx[0] += 1
        else:
            for child in module.children():
                add_deepspeed_moe_model(child, idx)

    add_deepspeed_moe_model(model, [0])

    model.eval()
    model.cuda()

    ds_engine = deepspeed.init_inference(
        model,
        dtype=(torch.float16 if is_qwen_moe else torch.float),
        replace_with_kernel_inject=False,
        moe={
            "enabled": True,
            "ep_size": args.world_size, 
            "moe_experts": [args.num_experts],
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=os.environ["HF_HOME"])

    flexible_dataset = FlexibleDataset(
        args.dataset, 
        tokenizer, 
        model, 
        seq_len=args.seq_len,
        num_samples=args.num_samples,
        model_name=args.model_name,
    )
    sampler = DistributedSampler(
        flexible_dataset, 
        num_replicas=args.world_size, 
        rank=args.local_rank, 
        shuffle=True, 
        seed=49
    )
    loader = DataLoader(
        flexible_dataset, 
        sampler=sampler, 
        batch_size=args.batch_size
    )

    latencies, run_start, run_end = run_standard_experiment(ds_engine, loader)

    path = f"{args.path}/{args.local_rank}"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    ############# E2E #######################
    file_path = f"{path}/e2e.csv"
    with open(file_path, "w") as f:
        fieldnames = ["iteration", "latency (s)"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, latency in enumerate(latencies):
            writer.writerow({
                "iteration": idx, 
                "latency (s)": latency,
            })
    
    ############# LAYER #######################
    for timing_module in get_timing_modules([], model):
        latencies = timing_module.get_latencies()[args.warmup_rounds:]
        file_path = f"{path}/layer-{timing_module.idx}.csv"
        with open(file_path, "w") as f:
            fieldnames = ["iteration", "latency (ms)"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx, latency in enumerate(latencies):
                writer.writerow({
                    "iteration": idx,
                    "latency (ms)": latency,
                })
    
    ############# META #######################
    run_info = vars(args).copy()
    run_info["system_name"] = "deepspeed-inference"
    with open(f"{args.path}/data.json", "w") as f:
        json.dump({ "start": run_start, "end": run_end, **run_info, "world_size": args.world_size}, f, indent=4)

def run_standard_experiment(ds_engine, loader):
    latencies = []
    
    with torch.no_grad():
        # WARMUP
        itr = 0
        for batch in loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            if "decoder_input_ids" in batch:
                ds_engine(
                    input_ids=batch["input_ids"], 
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=batch["decoder_input_ids"]
                )
            else:
                ds_engine(
                    input_ids=batch["input_ids"], 
                    attention_mask=batch["attention_mask"],
                )
            itr += 1
            if itr == args.warmup_rounds:
                break

        # RUN ACTUAL EXPERIMENT
        run_start = time.time()
        for iter_idx, batch in enumerate(tqdm(loader)):
            _ds_set_iteration(iter_idx)

            batch = {k: v.cuda() for k, v in batch.items()}
            _ds_trace_iter_start(batch)

            start = time.time()
            if "decoder_input_ids" in batch:
                ds_engine(
                    input_ids=batch["input_ids"], 
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=batch["decoder_input_ids"]
                )
            else:
                ds_engine(
                    input_ids=batch["input_ids"], 
                    attention_mask=batch["attention_mask"],
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.time()
            _ds_trace_iter_end(batch)

            latencies.append(end-start)
        run_end = time.time()
    
    return latencies, run_start, run_end

if __name__ == "__main__":
    stats = Stats(gpu=True, cpu=True, num_gpus=args.world_size)
    stats.start()

    run_inference_workload()

    stats.stop()
    stats.save(path=args.path)

    print("All done :)")