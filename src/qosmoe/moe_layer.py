import os
import torch
import torch.nn as nn
import torch.distributed as dist

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

from harmonymoe.router import Router, RouterConfig
from harmonymoe.scheduler import Scheduler as BaseScheduler

from qosmoe.config import QoSMoEConfig
from qosmoe.planner import QoSPlanner
from qosmoe.expert_manager import QoSExpertManager

try:
    from qosmoe_token_trace import trace_event as _qos_trace_event
except Exception:
    _qos_trace_event = None


def _qos_trace_token_ids(num_tokens):
    try:
        max_tokens = int(os.environ.get("QOSMOE_TRACE_MAX_TOKENS", "32"))
    except Exception:
        max_tokens = 32
    return list(range(min(int(num_tokens), max_tokens)))


def _qos_trace(event_name, token_ids, layer_idx, batch_id=0, extra=None):
    if _qos_trace_event is None:
        return
    try:
        _qos_trace_event(
            event_name,
            token_ids=token_ids,
            layer=int(layer_idx),
            batch_id=int(batch_id),
            extra=extra or {},
        )
    except Exception:
        pass



class QoSMoELayer(nn.Module):
    def __init__(self, config: QoSMoEConfig):
        super().__init__()
        self.config = config
        self.rank = config.rank
        self.layer_idx = config.layer_idx
        self.num_experts = config.num_experts
        self.num_gpus = config.world_size
        self.d_model = config.d_model
        self.args = config
        self.next_layer = None
        self.num_prefetch_launched = []

        self.router = Router(
            RouterConfig(
                d_model=config.d_model,
                num_experts=config.num_experts,
                weights=config.router_weights,
                enable_skew=config.enable_skew,
                enable_random=config.enable_random,
                enable_uniform=config.enable_uniform,
                skew=config.skew,
                num_experts_skewed=config.num_experts_skewed,
            )
        )

        self.base_scheduler = BaseScheduler(
            rank=config.rank,
            scheduling_policy="harmony",
            num_experts=config.num_experts,
            eq_tokens=config.eq_tokens,
            num_gpus=config.world_size,
            d_model=config.d_model,
            expert_placement=config.expert_placement,
            layer_idx=config.layer_idx,
        )

        full_cache = self.base_scheduler.get_cache()
        self.resident_cache = [
            list(gpu_experts[: config.expert_cache_size]) for gpu_experts in full_cache
        ]

        self.expert_manager = QoSExpertManager(
            rank=self.rank,
            world_size=self.num_gpus,
            experts=self.config.experts,
            expert_example=self.config.expert_example,
            cache_size=self.config.expert_cache_size,
            cache=self.resident_cache,
            fetching_strategy=self.config.fetching_strategy,
            layer_idx=self.layer_idx,
        )

        self.planner = QoSPlanner(
            args=config,
            rank=config.rank,
            num_experts=config.num_experts,
            num_gpus=config.world_size,
            d_model=config.d_model,
        )

        self.runtime_context = None

        self.tot_num_toks_send = []
        self.tot_num_toks_recv = []
        self.latencies = []
        self.expert_freqs = []

        self.start_pass = torch.cuda.Event(enable_timing=True)
        self.end_pass = torch.cuda.Event(enable_timing=True)

        self.single_gpu_all_resident = (
            self.num_gpus == 1 and self.config.expert_cache_size >= self.num_experts
        )

        if self.single_gpu_all_resident:
            local_cache = self.resident_cache[self.rank]
            self.single_gpu_slot_map = {
                expert_idx: slot_idx for slot_idx, expert_idx in enumerate(local_cache)
            }
        else:
            self.single_gpu_slot_map = None

    def set_runtime_context(self, ctx):
        self.runtime_context = ctx

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer

    def get_statistics(self):
        stats = []
        num_expert_swaps = self.expert_manager.get_statistics()["num_expert_swaps"]

        for i in range(len(self.tot_num_toks_send)):
            stats.append(
                {
                    "latency (ms)": self.latencies[i],
                    "total number of tokens sent": self.tot_num_toks_send[i],
                    "total number of tokens recv": self.tot_num_toks_recv[i],
                    "expert distribution": self.expert_freqs[i],
                    "num expert swaps": num_expert_swaps[i] if i < len(num_expert_swaps) else 0,
                    "num prefetch launched": self.num_prefetch_launched[i] if i < len(self.num_prefetch_launched) else 0,
                }
            )
        return stats

    def _maybe_prefetch_next_layer(self, expert_priority_meta: torch.Tensor):
        if not getattr(self.config, "qos_enable_prefetch", False):
            self.num_prefetch_launched.append(0)
            return

        if self.next_layer is None:
            self.num_prefetch_launched.append(0)
            return

        budget = int(getattr(self.config, "qos_prefetch_budget_experts", 0))
        if budget <= 0:
            self.num_prefetch_launched.append(0)
            return

        scores = expert_priority_meta.float().clamp_min(0)

        if scores.numel() == 0 or torch.all(scores <= 0):
            self.num_prefetch_launched.append(0)
            return

        power = float(getattr(self.config, "qos_prefetch_score_power", 1.0))
        if power != 1.0:
            scores = scores.pow(power)

        topk = int(getattr(self.config, "qos_prefetch_topk", budget))
        topk = min(topk, budget, scores.numel())

        candidate_ids = torch.topk(scores, k=topk).indices.detach().cpu().tolist()

        launched = self.next_layer.expert_manager.prefetch_experts(candidate_ids)
        self.num_prefetch_launched.append(len(launched))

    def _forward_single_gpu_resident_all(
        self,
        hidden_states,
        original_shape,
        router_probs,
        router_logits,
        expert_index,
    ):
        num_tokens = hidden_states.shape[0]

        counts = torch.bincount(expert_index, minlength=self.num_experts)
        num_toks_per_expert = counts.tolist()
        self.expert_freqs.append(num_toks_per_expert)

        self.tot_num_toks_send.append(num_tokens)
        self.tot_num_toks_recv.append(num_tokens)
        self.num_prefetch_launched.append(0)

        sorted_experts, sorted_pos = torch.sort(expert_index)
        sorted_tokens = hidden_states[sorted_pos]

        output_sorted = sorted_tokens.clone()

        rt_job_id, rt_job_ready_ms = _rt_job_start(
            hidden_states,
            None,
            tag=__name__ + "._forward_single_gpu_resident_all",
        )

        offset = 0
        for expert_idx in range(self.num_experts):
            cnt = int(counts[expert_idx].item())
            if cnt == 0:
                continue

            slot_idx = self.single_gpu_slot_map[expert_idx]
            expert = self.expert_manager.cached_experts[slot_idx]

            expert_tokens = sorted_tokens[offset : offset + cnt]
            original_token_idx = sorted_pos[offset : offset + cnt]

            rt_moe_start_ms = _rt_now_ms()
            output_sorted[offset : offset + cnt] = expert(expert_tokens)
            rt_moe_end_ms = _rt_now_ms()

            _rt_record_group(
                rt_job_id,
                rt_job_ready_ms,
                expert_idx,
                original_token_idx,
                rt_moe_start_ms,
                rt_moe_end_ms,
                tokens=hidden_states,
                note=__name__ + "._forward_single_gpu_resident_all",
                layer_idx=self.layer_idx,
            )

            offset += cnt

        hidden_states_out = torch.empty_like(hidden_states)
        hidden_states_out[sorted_pos] = output_sorted

        hidden_states_out = router_probs * hidden_states_out
        hidden_states_out = hidden_states_out.view(original_shape).contiguous()

        self.runtime_context.last_hidden_states = (
            hidden_states_out.view(-1, self.d_model).contiguous().detach()
        )

        self.end_pass.record()
        self.end_pass.synchronize()
        self.latencies.append(self.start_pass.elapsed_time(self.end_pass))

        return hidden_states_out, (router_logits, expert_index)

    @torch.no_grad()
    def forward(self, hidden_states):
        if self.runtime_context is None:
            raise RuntimeError("QoSMoELayer requires runtime_context before forward")

        self.start_pass.record()

        device = hidden_states.device
        original_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.d_model).contiguous()

        trace_token_ids = _qos_trace_token_ids(hidden_states.shape[0])
        _qos_trace("attn_end", trace_token_ids, self.layer_idx, batch_id=0)
        _qos_trace("moe_ready_time", trace_token_ids, self.layer_idx, batch_id=0)

        router_mask, router_probs, router_logits = self.router(hidden_states)
        expert_index = torch.argmax(router_mask, dim=-1)

        if self.single_gpu_all_resident:
            return self._forward_single_gpu_resident_all(
                hidden_states=hidden_states,
                original_shape=original_shape,
                router_probs=router_probs,
                router_logits=router_logits,
                expert_index=expert_index,
            )

        router_mask = router_mask.bool()

        expert_indices = [
            router_mask[:, j].nonzero(as_tuple=False).flatten()
            for j in range(self.num_experts)
        ]
        num_toks_per_expert = [idx.numel() for idx in expert_indices]
        self.expert_freqs.append(num_toks_per_expert)

        token_scores = self.planner.compute_token_scores(
            current_hidden_states=hidden_states.detach(),
            prev_hidden_states=self.runtime_context.last_hidden_states,
            router_logits=router_logits.detach(),
            expert_index=expert_index.detach(),
            runtime_context=self.runtime_context,
        )

        local_priority_meta = self.planner.aggregate_expert_priority(
            expert_indices, token_scores
        )

        if self.num_gpus == 1:
            expert_mask_local = expert_indices
            global_priority_meta = local_priority_meta

            exec_plan = self.planner.build_exec_plan(
                expert_mask=expert_mask_local,
                global_priority_meta=global_priority_meta,
                local_cache=self.resident_cache[self.rank],
                cache_size=self.config.expert_cache_size,
            )

            self._maybe_prefetch_next_layer(local_priority_meta)

            self.tot_num_toks_send.append(hidden_states.shape[0])
            self.tot_num_toks_recv.append(hidden_states.shape[0])

            _qos_trace("moe_launch_time", trace_token_ids, self.layer_idx, batch_id=0)
            _qos_trace("expert_fetch_start", trace_token_ids, self.layer_idx, batch_id=0)
            _qos_trace("expert_fetch_end", trace_token_ids, self.layer_idx, batch_id=0)
            _qos_trace("expert_compute_start", trace_token_ids, self.layer_idx, batch_id=0)

            self.expert_manager(
                hidden_states,
                expert_mask_local,
                schedule={
                    "expert_order": exec_plan.expert_order,
                    "slot_idxs": exec_plan.slot_idxs,
                    "need_loading": exec_plan.need_loading,
                },
            )

            _qos_trace("expert_compute_end", trace_token_ids, self.layer_idx, batch_id=0)
            _qos_trace("combine_start", trace_token_ids, self.layer_idx, batch_id=0)

            hidden_states = router_probs * hidden_states
            hidden_states = hidden_states.view(original_shape).contiguous()

            _qos_trace("combine_end", trace_token_ids, self.layer_idx, batch_id=0)

            self.runtime_context.last_hidden_states = (
                hidden_states.view(-1, self.d_model).contiguous().detach()
            )

            self.end_pass.record()
            self.end_pass.synchronize()
            self.latencies.append(self.start_pass.elapsed_time(self.end_pass))
            return hidden_states, (router_logits, expert_index)

        metadata_send = torch.tensor(
            [num_toks_per_expert] * self.num_gpus,
            dtype=torch.int,
            device=device,
        )
        metadata_recv = torch.zeros(
            (self.num_gpus, self.num_experts),
            dtype=torch.int,
            device=device,
        )
        dist.all_to_all_single(metadata_recv, metadata_send)

        global_priority_meta = local_priority_meta.clone()
        dist.all_reduce(global_priority_meta, op=dist.ReduceOp.SUM)

        self._maybe_prefetch_next_layer(global_priority_meta)

        schedule = self.base_scheduler(metadata_recv)
        schedule_list = schedule.tolist()

        _qos_trace("moe_launch_time", trace_token_ids, self.layer_idx, batch_id=0)
        _qos_trace("expert_fetch_start", trace_token_ids, self.layer_idx, batch_id=0)

        num_toks_send = schedule[self.rank].sum()
        self.tot_num_toks_send.append(num_toks_send.item())

        tokens_send, send_splits = self.base_scheduler.distribute_tokens(
            schedule_list, hidden_states, expert_indices, num_toks_send
        )
        tokens_recv, recv_splits = self.base_scheduler.allocate_recv_tensors(
            schedule, dtype=hidden_states.dtype
        )

        total_tokens = tokens_recv.shape[0]
        self.tot_num_toks_recv.append(total_tokens)

        dist.all_to_all_single(
            tokens_recv,
            tokens_send,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )

        expert_mask_recv = self.base_scheduler.generate_expert_mask(schedule, total_tokens)

        exec_plan = self.planner.build_exec_plan(
            expert_mask=expert_mask_recv,
            global_priority_meta=global_priority_meta,
            local_cache=self.resident_cache[self.rank],
            cache_size=self.config.expert_cache_size,
        )

        _qos_trace("expert_fetch_end", trace_token_ids, self.layer_idx, batch_id=0)
        _qos_trace("expert_compute_start", trace_token_ids, self.layer_idx, batch_id=0)

        self.expert_manager(
            tokens_recv,
            expert_mask_recv,
            schedule={
                "expert_order": exec_plan.expert_order,
                "slot_idxs": exec_plan.slot_idxs,
                "need_loading": exec_plan.need_loading,
            },
        )

        _qos_trace("expert_compute_end", trace_token_ids, self.layer_idx, batch_id=0)
        _qos_trace("combine_start", trace_token_ids, self.layer_idx, batch_id=0)

        dist.all_to_all_single(
            tokens_send,
            tokens_recv,
            output_split_sizes=send_splits,
            input_split_sizes=recv_splits,
        )

        hidden_states = self.base_scheduler.gather_tokens(
            schedule_list, tokens_send, hidden_states, expert_indices
        )

        hidden_states = router_probs * hidden_states
        hidden_states = hidden_states.view(original_shape).contiguous()

        _qos_trace("combine_end", trace_token_ids, self.layer_idx, batch_id=0)

        self.runtime_context.last_hidden_states = (
            hidden_states.view(-1, self.d_model).contiguous().detach()
        )

        self.end_pass.record()
        self.end_pass.synchronize()
        self.latencies.append(self.start_pass.elapsed_time(self.end_pass))

        return hidden_states, (router_logits, expert_index)