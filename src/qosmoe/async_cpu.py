
# ===== real token-level gantt trace =====
try:
    from real_token_gantt_trace import (
        job_start as rt_job_start,
        now_ms as rt_now_ms,
        record_group as rt_record_group,
    )
except Exception:
    import time as rt_time
    def rt_now_ms():
        return rt_time.perf_counter_ns() / 1e6
    def rt_job_start(tokens=None, expert_mask=None, tag=""):
        return -1, rt_now_ms()
    def rt_record_group(*args, **kwargs):
        return

rt_job_id = -1
rt_job_ready_ms = 0.0
# ===== end real token-level gantt trace =====

from dataclasses import dataclass

import torch


@dataclass
class QoSExecutorConfig:
    rank: int
    world_size: int
    experts: list
    cache: list
    cached_experts: list
    cache_size: int
    num_experts: int
    layer_idx: int = -1


class QoSAsyncCPU:
    """
    QoSMoE 独立 async CPU executor。
    只支持 expert-level schedule，不做 token-level 重排。

    这版只做一个保守优化：
    - 当 world_size == 1 且所有 expert 已常驻 cache 时，
      直接走 resident-all fast path
    - 不走 load_stream / comp_stream / loading schedule 的通用路径
    """

    def __init__(self, config: QoSExecutorConfig):
        self.config = config

        self.local_cache = list(self.config.cache[self.config.rank])
        if len(self.local_cache) < self.config.cache_size:
            self.local_cache.extend([-1] * (self.config.cache_size - len(self.local_cache)))
        self.local_cache = self.local_cache[: self.config.cache_size]

        # QoSMoE ablation:
        # QOSMOE_ENABLE_ASYNC=0 disables load/compute overlap by forcing
        # both loading and compute onto the default CUDA stream.
        if __import__("os").environ.get("QOSMOE_ENABLE_ASYNC", "1").lower() in ("0", "false", "no", "off"):
            self.load_stream = torch.cuda.default_stream()
            self.comp_stream = torch.cuda.default_stream()
            print("[QoSMoE_Ablation] async scheduling disabled: using default CUDA stream", flush=True)
        else:
            self.load_stream = torch.cuda.Stream()
            self.comp_stream = torch.cuda.Stream()

        self.slot_ready_events = [
            torch.cuda.Event(enable_timing=False) for _ in range(config.cache_size)
        ]
        self.expert_loaded = [
            torch.cuda.Event(enable_timing=False) for _ in range(config.num_experts)
        ]
        self.finished_event = torch.cuda.Event(enable_timing=False)

        self.num_expert_swaps = []

        for evt in self.slot_ready_events:
            evt.record()

        self._refresh_expert_to_slot()

        # single-GPU + all experts resident => 可直接走快路径
        self.all_resident_single_gpu = (
            self.config.world_size == 1
            and self.config.cache_size >= self.config.num_experts
            and all(expert_idx >= 0 for expert_idx in self.local_cache[: self.config.num_experts])
            and len(self.expert_to_slot) >= self.config.num_experts
        )

    def _refresh_expert_to_slot(self):
        self.expert_to_slot = {}
        for slot_idx, expert_idx in enumerate(self.local_cache):
            if expert_idx >= 0:
                self.expert_to_slot[int(expert_idx)] = int(slot_idx)

    def load_expert_into_slot(self, expert_idx, slot_idx):
        with torch.no_grad():
            with torch.cuda.stream(self.load_stream):
                self.load_stream.wait_event(self.slot_ready_events[slot_idx])

                pinned_state_dict = self.config.experts[expert_idx]
                cached_expert = self.config.cached_experts[slot_idx]

                for name, param in cached_expert.named_parameters():
                    cpu_param = pinned_state_dict[name]
                    param.copy_(cpu_param, non_blocking=True)

                self.local_cache[slot_idx] = int(expert_idx)
                self._refresh_expert_to_slot()

                self.expert_loaded[expert_idx].record(stream=self.load_stream)

    def _generate_default_work_order(self, expert_mask):
        active = []
        for expert_idx in range(self.config.num_experts):
            if expert_mask[expert_idx].numel() > 0:
                active.append(expert_idx)

        loaded = [e for e in active if e in self.expert_to_slot]
        unloaded = [e for e in active if e not in self.expert_to_slot]

        expert_order = loaded + unloaded
        slot_idxs = []
        need_loading = []

        for expert_idx in expert_order:
            if expert_idx in self.expert_to_slot:
                slot_idxs.append(self.expert_to_slot[expert_idx])
            else:
                slot_idxs.append(0)
                need_loading.append(expert_idx)

        return expert_order, need_loading, slot_idxs

    def _execute_resident_all(self, tokens, expert_mask, expert_order):

        rt_job_id, rt_job_ready_ms = rt_job_start(tokens, expert_mask, tag=__name__ + "._execute_resident_all")
        """
        所有 expert 都在 GPU 常驻时的快路径：
        - 不做 loading
        - 不做 event 等待
        - 直接按 active experts 执行
        """
        if len(expert_order) == 0:
            self.num_expert_swaps.append(0)
            return tokens

        for expert_idx in expert_order:
            token_idx = expert_mask[expert_idx]
            if token_idx.numel() == 0:
                continue

            slot_idx = self.expert_to_slot[expert_idx]
            expert = self.config.cached_experts[slot_idx]
            rt_moe_start_ms = rt_now_ms()
            tokens[token_idx] = expert(tokens[token_idx])

            rt_moe_end_ms = rt_now_ms()
            rt_record_group(
                rt_job_id,
                rt_job_ready_ms,
                expert_idx,
                token_idx,
                rt_moe_start_ms,
                rt_moe_end_ms,
                tokens=tokens,
                note=__name__,
                layer_idx=getattr(self.config, "layer_idx", -1),
            )
        self.num_expert_swaps.append(0)
        return tokens

    def execute_job(self, tokens, expert_mask, schedule=None):

        rt_job_id, rt_job_ready_ms = rt_job_start(tokens, expert_mask, tag=__name__ + ".execute_job")
        if schedule is not None and "expert_order" in schedule:
            expert_order = schedule["expert_order"]
            need_loading = schedule["need_loading"]
            slot_idxs = schedule["slot_idxs"]
        else:
            expert_order, need_loading, slot_idxs = self._generate_default_work_order(
                expert_mask
            )

        if len(expert_order) == 0:
            self.num_expert_swaps.append(0)
            return tokens

        # ===== resident-all fast path =====
        if self.all_resident_single_gpu and len(need_loading) == 0:
            return self._execute_resident_all(tokens, expert_mask, expert_order)

        # ===== generic path =====
        num_expert_swaps_this_round = 0
        last_expert_exec = expert_order[-1]
        next_load_idx = 0

        for idx, expert_idx in enumerate(expert_order):
            slot_idx = slot_idxs[idx]
            token_idx = expert_mask[expert_idx]

            if token_idx.numel() == 0:
                continue

            with torch.cuda.stream(self.comp_stream):
                if expert_idx not in self.expert_to_slot:
                    self.comp_stream.wait_event(self.expert_loaded[expert_idx])

                rt_moe_start_ms = rt_now_ms()

                tokens[token_idx] = self.config.cached_experts[slot_idx](
                    tokens[token_idx]
                )

                rt_moe_end_ms = rt_now_ms()

                rt_record_group(

                    rt_job_id,

                    rt_job_ready_ms,

                    expert_idx,

                    token_idx,

                    rt_moe_start_ms,

                    rt_moe_end_ms,

                    tokens=tokens,

                    note=__name__,
                    layer_idx=getattr(self.config, "layer_idx", -1),

                )
                self.slot_ready_events[slot_idx].record(stream=self.comp_stream)

            if next_load_idx < len(need_loading):
                next_expert = need_loading[next_load_idx]
                if next_expert != expert_idx:
                    self.load_expert_into_slot(next_expert, slot_idx)
                    num_expert_swaps_this_round += 1
                next_load_idx += 1

            if expert_idx == last_expert_exec:
                if expert_idx not in self.expert_to_slot:
                    self.finished_event.record(stream=self.load_stream)
                else:
                    self.finished_event.record(stream=self.comp_stream)

        self.finished_event.synchronize()
        self.num_expert_swaps.append(num_expert_swaps_this_round)
        return tokens

    def get_statistics(self):
        return {"num_expert_swaps": self.num_expert_swaps}