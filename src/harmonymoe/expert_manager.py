import copy
import random

import torch
import torch.nn as nn

from . import fetching_strategies
from .fetching_strategies import FetchingStrategyConfig


class ExpertManager:
    def __init__(
        self,
        rank,
        world_size,
        experts: nn.ModuleList,
        expert_example: nn.Module,
        cache_size: int,
        cache=None,
        fetching_strategy="async-cpu",
        layer_idx=-1,
    ):
        self.experts = experts
        self.num_experts = len(experts)
        self.cache_size = cache_size
        self.layer_idx = layer_idx
        self.rank = rank
        self.world_size = world_size
        self.fetching_strategy = fetching_strategy

        # 原始 cache 可能比 cache_size 大很多；这里统一截断到 cache_size
        self.original_cache = cache
        self.cache = self._normalize_cache(cache)

        self.validate_arguments()

        self.rng = random.Random(32)

        self.num_experts_per_gpu = self.num_experts // self.world_size
        self.load_stream = torch.cuda.Stream()
        self.comp_stream = torch.cuda.Stream()

        # 当前 rank 真实可用的本地 cache（去掉 pad 的 -1）
        self.local_cache = [e for e in self.cache[self.rank] if e >= 0]

        # 只分配 cache_size 个槽
        self.cached_experts = [
            copy.deepcopy(expert_example).cuda() for _ in range(self.cache_size)
        ]

        self.buffer_expert = (
            copy.deepcopy(expert_example).cuda()
            if self.fetching_strategy == "gpu"
            else None
        )

        self.expert_loaded_events = [
            torch.cuda.Event(enable_timing=False) for _ in range(self.num_experts)
        ]

        # 只加载截断后的 local_cache，避免 slot_idx 超出 cached_experts 长度
        for slot_idx, expert_idx in enumerate(self.local_cache):
            self.load_expert_into_slot(expert_idx, slot_idx)

        first_slot_expert_idx = self.local_cache[0] if len(self.local_cache) > 0 else -1
        last_slot_expert_idx = self.local_cache[-1] if len(self.local_cache) > 0 else -1

        config = FetchingStrategyConfig(
            rank=self.rank,
            world_size=self.world_size,
            experts=self.experts,
            cache=self.cache,
            cached_experts=self.cached_experts,
            buffer_expert=self.buffer_expert,
            expert_loaded_events=self.expert_loaded_events,
            cache_size=self.cache_size,
            num_experts=self.num_experts,
            first_slot_expert_idx=first_slot_expert_idx,
            last_slot_expert_idx=last_slot_expert_idx,
        )

        if fetching_strategy == "async-cpu":
            self.executor = fetching_strategies.AsynchronousCPU(config)
        elif fetching_strategy == "sync-cpu":
            self.executor = fetching_strategies.SynchronousCPU(config)
        elif fetching_strategy == "gpu":
            self.executor = fetching_strategies.GPU(config)
        else:
            raise NotImplementedError(
                f"Strategy {fetching_strategy} is not implemented"
            )

    def _normalize_cache(self, cache):
        """
        将每个 rank 的 cache 截断到 cache_size，并在不足时用 -1 pad。
        这样后续所有 slot 索引都只能落在 [0, cache_size-1]。
        """
        assert cache is not None, "ExpertManager requires a non-empty cache assignment."

        normalized = []
        for r in range(self.world_size):
            row = list(cache[r]) if r < len(cache) else []
            row = row[: self.cache_size]
            if len(row) < self.cache_size:
                row.extend([-1] * (self.cache_size - len(row)))
            normalized.append(row)
        return normalized

    def load_expert_into_slot(self, expert_idx, slot_idx):
        if expert_idx < 0:
            return

        if slot_idx < 0 or slot_idx >= len(self.cached_experts):
            raise IndexError(
                f"slot_idx={slot_idx} out of range for cached_experts "
                f"(len={len(self.cached_experts)}), cache_size={self.cache_size}, rank={self.rank}"
            )

        if expert_idx >= self.num_experts:
            raise IndexError(
                f"expert_idx={expert_idx} out of range, num_experts={self.num_experts}"
            )

        with torch.no_grad():
            pinned_state_dict = self.experts[expert_idx]
            cached_expert = self.cached_experts[slot_idx]
            for name, param in cached_expert.named_parameters():
                cpu_param = pinned_state_dict[name]
                param.copy_(cpu_param, non_blocking=False)

    def validate_arguments(self):
        assert self.num_experts > 0, "There must be atleast 1 expert to manage."
        assert self.cache_size > 0, "Cache must have room for at least 1 expert."
        assert self.world_size > 0, "world_size must be positive."
        assert self.cache is not None, "cache assignment cannot be None."

    def __call__(self, tokens, expert_mask, schedule=None):
        return self.executor.execute_job(tokens, expert_mask, schedule=schedule)

    def get_statistics(self):
        return self.executor.get_statistics()
