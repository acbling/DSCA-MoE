import copy
import torch
import torch.nn as nn

from qosmoe.async_cpu import QoSAsyncCPU, QoSExecutorConfig


class QoSExpertManager:
    """
    QoSMoE 自己的 ExpertManager
    - 不修改 harmonymoe/*
    - 当前只支持 async-cpu
    """

    def __init__(
        self,
        rank,
        world_size,
        experts: list,
        expert_example: nn.Module,
        cache_size: int,
        cache=None,
        fetching_strategy="async-cpu",
        layer_idx=-1,
    ):
        self.experts = experts
        self.num_experts = len(experts)
        self.cache = cache
        self.cache_size = cache_size
        self.layer_idx = layer_idx
        self.rank = rank
        self.world_size = world_size
        self.fetching_strategy = fetching_strategy
        self._prefetch_cursor = 0

        self.validate_arguments()

        full_local_cache = list(self.cache[self.rank]) if self.cache is not None else []
        self.local_cache = list(full_local_cache[: self.cache_size])

        self.cached_experts = [
            copy.deepcopy(expert_example).cuda() for _ in range(self.cache_size)
        ]

        for slot_idx, expert_idx in enumerate(self.local_cache):
            self.load_expert_into_slot(expert_idx, slot_idx)

        normalized_cache = []
        for i in range(self.world_size):
            normalized_cache.append(list(self.cache[i][: self.cache_size]))

        config = QoSExecutorConfig(
            rank=self.rank,
            world_size=self.world_size,
            experts=self.experts,
            cache=normalized_cache,
            cached_experts=self.cached_experts,
            cache_size=self.cache_size,
            num_experts=self.num_experts,
            layer_idx=self.layer_idx,
        )

        if fetching_strategy == "async-cpu":
            self.executor = QoSAsyncCPU(config)
        else:
            raise NotImplementedError(
                f"QoSMoE currently only supports fetching_strategy='async-cpu', got: {fetching_strategy}"
            )

    def load_expert_into_slot(self, expert_idx, slot_idx):
        with torch.no_grad():
            pinned_state_dict = self.experts[expert_idx]
            cached_expert = self.cached_experts[slot_idx]
            for name, param in cached_expert.named_parameters():
                cpu_param = pinned_state_dict[name]
                param.copy_(cpu_param, non_blocking=False)

    def validate_arguments(self):
        assert self.num_experts > 0, "There must be at least 1 expert to manage."
        assert self.cache_size > 0, "Cache must have room for at least 1 expert."
        assert self.cache is not None, "QoSMoE ExpertManager requires cache assignment."

    def __call__(self, tokens, expert_mask, schedule=None):
        return self.executor.execute_job(tokens, expert_mask, schedule=schedule)

    def get_statistics(self):
        return self.executor.get_statistics()

    def peek_local_cache(self):
        if hasattr(self.executor, "local_cache"):
            return list(self.executor.local_cache)
        return list(self.local_cache)

    def prefetch_experts(self, expert_ids):
        """
        最小版预取：
        - 已在 cache 中的 expert 跳过
        - round-robin 选 slot
        - 优先复用 async CPU executor 的 load_expert_into_slot
        """
        launched = []
        local_cache = self.peek_local_cache()
        local_cache_set = set([e for e in local_cache if e >= 0])

        for expert_idx in expert_ids:
            expert_idx = int(expert_idx)

            if expert_idx < 0 or expert_idx >= self.num_experts:
                continue
            if expert_idx in local_cache_set:
                continue

            slot_idx = self._prefetch_cursor % self.cache_size
            self._prefetch_cursor = (self._prefetch_cursor + 1) % self.cache_size

            if hasattr(self.executor, "load_expert_into_slot"):
                self.executor.load_expert_into_slot(expert_idx, slot_idx)
            else:
                self.load_expert_into_slot(expert_idx, slot_idx)

            # 本地视图也同步一下，避免重复预取
            if slot_idx < len(self.local_cache):
                self.local_cache[slot_idx] = expert_idx

            launched.append((expert_idx, slot_idx))
            local_cache_set.add(expert_idx)

        return launched