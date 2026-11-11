from dataclasses import dataclass
from itertools import cycle

import torch


@dataclass
class QoSExecPlan:
    expert_order: list
    slot_idxs: list
    need_loading: list


class QoSPlanner:
    """
    轻量 QoSMoE planner：
    - 保留 token score 计算
    - 只聚合成 expert-level priority
    - 不做 token-level 重排
    """

    def __init__(self, args, rank: int, num_experts: int, num_gpus: int, d_model: int):
        self.args = args
        self.rank = rank
        self.num_experts = num_experts
        self.num_gpus = num_gpus
        self.d_model = d_model

    def _match_token_length(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        x = x.reshape(-1)

        if x.numel() == target_len:
            return x

        if x.numel() == 0:
            return torch.zeros(target_len, device="cuda", dtype=torch.float)

        if target_len % x.numel() == 0:
            repeat_factor = target_len // x.numel()
            return x.repeat_interleave(repeat_factor)

        if x.numel() > target_len:
            return x[:target_len]

        pad = torch.full(
            (target_len - x.numel(),),
            x[-1].item(),
            device=x.device,
            dtype=x.dtype,
        )
        return torch.cat([x, pad], dim=0)

    def _attention_factor(self, router_logits: torch.Tensor) -> torch.Tensor:
        mode = getattr(self.args, "qos_attention_mode", "router_var")
        if mode == "router_var":
            return torch.var(router_logits.float(), dim=-1, unbiased=False)
        return torch.zeros(
            router_logits.shape[0],
            device=router_logits.device,
            dtype=torch.float,
        )

    def compute_token_scores(
        self,
        current_hidden_states: torch.Tensor,
        prev_hidden_states: torch.Tensor | None,
        router_logits: torch.Tensor,
        expert_index: torch.Tensor,
        runtime_context,
    ) -> torch.Tensor:
        """
        p_ij^(l) = P_i * (1 + V_ij^(l))
        这里只算 token score，但后面只聚合成 expert priority。
        """
        if current_hidden_states.dim() > 2:
            current_hidden_states = current_hidden_states.reshape(
                -1, current_hidden_states.shape[-1]
            )
        else:
            current_hidden_states = current_hidden_states.reshape(
                -1, current_hidden_states.shape[-1]
            )

        if prev_hidden_states is not None:
            if prev_hidden_states.dim() > 2:
                prev_hidden_states = prev_hidden_states.reshape(
                    -1, prev_hidden_states.shape[-1]
                )
            else:
                prev_hidden_states = prev_hidden_states.reshape(
                    -1, prev_hidden_states.shape[-1]
                )

        if router_logits.dim() > 2:
            router_logits = router_logits.reshape(-1, router_logits.shape[-1])
        else:
            router_logits = router_logits.reshape(-1, router_logits.shape[-1])

        expert_index = expert_index.reshape(-1)
        token_len = current_hidden_states.shape[0]

        if (
            prev_hidden_states is None
            or prev_hidden_states.shape != current_hidden_states.shape
        ):
            info_gain = torch.norm(current_hidden_states.float(), dim=-1)
        else:
            info_gain = torch.norm(
                (current_hidden_states - prev_hidden_states).float(),
                dim=-1,
            )

        attn_factor = self._attention_factor(router_logits).reshape(-1)
        attn_factor = self._match_token_length(attn_factor, token_len)

        per_expert_count = torch.bincount(
            expert_index,
            minlength=self.num_experts,
        ).float()

        small_batch_penalty = self.args.qos_target_batch / (
            per_expert_count[expert_index] + self.args.qos_epsilon
        )
        small_batch_penalty = self._match_token_length(
            small_batch_penalty, token_len
        )

        flat_remaining_tokens = self._match_token_length(
            runtime_context.flat_remaining_tokens.float(), token_len
        )
        flat_seq_priority = self._match_token_length(
            runtime_context.flat_seq_priority.float(), token_len
        )
        flat_valid_mask = self._match_token_length(
            runtime_context.flat_valid_mask.float(), token_len
        ).bool()

        hidden_cost = torch.full_like(info_gain, float(self.d_model))

        denom = (
            self.args.qos_alpha * hidden_cost
            + self.args.qos_beta * flat_remaining_tokens
            + self.args.qos_delta * small_batch_penalty
        )

        token_value = (info_gain * (1.0 + attn_factor)) / denom
        token_priority = flat_seq_priority * (1.0 + token_value)

        token_priority = token_priority * flat_valid_mask.float()
        token_priority[~flat_valid_mask] = -1e9
        return token_priority

    def aggregate_expert_priority(self, expert_indices, token_scores):
        """
        把 token-level score 聚合成 expert-level priority。
        不做 token-level 重排。
        """
        priority_meta = torch.zeros(
            (self.num_experts,),
            device=token_scores.device,
            dtype=torch.float,
        )

        for expert_idx in range(self.num_experts):
            idx = expert_indices[expert_idx]
            if idx.numel() == 0:
                continue
            priority_meta[expert_idx] = token_scores[idx].sum()

        return priority_meta

    def build_exec_plan(
        self,
        expert_mask,
        global_priority_meta: torch.Tensor,
        local_cache: list,
        cache_size: int,
    ) -> QoSExecPlan:
        """
        只决定 expert 的执行顺序，不决定 token 内部顺序。
        """
        if local_cache is None:
            local_cache = []

        active = [e for e in range(self.num_experts) if expert_mask[e].numel() > 0]
        loaded = [e for e in active if e in local_cache]
        unloaded = [e for e in active if e not in local_cache]

        loaded = sorted(
            loaded,
            key=lambda e: global_priority_meta[e].item(),
            reverse=True,
        )
        unloaded = sorted(
            unloaded,
            key=lambda e: global_priority_meta[e].item(),
            reverse=True,
        )

        if len(unloaded) == 0:
            return QoSExecPlan(
                expert_order=loaded,
                slot_idxs=[local_cache.index(e) for e in loaded],
                need_loading=[],
            )

        all_slots = list(range(cache_size))
        loaded_slots_low_to_high = [
            local_cache.index(e)
            for e in sorted(loaded, key=lambda e: global_priority_meta[e].item())
        ]
        used_slots = list(set(loaded_slots_low_to_high))
        free_slots = [s for s in all_slots if s not in used_slots]

        candidate_slots = free_slots + loaded_slots_low_to_high
        if len(candidate_slots) == 0:
            candidate_slots = [0]

        slot_cycle = cycle(candidate_slots)

        expert_order = loaded + unloaded
        slot_idxs = []
        need_loading = []

        for expert_idx in expert_order:
            if expert_idx in local_cache:
                slot_idxs.append(local_cache.index(expert_idx))
            else:
                slot = next(slot_cycle)
                slot_idxs.append(slot)
                need_loading.append(expert_idx)

        return QoSExecPlan(
            expert_order=expert_order,
            slot_idxs=slot_idxs,
            need_loading=need_loading,
        )
