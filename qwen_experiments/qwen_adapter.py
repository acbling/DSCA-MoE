"""Correctness-preserving Qwen2-MoE per-expert execution ordering prototype.
NOT multi-GPU expert-parallel DSCA-MoE.
"""
import torch
import torch.nn.functional as F


def make_ordered_forward(module, mode="count_desc"):
    if mode not in ("count_desc", "expert_id"):
        raise ValueError(mode)

    def forward(hidden_states):
        b, s, d = hidden_states.shape
        flat = hidden_states.reshape(-1, d)
        logits = module.gate(flat)
        prob = F.softmax(logits, dim=-1, dtype=torch.float32)
        weights, indices = torch.topk(prob, module.top_k, dim=-1)
        if module.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights.to(flat.dtype)
        count = torch.bincount(indices.flatten(), minlength=module.num_experts)
        if mode == "count_desc":
            ids = sorted(range(module.num_experts), key=lambda i: (-int(count[i]), i))
        else:
            ids = range(module.num_experts)
        result = torch.zeros_like(flat)
        for expert_id in ids:
            if not count[expert_id]:
                continue
            token_pos, slot = torch.where(indices == expert_id)
            expert_values = module.experts[expert_id](flat.index_select(0, token_pos))
            result.index_add_(0, token_pos, expert_values * weights[token_pos, slot, None])
        shared = torch.sigmoid(module.shared_expert_gate(flat)) * module.shared_expert(flat)
        result = result + shared
        return result.reshape(b, s, d), logits
    return forward


def patch_model(model, mode="count_desc"):
    n = 0
    for m in model.modules():
        if type(m).__name__ == "Qwen2MoeSparseMoeBlock":
            m.forward = make_ordered_forward(m, mode)
            n += 1
    if n == 0:
        raise RuntimeError("No Qwen2MoeSparseMoeBlock found")
    return n
