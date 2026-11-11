from dataclasses import dataclass


@dataclass
class QoSMoEConfig:
    rank: int
    world_size: int

    experts: list = None
    expert_example: object = None
    router_weights: object = None

    layer_idx: int = 0
    d_model: int = 768
    num_experts: int = 8
    model_name: str = ""

    cache_policy: str = "RAND"
    expert_cache_size: int = 2
    eq_tokens: int = 150
    expert_placement: str | None = None
    fetching_strategy: str = "async-cpu"

    enable_skew: bool = False
    enable_random: bool = False
    enable_uniform: bool = False
    skew: float = 0.0
    num_experts_skewed: int = 1

    qos_w1: float = 1.0
    qos_w2: float = 0.1
    qos_w3: float = 0.5

    qos_alpha: float = 1.0
    qos_beta: float = 0.1
    qos_delta: float = 0.5

    qos_epsilon: float = 1e-6
    qos_avg_token_cost: float = 0.005
    qos_default_deadline_s: float = 2.0

    qos_phase_prefill: float = 0.5
    qos_phase_decode: float = 1.0

    qos_target_batch: int = 64
    qos_attention_mode: str = "router_var"
    qos_budget_tokens_per_gpu: int = 0

    qos_enable_prefetch: bool = False
    qos_enable_async: bool = True
    qos_enable_load_balance: bool = True
    qos_prefetch_topk: int = 2
    qos_prefetch_budget_experts: int = 1
    qos_prefetch_score_power: float = 1.0
