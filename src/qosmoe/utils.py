import dataclasses
import torch.nn as nn

from qosmoe.moe_layer import QoSMoELayer
from qosmoe.config import QoSMoEConfig


def get_tensor_by_path(module, path):
    parts = path.split(".")
    current = module
    for part in parts:
        current = getattr(current, part)
    return current


def replace_qosmoe_layer(
    model,
    moe_parent_type,
    moe_type,
    name_experts,
    router_tensor_path,
    config: QoSMoEConfig,
):
    _replace_qosmoe_layer(
        model,
        moe_parent_type,
        moe_type,
        [0],
        name_experts,
        router_tensor_path,
        config,
    )


def _replace_qosmoe_layer(
    model,
    moe_parent_type,
    moe_type,
    layer_idx,
    name_experts,
    router_tensor_path,
    config: QoSMoEConfig,
):
    if type(model).__name__ == moe_parent_type:
        for child_name, child in model.named_children():
            if type(child).__name__ == moe_type:
                experts = get_tensor_by_path(child, name_experts)

                if isinstance(experts, nn.ModuleDict):
                    experts = list(experts.values())
                elif isinstance(experts, nn.ModuleList):
                    experts = list(experts)

                pinned_experts = []
                for expert in experts:
                    pinned_state_dict = {
                        name: param.detach().cpu().pin_memory()
                        for name, param in expert.state_dict().items()
                    }
                    pinned_experts.append(pinned_state_dict)

                local_config = dataclasses.replace(config)
                local_config.layer_idx = layer_idx[0]
                local_config.experts = pinned_experts
                local_config.expert_example = experts[0]

                router = get_tensor_by_path(child, router_tensor_path)
                local_config.router_weights = router.weight

                new_layer = QoSMoELayer(local_config)
                layer_idx[0] += 1

                setattr(model, child_name, new_layer)
    else:
        for child in model.children():
            _replace_qosmoe_layer(
                child,
                moe_parent_type,
                moe_type,
                layer_idx,
                name_experts,
                router_tensor_path,
                config,
            )


def get_qosmoe_layers(model):
    acc = []
    _get_qosmoe_layers(acc, model)
    return acc


def _get_qosmoe_layers(acc, model):
    for module in model.children():
        if isinstance(module, QoSMoELayer):
            acc.append(module)
        else:
            _get_qosmoe_layers(acc, module)