import inspect

from transformers.models.qwen2_moe.modeling_qwen2_moe import (
    Qwen2MoeSparseMoeBlock
)

print("=" * 70)
print("Qwen MoE module")
print("=" * 70)

print("Class:", Qwen2MoeSparseMoeBlock.__name__)

print("\nForward signature:")
print(inspect.signature(Qwen2MoeSparseMoeBlock.forward))

print("\nForward implementation:")
print(inspect.getsource(Qwen2MoeSparseMoeBlock.forward))
