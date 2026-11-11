"""Compare the original and reordered Qwen expert execution on the same inputs."""
import argparse
import torch
from transformers import AutoModelForCausalLM
from qwen_adapter import patch_model

p = argparse.ArgumentParser()
p.add_argument('--model', default='/home/lcf/MoE_Syn/models/Qwen1.5-MoE-A2.7B')
p.add_argument('--seq-len', type=int, default=32)
a = p.parse_args()
torch.manual_seed(42)
assert torch.cuda.is_available()
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, local_files_only=True).to('cuda').eval()
ids = torch.randint(100, min(10000,model.config.vocab_size), (1,a.seq_len),device='cuda')
with torch.inference_mode():
    ref = model(input_ids=ids, use_cache=False).logits.float().cpu()
count = patch_model(model)
with torch.inference_mode():
    new = model(input_ids=ids, use_cache=False).logits.float().cpu()
error = (ref-new).abs()
print(f'patched_layers={count} max_abs_error={error.max().item():.6g} mean_abs_error={error.mean().item():.6g}')
# BF16 computational reordering may differ slightly. Error metrics always reported, not concealed.
print('Validation completed. Inspect error before benchmarking.')
