import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home/lcf/MoE_Syn/models/Qwen1.5-MoE-A2.7B"

print("[1/4] Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("[2/4] Loading Qwen model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
    low_cpu_mem_usage=True
)
model.eval()

print("[3/4] Running inference...", flush=True)

inputs = tokenizer(
    "The main advantage of mixture-of-experts models is",
    return_tensors="pt"
).to(next(model.parameters()).device)

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False
    )

print("[4/4] Inference completed", flush=True)
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("QWEN_FORWARD_SUCCESS")
