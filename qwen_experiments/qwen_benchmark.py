"""4-GPU independent-replica forward benchmark. NOT expert-parallel DeepSpeed vs DSCA.
Modes: original model, or expert_count_priority prototype. Results must not be labeled DSCA.
"""
import argparse, json, os, time, statistics
from pathlib import Path
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM
from qwen_adapter import patch_model

p = argparse.ArgumentParser()
p.add_argument('--model', default='/home/lcf/MoE_Syn/models/Qwen1.5-MoE-A2.7B')
p.add_argument('--mode', choices=['native','priority_prototype'], required=True)
p.add_argument('--out', required=True)
p.add_argument('--batch-size-per-gpu',type=int,default=4)
p.add_argument('--seq-len',type=int,default=256)
p.add_argument('--warmup',type=int,default=3)
p.add_argument('--steps',type=int,default=20)
a=p.parse_args()
rank=int(os.environ.get('RANK','0')); local=int(os.environ.get('LOCAL_RANK','0')); world=int(os.environ.get('WORLD_SIZE','1'))
if world>1: dist.init_process_group('nccl')
torch.cuda.set_device(local)
model=AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, local_files_only=True).to(f'cuda:{local}').eval()
if a.mode=='priority_prototype': print(f'rank={rank} patched={patch_model(model)}',flush=True)
g=torch.Generator(device=f'cuda:{local}');g.manual_seed(1200)
ids=torch.randint(100,min(10000,model.config.vocab_size),(a.batch_size_per_gpu,a.seq_len),device=f'cuda:{local}',generator=g)
lat=[]
with torch.inference_mode():
    for i in range(a.warmup+a.steps):
        torch.cuda.synchronize(); t0=time.perf_counter()
        model(input_ids=ids,use_cache=False)
        torch.cuda.synchronize(); t1=time.perf_counter()
        if i>=a.warmup: lat.append(t1-t0)
local_stats={'rank':rank,'latencies_s':lat,'elapsed_s':sum(lat)}
all_stats=[None]*world
if world>1: dist.all_gather_object(all_stats,local_stats)
else: all_stats=[local_stats]
if rank==0:
    max_elapsed=max(x['elapsed_s'] for x in all_stats)
    # synchronized, independent inference: total processed samples / slowest rank total compute time
    throughput=world*a.batch_size_per_gpu*a.steps/max_elapsed
    import numpy as np
    # Each step processes a batch; latency here is batch-forward latency, not per-request online latency.
    arr=np.array([v for x in all_stats for v in x['latencies_s']])
    obj={'model':a.model,'mode':a.mode,'parallelism':'4 independent data replicas, NOT expert parallel','world_size':world,'batch_size_per_gpu':a.batch_size_per_gpu,'global_batch_size':world*a.batch_size_per_gpu,'seq_len':a.seq_len,'steps':a.steps,'throughput_samples_s':throughput,'mean_batch_latency_ms':float(arr.mean()*1000),'p95_batch_latency_ms':float(np.percentile(arr,95)*1000),'ranks':all_stats}
    output=Path(a.out);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(obj,indent=2));print(json.dumps({k:v for k,v in obj.items() if k!='ranks'},indent=2))
if world>1: dist.destroy_process_group()
