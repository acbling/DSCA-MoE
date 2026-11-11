#!/usr/bin/env python3
"""Compare real 4-rank EP output to HF native top-4 MoE on a tiny test layer."""
import copy
import os
import sys
import torch
import torch.distributed as dist
from transformers import AutoConfig
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
from qwen_ep import ShardedQwenMoE


def main():
    dist.init_process_group('nccl')
    rank=dist.get_rank();world=dist.get_world_size()
    if world!=4:raise RuntimeError('run on exactly four GPUs')
    torch.cuda.set_device(rank)
    torch.manual_seed(20261016)
    cfg=AutoConfig.from_pretrained(os.environ.get('QWEN_MODEL','/home/lcf/MoE_Syn/models/Qwen1.5-MoE-A2.7B'),local_files_only=True)
    cfg.hidden_size=64;cfg.moe_intermediate_size=32
    cfg.shared_expert_intermediate_size=64
    cfg.num_experts=8;cfg.num_experts_per_tok=4
    ref=Qwen2MoeSparseMoeBlock(cfg).to(f'cuda:{rank}').eval()
    ours=ShardedQwenMoE(copy.deepcopy(ref),rank,world,'fifo').to(f'cuda:{rank}').eval()
    x=torch.randn(2,8,64,device=f'cuda:{rank}')
    with torch.inference_mode():
        gold,rg=ref(x)
        actual,ag=ours(x)
    diff=(gold-actual).abs()
    stats=torch.tensor([diff.max().item(),diff.mean().item()],device=x.device)
    dist.all_reduce(stats,op=dist.ReduceOp.MAX)
    if rank==0:print('MAX_ERR:',stats[0].item(),'MAX_MEAN_ERR:',stats[1].item(),flush=True)
    valid=torch.allclose(gold,actual,atol=3e-3,rtol=3e-3) and torch.equal(rg,ag)
    okay=torch.tensor(int(valid),device=x.device)
    dist.all_reduce(okay,op=dist.ReduceOp.MIN)
    if rank==0:print('QWEN_TOP4_EP_VALIDATION_'+('PASS' if okay.item()==1 else 'FAIL'),flush=True)
    dist.destroy_process_group()
    if okay.item()!=1:sys.exit(1)
if __name__=='__main__':main()
