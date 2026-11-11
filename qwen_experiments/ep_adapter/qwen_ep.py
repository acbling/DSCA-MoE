#!/usr/bin/env python3
"""Qwen2-MoE 4-way *real* expert parallel proof-of-concept.

Replicates dense/shared components and shards routed experts. Routed token
assignments travel between ranks via NCCL all_to_all_single. No synthetic
latencies or emulated EP. The 'priority' mode is a research prototype, NOT
an implementation or reproduction of the project's original DSCA-MoE.
"""
import argparse, csv, gc, json, os, statistics, time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def a2a(x, send_counts, recv_counts):
    shape=(sum(recv_counts),)+tuple(x.shape[1:])
    y=torch.empty(shape, dtype=x.dtype, device=x.device)
    dist.all_to_all_single(y, x.contiguous(), output_split_sizes=recv_counts, input_split_sizes=send_counts)
    return y

class ShardedQwenMoE(nn.Module):
    def __init__(self, old, rank, world, policy):
        super().__init__()
        assert old.num_experts % world == 0
        assert old.top_k == 4, 'This adapter is specific to the Qwen top-4 model'
        self.rank=rank; self.world=world; self.policy=policy
        self.num_experts=old.num_experts; self.top_k=old.top_k
        self.norm_topk_prob=old.norm_topk_prob
        self.per_rank=self.num_experts//world
        self.gate=old.gate
        self.shared_expert=old.shared_expert
        self.shared_expert_gate=old.shared_expert_gate
        self.experts=nn.ModuleDict()
        for eid, expert in enumerate(old.experts):
            if eid//self.per_rank == rank:
                self.experts[str(eid)] = expert
        self.route_calls=0
        self.remote_assignments=0

    @torch.no_grad()
    def forward(self, hidden_states):
        b,s,d=hidden_states.shape
        flat=hidden_states.reshape(-1,d)
        logits=self.gate(flat)
        probs=F.softmax(logits,dim=1,dtype=torch.float)
        weights, choices=torch.topk(probs,self.top_k,dim=-1)
        if self.norm_topk_prob:
            weights=weights/weights.sum(dim=-1,keepdim=True)
        weights=weights.to(flat.dtype)
        # Each (token, route-slot) constitutes one expert job.
        ids=choices.flatten(); w=weights.flatten()
        src_tok=torch.arange(flat.shape[0],device=flat.device).repeat_interleave(self.top_k)
        owner=torch.div(ids,self.per_rank,rounding_mode='floor')
        # All-to-all messages sorted by destination rank (with optional locally
        # prioritized ties; this changes message order, not numerical semantics).
        if self.policy=='priority':
            # Prioritize experts with larger active queues, a prototype policy.
            demand=torch.bincount(ids,minlength=self.num_experts)
            score=demand[ids]
            # Stable sort on score within destination, without CPU sync.
            ix_score=torch.argsort(score,descending=True,stable=True)
            ix=ix_score[torch.argsort(owner[ix_score],stable=True)]
        else:
            ix=torch.argsort(owner,stable=True)
        send_counts=torch.bincount(owner,minlength=self.world).cpu().tolist()
        recv_counts=[0]*self.world
        send_num=torch.tensor(send_counts,dtype=torch.int64,device=flat.device)
        recv_num=torch.empty_like(send_num)
        dist.all_to_all_single(recv_num,send_num)
        recv_counts=recv_num.cpu().tolist()
        dispatched=a2a(flat.index_select(0,src_tok[ix]),send_counts,recv_counts)
        recv_ids=a2a(ids[ix].to(torch.int64),send_counts,recv_counts)
        answer=torch.empty_like(dispatched)
        # Expert compute only on locally owned experts.
        start=self.rank*self.per_rank
        local_counts=torch.bincount(recv_ids-start,minlength=self.per_rank)
        if self.policy=='priority':
            order=torch.argsort(local_counts,descending=True,stable=True).tolist()
        else:
            order=range(self.per_rank)
        for local in order:
            mask=(recv_ids==start+local).nonzero(as_tuple=True)[0]
            if mask.numel():
                z=dispatched.index_select(0,mask)
                answer.index_copy_(0,mask,self.experts[str(start+local)](z))
        # Return result to the original rank in the exact transmitted order.
        returned=a2a(answer,recv_counts,send_counts)
        outputs=torch.zeros_like(flat)
        outputs.index_add_(0,src_tok[ix],returned*w[ix,None])
        outputs=outputs+self.shared_expert(flat)*torch.sigmoid(self.shared_expert_gate(flat))
        self.route_calls+=1
        self.remote_assignments+=int((owner!=self.rank).sum().item())
        return outputs.reshape(b,s,d),logits


def patch(model, rank, world, policy):
    layers=model.model.layers
    changed=0
    for layer in layers:
        block=layer.mlp
        if block.__class__.__name__=='Qwen2MoeSparseMoeBlock':
            layer.mlp=ShardedQwenMoE(block,rank,world,policy)
            changed+=1
    if not changed:raise RuntimeError('No Qwen2MoeSparseMoeBlock found: wrong checkpoint/model version')
    return changed


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='/home/lcf/MoE_Syn/models/Qwen1.5-MoE-A2.7B')
    ap.add_argument('--policy',choices=['fifo','priority'],default='fifo')
    ap.add_argument('--seq-len',type=int,default=256)
    ap.add_argument('--batch-per-rank',type=int,default=4)
    ap.add_argument('--iterations',type=int,default=8)
    ap.add_argument('--warmup',type=int,default=2)
    ap.add_argument('--result',required=True)
    ap.add_argument('--validation',action='store_true')
    a=ap.parse_args()
    dist.init_process_group('nccl')
    rank=dist.get_rank();world=dist.get_world_size()
    assert world==4,'Expected four distributed ranks'
    torch.cuda.set_device(rank)
    os.environ['TOKENIZERS_PARALLELISM']='false'
    # Sequential checkpoint loading prevents four simultaneous full-checkpoint
    # CPU load peaks. System still needs sufficient RAM for each rank's live shard.
    model=None
    for r in range(world):
        if rank==r:
            print(f'[Rank {rank}] loading checkpoint on CPU',flush=True)
            model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map='cpu',low_cpu_mem_usage=True,local_files_only=True)
            model.eval()
            changed=patch(model,rank,world,a.policy)
            print(f'[Rank {rank}] patched {changed} MoE blocks',flush=True)
            model.to(f'cuda:{rank}')
            gc.collect()
        dist.barrier()
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    # Deterministic identical input across ranks for distributed-output validation.
    torch.manual_seed(2026)
    vocab=model.config.vocab_size
    ids=torch.randint(10,vocab,(a.batch_per_rank,a.seq_len),dtype=torch.long,device=f'cuda:{rank}')
    mask=torch.ones_like(ids)
    # Each rank serves its own batch. Routing communication does cross GPUs.
    with torch.inference_mode():
        for _ in range(a.warmup):
            _=model(input_ids=ids,attention_mask=mask,use_cache=False)
        torch.cuda.synchronize()
        times=[]
        dist.barrier()
        t0=time.perf_counter()
        for _ in range(a.iterations):
            t1=time.perf_counter()
            _=model(input_ids=ids,attention_mask=mask,use_cache=False)
            torch.cuda.synchronize()
            times.append(time.perf_counter()-t1)
        dist.barrier()
        ttotal=time.perf_counter()-t0
    gathered=[None]*world
    dist.all_gather_object(gathered, {'rank':rank,'latencies':times,'elapsed':ttotal,
        'moe_calls':sum(x.mlp.route_calls for x in model.model.layers if isinstance(x.mlp,ShardedQwenMoE))})
    if rank==0:
        # True aggregate rate: total items processed across all ranks / longest elapsed.
        flat=[v for item in gathered for v in item['latencies']]
        vals=sorted(flat)
        p95=vals[max(0,__import__('math').ceil(len(vals)*.95)-1)]
        result={'model':'Qwen1.5-MoE-A2.7B','method': 'Torch-EP-FIFO' if a.policy=='fifo' else 'Torch-EP-Priority-Prototype',
           'algorithm_validated_as_original_DSCAMoE':False,'expert_parallel':True,'top_k':4,
           'shared_expert_preserved':True,'world_size':world,'batch_per_gpu':a.batch_per_rank,
           'seq_len':a.seq_len,'iterations':a.iterations,'latency_definition':'per-rank batched forward',
           'mean_ms':statistics.mean(flat)*1000,'p95_ms':p95*1000,
           'throughput':a.iterations*a.batch_per_rank*world/max(x['elapsed'] for x in gathered),
           'validation_status':'NOT_YET_CHECKED_AGAINST_UNSHARDED_REFERENCE',
           'moe_calls':[x['moe_calls'] for x in gathered]}
        p=Path(a.result);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2),flush=True)
    dist.destroy_process_group()

if __name__=='__main__':main()
