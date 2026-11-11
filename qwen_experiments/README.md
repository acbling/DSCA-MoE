# Qwen correctness / scheduling prototype (not the final 2×2 benchmark)

Place these files inside `/home/lcf/MoE_Syn/HarMoEny-main/qwen_experiments/` (new files only, no deletions).

```
conda activate qwen-moe
cd /home/lcf/MoE_Syn/HarMoEny-main/qwen_experiments
CUDA_VISIBLE_DEVICES=0 python qwen_validate.py
bash run_qwen_prototype.sh
```

Important distinction: Qwen native and priority_prototype use 4 *independent data replicas*, and the prototype only reorders local expert execution. They are **NOT** DeepSpeed-MoE vs DSCA-MoE, not cross-GPU expert parallel, not online request serving. Never attribute any performance delta to DSCA-MoE. The existing Switch code remains the valid, separate experimental baseline. Implementing correct Qwen Top-4 EP requires all-to-all dispatch/return, weighted combine, and shared experts plus QoSPlanner/ExpertManager integration. Validate logits before performance claims.
