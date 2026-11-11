# Qwen 4 GPU EP Prototype Results

**This is NOT the DeepSpeed vs DSCA-MoE comparison.** Both methods use custom torch.distributed expert parallel; priority is only an expert-order prototype.

| Mode | Throughput (samples/s) | Mean (ms) | P95 (ms) |
|---|---:|---:|---:|
| Torch-EP-FIFO | 119.63 | 133.73 | 138.06 |
| Torch-EP-Priority-Prototype | 112.10 | 142.72 | 152.24 |

Priority vs FIFO throughput change: -6.30%

The model-wide semantic equivalence has not yet been checked. The included tiny-layer distributed test is necessary but insufficient for publication-grade validation.
