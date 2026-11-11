# DSCA-MoE

DSCA-MoE is a runtime scheduling implementation for
multi-GPU Mixture-of-Experts (MoE) inference.

The repository includes:
- DSCA-MoE expert scheduling and runtime modules
- DeepSpeed-MoE baseline adaptation
- Switch-base-64 and Qwen1.5-MoE-A2.7B benchmark interfaces
- Reproducibility scripts and environment specifications

## Project Structure

- `src/start_qosmoe.py`: DSCA-MoE entry point
- `src/start_deepspeed.py`: DeepSpeed-MoE entry point
- `src/run_all_moe_benchmarks.py`: Benchmark runner
- `src/qosmoe/`: DSCA-MoE scheduling and expert execution
- `src/harmonymoe/`: Supporting components
- `qwen_ep_adapter/`: Qwen compatibility components
- `src/flexible_dataset.py`: Dataset loading
- `compare_third_party_logs_deepspeed.py`: Result comparison

## Environment

The original experiments used two Conda environments:

- `harmony-clean` for Switch-base-64
- `qwen-moe` for Qwen1.5-MoE-A2.7B

Environment specifications are in `envs/`.

Example:

```bash
conda env create -f envs/qwen-moe.yml
conda activate qwen-moe
```

## Reproduction

The original four-GPU benchmark configuration used:

- Four NVIDIA A100 GPUs
- Batch size: 16 per GPU
- Sequence length: 256
- Number of samples: 512
- Warm-up rounds: 1
- Dataset: WikiText

The model weights and dataset cache are not included.

For Qwen1.5-MoE-A2.7B:

```bash
python src/run_all_moe_benchmarks.py \
  --systems qosmoe \
  --model_name /path/to/Qwen1.5-MoE-A2.7B \
  --world_size 4 \
  --cuda_visible_devices 0,1,2,3 \
  --batch_size 16 \
  --seq_len 256 \
  --num_samples 512 \
  --warmup_rounds 1 \
  --num_experts 60 \
  --d_model 2048 \
  --type_moe_parent Qwen2MoeDecoderLayer \
  --type_moe Qwen2MoeSparseMoeBlock \
  --router_tensor_path gate \
  --name_experts experts \
  --dataset wikitext \
  --project_root "$PWD" \
  --outputs_root "$PWD/outputs/qwen_dsca" \
  --tag qwen_dsca
```

For the DeepSpeed baseline, change:
`--systems qosmoe` to `--systems deepspeed`
and use a separate output directory and tag.

The DeepSpeed Qwen path includes a compatibility
adaptation for the original model's Top-4 routed
experts and shared expert.

## Acknowledgement

This implementation is developed from and extends
components of HarMoEny:

https://github.com/sacs-epfl/HarMoEny

Third-party copyright and licensing notices must
be respected before public redistribution.
