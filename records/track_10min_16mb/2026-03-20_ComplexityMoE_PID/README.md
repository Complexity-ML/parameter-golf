# Partitioned MoE + PID + BigramHash + SmearGate + SlidingWindow + Int5/Int6

**Author:** Boris Peyriguere (Complexity-ML)
**Date:** 2026-03-20
**Score:** Pending (awaiting compute credits)

## Summary

Architecture combining **hash-partitioned MoE**, **PID dynamics**, and all proven leaderboard techniques from the [Complexity Framework](https://github.com/Complexity-ML/complexity-framework):

1. **Per-Layer Hash Routing** — `hash(layer_id, token_id) = (token_id * 36313) ^ (layer_id * 27191) % E`. Each layer routes tokens to different experts, breaking co-activation bias.

2. **Layer Partitioning** — 3 partitions with different budgets:
   - **P0 (layers 0-2):** Classical GQA, 2 experts, MLP 2x — input features
   - **P1 (layers 3-5):** INL BetaMu, 4 experts, MLP 3x — max capacity
   - **P2 (layers 6-8):** Classical GQA, 4 experts, MLP 2x — output precision

3. **PID Dynamics (INL BetaMu)** — Error-gated causal conv1d with learnable equilibrium `mu`. Layers 3-5 use INL, rest uses classical GQA with RoPE.

4. **BigramHash(10240) + SmearGate** — Hash consecutive token pairs into 10240-bucket embedding table (dim=128), projected to model_dim. SmearGate blends each token with its predecessor.

5. **Sliding Window Eval** (stride=64) — Overlapping windows score only the last `stride` tokens, increasing effective context. Extracted into `eval_sliding.py`.

6. **Int5/Int6 Mixed Quantization** — Int5 for MLP weights (better zstd compression), Int6 for attention (precision-sensitive), FP16 for embeddings. + 3% magnitude pruning before quant.

7. **SWA** (start_frac=0.4) — Stochastic Weight Averaging over the last 40% of warmdown, every 50 steps.

8. **Test-Time Training (LoRA)** — Per-document LoRA adaptation at eval time.

## Architecture

```
Input -> Embedding + BigramHash(10240) -> RMSNorm -> SmearGate -> [Block x 9] -> FinalNorm -> LM Head

Partitions:
  P0 (layers 0-2): GQA attention,  2 experts, MLP 2x   [input]
  P1 (layers 3-5): INL BetaMu,     4 experts, MLP 3x   [middle]
  P2 (layers 6-8): GQA attention,  4 experts, MLP 2x   [output]

Block:
  ResidMix -> Attention -> AttnScale -> MLP(TokenRoutedMoE) -> MLPScale
  U-Net skip connections between encoder/decoder halves

MoE routing per layer:
  expert_id = ((token_id * 36313) ^ (layer_id * 27191)) % num_experts_for_layer
```

## Key Hyperparameters

| Param | Value |
|-------|-------|
| model_dim | 512 |
| num_layers | 9 (4 encoder + 5 decoder) |
| num_heads / kv_heads | 8 / 4 (GQA) |
| train_seq_len | 2048 |
| train_batch_tokens | 786K |
| Muon momentum | 0.99 (warmup 0.92 over 1500 steps) |
| Weight decay | 0.04 |
| Warmdown | 3000 iters |
| Eval stride | 64 (sliding window) |
| Quantization | Int5 MLP + Int6 attn + FP16 embed |
| SWA | start_frac=0.4, every 50 steps |
| Pruning | 3% magnitude before quant |

## Files

| File | Description |
|------|-------------|
| `train_gpt.py` | Training script (1435 lines, under 1500 limit) |
| `eval_sliding.py` | Sliding window eval (standalone or imported) |
| `config.json` | All hyperparameters |
| `i64_moe_kernel.cu` | Optional CUDA kernel for MoE dispatch |
| `submission.json` | Submission metadata |

## How to Run

```bash
# Download data
python3 data/cached_challenge_fineweb.py --variant sp1024

# Train (8xH100)
RUN_ID=partition_v2 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-20_ComplexityMoE_PID/train_gpt.py

# Standalone sliding window eval
MODEL_PATH=final_model.int8.ptz python3 \
  records/track_10min_16mb/2026-03-20_ComplexityMoE_PID/eval_sliding.py
```

## Status

Awaiting RunPod compute credits to produce training logs and final val_bpb score.
