# Token-Routed MoE + Mu-Guidance + Shared Expert + Zipf Routing

**Author:** Boris Peyriguere (Complexity-ML)
**Date:** 2026-03-29
**Score:** Pending

---

## Summary

Novel architecture combining **deterministic Token-Routed MoE** with **Mu-Guidance** (inter-layer equilibrium signal) and a **Shared Lexical Expert** for maximum parameter efficiency under the 16MB constraint.

### Key innovations

- **Mu-Guidance** -- learnable equilibrium signal flowing layer-to-layer
  Each layer produces `mu_current = clamp(mu_param + mu_proj(h), -2, 2)` which biases the next layer's Q/K/V projections. Provides top-down context without recurrence cost. Ablation shows removing Mu degrades loss below dense baseline.

- **Zipf-balanced deterministic routing** -- no learned router, no load balancing loss
  Tokens are assigned to experts via greedy bin-packing on corpus frequency (Zipf distribution). Each expert gets equal token load by construction. Zero overhead: `expert_id = token_to_expert[token_id]`, fully compatible with CUDA graph capture.

- **Shared Lexical Expert** -- dense SwiGLU MLP all tokens pass through
  Captures common patterns (function words, syntax) while routed experts specialize. Output = shared(x) + expert(x).

- **BigramHash(10240) + SmearGate** -- hash consecutive token pairs into learned embeddings + previous-token gating.

- **Int5/Int6 mixed quantization** -- Int5 for MLP, Int6 for attention, FP16 for tied embeddings. 3% magnitude pruning before zstd-22 compression.

- **Sliding window eval** (stride=64, window=2048)

---

## Architecture

```
Embedding + BigramHash(10240) --> RMSNorm --> SmearGate --> [Block x 9] --> FinalNorm --> LM Head

Each block:
  1. RMSNorm --> GQA Attention (12h/4kv, QK-Norm, RoPE)
     + Mu-Guided Q/K/V bias from previous layer
  2. RMSNorm --> Token-Routed MLP (4 experts SwiGLU) + Shared Expert
     Routing: Zipf bin-packing (deterministic, no router)
  3. Mu-Guidance: mu = clamp(mu_param + mu_proj(h), -2, 2)
     --> flows to next layer

mu_init (learnable) --> layer 0 --> mu_0 --> layer 1 --> ... --> layer 8
```

### Specs

| Component | Value |
|-----------|-------|
| Model dim | 512 |
| Layers | 9 |
| Attention | GQA (8 heads, 4 KV heads) |
| MLP | Token-Routed, 4 experts SwiGLU |
| Shared expert | Yes (same size as one expert) |
| Routing | Zipf bin-packing (deterministic) |
| Mu-Guidance | Yes (clamp [-2, 2]) |
| Vocab | 1024 (SentencePiece) |
| Tied embeddings | Yes (FP16) |
| Quantization | Int5 MLP / Int6 Attention |

---

## Ablation evidence (from TMLR submission)

On 170-187M models trained on 500M tokens FineWeb-Edu:

| Configuration | Avg Loss |
|---------------|----------|
| Dense SwiGLU (171M) | 4.905 |
| TR + Shared + Mu + Zipf (187M) | **4.793** |
| TR + Shared + Zipf, no Mu (187M) | 4.916 |

Mu-Guidance is the key component: without it, Token-Routed is worse than dense.

---

## Files

| File | Description |
|------|-------------|
| `train_gpt.py` | Training script |
| `submission.json` | Submission metadata |

---

## How to Run

```bash
# Download data
python3 data/cached_challenge_fineweb.py --variant sp1024

# Train (8xH100)
RUN_ID=complexity_mu torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-29_ComplexityMoE_MuGuidance/train_gpt.py
```

---

## Status

Awaiting compute credits to produce training logs and final val_bpb score.

## References

- Framework: https://github.com/Complexity-ML/complexity-framework
- vLLM integration: https://github.com/Complexity-ML/vllm-cuda_graph
- Paper: https://github.com/Complexity-ML/tmlr-paper-pool
