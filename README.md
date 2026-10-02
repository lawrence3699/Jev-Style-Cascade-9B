---
license: apache-2.0
library_name: jev-style
base_model:
- chaoliangUNSW/Jev-Style-2B-Decision-v3
- alibiserikbay/JevK5-9B
tags:
- decision-model
- typed-decisions
- system-one
- cascade
- calibration
- jev-style
language:
- en
pipeline_tag: text-classification
---

# Jev-Style-Cascade-9B

**One `/v1/systemone` model built from two:** [Jev-Style-2B-Decision-v3](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3) answers every question first, and only the questions it is not confident about go to [JevK5-9B](https://huggingface.co/alibiserikbay/JevK5-9B) (by alibiserikbay, Apache-2.0). This repository holds no new weights: it is the frozen cascade (`cascade.json`), how its threshold was chosen, and its evaluation. Both models are downloaded from their own repositories at pinned revisions.

| | Jev-Style-Cascade-9B |
|---|---|
| JevBench public items (231) | **87.9 %** (203 / 231) — Jev 1.13.0's published figure on the same items is 86.6 % |
| Questions the 2B settles on its own | **53 %** of JevBench items, **43 %** of our 4,992-question calibration set |
| Calibration error on JevBench (ECE) | **0.026** |
| Decision Index 0.2.1 (offline forecast) | **46.08**, from 34.83 for the 2B alone |
| Latency, RTX 5090 (median per request) | **29 ms** on JevBench items, **38 ms** on the calibration set |
| Context | up to 25,600 tokens per request (tested to 25,275) |

<sub>JevBench: harness github.com/fstandhartinger/jevbench @ bb05a335 (v1.4.2.2 + 5 commits), the 231 public items, `typesafe` adapter against our server, one run on 2026-10-02 (UTC). This is our own run on the public items only, not a board row; the board's ranking also uses sealed items. The Jev 1.13.0 figure is `public_accuracy` 0.8658 in the harness's results/v1.4.2/jevbench-v1.4.2-results.json. Decision Index: see [Decision Index](#decision-index-offline-forecast).</sub>

## Quick start

It needs one NVIDIA GPU with 32 GB (both models stay resident; see [Memory](#memory)), PyTorch, [`jev-style`](https://github.com/lawrence3699/jev-style) 0.4.0 or later, and JevK5's runtime [allebee/jevk5](https://github.com/allebee/jevk5), which is not on PyPI:

```bash
pip install "jev-style[torch]>=0.4.0" "jevk5[fast] @ git+https://github.com/allebee/jevk5@v0.3.3"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
jev-style download --cascade cascade-9b          # both tiers; JevK5-9B is checked against its SHA256SUMS
jev-style serve --cascade cascade-9b --backend torch --device cuda
```

```bash
curl -s localhost:8765/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice for my order, please refund one.",
  "questions": {
    "team":   {"type": "choice", "instructions": "Which team should handle this?",
               "criteria": {"billing": "payments and refunds", "tech": "bugs and outages"}},
    "refund": {"type": "noul", "instructions": "The customer asks for money back."}}}'
```

From Python: `from jev_style import JevStyle; JevStyle(cascade="cascade-9b").decide(state, questions)`. Every answer has the usual systemone shape plus `tier` (1 = the 2B, 2 = JevK5-9B) and `tier_model`; `timing.tiers` reports what each tier did.

## How it works

1. The 2B (PyTorch float32, [CUDA graphs](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3#cuda-graphs-6-faster-short-calls-on-cuda)) reads the state once and answers every question in one call.
2. A question's confidence is `(k · p_max − 1) / (k − 1)` for k options (`|2 · P(true) − 1|` for true/false). Questions at **τ = 0.75** or above keep the 2B's answer.
3. The rest go to JevK5-9B in one call with the same state; its answer is final. If the 2B fails on a question (for example an input it cannot take), JevK5-9B answers it; if JevK5-9B fails, the 2B's answer is used.

JevK5-9B runs in the same process through its own runtime (the code `jevk5-serve` uses, at v0.3.3), with its calibration temperatures from `jevk5_config.json`; the cascade refuses to load if that file is missing, since JevK5 would otherwise run uncalibrated.

## How τ was chosen

The rule was written down before any result existed ([validation/PREDECLARATION.en.md](validation/PREDECLARATION.en.md); the binding Chinese original and its sha256 are next to it):

- **Data:** a 4,992-question calibration set from 29 public datasets (human or gold labels, about a quarter non-English, states up to 12K tokens). It was de-duplicated against Decision Index and JevBench, and contains nothing that either model was trained on according to their cards. The source list and licences are in [validation/calibration_manifest.json](validation/calibration_manifest.json).
- **Constraint:** pick the τ with the lowest expected latency whose calibration accuracy stays within 1.0 point of JevK5-9B alone.

| Calibration set (4,992 questions) | Accuracy |
|---|---:|
| Jev-Style-Cascade-9B (τ = 0.75) | 73.1 % |
| JevK5-9B alone | 74.1 % |
| Jev-Style-2B-Decision-v3 alone | 66.2 % |

The 2B settled 42.8 % of the calibration questions; JevK5-9B was called for 55.5 % of the requests. The full search output is [validation/freeze_result.json](validation/freeze_result.json). An earlier run without the 2B's CUDA-graph runtime failed the pre-declared speed gate and is kept as [validation/freeze_result_first_run_without_cuda_graphs.json](validation/freeze_result_first_run_without_cuda_graphs.json).

## Results

### JevBench public items

| | All (231) | Easy (48) | Standard (72) | Hard (111) | ECE | p50 latency |
|---|---:|---:|---:|---:|---:|---:|
| **Jev-Style-Cascade-9B** | **203** | 48 | 70 | 85 | **0.026** | 29.0 ms |
| JevK5-9B alone (our run) | 203 | 48 | 69 | 86 | 0.034 | 28.1 ms |
| Jev-Style-2B-Decision-v3 alone (CUDA graphs) | 170 | 48 | 69 | 53 | 0.062 | 15.5 ms |

The 2B answered 123 of the 231 items itself. Per-item results are in [validation/jevbench/](validation/jevbench/).

### Decision Index (offline forecast)

**46.08** on Decision Index 0.2.1, against **34.83** for the 2B alone. This is a forecast, not a board entry: the cascade's answers were composed at the frozen τ from two public per-row result sets — our 2B's run ([chaoliangUNSW/decision-index-results](https://huggingface.co/datasets/chaoliangUNSW/decision-index-results)) and JevK5-9B v0.3.3's run ([alibiserikbay/decision-index-results](https://huggingface.co/datasets/alibiserikbay/decision-index-results)) — and scored with the official kit (apolinario/decision-index @ 87d4650) on the full rebuilt suite. Composing the same way reproduced the live cascade exactly on all 231 JevBench items. Scores: [validation/decision_index_0.2.1_offline_forecast.scores.json](validation/decision_index_0.2.1_offline_forecast.scores.json).

### Latency

| RTX 5090, in-process, one request at a time | median | mean | p95 |
|---|---:|---:|---:|
| Calibration set (1,000 requests, up to 12K tokens) | 37.5 ms | 139.0 ms | 673.6 ms |
| JevBench public items (231, one question each) | 29.0 ms | | 477.4 ms |

<sub>torch 2.14.1+cu130, transformers 5.18.0, flash-linear-attention 0.5.2, jevk5 0.3.3; the models are loaded and their CUDA graphs recorded before timing (start-up 47 s with the weights cached, including the SHA256SUMS check of JevK5-9B). A 25,275-token request with two questions took 2.4 s.</sub>

## Memory

On the RTX 5090 (32 GB) the cascade used 29.7 GB after start-up and at most 31.8 GB while serving the calibration set, a 25,275-token request included, with no out-of-memory errors. Keep the two models in one process (as `jev-style serve --cascade cascade-9b` does) and set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; a smaller GPU will not hold both models.

## Scope and notes

- JevK5-9B is English-focused; about a quarter of the calibration set was non-English.
- JevK5-9B's model card says part of its training labels were generated by OpenAI's GPT-6 Luna under OpenAI's terms; check that those terms suit your use.
- No output of TypeSafe's Jev was used to train either model, to choose τ, or to evaluate the cascade; the Jev figure above is the one JevBench publishes.

## Credits and licences

- Jev-Style-2B-Decision-v3 (Apache-2.0), fine-tuned from Qwen/Qwen3.5-2B (Apache-2.0).
- JevK5-9B by alibiserikbay (Apache-2.0) and its runtime allebee/jevk5 (Apache-2.0).

Neither model's weights are redistributed here. This cascade is not affiliated with, endorsed by or connected to TypeSafe or Jev, the JevK5 author, or Alibaba Cloud / the Qwen team. "Jev-style" only describes the kind of model.

## Citation

```bibtex
@misc{jevstylecascade9b2026,
  title  = {Jev-Style-Cascade-9B: a confidence cascade of Jev-Style-2B-Decision-v3 and JevK5-9B},
  author = {chaoliangUNSW},
  year   = {2026},
  url    = {https://huggingface.co/chaoliangUNSW/Jev-Style-Cascade-9B}
}
```
