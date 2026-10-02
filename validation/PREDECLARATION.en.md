# Pre-declaration (English translation)

The binding text is `PREDECLARATION.zh.md`, written on 2026-10-03 before any calibration or evaluation result existed.
- sha256 of the Chinese file: `e427a671bbe6cc3a07c3c1819125f7169b6d547fc37d067f5f3ccff77ad4e0a6`
- Committed in the project's local git history before any model was run on the calibration set.

## Systems
- **Glue-2**: chaoliangUNSW/Jev-Style-2B-Decision-v3 → alibiserikbay/JevK5-9B v0.3.3. This system is released here as `cascade-9b`.
- **Glue-3**: Jev-Style-0.8B-Decision-v3 → Jev-Style-2B-Decision-v3 → JevK5-9B. It is not released.

## Confidence and routing
- **Confidence** is the normalized top probability:
  - choice and score: (k·p_max − 1)/(k − 1), where k is the number of options;
  - noul: |2·P(true) − 1|.
- **Thresholds.** There is one threshold τ per non-top tier, shared by all question types.
- **Routing.** A tier answers a question when its confidence is at least τ. Otherwise the question is escalated.
- **Errors.**
  - A tier error (input budget, invalid question, unsupported) escalates the question.
  - If the top tier fails, the answer of the last tier that succeeded is used.

## Choosing τ
- **Data:** the calibration set (sha256 `580962e4…`). It contains:
  - no Decision Index test items (deduplicated by normalized text);
  - no JevBench items;
  - no Jev output;
  - no labels produced by any of the cascade's models.
- **Grid:** τ ∈ {0.00, 0.01, …, 0.99}.
- **Constraint:** cascade accuracy ≥ JevK5-9B-alone accuracy − 1.0 point.
- **Objective:** the lowest expected cost.
  - Cost = Σ over tiers of (share of requests that call the tier) × (that tier's measured median latency on the RTX 5090).
  - Ties go to the lower escalation share.
- **Freeze:** the chosen τ is written to the cascade file with its sha256 and time, and never changed afterwards.

## Evaluations (each system run once, at the frozen checkpoint)
- **JevBench public items:**
  - 2B alone
  - JevK5-9B alone
  - Glue-2
  - Glue-3
  - 0.8B alone (reference)
- **Decision Index 0.2.1 offline forecast:**
  - Composed from public per-row results at the frozen τ and scored with the kit's scorer.
  - Only that one point is reported; no curve is reported and nothing is re-tuned on it.
- **Latency:** median and p95 for each system on the same RTX 5090.

## Speed go / no-go
- The 2B's fastest usable path must have a median latency ≤ 0.6 × JevK5-9B's median latency.
- "Usable" means ≤ 0.5 % top-1 changes vs the float32 reference path on the calibration set.
- If the condition is not met: report to the user and pause the release.

## What happened
- **First run (`freeze_result_first_run_without_cuda_graphs.json`): the gate failed.**
  - The 2B on its ordinary float32 path had a median of 90.0 ms vs 28.8 ms for JevK5-9B (ratio 3.13).
  - bfloat16 was not usable: 0.76 % top-1 changes, and it was slower.
  - The release was paused.
- **Second run (`freeze_result.json`): the gate passed.**
  - The 2B's CUDA-graph runtime was then added as a third candidate path: 0 top-1 changes on all 4,992 calibration questions, and a median of 13.97 ms (ratio 0.485).
  - The same rule then gave τ = 0.75.
- **Accuracy:** at τ = 0.75 the calibration accuracy was 73.12 % vs 74.10 % for JevK5-9B alone.
