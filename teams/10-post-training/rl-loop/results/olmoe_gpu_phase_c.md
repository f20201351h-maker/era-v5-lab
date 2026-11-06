# Frozen-router RL on a real MoE: OLMoE-1B-7B on MATH (6 Nov 2026)

**Question.** Gate 4 freezes the router. Does group-relative RL with a frozen router raise held-out reward on a real MoE, and does routing stay stable? We compared grpo vs gspo, plus a control run with the router trainable.

**Pre-registration.** The success bar, the learning-rate rule and the setup were written before the first run. Two deviations were logged during the work: chunked backward after an out-of-memory crash, and an 80-minute cap. Numbers: `olmoe_gpu_phase_c.json`.

## Setup

- **Model:** `allenai/OLMoE-1B-7B-0125-DPO` (revision `0f2e2e1`). This checkpoint has had no RL stage yet.
- **What trains:** attention only (268.5M parameters, fp32 masters). The router and the experts stay frozen.
- **Data:** hendrycks MATH train, decontaminated with fingerprints v2. Only problems the start model sometimes solves (0 < pass rate < 1 in survey B1; see `rl-data/results/survey_b1.md`) are used: 556 train and 199 held-out, stratified by level.
- **Prompt:** the raw problem in the chat template. No instruction text.
- **Reward:** math-verify equality with the last `\boxed{}` of the human solution.
- **Sampling:** vLLM on the same GPU. k = 8, 16 prompts per step, temperature 1.0, at most 1,024 new tokens.
- **Optimiser:** lr 5e-6, clip 0.2, no KL penalty, TIS clip 2.0, 200 steps, seed 1234.
- **Hardware:** one H100. About 12 to 13 s per step, peak 49 GB.
- **Learning rate:** 5e-6 was picked over 1e-6 by the pre-set pilot rule (30 steps each), using KL and training reward only. Held-out reward was not used. The rule's clip-fraction criterion could never trigger, because with one PPO epoch the ratio is always 1.

## Results

Held-out score = pass@1 over 199 problems × 4 fresh samples, measured at step 0 and again at step 200. The CI is a paired bootstrap 95% interval over problems.

| Run | Held-out step 0 → 200 | Change (95% CI) | Router checksum | KL to start | Worst per-layer load-ratio rise | Meets bar |
|---|---|---|---|---|---|---|
| C1 grpo, router frozen | 37.2% → 52.1% | **+15.0** (+10.4 to +19.4) | unchanged | 0.082 | +2.5% | yes |
| C2 gspo, router frozen | 37.2% → 53.3% | **+16.1** (+11.6 to +20.6) | unchanged | 0.093 | +4.4% | yes |
| C3 grpo, router trainable (control) | 37.2% → 52.1% | **+15.0** (+10.6 to +19.2) | changed (as intended) | 0.055 | +6.9% | n/a (control) |

The bar was all four of: at least +5 points with the CI above 0, router checksum unchanged, KL ≤ 0.1, and load-ratio rise ≤ 10%.

**Held-out curves** (steps 0, 25, 50, ..., 200)

| Run | Pass@1 at each eval |
|---|---|
| grpo | .372 .494 .477 .515 .500 .536 .539 .492 .521 |
| gspo | .372 .464 .491 .501 .504 .534 .544 .531 .533 |
| router trainable | .372 .487 .485 .494 .497 .504 .484 .513 .521 |

- Most of the gain comes in the first 25 steps.
- C1 and C3 both end at exactly 415 of 796 correct samples. Their curves differ at every other eval, so these are different runs that happen to end on the same count.

**grpo vs gspo:** the CIs overlap almost completely, so neither is better on this evidence (one seed each).

## Routing with a frozen router

| Measure | grpo | gspo | router trainable |
|---|---|---|---|
| Per step: top-8 set changed in ≥ 1 layer | ~45% | ~45% | ~45% |
| Per step: top-1 expert changed | 1.2% | 1.2% | 1.2% |
| Per step: routing weight moved | 0.65% | 0.66% | 0.66% |
| Step 200 vs step 0: top-1 expert changed | 8.9% | 10.3% | 8.5% |
| Step 200 vs step 0: routing weight moved | 4.7% | 5.2% | 4.4% |

Measured on 8 held-out prompts with their step-0 completions. Two identical forwards give exactly the same routing.

- Freezing the router does not freeze routing. Training attention changes the router's input, so about 9 to 10% of tokens change their top expert over 200 steps, and about 5% of routing weight moves.
- Most per-step set changes are near-tie swaps that carry almost no weight. The router scores in bf16, so the 8th and 9th expert are often nearly tied.
- **The control changes little at this scale.** With the router trainable, routing moved about as much (4.4% vs 4.7%) and held-out gain was the same. Expert load became slightly less even (+6.9% vs +2.5%), but did not collapse.
  - The collapse reported for a trainable router (MiMo-V2.6, arXiv 2610.11959) did not appear here. That is one seed on OLMoE at lr 5e-6 for 200 steps; larger learning rates, longer runs or Team 5's model may differ.
  - The router checksum is still the cheap, exact proof of "frozen" that Gate 4 asks for.
- At step 0 OLMoE's load is already uneven on this text: the busiest expert in a layer gets 4.1 to 7.5 times the mean load. That is why the bar uses the change from step 0, not a fixed limit.

## Is the gain just answers fitting the token limit? (C4, pre-registered)

Completions got about 20% shorter during training (grpo 527 → 436 tokens, gspo 538 → 405, router trainable 541 → 466), and in survey B1 16% of samples hit the 1,024-token cap. So we reran C1 (same seed and settings) and scored the held-out set at steps 0 and 200 a second time, with a 2,048-token limit.

| Held-out pass@1 (199 x 4) | Step 0 | Step 200 | Change (95% CI) | Hit the limit (start → end) | Mean length (start → end) |
|---|---|---|---|---|---|
| 1,024-token limit | 37.2% | 54.0% | +16.8 (+12.6 to +21.1) | 8.7% → 4.0% | 544 → 425 |
| 2,048-token limit | 37.9% | 51.8% | **+13.8 (+9.1 to +18.6)** | 0.75% → 0.13% | 569 → 431 |

- The pre-registered rule was: a gain of at least +10 at 2,048 with the CI above 0 means most of the gain is not explained by the cap. That holds.
- With twice the room, the start model improves by only 0.7 points, although 8.7% of its answers had been cut off. The cut answers were mostly wrong anyway. At most about 3 of the ~17 points come from the cap.
- Not ruled out: part of the gain may be the model stating its final answer in a form the checker accepts (format) rather than solving more problems. The completions were not kept. The trained attention weights from C4 are saved on the Modal Volume (`/vol/ckpt/c4_grpo_attention_bf16.pt`, 537 MB, sha256 `102b417c…`), so this can be checked without retraining.
- C4 also repeats C1 under GPU non-determinism: +16.8 vs +15.0 at 1,024 tokens, with the same seed. This is not an independent seed.

## Caveats
- **The held-out problems come from the same distribution** (MATH train, pre-selected as "sometimes solved" by the start model). This is not a benchmark score; MATH-500 and other G9 benchmarks were not run.
- One seed per run. Attention-only training. OLMoE, not Team 5's model (which arrives 18 Nov).
- KL to start was close to the limit at step 200 for grpo and gspo (0.082, 0.093). Longer runs would need a KL penalty or a lower learning rate.

**Cost:** pilots $1.47, the two crashed first attempts $3.40, C1 $3.66, C2 $3.95, C3 $4.07, C4 $4.28 (Modal list rates, from our run ledger). All phases together: $24.19.
