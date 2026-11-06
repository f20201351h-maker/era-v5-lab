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

## Does it transfer to a standard benchmark, and is it format? (E1, pre-registered)

The C4 end model (base + saved attention weights) and the start model were scored on **MATH-500** and on our held-out set. Settings: 4 samples, temperature 1.0, up to 2,048 tokens, raw problem.
- MATH-500 is the 500-problem MATH test subset from OpenAI's PRM800K. It covers all levels, is not filtered by pass rate, and has 0 exact text overlap with our training pool.
- For every sample we recorded why it failed and whether it boxed an answer. Manifests (ids and numbers): `e1_eval/`.

| Set | Start | End | Change (95% CI) |
|---|---|---|---|
| MATH-500 | 19.0% | 23.1% | **+4.1** (+1.9 to +6.4) |
| Held-out (199, "sometimes solved") | 36.8% | 51.8% | +15.0 (+10.7 to +19.2) |

MATH-500 by level:

| Level | Problems | Start | End |
|---|---|---|---|
| 1 | 43 | 48.3% | 58.7% |
| 2 | 90 | 31.9% | 42.5% |
| 3 | 105 | 21.2% | 25.7% |
| 4 | 128 | 12.1% | 12.5% |
| 5 | 134 | 5.6% | 6.5% |

- **Transfer: real, but below the pre-registered +5 bar.** The CI excludes 0, but +4.1 < +5.
  - The start score (19.0%) matches the model card's MATH figure for this checkpoint (20.36).
  - The gain sits in levels 1 to 3. Levels 4 and 5 did not move.
  - The held-out set was selected for "sometimes solved", the band where RL helps most, so it overstates what a full benchmark shows. About +4 points (+22% relative) is the honest benchmark number.
- **Format: ruled out.** Before and after, 97 to 99.6% of answers contain `\boxed{}`, and no failure is "no answer found". All failures are wrong answers, so every gained point is a wrong answer turned right.
- The start model's held-out score here (36.8%) agrees with the loop's own 2,048-token eval (37.9%), so the two evaluation paths are consistent.

## Independent second seed (E2, pre-registered)

Same as C1 but seed 4242 (different data order, sampling and probe) and 100 steps.

| Step | 0 | 25 | 50 | 75 | 100 |
|---|---|---|---|---|---|
| seed 1234 (C1), held-out | .372 | .494 | .477 | .515 | .500 |
| seed 4242 (E2), held-out | .379 | .469 | .495 | .508 | .504 |

- E2 at step 100: **+12.4 points** (CI +8.0 to +16.8). C1 at the same step: +12.8. The bar (at least +5, CI above 0) is met.
- Router checksum unchanged. KL to start 0.039. Worst load-ratio rise +8.0%.
- Two seeds agree closely. That is still only two values, so nothing is claimed about the spread.

## Caveats
- **The held-out problems come from the same distribution** (MATH train, pre-selected as "sometimes solved" by the start model), which favours RL. On the standard MATH-500 benchmark the gain is +4.1 points (see E1). Other G9 benchmarks were not run.
- Seeds: two for grpo (1234 for 200 steps, 4242 for 100 steps); one each for gspo and the router-trainable control. Attention-only training. OLMoE, not Team 5's model (which arrives 18 Nov).
- KL to start was close to the limit at step 200 for grpo and gspo (0.082, 0.093). Longer runs would need a KL penalty or a lower learning rate.

**Cost:** pilots $1.47, the two crashed first attempts $3.40, C1 $3.66, C2 $3.95, C3 $4.07, C4 $4.28, E1 $0.70 (incl. a failed first try) plus a $0.01 merge, E2 $2.00 (Modal list rates, from our run ledger). All phases together: $26.89.
