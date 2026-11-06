# RL loop on OLMoE-1B-7B: step time, memory, router checksum (5 Nov 2026)

This answers the job request in #36 (step time and peak memory for the loop on OLMoE-1B-7B, router checksum unchanged) without a TSAI job. Runs on Modal with `rl-loop/modal/app.py`. Numbers: `olmoe_gpu_phase_a.json`.

**Setup**
- Model: `allenai/OLMoE-1B-7B-0125-Instruct` at revision `b89a7c4`, bf16.
- Training: router frozen. Only the attention weights train (268.5M parameters), kept in fp32 under bf16 autocast.
- Task: the synthetic parity token task, so no training data is involved.
- Algorithm: grpo, lr 1e-6, one PPO epoch.
- Rollouts: either one batched Hugging Face `generate`, or a vLLM 0.32.0 engine on the same GPU (30% of memory) with the trained weights pushed after every step.

## Step time and memory

| GPU | Config (prompts x k x new tokens) | Sampler | Seconds per step (after step 1) | Rollout tokens/s | Peak torch memory (GB) |
|---|---|---|---|---|---|
| H100 80GB | 8 x 8 x 64 (#36's config) | vLLM | 1.15 | 8,427 | 41.0 |
| H100 80GB | 8 x 8 x 64 (#36's config) | HF batch | 6.13 | 651 | 24.1 |
| H100 80GB | 16 x 8 x 512 | vLLM | 7.24 (rollout 4.33, old logprobs 0.75, train 2.15, sync 0.006) | 15,146 | 50.7 |
| H100 80GB | 16 x 8 x 512 | HF batch | 93.7 (rollout 90.9) | 721 | 45.2 |
| A100 80GB | 16 x 8 x 512 | vLLM | 14.09 | 9,183 | 51.3 |

- vLLM rollouts are about 13x faster than batched HF `generate` at the realistic size.
- The H100 is cheaper per step than the A100: about $0.009 vs $0.012, counting Modal's GPU, CPU and memory rates.
- The run with the vLLM engine used 53 GB of the GPU in total, which fits on an 80 GB card.

## Checks

- **Router checksum:** identical before and after in every run. `groups_used` > 0 on every step.
- **vLLM weight sync works.** In the sync check, all q_proj weights were zeroed in the trainer and the sampler/trainer gap (K3 on vLLM's samples) was measured:

  | Condition | K3 |
  |---|---|
  | baseline | 0.0048 |
  | zeroed, not synced | 3.17 |
  | zeroed, synced | 0.00036 |
  | restored, synced | 0.0056 |

- **Sampler/trainer gap during training:** K3 0.004 to 0.009 per step. That is within the 1e-3 to 1e-1 range reported for MoE models.
- **Weights move:** 99.99% of trainable entries changed.
  - Before the fp32-master fix, a CPU check on the same config (lr 1e-5) changed only 14.6% of entries.
  - At lr 1e-6, a Mac test changed only 2%. The cause was bf16 rounding.

## Routing with a frozen router (first look; random-token prompts, not text)

- Two identical forwards give identical routing (noise floor 0.0).
- After one optimiser step, 43% to 63% of tokens have a different top-8 expert set in at least one of the 16 layers. Per layer it is 1% to 15%.
- Against step 0, the rate levels off: 0.59 at step 10 and 0.59 at step 20 in the 20-step run.
- Our reading, not yet tested: the router scores in bf16, so the 8th and 9th expert are often nearly tied, and small changes swap them back and forth.
- Update (6 Nov): measured on real MATH text in phase C (`olmoe_gpu_phase_c.md`). Per step only 1.2% of tokens change their top expert and 0.65% of routing weight moves, so most set changes are near-tie swaps. Over 200 steps, about 9 to 10% of top-1 choices and about 5% of routing weight change.

## Not covered

- Team 5's model (arrives 18 Nov).
- Learning on a real task (phase C).
- The routing difference inside vLLM's own forward (R3), since vLLM does not return its expert choices here.
