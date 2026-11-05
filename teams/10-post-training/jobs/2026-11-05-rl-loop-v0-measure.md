# RL loop v0 on OLMoE-1B-7B-Instruct: step time, peak memory, frozen router (1 GPU)

```
Team:            10-post-training
Owner:           Anirudh Narla (f20201351h-maker)
Lead approval:   saitej123
Question:        On one GPU, how long does one step of the G10 RL loop take on OLMoE-1B-7B-0125-Instruct, how much GPU memory does it need, and does the router checksum stay the same?
Script:          teams/10-post-training/rl-loop/train.py   (PR #35)
Command:         python teams/10-post-training/rl-loop/train.py --config teams/10-post-training/rl-loop/config/olmoe_1b7b_measure.yaml --model-path ./inputs/OLMoE-1B-7B-0125-Instruct
Config:          teams/10-post-training/rl-loop/config/olmoe_1b7b_measure.yaml
Inputs:          allenai/OLMoE-1B-7B-0125-Instruct (public weights, Apache-2.0; config.json and safetensors), placed under ./inputs/. Task inputs are synthetic: random token ids, no text, no dataset.
Hardware:        1 GPU
Max time:        45
Precision:       bf16
Measures:        seconds per step, peak GPU memory (torch.cuda.max_memory_allocated), router checksum before and after (SHA-256 of all 16 mlp.gate weights), train and held-out reward on the synthetic task
Expected result: router_unchanged true; peak memory about 18 to 25 GB; about 5 to 60 s per step (see below)
Outputs:         out/metrics.jsonl, out/summary.json (well under 1 MB)
Test log:        teams/10-post-training/jobs/2026-11-05-rl-loop-v0-measure.log (own machine, CPU, network off, shrunk model, router unchanged)
```

## Why this job

- **It's the first real-size run of the loop that Gate 4 needs.** It checks three things on the real model:
  - the loop fits on one GPU;
  - how long a step takes;
  - the 16 router weights stay exactly the same with `requires_grad=False`.
- **We can't run it ourselves.** The Instruct model is 13.8 GB in bf16, more than a free T4 or a 16 GB laptop can train.
- **No training data is involved.** The task is synthetic: the reward is 1 when the first sampled token has the same parity as the last prompt token. Random guessing gets about half right, so every step has a training signal. The weights are thrown away after the job.
- **A learning run comes later.** It waits for RL data v1 and TSAI's choice of data and algorithm.

## Expected numbers (my estimate)

- **Weights:** 6.92B parameters in bf16 is about 13.8 GB.
- **Trainable parameters:** attention only, `train.trainable: ['\.self_attn\.']`. That's 16 layers × 4 × 2048 × 2048 ≈ 268M parameters. bf16 gradients take about 0.5 GB, and AdamW states in bf16 about 1.1 GB.
- **Logits:** for one group (8 sequences × 72 tokens × 50,304 vocab, in fp32) they take about 0.12 GB. The loss processes one group at a time.
- **Generation:** 8 × 64 new tokens per prompt, 8 prompts per step.
- **Total:** about 18 to 25 GB peak. Step time is a wide guess (5 to 60 s), which is why we're asking.

## Test log

The same script and config ran on my machine (Apple M1 Pro CPU) inside Docker with `--network=none`. The model was a shrunk random OLMoE with the same vocabulary and router layout (hidden size 64, 2 layers, 8 experts), for 6 steps. All 8 groups were used every step, and the router checksum was unchanged. See the `.log` file next to this one.
