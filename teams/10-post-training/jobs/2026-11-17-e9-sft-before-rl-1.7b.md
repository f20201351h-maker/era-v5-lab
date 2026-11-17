# Job request: E9 at SmolLM2-1.7B base, can SFT make RL start (draft, not sent)

```
Team:            10-post-training
Owner:           f20201351h-maker
Lead approval:   saitej123
Question:        At 1.7B, after full SFT on execution-verified code (full reasoning traces vs short code-only answers), does the share of easiest-rung RL prompts with a mixed (gradient-carrying) group of 8 samples reach 10%?
Script:          teams/10-post-training/rl-loop/pilots/recipe_e9_e10_20261116/recipe_job.py
Command:         python teams/10-post-training/rl-loop/pilots/recipe_e9_e10_20261116/recipe_job.py --config teams/10-post-training/rl-loop/pilots/recipe_e9_e10_20261116/cfg_e9_17b.yaml
Config:          teams/10-post-training/rl-loop/pilots/recipe_e9_e10_20261116/cfg_e9_17b.yaml
Inputs:          HuggingFaceTB/SmolLM2-1.7B @ effd688a12921b4cc83e3312b6feb579f70f9c71 (Apache-2.0) as inputs/SmolLM2-1.7B-effd688a; inputs/g10-e9/: ocr2_clean_v2.tok.jsonl (OCR-2 pack v1 slice, execution-verified, minus every row whose problem family matches an evaluation problem), ocr2_short_v2.tok.jsonl (the same rows with the reasoning block deleted), cc.jsonl (RL data v1 code records), rung1_s128_ids.txt, rest_s32_ids.txt
Hardware:        1 GPU, A100 or H100 (>= 40 GB: full fine-tune of 1.7B with fp32 AdamW state needs ~27 GB + activations)
Max time:        180   (estimate below: ~140 min)
Precision:       training fp32 weights / gradients / AdamW state with bf16 autocast; sampling bf16 (fp32 on every smaller run)
Measures:        per arm: closed-turn, code-block, parses, runs, pass shares with 95% intervals over problems; failure reasons; mixed-group share; pass@8; forgetting bpb on G5 held-out v1; step time; peak memory
Expected result: A2v2 mixed share 10-25%, pass 3-8%; A4 mixed share at least A2v2's with far fewer tokens; A4 forgetting above 0.05 (see the trend below)
Outputs:         out/e9/{A2v2,A4}/screen/screen_report.json, train_report.json, forget.json, e9_summary.json; generations.jsonl hold model text (keep private); < 50 MB without generations and model folders
Test log:        teams/10-post-training/jobs/2026-11-17-e9-sft-before-rl-1.7b.log
```

Why 1.7B (results so far, SmolLM2 base, rung 1, 128 problems x 8 samples, one seed):

| size, code SFT | new tokens | code block | program runs | pass | mixed groups |
|---|---|---|---|---|---|
| 135M full traces | 1,024 | 8.6% | 1.9% | 0 | 0/128 |
| 135M full traces | 2,048 (60 problems) | 55% | 10.8% | 0.2% | 1/60 |
| 135M short answers | 1,024 | 99% | 35% | 0.4% | 4/128 = 3.1% |
| 360M full traces, full FT | 2,048 | 62% | 28% | 1.0% (strict 9/10) | 8/128 = 6.3% [2.3, 10.9] |
| 360M full traces, LoRA r4 + marker rows | 2,048 | 54% | 18% | 0.6% | 4/128 = 3.1% |

The mixed share grows with size and has not crossed 10% at 360M. 1.7B is the first size where it plausibly does, and it is
too large for full fine-tuning on our free hardware (2x T4). Passes were re-checked on 80 unused source tests ("strict").

Time estimate (labelled an estimate; basis measured): the 360M full-traces screen on a Kaggle T4 (fp32, KV budget 6 GB)
generated 1,675,500 tokens in 15,690 s (106.8 tokens/s) for 128 x 8 samples at 2,048 new tokens. Scaling to 1.7B on one
A100 assumes decoding is memory-bandwidth bound: bytes per generated token x4.7 (weights 1.7B vs 360M; KV cache 0.39 vs
0.08 MB per token), bandwidth x6 (A100 ~2.0 TB/s vs T4 0.32 TB/s), bf16 sampling halves bytes: 15,690 s x 4.7 / 6 / 2
= ~6,100 s (~100 min) for the A2v2 screen, before any gain from the 5x larger KV budget (30 GB) on the A100. The A4 screen
generates ~1/15 of the tokens (mean 93 vs 1,636 per sample): ~7 min. Training: 135M took 14 s per update on an M1 Pro
(fp32 MPS); 1.7B has 12.6x the FLOPs and an A100 with bf16 autocast is ~60x the M1's throughput: ~3 s per update, ~8 min
per arm for 150 updates. Forgetting: ~3 min per arm. Total ~140 min; Max time 180.

The script runs offline (model and data staged as inputs) and writes only to `out/`. Programs run in Docker
(`stdio_runner: docker`) or, without Docker on Linux as root, with `stdio_runner: local`; the expected outputs never reach
the program. Training state is saved every 10 updates (`--save-every`), so a killed job resumes with the same command.
Test log: the same script and config shape on a Mac CPU with `dry_run: true` (SmolLM2-135M, tiny sizes), 69 s, exit 0;
it proves the commands, files and resume markers, not a result.
