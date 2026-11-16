# Job request: E9 at SmolLM2-1.7B base, what SFT must deliver before RL (draft, not sent)

```
Team:            10-post-training
Owner:           f20201351h-maker
Lead approval:   saitej123
Question:        After full SFT on execution-verified code (vs Dolly only, vs none), what share of k=8 samples on the easiest RL rung close the turn, contain a program, run and pass, and what share of prompts give a mixed (gradient-carrying) group at 1.7B?
Script:          teams/10-post-training/rl-loop/pilots/recipe_e9_e10_20261116/recipe_job.py
Command:         python teams/10-post-training/rl-loop/pilots/recipe_e9_e10_20261116/recipe_job.py --config teams/10-post-training/rl-loop/pilots/recipe_e9_e10_20261116/cfg_e9_17b.yaml
Config:          teams/10-post-training/rl-loop/pilots/recipe_e9_e10_20261116/cfg_e9_17b.yaml
Inputs:          HuggingFaceTB/SmolLM2-1.7B @ effd688a12921b4cc83e3312b6feb579f70f9c71 (Apache-2.0) as inputs/SmolLM2-1.7B-effd688a; inputs/g10-e9/: dolly.tok.jsonl (databricks-dolly-15k, CC-BY-SA-3.0, human-written), ocr2_clean.tok.jsonl (OCR-2 pack v1 slice, execution-verified, minus every row whose problem is an RL eval prompt), cc.jsonl (RL data v1 code records, sha256 91ee9951...), rung1_s128_ids.txt, rest_s32_ids.txt
Hardware:        1 GPU (A100 or H100, >= 40 GB; full fine-tune of 1.7B with FP32 AdamW state needs ~27 GB + activations); estimate 60-90 min
Max time:        120
Precision:       bf16 autocast compute, fp32 weights / gradients / AdamW state; sampling fp32
Measures:        per arm (A2 OCR-2 SFT, A0 base): closed-turn, code-block, parses, runs, pass shares with 95% intervals over problems; failure outcomes; mixed-group share; pass@8; forgetting bpb on G5 held-out v1; train step time; peak memory
Expected result: <filled from the 135M / 360M trend when sent; see work/recipe-e9-2026-11-16/RESULTS.md>
Outputs:         out/e9/{A0,A2}/screen/screen_report.json, train_report.json, forget.json, e9_summary.json (generations.jsonl stays in out/ but holds model text: TSAI keeps it private); < 50 MB without generations
Test log:        teams/10-post-training/jobs/2026-11-17-e9-sft-before-rl-1.7b.log
```

The script runs offline (model and data are staged inputs) and writes only to `out/`. The reward runs each program
in Docker (`stdio_runner: docker`) or, without Docker on Linux as root, with `stdio_runner: local` (per-uid isolation);
the expected outputs never reach the program. The test log is the same script and config shape on a Mac (CPU, `dry_run:
true`, SmolLM2-135M, tiny sizes): it proves the commands, the files and the resume markers, not a result.
