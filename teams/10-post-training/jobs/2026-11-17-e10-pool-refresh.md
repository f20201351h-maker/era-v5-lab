# Job request: E10, fixed cohort vs refreshing pool in RL after SFT (draft, not sent; only if E9 passes its gate)

```
Team:            10-post-training
Owner:           f20201351h-maker
Lead approval:   saitej123
Question:        On the post-SFT model, does refreshing the RL prompt pool when kept groups halve beat a fixed screened cohort on held-out strict pass after step 10 (steps, lr, rollouts, evaluation matched)?
Script:          teams/10-post-training/rl-loop/pilots/recipe_e9_e10_20261116/recipe_job.py
Command:         python teams/10-post-training/rl-loop/pilots/recipe_e9_e10_20261116/recipe_job.py --config teams/10-post-training/rl-loop/pilots/recipe_e9_e10_20261116/cfg_e10.yaml
Config:          teams/10-post-training/rl-loop/pilots/recipe_e9_e10_20261116/cfg_e10.yaml
Inputs:          inputs/e9-A2-model (the E9 A2 model folder at the gated size); inputs/g10-e9/cc.jsonl, e10_train_ids.txt (626 rung-1 prompts), e10_heldout_ids.txt (128 rung-1 + 32 rest, never trained)
Hardware:        1 GPU (A100 / H100)
Max time:        360
Precision:       fp32 master weights, fp32 sampling (hf_batch); attention weights trainable
Measures:        held-out pass (kept and strict) at steps 0/10/20/40 per arm, paired by problem; kept / all-pass / all-fail groups and unique prompts per step; refresh events; step time split (rollout, reward, update)
Expected result: <filled from E9 when sent; preregistered readings H1 / H0 / H2 in PREREG_E10.md>
Outputs:         out/e10/{fixed,refresh}/{metrics.jsonl,evals.jsonl,summary.json,heldout_*/screen_report.json}; state/ and ckpt_*/ are large: keep or delete per TSAI policy
Test log:        teams/10-post-training/jobs/2026-11-17-e10-pool-refresh.log
```

Durable state: a killed or time-limited job continues from `out/e10/<arm>/state` with the same command (the trainer
refuses a state from a different config). The test log is the same script on a Mac CPU with `dry_run: true`: both
arms, pause after step 2, export and screen, resume to step 4, export and screen.
