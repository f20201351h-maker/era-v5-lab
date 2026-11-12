# eval: reasoning-effort suite (Package T2)

## What the files are

| File | What |
|---|---|
| `effort_suite.py` | Runner for a fixed prompt set crossed with the four effort labels `1k, 4k, 16k, 64k`. Commands: `verify`, `plan`, `run`, `guard`, `compare`. |
| `effort_manifest.json` | The 100-slot manifest. It stores ids, dataset revisions, file and text sha256 values, licence and generator for each slot. It holds no prompt text. `manifest_sha256` seals it. |
| `test_effort_suite.py` | pytest file for the suite (30 tests as of 11 Nov, per `results/effort_probe.md`). |
| `stage_b_qwen_modal.py` | Generation-only Stage B job for Qwen3.6-35B-A3B-FP8 on one H100 on Modal. |
| `stage_b_qwen_report.py` | Local scorer for that job's raw output. |

## Gate

No gate is named in these files. The suite is development evidence for the effort labels in `sft/sft_mix_proposal.md` section 4. G9's own final score is separate.

## What has been measured (committed results only)

- **Suite and CPU checks, 11 Nov:** `results/effort_probe.md` records the 30 tests, the manifest `verify` run, a scripted-completion pipeline smoke (no model), and a `guard` run that flags the 55 resolved tasks in the local RL v1 files.
- **Qwen3.6-35B-A3B-FP8, Stage B, 12 Nov:** 55 prompts (30 code, 25 math) x 4 labels x 2 samples = 440 cells at a 1024-token cap. `results/effort_stageB_qwen_2026-11-12/report.md` records 161 of 440 cells correct, 262 cells censored by the cap, 0 parse failures, and no visible effect of the label at a fixed cap. That report calls Qwen an above-floor reference, not the capstone model.
- **OLMoE-1B-7B-0924-Instruct, Stage A and Stage B:** the Qwen report quotes these (Stage B: 5, 5, 5 and 7 correct of 110 cells at 1k, 4k, 16k, 64k). The OLMoE result files are in separate PRs (#61 and #63 in the Qwen report) and are not committed on this branch.

## What has not been measured

- **Stage C** (label budget as the cap) has not been run.
- **The G10 chat template on the G5 model** (`--backend team5`). `results/effort_probe.md` says this waits for the G5 bundle; the Qwen and OLMoE runs use each model's own template.
- **Tool/agentic (20), Indic (15) and general (10) slots.** 45 of the 100 slots are unavailable, with the reason in the manifest.
- **Whether a label above 1k changes behaviour when the cap allows it.** Every Stage A and Stage B run uses a 1024 cap for all four labels.
- **Code ability of Qwen.** 195 of 240 code cells hit the cap, so the code numbers are a floor (Qwen report).
- `results/effort_probe.md` is dated 11 Nov and says no model had generated anything for the suite. The Stage B report is later. Read the two together.

## Open items stated in the results

- RL v1 must drop the 55 resolved tasks before any training while this suite is in use (`results/effort_probe.md`). Whether that drop has been made is not recorded in this folder.
- No licence-table row exists for the hendrycks MATH source (`results/effort_probe.md`).

## Reports

- [results/effort_probe.md](../results/effort_probe.md)
- [results/effort_stageB_qwen_2026-11-12/report.md](../results/effort_stageB_qwen_2026-11-12/report.md)

```bash
cd teams/10-post-training
python -m pytest eval/test_effort_suite.py -q -p no:cacheprovider
```
