# Pass-rate filter: OLMoE-1B-7B-0125-Instruct on 36 Codeforces problems

Generated on 2026-11-05 with `rl-data/build.py` and `rl-data/filter.py`.
- **Problems:** `deepmind/code_contests` train, parquet shard `partial-train/0001` (CC-BY-4.0, human problems and tests). Codeforces rows rated 800 to 1200, up to 20 tests each, giving 36 tasks.
- **Model:** allenai/OLMoE-1B-7B-0125-Instruct (Q8_0 GGUF, llama.cpp b12125).
- **Sampling:** k = 8 per task, temperature 0.8, top_p 0.95, up to 1,024 new tokens.
- **Scoring:** `stdio_check.py` (python:3.12-slim, no network).
- **Run time:** 34.1 minutes on an Apple M1 Pro.

**Result: keep 1, too hard 35, too easy 0.** Only `cf918A` (rated 800) was solved, once in 8 tries.

| Rating | Tasks | Kept |
|---|---|---|
| 800 | 8 | 1 |
| 900 | 3 | 0 |
| 1000 | 4 | 0 |
| 1100 | 11 | 0 |
| 1200 | 10 | 0 |

Each task's id, pass rate and decision is in the manifest `filter.py` writes; it isn't committed, to keep data files out of the repo.

## What it means

- Codeforces problems rated 800 to 1200 give this model almost no RL signal. GRPO would drop 35 of 36 groups.
- An earlier base-rate run with a different harness (same model and settings, the 33 tasks of that shard that have public and private tests) gave 1 of 264 tries, also on `cf918A`. So the result is stable.
- Most failures in that run were format failures, not wrong algorithms:
  - a function defined but never called;
  - sample input pasted in as code;
  - a missing import.
- **Before RL on this kind of task:**
  - an SFT step that teaches the stdin/stdout format;
  - easier tasks: the easy rung, with templates written by the team;
  - and the filter rerun on the actual starting model (Team 5's, 18 Nov).
