# Pass-rate survey B1: which pools give OLMoE-1B-7B any signal (5 Nov 2026)

**Setup**
- Script: `rl-data/survey.py`.
- Hardware: vLLM 0.32.0 on one L40S, on Modal.
- Sampling: k = 8, temperature 1.0, top_p 1.0, max 1,024 new tokens, seed 1234.
- **No instruction text.** The prompt is the raw problem in the model's chat template, so no AI-written text is in any prompt.
- Manifests: `survey_*.json`. They hold ids and numbers only.

**Pools** (both decontaminated with `sft/decontam_run.py` on fingerprints v2, full set `2826a3b8…`; a 0 recheck on the clean files)

| Pool | Licence | Generator | Rows | Flagged | Kept |
|---|---|---|---|---|---|
| hendrycks MATH train (`EleutherAI/hendrycks_math` @ `21a5633`) | MIT | none, human | 7,498 | 175 (MMLU 92, AIME 2024 67, AIME 2025 61, HLE 4, LiveCodeBench 2) | 7,323 |
| code_contests train, non-Codeforces judges (`deepmind/code_contests` @ `802411c`) | CC-BY-4.0 | none, human | 5,092 | 1,369 (LiveCodeBench 1,222, Codeforces 207, other 5) | 3,723 |

- Survey samples: MATH 1,505 (43 per subject x level); MATH on Instruct 301 (every 5th of those); code 300 (75 per judge).
- The AIME flags can't be true copies, since MATH predates those exams. They were removed anyway.

## Results

| Model (revision) | Pool | Tasks | Mean pass@1 | Kept (0 < p < 1) | Too hard | Too easy | Hit the 1,024-token limit |
|---|---|---|---|---|---|---|---|
| OLMoE-0125-DPO (`0f2e2e1`) | MATH | 1,505 | 23.0% | **755** | 701 | 49 | 16.1% |
| OLMoE-0125-Instruct (`b89a7c4`) | MATH | 301 | 25.7% | 169 | 119 | 13 | 15.8% |
| OLMoE-0125-DPO | code_contests non-CF | 300 | 1.4% | 20 | 280 | 0 | 21.7% |

MATH on DPO by level:

| Level | Mean pass | Kept / tasks |
|---|---|---|
| 1 | 45.7% | 227 / 299 |
| 2 | 30.9% | 192 / 301 |
| 3 | 18.8% | 145 / 301 |
| 4 | 14.6% | 130 / 301 |
| 5 | 5.0% | 60 / 301 |

**What it shows**
- MATH train gives plenty of signal: about half the tasks are mixed. 23% matches the model card's MATH (Flex) score for the DPO checkpoint, 20.36. So the earlier near-zero base rates came from the task choice, not from a broken pipeline.
- DPO vs Instruct on the same 301 problems: 24.4% vs 25.7%, a difference of +1.3 points with a paired bootstrap 95% CI of -0.3 to +3.0, so no clear difference.
- Code without an instruction is weak: 20 of 300 tasks are mixed. A program-format instruction (to be written by a team member) may change this. That is the next survey.
- About 16% of math samples hit the token limit. Some correct answers may be lost there. Check this before fixing `max_new` for training.

**Cost:** 38 min on an L40S, $1.57 (Modal list rates).
