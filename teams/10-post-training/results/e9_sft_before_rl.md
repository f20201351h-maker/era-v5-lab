# E9: what SFT has to deliver before RL can start on a weak base model (16-17 Nov 2026)

**Evidence level: surrogate, one seed.** SmolLM2 base models (135M, 360M; not instruct), dense with a tied head and their
own tokenizer. What transfers to the capstone is the order of the stages, the shape of the failure ladder and the trend
across sizes, not the numbers. Preregistration with seven dated amendments (each written before the results it governs)
is kept with the run files; the decision numbers below are the ones the amendments name.

## Question and design

How much of "closed turn -> code block -> program runs -> right answer" does SFT deliver, and what share of the easiest RL
prompts then gives a mixed group of 8 samples (unequal rewards: the only groups that carry gradient under objective_v1)?
Arms differ only in data: A0 no SFT; A1 Dolly (human-written chat, no code); A2 OCR-2 (execution-verified code with
reasoning traces, rows <= 2,048 tokens, every row whose problem family matches an evaluation problem removed); A4 the A2
rows with the reasoning block deleted (code-only answers; a deletion of admitted text). Full fine-tune, FP32, AdamW lr
1e-4, 150 updates x 8,000 supervised tokens each (A4 therefore repeats its short rows ~10.7 times). LoRA arm at 360M:
r4 on q/k/v/o/gate/up/down + the chat-marker rows, lr 1e-3 (marker rows 1e-4), same data and schedule.
Screen (`sft/bench/rl_screen.py`): the RL prompt (statement as the user turn), k = 8 samples at temperature 1, no top-k /
top-p, scored by the RL loop's own reward in a Docker sandbox. Rung 1 = codechef / hackerearth prompts in the shortest
third of statements (`rl-data/rung_pool.py`, 754 prompts); the decision set is a fixed seeded 128 of them, plus 32
unfiltered prompts from the rest of the pool. "Strict" = passes the kept tests and 80 unused source tests. Intervals: 95%
bootstrap over problems.

## Results (rung 1, 128 problems x 8 samples unless noted)

| size | arm | new tokens | closed with content | code block | program runs | pass (strict) | mixed groups [95% CI] | truncated |
|---|---|---|---|---|---|---|---|---|
| 135M | A0 base | 1,024 | 0% | 1.3% | 0% | 0 | 0/128 | 35% |
| 135M | A1 Dolly | 1,024 | 92% | 0.2% | 0% | 0 | 0/128 | 7% |
| 135M | A2 code, full traces | 1,024 | 9% | 8.6% | 1.9% | 0 | 0/128 | 74% |
| 135M | A2 code, full traces | 2,048 (60 problems) | 60% | 55% | 10.8% | 1/480 (1) | 1/60 | 24% |
| 135M | A4 code, short answers | 1,024 | 99% | 99% | 35.4% | 4/1,024 (4) | 4/128 = 3.1% | 0% |
| 360M | A0 base | 1,024 | 0.1% | 1.2% | 0.1% | 0 | 0/128 | 43% |
| 360M | A1 Dolly | 1,024 | 96% | 0% | 0% | 0 | 0/128 | 3% |
| 360M | A2 code, full traces | 1,024 | 9% | 8.8% | 4.5% | 0 | 0/128 | 90% |
| 360M | A2 code, full traces | 2,048 | 63% | 62% | 27.9% | 10/1,024 (9) | 8/128 = 6.3% [2.3, 10.9]; strict 7/128 | 35% |
| 360M | A2 code, LoRA r4 + markers | 2,048 | 58% | 54% | 18.4% | 6/1,024 (6) | 4/128 = 3.1% [0.8, 6.3] | 37% |
| control | constant `print(0)` (754 x 1) | - | 100% | 100% | 100% | 1/754 | n/a | 0 |
| control | copy the statement (754 x 1) | - | 100% | 0% | 0% | 0 | n/a | 0 |

Forgetting (G5 held-out v1 bits per byte, base 1.0981; 135M): A1 +0.011, A2 +0.019, A4 **+0.070** (above the team's 0.05
limit; the repetition of short rows is the likely cause, not tested).

Failure reasons, 135M A2 at 1,024 tokens: truncated 743 of 1,024, empty immediate reply 169, runtime error 54, wrong
answer 20. At 360M / 2,048: truncated 354, runtime error 306, wrong answer 276, pass 10. Raw outputs: A2 writes fluent
think-style reasoning that does not track the problem; A1 answers with a guessed number or a paraphrase; the passing
programs are short (sum two numbers, a YES / NO condition).

## What it says

1. **A 1,024-token screen of a model trained on reasoning traces measures the budget, not the model.** At 1,024 tokens
   74% (135M) and 90% (360M) of code-SFT samples ran out mid-reasoning; at 2,048 code blocks rose from 9% to 55-62% and
   running programs from 2-4% to 11-28%. Any post-SFT screen or RL run needs a budget matched to the trace length the SFT
   taught (the OCR-2 1k-effort traces have a median of ~960 target tokens).
2. **Short answers buy the answer shape cheaply.** Deleting the reasoning (A4) gave 99% code blocks and 35% running
   programs at 135M within 93 tokens per sample, against 1.9% at the same budget with traces. Correctness did not follow:
   0.4% pass.
3. **Correctness is the wall, and it moves with size.** Mixed groups: 135M 0-3%, 360M 6.3% (strict 5.5%). Below the
   10% line at both sizes, so RL would start with almost no gradient-carrying prompts below ~1B on this rung.
4. **LoRA reaches about half of full fine-tuning's signal at 360M** (mixed 3.1% vs 6.3%, runs 18% vs 28%).
5. **Format metrics do not measure capability.** Dolly-only SFT closes 92-96% of turns with no code; a constant program
   scores 100% on closed turn, code block and "runs"; 16.5 points of 135M A2's closed turns were an immediate empty reply.
   Only pass, preferably strict, measures content.

Answer to the question: before RL can start, SFT must deliver a closed turn with a running program inside the RL token
budget (short answers do this even at 135M) and enough correctness that ~10% of prompts give mixed groups. The second part
is a capability threshold: not reached at 135M or 360M; 1.7B is the next point (job request
`jobs/2026-11-17-e9-sft-before-rl-1.7b.md`).

## Found on the way

- **SFT data v1 already contains most RL data v1 code problems**: OCR-2 pack shards 0-3 hold 67% of the RL code pool and
  81% of rung 1 (13-word shingles); PR 106 confirms it at full scale by family key. E9 removed every evaluation problem
  from its SFT rows by text and by family key (72 rows were caught only by family key; none touch the decision sets).
- **One rung-1 task rewards a constant**: `print(0)` passes all its kept tests. It should leave the RL pool.
- **transformers 5.22 + SmolLM2**: `AutoTokenizer` rebuilds a tokenizer that splits a blank line into two tokens, unlike
  `tokenizer.json`; an RL run through `AutoTokenizer` sees different prompt tokens than SFT did. The E10 route loads
  `tokenizer.json` and checks prompt-token identity.

## What does not transfer

Tokenizer, architecture (KE2 MoE with an untied head), pre-training data and token count, optimizer (Muon), and the SFT
context. The thresholds (sizes where mixed groups appear) must be re-measured on the capstone checkpoint.
