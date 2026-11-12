# swe-task: small SWE smoke tasks (Python)

## What the files are

Four small repair tasks. Each has a seeded failing state, a test, a runner that applies a patch and re-runs the test, and a check that the trajectory renders and parses with `chat-format/render.py`.

| Task | Files |
|---|---|
| Calculator (`add` returns `a + b + 1`) | `calc.py`, `test_calc.py`, `run_e2e.py`, `Dockerfile` |
| Hindi text preparation | `hindi_text_prep.py`, `run_hindi_e2e.py`, `Dockerfile.hindi` |
| Kannada text preparation | `kannada_text_prep.py`, `kannada_examples.py`, `preview_kannada.py`, `run_kannada_e2e.py`, `Dockerfile.kannada` |
| Tamil UTF-8 writer | `tamil_utf8_io.py`, `run_tamil_e2e.py` (no Docker) |

## Gate

Week 1 work on the G10 chat format and one SWE task end to end. These are not the RL environments; those are in `../swe-envs/`.

## What has been measured (committed reports)

- `results/swe_e2e.md`: the calculator task, run in Docker, before and after the patch.
- `results/hindi_swe_e2e.md`: run 30 Oct 2026 in Docker. The report says it is an authored smoke task and that it does not run a model, train a model or report a SWE-bench score.
- `results/tamil_swe_e2e.md`: run 28 Oct 2026 with local Python subprocesses, no Docker. Same statement about no model and no score.
- `results/kannada_swe_e2e.md`: Kannada serialization and a before/after run. The report also lists a browser check of eight examples; that check is not reproduced by these files.

## What has not been measured

- No model has attempted any of these tasks.
- No score of any kind. These tasks have one bug each and are too small to compare models.
- Hindi and Tamil use authored strings, not diagnosis-pack text (their reports say so).
- The Docker runs have not been repeated for this README.

## Tests

Task tests sit in `test_calc.py` and in `../tests/` (`test_hindi_text_prep.py`, `test_kannada_text_prep.py`, `test_tamil_utf8_io.py`, `test_kannada_chat_format.py`).

## Reports

- [../results/swe_e2e.md](../results/swe_e2e.md)
- [../results/hindi_swe_e2e.md](../results/hindi_swe_e2e.md)
- [../results/kannada_swe_e2e.md](../results/kannada_swe_e2e.md)
- [../results/tamil_swe_e2e.md](../results/tamil_swe_e2e.md)
