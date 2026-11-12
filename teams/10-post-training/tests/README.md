# tests: chat-format, special-token and small SWE task tests

## What the files are

pytest and unittest files for the Week 1 deliverables. Tests for other folders sit next to their code (`../eval/`, `../rl/`, `../rl-data/`, `../rl-loop/`, `../sft/`, `../swe-envs/`).

| File | What it checks |
|---|---|
| `test_specs.py` | Week 1 spec checks: the assigned markers, files present for the SWE task, and the docs. |
| `test_render.py` | `chat-format/render.py` against the special-token list and chat format v0 (uses a mock tokenizer for some cases). |
| `test_chat_template.py` | `chat_template.jinja` renders like `render.py`. The comparison with Team 12's copy skips when that copy is absent. |
| `test_frozen_tokenizer.py` | `render.py` and the format on Team 3's tokenizer v0.1. It rebuilds or loads the tokenizer through `verify_frozen_tokenizer.py` (cache in `~/.cache/era-v5` or `G10_TOKENIZER_CACHE`), and the baseline download needs network. |
| `test_code_python_chat_format.py` | Python source and StarCoder tags through `render.py`. No Docker. |
| `test_go_chat_format.py` | Go cases for the chat format and the Go task files. No Docker. |
| `test_hindi_text_prep.py`, `test_kannada_text_prep.py`, `test_kannada_chat_format.py`, `test_tamil_utf8_io.py` | Text-preservation cases for the Hindi, Kannada and Tamil helpers. Standard library only. |

## Gate

Gate 1 deliverables (special tokens, chat format, SWE task). The tokenizer test is also the code behind the Gate 2 frozen-tokenizer evidence.

## What has been measured

No result file for this folder as a whole is committed. Two reports quote counts for the tests at the time they were written: `../results/kannada_swe_e2e.md` ("All 31 G10 tests passed") and `../results/render_frozen_tokenizer.txt` (the marker and round-trip lines). Neither is a fresh run.

## What has not been measured

- The tests were not run for this README. No pytest result is attached to this folder at this commit.
- `test_frozen_tokenizer.py` has been written for v0.1 (sha256 `a33c9907...`). It has not been run on the Gate 2 tokenizer (sha 15468e66...), which is not in the repo as of 12 Nov.
- Template parity with Team 12's copy has not been compared.

```bash
cd teams/10-post-training
python -m pytest tests -q -p no:cacheprovider
```
