# chat-format: ChatML-style format v0

## What the files are

| File | What |
|---|---|
| `chat_format_v0.md` | The format spec: special tokens for structure, plain text for values, effort as the plain-text line `Reasoning effort: {1k\|4k\|16k\|64k}`. |
| `render.py` | `render()` and `parse()` for that format. It reads token ids from Team 3's specification or a `tokenizer.json` and rejects literal markers and look-alikes inside message content. |
| `chat_template.jinja` | Jinja2 version of the same template, written for Team 12. |
| `team12_chat_template_handoff.md` | Hand-off note to Team 12 with the paths to use. |
| `verify_frozen_tokenizer.py` | Builds or loads Team 3's tokenizer v0.1, checks its sha256 (`a33c9907...`) and checks the nine G10 markers and look-alikes. |

## Gate

Gate 1 deliverable "Special-token list and chat format" (final 31 Oct). The frozen-tokenizer check is evidence for Gate 2 (14 Nov).

## What has been measured

`../results/render_frozen_tokenizer.txt` is a committed report of one run on Team 3's tokenizer **v0.1** (sha256 `a33c9907...`, matched against `teams/03-tokenizer/v0/SHA256SUMS`). Its lines record the nine markers as single tokens, look-alikes encoded as plain text, round trips of five Indic scripts, and a multi-turn round trip at four effort labels. The tests that exercise `render.py` are in `../tests/` and have not been run for this README.

## What has not been measured

- **The Gate 2 tokenizer.** Gate 2 references sha 15468e66... (the C3 sample-v0 rebuild in `teams/03-tokenizer/README.md`). The tokenizer file is not in the repo as of 12 Nov, so nothing here has been run on it. The report above is for v0.1, a different file.
- **Parity with Team 12's copy of the template.** The test that compares them skips when Team 12's copy is absent.
- **A model reading or producing this format.** No SFT run exists (see `../sft/README.md`).
- Browser glyph rendering for Indic text is a separate review (`../tests/test_kannada_chat_format.py` says so).

## Reports

- [../results/render_frozen_tokenizer.txt](../results/render_frozen_tokenizer.txt)
- [../special-tokens/special_tokens_v1.md](../special-tokens/special_tokens_v1.md)
