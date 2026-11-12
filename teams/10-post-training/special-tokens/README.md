# special-tokens: G10 marker allocation

## What the files are

| File | What |
|---|---|
| `special_tokens_v0.md` | First list, sent to Team 3 before tokenizer training. It numbers the markers in a reserved block `000` to `255`. Rule: special tokens only for structure the model must never confuse with normal text (turn start and end, tool call, tool result, thinking); values such as effort labels stay plain text. |
| `special_tokens_v1.md` | Final consolidated allocation. It replaces the `000` to `255` numbering with Team 3's real token ids (for example `<|im_start|>` 129994, `<|im_end|>` 130816, `<|endoftext|>` 130822, `<|think|>` 130823). It also records the pre-tokenization rule (no NFKC) and per-dataset notes on ghost markers and joiners. |

## Gate

Gate 1 deliverable "Special-token list and chat format" (draft 28 Oct, final 31 Oct, plan sheet). `v1` is addressed to Teams 3, 4 and 9.

## What has been measured

These files are specifications. The only committed run that exercises the ids is `../results/render_frozen_tokenizer.txt`: one report on Team 3's tokenizer v0.1 that records the nine active markers as single tokens with the ids in `v1`. See `../chat-format/README.md`.

`v1` says some dataset checks found no ghost tokens (for example 0 of 500 documents in the `code-python` pack). Those checks are described in the file; this README has not re-run them.

## What has not been measured

- No check on the Gate 2 tokenizer (sha 15468e66...). That file is not in the repo as of 12 Nov.
- No model has been trained with these markers.
- The `v0` numbering is superseded. Use `v1` for ids.

## Reports

- [../results/render_frozen_tokenizer.txt](../results/render_frozen_tokenizer.txt)
- [../chat-format/chat_format_v0.md](../chat-format/chat_format_v0.md)
