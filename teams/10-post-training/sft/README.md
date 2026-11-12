# sft: SFT data v1 tooling and records

## What the files are

| Group | Files |
|---|---|
| Proposal | `sft_mix_proposal.md` (area shares 35 / 25 / 15 / 15 / 7 / 3, effort distribution, revision notes to 9 Nov). |
| Conversion | `convert.py` turns one candidate set into chat-format v0 JSONL with loss masks. `convert_report.md` records a 1,000-row test on four sets. |
| Decontamination | `decontam_run.py`, `decontam_pipeline.py`, `decontam_report.py` (writes `decontam_report.md`), `decontam_kept_rows.md` (rows kept per set). |
| OpenCodeReasoning-2 (OCR-2) | `source_index.py`, `prepare_ocr2.py`, `run_ocr2.py`, `pack_admission.py`, `ocr2_policy.json`, `ocr2_fixtures.py`. |
| Inventory | `mix_inventory.py` builds `results/eligibility_ledger.jsonl` and `../docs/eligibility_ledger.md`. |
| Tests | `test_*.py` for convert, decontam_run, mix_inventory, ocr2_resume, pack_admission and prepare_ocr2. `verify.sh` re-runs the OCR-2 tests and fails if any is skipped. |
| `results/` | `eligibility_ledger.jsonl`, `ocr2_coverage.md`, `ocr2_full_run.json`, `ocr2_full_run_inputs.tsv`. |

## Gate

Workstream "SFT data v1" (1 to 21 Nov, plan sheet). The SFT mix is one of the items TSAI decides at Gate 2 (14 Nov).

## What has been measured (from the committed reports)

- **Decontamination** against Team 9 fingerprints v2, full set (`set_sha256` 2826a3b8...): 34 sets, 18,376,709 rows in, 464,365 removed (2.53%), 17,912,344 kept, 0 canaries. See `decontam_report.md`.
- **OCR-2, all 70 Python shards** (`results/ocr2_coverage.md`, `../docs/eligibility_ledger.md`): raw 1,398,166 rows; 786,190 excluded by policy; 69,690 held; 542,286 joined; 67 rejected by the converter; 542,219 scanned; 115,067 removed; 427,152 kept; **admitted 0**.
- **Kept OCR-2 rows, token lengths** (ledger): p50 5,007, max 29,851; 2,714,970,210 sequence tokens in total.
- **Conversion test:** 1,000 rows each on four sets (`convert_report.md`). No data file is committed.

## What has not been done

- **No SFT run.** No model has been fine-tuned on these rows. No pack is admitted. The ledger counts 34 scanned entries and 0 approved complete training packs.
- **Token records exist for OCR-2 only.** 32 other scanned sets have row counts and no token record (ledger, section 6, item 4).
- **Overlap between OCR-2, OpenCodeReasoning and codeforces-cots** is not measured.
- **Safety share.** The table in the proposal says 3% Safety, but the revision notes say to use 0% Safety until TSAI names an allowed safety set with chat rows.
- **The tests in this folder were not run for this README.** The only run record is whatever the reports above quote.

## Blockers

- **max_sequence_tokens is not set.** `configs/config_v0.yaml` `data.seq_len` is null and the OCR-2 manifest has `max_sequence_tokens` null, so the admission check reports `context_limit_not_set` (Gate 2).
- **Source identity is not verified** for `apps/train`, `code_contests/train` and `taco/train` (`source_identity_not_verified:*`), and those sources are not approved for training. The pack report also lists `set_not_allowed_in_licence_table`, `set_not_allowed_in_policy`, `join_not_in_admission_mode` and `tokenizer_not_approved`.
- **The tokenizer is not in the repo as of 12 Nov.** Gate 2 references sha 15468e66...; `teams/03-tokenizer/` has build scripts and `SHA256SUMS` but no `tokenizer.json` for it. The OCR-2 pack check lists `tokenizer_not_approved`, and `../docs/eligibility_ledger.md` notes that token counts taken before the frozen tokenizer arrives would need a rerun.
- **Codeforces pool: 69,690 `open-r1/codeforces` train rows are held** until the producer revision is settled.

## Reports

- [sft_mix_proposal.md](sft_mix_proposal.md)
- [decontam_report.md](decontam_report.md), [decontam_kept_rows.md](decontam_kept_rows.md)
- [convert_report.md](convert_report.md)
- [results/ocr2_coverage.md](results/ocr2_coverage.md)
- [../docs/eligibility_ledger.md](../docs/eligibility_ledger.md)
