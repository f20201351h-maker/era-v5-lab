# licences: licence table

## What the file is

`licence_table.md` is the shared G10 licence record (about 1,500 lines). It has:
- a consolidated table in three stages: Stage 0 foundation data, Stage 1 SFT mixture, Stage 2 preference and RLVR;
- one section per core-team member for the dataset they checked;
- one section per Stage 1 candidate set, with source, generator model, licence and decision.

Decision values used in the table include `TSAI reserves`, `excluded`, `allowed` (with a date or condition) and `needs TSAI decision`. The data rule it applies is the TSAI rule of 3 Nov: human-written data, execution-verified synthesis and output of open-weight models whose licence allows training on it are in; output of closed models, Llama and use-restricted licences are out.

## Gate

Week 1 deliverable "Licence table and strategy v0" (plan sheet, 24 to 31 Oct). Rows have been added and revised since, for example the RL and SWE rows dated 4, 6 and 11 Nov.

## What has been measured

Nothing. This is a record of licences, generators and decisions, not a measurement. Other files in the folder tree cite it (`sft/sft_mix_proposal.md`, `sft/decontam_report.md`, `docs/eligibility_ledger.md`).

## What has not been done

- No row-level licence audit of any set. `docs/eligibility_ledger.md` says it does not audit CommitPackFT row or repository licences.
- No licence-table row for hendrycks MATH, which `eval/effort_manifest.json` uses (`results/effort_probe.md` records the gap).
- Many Stage 1 sets are still `TSAI reserves`. A `TSAI reserves` set is not approved for training.

## Reports

- [docs/eligibility_ledger.md](../docs/eligibility_ledger.md) reads this table together with the decontamination report.
