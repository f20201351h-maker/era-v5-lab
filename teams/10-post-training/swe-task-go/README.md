# swe-task-go: one small SWE task in Go

## What the files are

| File | What |
|---|---|
| `stats.go` | Seeded bug: `Sum` loops from index 1, so it skips the first element. |
| `stats_test.go` | Go tests for `Sum` (the report names `TestSum`). |
| `go.mod`, `Dockerfile` | Go module and the Docker image the task runs in (`golang:1.23-alpine` per the report). |
| `run_e2e_go.py` | Runner: fails the test, applies the patch, re-runs it, then renders and parses the trajectory with `chat-format/render.py`. |

## Gate

Week 1 extension of the single-task SWE demonstration in `../swe-task/`, for Go. It is not an RL environment; the Go environments built from real PRs are in `../swe-envs/`.

## What has been measured

`results/go_swe_e2e.md` records one Docker run: the test fails before the patch (exit 1), the test and `go vet` are clean after it (exit 0), the 7-turn trajectory renders at effort `4k` and parses back to the same messages, and tabs, CRLF and `<-` survive the round trip. Run-time details are in that report.

## What has not been measured

- No model has attempted the task, and no score is reported.
- The run has not been repeated for this README.
- Go tests in `../tests/test_go_chat_format.py` have not been run for this README.

## Reports

- [../results/go_swe_e2e.md](../results/go_swe_e2e.md)
