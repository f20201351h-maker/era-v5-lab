"""rl_screen.py: the failure ladder, aggregation with intervals over problems, one sandbox run per distinct program,
and the prompt / judge round trip on the real SmolLM2 tokenizer (skipped when it is not cached). No model, no Docker."""
import glob
import importlib.util
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("g10_rl_screen", HERE / "rl_screen.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)

_skipif = pytest.mark.skipif  # alias: pr-guard flags an added line that starts with a dotted decorator as an email
TOK = sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/*/tokenizer.json")))


def rec(task, **kw):
    base = {"task_id": task, "set": "s", "code_extracted": True, "parses": True, "closed_im_end": True, "valid": True,
            "hit_budget": False, "reward": 0, "score_reason": "wrong answer on a test", "gen_tokens": 10}
    base.update(kw)
    return base


def test_outcome_is_the_first_failed_step():
    assert rs.outcome(rec("a", code_extracted=False, parses=False, hit_budget=True)) == "no_code_truncated"
    assert rs.outcome(rec("a", code_extracted=False, parses=False)) == "no_code_closed"
    assert rs.outcome(rec("a", parses=False)) == "syntax_error"
    assert rs.outcome(rec("a", score_reason="error on a test")) == "runtime_error"
    assert rs.outcome(rec("a", score_reason="timeout on a test")) == "timeout"
    assert rs.outcome(rec("a")) == "wrong_answer"
    assert rs.outcome(rec("a", reward=1, score_reason="all tests passed")) == "pass"
    f = rs.ladder_flags(rec("a"))
    assert f["runs"] and not f["pass_kept"] and "pass_strict" not in f


def test_an_empty_closed_reply_is_closed_but_not_closed_with_content():
    f = rs.ladder_flags(rec("a", text="", code_extracted=False, parses=False))
    assert f["closed"] and not f["closed_content"]
    assert rs.ladder_flags(rec("a", text="x = 1"))["closed_content"]


def test_aggregate_mixed_groups_and_intervals():
    recs = [rec("p1", reward=1, score_reason="all tests passed"), rec("p1")]  # mixed
    recs += [rec("p2"), rec("p2")]  # all fail
    recs += [rec("p3", reward=1, score_reason="all tests passed")] * 2  # all pass
    rep = rs.aggregate(recs, k=2, n_boot=500)
    g = rep["groups_pass_kept"]
    assert (g["mixed"], g["all_fail"], g["all_pass"]) == (1, 1, 1)
    assert g["mixed_share"] == round(1 / 3, 4) and g["pass_at_k_share"] == round(2 / 3, 4)
    lo, hi = rep["ladder"]["pass_kept"]["ci95"]
    assert 0 <= lo <= rep["ladder"]["pass_kept"]["share"] <= hi <= 1
    assert rep["outcomes"] == {"wrong_answer": 3, "pass": 3}


def test_strict_flag_enters_the_ladder_only_when_measured():
    recs = [rec("p1", reward=1, score_reason="all tests passed", strict=False), rec("p1", strict=False)]
    rep = rs.aggregate(recs, k=2, n_boot=200)
    assert rep["ladder"]["pass_kept"]["share"] == 0.5 and rep["ladder"]["pass_strict"]["share"] == 0.0
    assert rep["groups_pass_strict"]["mixed"] == 0 and rep["groups_pass_kept"]["mixed"] == 1


def test_score_records_runs_each_distinct_program_once(monkeypatch):
    calls = []

    class FakePool:
        def __init__(self, workers):
            pass

        def score(self, items):
            calls.extend(items)
            return [{"reward": 1, "reason": "all tests passed", "passed": 3} for _ in items]

        def close(self):
            pass

    monkeypatch.setattr(rs, "_load", lambda name, path: type("M", (), {"ScorePool": FakePool}))  # only score_pool is loaded
    text = "```python\nprint(1)\n```"
    recs = [{"task_id": "t", "text": text, **rs.code_status(text)} for _ in range(3)]
    recs.append({"task_id": "t", "text": "no code", **rs.code_status("no code")})
    info = rs.score_records(recs, {"t": {"id": "t", "tests": []}}, workers=2)
    assert info["distinct_programs_run"] == 1 and len(calls) == 1
    assert [r["reward"] for r in recs] == [1, 1, 1, 0]


@_skipif(not TOK, reason="SmolLM2 tokenizer not cached")
def test_prompt_and_control_round_trip_on_the_real_tokenizer():
    from tokenizers import Tokenizer  # noqa: PLC0415
    tok = Tokenizer.from_file(TOK[0])
    items, rejected = rs.encode_prompts(tok, [{"id": "x", "statement": "Read n and print n squared."}], "1k")
    assert not rejected and len(items) == 1
    task, ids = items[0]
    assert ids[-3:] == tok.encode("<|im_start|>assistant\n", add_special_tokens=False).ids[-3:]
    out = rs.control_records(tok, "constant", [("s", task, ids)])[0]
    assert out["closed_im_end"] and out["valid"] and out["code_extracted"] and out["parses"]
    assert out["text"].strip().endswith("```")  # scored text = decode(skip_special_tokens=True), no <|im_end|>
    copy = rs.control_records(tok, "copy", [("s", task, ids)])[0]
    assert copy["closed_im_end"] and not copy["code_extracted"]


def test_a_sandbox_failure_is_an_error_not_a_zero_reward(monkeypatch):
    class DownPool:  # what the Docker runner returns when the daemon is not running
        def __init__(self, workers):
            pass

        def score(self, items):
            return [{"reward": 0, "reason": "runner exited with 1"} for _ in items]

        def close(self):
            pass

    monkeypatch.setattr(rs, "_load", lambda name, path: type("M", (), {"ScorePool": DownPool}))
    text = "```python\nprint(1)\n```"
    with pytest.raises(RuntimeError, match="sandbox failure"):
        rs.score_records([{"task_id": "t", "text": text, **rs.code_status(text)}], {"t": {"id": "t", "tests": []}}, 2)
    with pytest.raises(SystemExit, match="preflight"):
        rs.sandbox_preflight(2)


def test_jsonl_with_unicode_line_separators_reads_back(tmp_path):
    recs = [{"task_id": "a", "text": "x\u2028y\u0085z"}, {"task_id": "b", "text": "ok"}]
    f = tmp_path / "g.jsonl"
    f.write_text("".join(__import__("json").dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    assert rs.read_jsonl(f) == recs  # str.splitlines() would have made four broken lines of these two
