"""train.pool inside the real trainer (toy task, tiny random OLMoE, CPU, objective_v1): logging, exact resume, the
matched no-refresh arm, and the stand-in engine (sampler-only) branch. Fixtures, not training data."""
import copy
import json
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train  # noqa: E402
from test_resume_state import TIMING, final_state, same_state  # noqa: E402
from test_sampler_branches import rows as engine_rows  # noqa: E402
from test_sampler_branches import run as engine_run  # noqa: E402

# refresh_below 2.0: on the toy task kept groups RISE as it learns, so the 0.5 rule would never fire here; 2.0 makes the
# refresh branch run inside the trainer (every `window` steps after the baseline), which is what these tests need
POOL = {"mode": "refresh", "cohort_size": 24, "refresh_below": 2.0, "baseline_steps": 2, "window": 2}


def cfg_for(out: Path, steps: int, pool: dict | None = POOL, **train_kw) -> dict:
    cfg = yaml.safe_load((HERE / "config" / "smoke_objective_v1.yaml").read_text())
    cfg["out_dir"], cfg["verbose"] = str(out), False
    cfg["train"].update(steps=steps, eval_every=4, state_every=2, state_dir=str(out / "state"), ppo_epochs=2,
                        prompts_per_step=8, **train_kw)
    if pool is not None:
        cfg["train"]["pool"] = dict(pool)
    return cfg


def metric_rows(out: Path) -> list[dict]:
    return [{k: v for k, v in json.loads(line).items() if k not in TIMING}
            for line in (out / "metrics.jsonl").read_text().splitlines()]


def test_pool_run_logs_every_step_and_refreshes(tmp_path):
    s = train.train(cfg_for(tmp_path / "run", 16))
    assert s["status"] == "COMPLETE"
    recs = metric_rows(tmp_path / "run")
    assert len(recs) == 16 and all("pool" in r for r in recs)
    for r in recs:
        p = r["pool"]
        assert p["kept"] + p["all_pass"] + p["all_fail"] <= 8 and p["kept"] == r["group_kinds"]["mixed"]
        assert p["all_pass"] == r["group_kinds"]["all_one"] and p["all_fail"] == r["group_kinds"]["all_zero"]
    seen = [r["pool"]["unique_seen"] for r in recs]
    assert seen == sorted(seen) and seen[-1] >= 24  # the whole starting cohort was drawn
    summ = s["prompt_pool"]
    assert summ["kept_per_step"] == [r["pool"]["kept"] for r in recs]
    assert summ["refreshes"] and summ["refreshes"][0]["step"] == 3  # baseline after 2 steps, window 2
    assert summ["pool_used"] > 24 and seen[-1] > 24  # fresh prompts entered from the pool and were drawn


def test_pool_stop_and_continue_equals_uninterrupted(tmp_path):
    whole = train.train(cfg_for(tmp_path / "whole", 12))
    cfg = cfg_for(tmp_path / "split", 12)
    cfg["train"]["max_steps_this_call"] = 5
    assert train.train(copy.deepcopy(cfg))["status"] == "PAUSED_AT_DEADLINE"
    del cfg["train"]["max_steps_this_call"]
    done = train.train(copy.deepcopy(cfg))
    assert done["status"] == "COMPLETE" and done["resumed_from_step"] == 5
    assert metric_rows(tmp_path / "whole") == metric_rows(tmp_path / "split")
    a, b = final_state(tmp_path / "whole"), final_state(tmp_path / "split")
    same_state(a, b)
    assert a["pool"] == b["pool"] and a["pool"]["refreshes"]  # a refresh happened on each side of the split
    assert whole["prompt_pool"] == done["prompt_pool"] and whole["heldout_change"] == done["heldout_change"]


def test_no_refresh_arm_matches_refresh_arm_until_the_first_refresh(tmp_path):
    fresh = train.train(cfg_for(tmp_path / "fresh", 12))
    fixed = train.train(cfg_for(tmp_path / "fixed", 12, {**POOL, "refresh_below": 0}))
    assert not fixed["prompt_pool"]["refreshes"]
    ra, rb = metric_rows(tmp_path / "fresh"), metric_rows(tmp_path / "fixed")
    first = next((r["step"] for r in ra if r["pool"]["refreshed"]), None)
    upto = first if first is not None else len(ra)
    for x, y in zip(ra[:upto], rb[:upto]):  # same draws, same samples, same updates until the cohorts differ
        x = {k: v for k, v in x.items() if k != "pool"}
        y = {k: v for k, v in y.items() if k != "pool"}
        assert x == y


def test_pool_identity_guards_resume(tmp_path):
    cfg = cfg_for(tmp_path / "run", 4)
    cfg["train"]["max_steps_this_call"] = 2
    train.train(copy.deepcopy(cfg))
    cfg["train"]["pool"]["refresh_below"] = 0.9
    import pytest  # noqa: PLC0415
    with pytest.raises(RuntimeError, match="different run"):
        train.train(cfg)


def test_fixed_path_state_has_no_pool_key(tmp_path):
    train.train(cfg_for(tmp_path / "run", 2, pool=None))
    st = final_state(tmp_path / "run")
    assert "pool" not in st and "pool" not in st["identity"]["train"]


def test_stand_in_engine_with_pool_holds_the_policy_and_resumes(monkeypatch, tmp_path):
    out = tmp_path / "eng"
    extra = {"pool": {"mode": "refresh", "cohort_size": 32, "refresh_below": 0.5, "baseline_steps": 2, "window": 2},
             "max_steps_this_call": 3}
    first, _ = engine_run(monkeypatch, out, 6, False, extra)
    assert first["status"] == "PAUSED_AT_DEADLINE"
    extra.pop("max_steps_this_call")
    done, fake = engine_run(monkeypatch, out, 6, False, extra)
    assert done["status"] == "COMPLETE" and done["resumed_from_step"] == 3
    recs = engine_rows(out)
    assert [r["step"] for r in recs] == [1, 2, 3, 4, 5, 6] and all("pool" in r for r in recs)
    late = [r["mismatch_k3"] for r in recs[3:] if r["groups_used"]]
    assert late and max(late) < 1e-9 and fake.syncs


def test_stream_mode_runs(tmp_path):
    s = train.train(cfg_for(tmp_path / "run", 6, {"mode": "stream"}))
    recs = metric_rows(tmp_path / "run")
    assert s["status"] == "COMPLETE" and recs[-1]["pool"]["unique_seen"] == 48  # 6 x 8 distinct prompts
    assert torch.isfinite(torch.tensor([r["loss"] for r in recs])).all()
