"""prompt_pool.PromptPool: draws without replacement, the refresh rule, exact resume, stream mode. No model."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prompt_pool as pp  # noqa: E402


def simulate(pool, steps, kind_of, start=1):
    logs = []
    for step in range(start, start + steps):
        idx = pool.draw()
        logs.append((idx, pool.observe(step, idx, [kind_of(i, step) for i in idx])))
    return logs


def test_group_kind():
    assert pp.group_kind([1.0, 0.0]) == "mixed"
    assert pp.group_kind([1.0, 1.0]) == "all_pass"
    assert pp.group_kind([0.0, 0.0]) == "all_fail"
    assert pp.group_kind([0.5, 0.5]) == "tied_other"
    assert pp.group_kind([1.0]) == "tied_other"


def test_no_repeat_within_a_step_and_each_pass_covers_the_cohort():
    pool = pp.PromptPool(100, per_step=3, seed=1, cohort_size=7, refresh_below=0)
    seen = []
    for step in range(1, 15):
        idx = pool.draw()
        assert len(set(idx)) == 3 and set(idx) <= set(pool.initial)
        pool.observe(step, idx, ["mixed"] * 3)
        seen += idx
    # 14 steps x 3 = 42 draws = 6 full passes over 7 prompts: every prompt drawn exactly 6 times
    assert sorted(seen.count(i) for i in pool.initial) == [6] * 7


def test_refresh_fires_below_half_and_replaces_saturated_prompts():
    pool = pp.PromptPool(50, per_step=4, seed=2, cohort_size=8, refresh_below=0.5, baseline_steps=2, window=2)
    first = list(pool.active)
    # steps 1-2 all mixed (baseline 4), then every group is all-pass: kept 0 < 0.5 * 4 over a 2-step window
    logs = simulate(pool, 6, lambda i, s: "mixed" if s <= 2 else "all_pass")
    fired = [lg for _, lg in logs if lg["refreshed"]]
    assert fired and fired[0]["refreshed"]["step"] == 4
    ev = fired[0]["refreshed"]
    assert ev["added"] == ev["retired"] > 0 and not ev["pool_exhausted"]
    assert len(pool.active) == 8 and set(pool.active) != set(first)
    assert pool.baseline == 4.0
    # newcomers come from the pool order after the initial cohort, in order
    assert set(pool.active) - set(first) <= set(pool.pool_order[8:8 + ev["added"] + 8])


def test_cooldown_prevents_back_to_back_refreshes():
    pool = pp.PromptPool(200, per_step=2, seed=3, cohort_size=4, refresh_below=0.5, baseline_steps=1, window=3)
    logs = simulate(pool, 12, lambda i, s: "mixed" if s == 1 else "all_fail")
    steps = [lg["refreshed"]["step"] for _, lg in logs if lg["refreshed"]]
    assert all(b - a >= 3 for a, b in zip(steps, steps[1:]))


def test_refresh_below_zero_never_refreshes_and_draws_like_refresh_until_it_fires():
    fixed = pp.PromptPool(60, per_step=4, seed=9, cohort_size=8, refresh_below=0)
    fresh = pp.PromptPool(60, per_step=4, seed=9, cohort_size=8, refresh_below=0.5, baseline_steps=2, window=2)
    kind = lambda i, s: "mixed" if s <= 2 else "all_pass"  # noqa: E731
    a, b = simulate(fixed, 8, kind), simulate(fresh, 8, kind)
    assert not any(lg["refreshed"] for _, lg in a)
    first_fire = next(n for n, (_, lg) in enumerate(b) if lg["refreshed"])
    assert [i for i, _ in a[: first_fire + 1]] == [i for i, _ in b[: first_fire + 1]]


def test_pool_exhaustion_is_logged_and_drawing_still_works():
    pool = pp.PromptPool(10, per_step=3, seed=4, cohort_size=8, refresh_below=0.5, baseline_steps=1, window=1)
    logs = simulate(pool, 10, lambda i, s: "mixed" if s == 1 else "all_pass")
    evs = [lg["refreshed"] for _, lg in logs if lg["refreshed"]]
    assert any(e["pool_exhausted"] for e in evs)
    assert all(len(set(i)) == 3 for i, _ in logs)


def test_state_roundtrip_through_json_gives_identical_future():
    kind = lambda i, s: "mixed" if (i + s) % 3 else ("all_pass" if s > 3 else "all_fail")  # noqa: E731
    whole = pp.PromptPool(80, per_step=4, seed=5, cohort_size=10, refresh_below=0.5, baseline_steps=2, window=2)
    a = simulate(whole, 20, kind)
    part = pp.PromptPool(80, per_step=4, seed=5, cohort_size=10, refresh_below=0.5, baseline_steps=2, window=2)
    b = simulate(part, 7, kind)
    saved = json.loads(json.dumps(part.state_dict()))
    resumed = pp.PromptPool(80, per_step=4, seed=5, cohort_size=10, refresh_below=0.5, baseline_steps=2, window=2)
    resumed.load_state_dict(saved)
    b += simulate(resumed, 13, kind, start=8)
    assert a == b
    assert whole.state_dict() == resumed.state_dict()


def test_stream_mode_takes_unseen_prompts_in_order_then_reshuffles():
    pool = pp.PromptPool(10, per_step=4, seed=6, mode="stream")
    first = pool.pool_order[:]
    logs = simulate(pool, 2, lambda i, s: "all_fail")
    assert [i for i, _ in logs] == [first[:4], first[4:8]]
    logs = simulate(pool, 2, lambda i, s: "all_fail", start=3)
    assert logs[0][1]["unique_seen"] == 10 and all(len(set(i)) == 4 for i, _ in logs)


def test_screened_cohort_and_fresh_list():
    pool = pp.PromptPool(30, per_step=2, seed=7, initial=[3, 5, 7], fresh=list(range(20, 30)))
    assert pool.active == [3, 5, 7] and sorted(pool.pool_order) == list(range(20, 30))


def test_bad_configs_are_refused():
    with pytest.raises(ValueError):
        pp.PromptPool(10, per_step=4, seed=1, cohort_size=2)
    with pytest.raises(ValueError):
        pp.from_config({"cohort": 3}, 10, 2, 1)
    with pytest.raises(ValueError):
        pp.PromptPool(10, per_step=2, seed=1, mode="bogus", cohort_size=4)
    pool = pp.PromptPool(10, per_step=2, seed=1, cohort_size=4)
    other = pp.PromptPool(10, per_step=3, seed=1, cohort_size=4)
    with pytest.raises(RuntimeError):
        other.load_state_dict(pool.state_dict())


def test_short_screened_cohort_is_topped_up_from_unscreened_prompts():
    pool = pp.PromptPool(30, per_step=4, seed=8, cohort_size=6, initial=[1], fresh=list(range(10, 30)))
    assert len(pool.active) == 6 and pool.active[0] == 1 and pool.topped_up == 5 and pool.cursor == 5
    assert set(pool.active[1:]) <= set(range(10, 30))
    empty = pp.PromptPool(30, per_step=4, seed=8, initial=[], fresh=list(range(10, 30)))
    assert len(empty.active) == 4 and len(set(empty.draw())) == 4
