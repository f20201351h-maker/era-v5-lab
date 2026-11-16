"""Pool-refresh prompt sampler for the RL loop (config `train.pool`, optional; without it nothing changes).

Why (RESULTS R4, 15 Nov): with 56 fixed prompts all of the held-out gain was in by step 10; afterwards the groups
turned all-pass, kept groups per step fell from 12.6 to 3.0, and the run memorised its cohort. This sampler keeps the
cohort fresh instead.

mode: refresh (default)
    An active cohort of `cohort_size` prompts (or the screened cohort the trainer hands over). Each step takes
    `prompts_per_step` of them WITHOUT replacement: a queue holds one shuffled pass over the cohort, and when fewer
    than a step's worth remain, the next pass (a fresh shuffle of the other cohort prompts) is appended, so a step
    never repeats a prompt. Kept groups (groups with unequal rewards) are counted every step. The starting level is
    their mean over the first `baseline_steps` steps. When the mean over the last `window` steps falls below
    `refresh_below` x the starting level, every cohort prompt whose most recent group was all-pass or all-fail
    (the kinds in `retire`) leaves the cohort, and as many unseen prompts as left enter it from the pool, in the
    pool's fixed shuffled order. Then nothing can fire again for `window` steps. When the pool is used up, the cohort
    shrinks and the log says so. `refresh_below: 0` never refreshes (the matched fixed-cohort arm: identical draws).
mode: stream
    No cohort: every step takes the next `prompts_per_step` unseen prompts of the shuffled pool (a new shuffled pass
    when it runs out) and lets group dropping do the screening.

The whole sampler state (pool order, cursor, cohort, queue, kept history, last kind per prompt, refresh events) is a
JSON-serialisable dict saved in the trainer's durable state, so a resumed run draws exactly what the uninterrupted
run would have drawn. Indices refer to the trainer's prompt list as built from the config (deterministic).
"""
from __future__ import annotations

import random

KINDS = ("mixed", "all_pass", "all_fail", "tied_other")


def group_kind(rewards: list[float]) -> str:
    if len(rewards) < 2 or max(rewards) != min(rewards):
        return "mixed" if len(rewards) >= 2 else "tied_other"
    return "all_pass" if rewards[0] == 1.0 else "all_fail" if rewards[0] == 0.0 else "tied_other"


class PromptPool:
    def __init__(self, n_pool: int, per_step: int, seed: int, mode: str = "refresh", cohort_size: int = 0,
                 refresh_below: float = 0.5, baseline_steps: int = 3, window: int = 3,
                 retire: tuple[str, ...] = ("all_pass", "all_fail"), initial: list[int] | None = None,
                 fresh: list[int] | None = None):
        if mode not in ("refresh", "stream"):
            raise ValueError(f"pool mode {mode!r}: refresh | stream")
        if per_step < 1 or n_pool < 1:
            raise ValueError("pool needs prompts_per_step >= 1 and a non-empty pool")
        self.mode, self.per_step, self.refresh_below = mode, int(per_step), float(refresh_below)
        self.baseline_steps, self.window, self.retire = int(baseline_steps), int(window), tuple(retire)
        self.rng = random.Random(seed)
        self.topped_up = 0
        if initial is not None:  # the trainer's screened cohort; `fresh` = the prompts that may enter later
            self.active = list(initial)
            init_set = set(initial)
            self.pool_order = list(fresh if fresh is not None else [i for i in range(n_pool) if i not in init_set])
            self.rng.shuffle(self.pool_order)
            # a screen that kept fewer than cohort_size (or than one step needs) is topped up with unscreened prompts
            want = max(int(cohort_size), per_step) if mode == "refresh" else 0
            self.topped_up = max(0, min(want - len(self.active), len(self.pool_order)))
            self.active += self.pool_order[: self.topped_up]
            self.cursor = self.topped_up
        else:
            self.pool_order = list(range(n_pool))
            self.rng.shuffle(self.pool_order)
            take = int(cohort_size) if mode == "refresh" else 0
            if mode == "refresh" and take < per_step:
                raise ValueError("refresh mode needs cohort_size >= prompts_per_step")
            self.active, self.cursor = self.pool_order[:take], take
        if mode == "refresh" and len(self.active) < per_step:
            raise ValueError(f"cohort of {len(self.active)} prompts is smaller than prompts_per_step {per_step}")
        self.initial = list(self.active)
        self.queue: list[int] = []
        self.kept_hist: list[int] = []
        self.baseline: float | None = None
        self.last_kind: dict[int, str] = {}
        self.seen: set[int] = set()
        self.refreshes: list[dict] = []
        self.cooldown_until = 0
        self.epochs = 0  # stream mode: completed passes over the pool

    # ------------------------------------------------------------ drawing
    def draw(self) -> list[int]:
        if self.mode == "stream":
            out = []
            while len(out) < self.per_step:
                if self.cursor >= len(self.pool_order):
                    self.rng.shuffle(self.pool_order)
                    self.cursor, self.epochs = 0, self.epochs + 1
                cand = self.pool_order[self.cursor]
                self.cursor += 1
                if cand not in out:
                    out.append(cand)
            return out
        if len(self.queue) < self.per_step:
            # next full pass over the cohort; prompts still waiting in the queue are moved behind the slots this
            # step takes from the new pass, so every pass holds each prompt once and no step repeats one
            rest = set(self.queue)
            nxt = list(self.active)
            self.rng.shuffle(nxt)
            need = self.per_step - len(self.queue)
            head = [i for i in nxt if i not in rest][:need]
            hs = set(head)
            self.queue += head + [i for i in nxt if i not in hs]
        out, self.queue = self.queue[: self.per_step], self.queue[self.per_step:]
        return out

    # ------------------------------------------------------------ feedback
    def observe(self, step: int, idxs: list[int], kinds: list[str]) -> dict:
        if len(idxs) != len(kinds):
            raise ValueError("one kind per drawn prompt")
        for i, k in zip(idxs, kinds):
            if k not in KINDS:
                raise ValueError(f"unknown group kind {k!r}")
            self.last_kind[i] = k
            self.seen.add(i)
        kept = sum(k == "mixed" for k in kinds)
        self.kept_hist.append(kept)
        if self.baseline is None and len(self.kept_hist) >= self.baseline_steps:
            self.baseline = sum(self.kept_hist[: self.baseline_steps]) / self.baseline_steps
        rec = {"kept": kept, "all_pass": sum(k == "all_pass" for k in kinds),
               "all_fail": sum(k == "all_fail" for k in kinds), "unique_seen": len(self.seen),
               "active": len(self.active), "pool_cursor": self.cursor, "pool_size": len(self.pool_order),
               "baseline_kept": self.baseline, "refreshed": None}
        if self.mode == "stream":
            rec["pool_epochs"] = self.epochs
            return rec
        recent = self.kept_hist[-self.window:]
        if (self.refresh_below > 0 and self.baseline is not None and len(self.kept_hist) > self.baseline_steps
                and step >= self.cooldown_until and len(recent) == self.window
                and sum(recent) / self.window < self.refresh_below * self.baseline):
            rec["refreshed"] = self._refresh(step)
            rec["active"] = len(self.active)
            rec["pool_cursor"] = self.cursor
        return rec

    def _refresh(self, step: int) -> dict:
        retired = [i for i in self.active if self.last_kind.get(i) in self.retire]
        gone = set(retired)
        room = len(self.pool_order) - self.cursor
        added = self.pool_order[self.cursor:self.cursor + min(len(retired), room)]
        self.cursor += len(added)
        self.active = [i for i in self.active if i not in gone] + added
        self.queue = [i for i in self.queue if i not in gone]
        self.cooldown_until = step + self.window
        ev = {"step": step, "retired": len(retired), "added": len(added),
              "retired_kinds": {k: sum(self.last_kind.get(i) == k for i in retired) for k in self.retire},
              "pool_exhausted": len(added) < len(retired)}
        self.refreshes.append(ev)
        if len(self.active) < self.per_step:  # keep drawing possible: refill from retired prompts, oldest first
            back = [i for i in retired if i not in set(self.active)][: self.per_step - len(self.active)]
            self.active += back
            ev["readmitted_for_minimum"] = len(back)
        return ev

    # ------------------------------------------------------------ durable state
    def state_dict(self) -> dict:
        return {"mode": self.mode, "per_step": self.per_step, "pool_order": list(self.pool_order),
                "cursor": self.cursor, "active": list(self.active), "initial": list(self.initial),
                "queue": list(self.queue), "kept_hist": list(self.kept_hist), "baseline": self.baseline,
                "last_kind": {str(k): v for k, v in self.last_kind.items()}, "seen": sorted(self.seen),
                "refreshes": list(self.refreshes), "cooldown_until": self.cooldown_until, "epochs": self.epochs,
                "topped_up": self.topped_up, "rng": _rng_to_json(self.rng.getstate())}

    def load_state_dict(self, s: dict) -> None:
        if s["mode"] != self.mode or s["per_step"] != self.per_step:
            raise RuntimeError("saved pool state belongs to a different pool configuration")
        self.pool_order, self.cursor = list(s["pool_order"]), int(s["cursor"])
        self.active, self.initial, self.queue = list(s["active"]), list(s["initial"]), list(s["queue"])
        self.kept_hist, self.baseline = list(s["kept_hist"]), s["baseline"]
        self.last_kind = {int(k): v for k, v in s["last_kind"].items()}
        self.seen, self.refreshes = set(s["seen"]), list(s["refreshes"])
        self.cooldown_until, self.epochs = int(s["cooldown_until"]), int(s["epochs"])
        self.topped_up = int(s.get("topped_up", 0))
        self.rng.setstate(_rng_from_json(s["rng"]))

    def summary(self) -> dict:
        return {"mode": self.mode, "unique_prompts_seen": len(self.seen), "refreshes": list(self.refreshes),
                "starting_cohort": len(self.initial), "topped_up_from_unscreened": self.topped_up,
                "baseline_kept": self.baseline, "kept_per_step": list(self.kept_hist),
                "pool_used": self.cursor, "pool_size": len(self.pool_order), "final_active": len(self.active)}


def _rng_to_json(state):
    version, internal, gauss = state
    return [version, list(internal), gauss]


def _rng_from_json(s):
    return (s[0], tuple(s[1]), s[2])


def from_config(pcfg: dict, n_pool: int, per_step: int, seed: int, initial=None, fresh=None) -> PromptPool:
    allowed = {"mode", "cohort_size", "refresh_below", "baseline_steps", "window", "retire"}
    unknown = set(pcfg) - allowed
    if unknown:
        raise ValueError(f"unknown train.pool keys {sorted(unknown)}")
    return PromptPool(n_pool, per_step, seed, mode=pcfg.get("mode", "refresh"), cohort_size=int(pcfg.get("cohort_size", 0)),
                      refresh_below=float(pcfg.get("refresh_below", 0.5)), baseline_steps=int(pcfg.get("baseline_steps", 3)),
                      window=int(pcfg.get("window", 3)), retire=tuple(pcfg.get("retire", ("all_pass", "all_fail"))),
                      initial=initial, fresh=fresh)
