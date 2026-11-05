"""RL loop v0: group-relative policy optimisation with a frozen MoE router.

    python teams/10-post-training/rl-loop/train.py --config teams/10-post-training/rl-loop/config/smoke_tiny_olmoe.yaml

One step: take `prompts_per_step` training prompts, sample `k` completions for each, score them, drop groups whose
rewards are all equal, and take `ppo_epochs` optimiser steps on the clipped surrogate of the chosen algorithm (grpo,
dr_grpo, rloo or gspo; see grpo.py). Every `eval_every` steps the held-out prompts are scored (that is the number
Gate 4 should be judged on, not the training reward). The router is frozen before training and its checksum must
be identical at the end; the run fails otherwise.

Writes to `out_dir` (default ./out, the TSAI job rule): metrics.jsonl (one line per step), summary.json.

Model kinds (config `model.kind`):
    tiny_olmoe   a random tiny OLMoE from a config (no download); for smoke tests on a laptop
    hf           AutoModelForCausalLM.from_pretrained(model.path), e.g. a local copy of allenai/OLMoE-1B-7B-0125-Instruct
    team5        Team 5's harness LanguageModel (teams/05-architecture/harness/week2/model.py), built with their own
                 config loader from model.arm + model.resolution, plus model.overrides; the KE2 code table is a
                 seeded stand-in until Team 5 ships the real model on 18 Nov
Task kinds (config `task.kind`): toy (see tasks.py), code_tasks (rl/tasks via rl/reward.py; needs an hf tokenizer).
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import grpo  # noqa: E402
import router  # noqa: E402
import tasks  # noqa: E402
from policy import completion_logprobs, hf_policy, module_policy  # noqa: E402

TEAM5_HARNESS = HERE.parents[1] / "05-architecture" / "harness"


# --- building the model and the task --------------------------------------------------------------

def build_model(mcfg: dict, device: str):
    kind = mcfg["kind"]
    if kind == "tiny_olmoe":
        from transformers import OlmoeConfig, OlmoeForCausalLM  # noqa: PLC0415
        cfg = OlmoeConfig(eos_token_id=None, bos_token_id=None, pad_token_id=None, **mcfg["config"])
        return hf_policy(OlmoeForCausalLM(cfg), None, device), None
    if kind == "hf":
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        dtype = getattr(torch, mcfg.get("dtype", "float32"))
        model = AutoModelForCausalLM.from_pretrained(mcfg["path"], dtype=dtype, local_files_only=True)
        try:
            tok = AutoTokenizer.from_pretrained(mcfg["path"], local_files_only=True)
        except (OSError, ValueError):
            tok = None  # the toy task needs no tokenizer
        if mcfg.get("gradient_checkpointing"):
            model.gradient_checkpointing_enable()
        eos = tok.eos_token_id if tok is not None and mcfg.get("stop_at_eos", True) else None
        return hf_policy(model, eos, device), tok
    if kind == "team5":
        import numpy as np  # noqa: PLC0415
        sys.path.insert(0, str(TEAM5_HARNESS))
        from week2.config import load, load_runtime, merge  # noqa: PLC0415
        from week2.model import LanguageModel  # noqa: PLC0415
        team5 = TEAM5_HARNESS.parent
        spec = merge(load_runtime(team5 / mcfg["arm"]), load(team5 / mcfg["resolution"]))["model"]
        moe = mcfg.get("overrides", {}).pop("moe", {}) if mcfg.get("overrides") else {}
        spec.update(mcfg.get("overrides", {}))
        spec["moe"].update(moe)
        rng = np.random.default_rng(mcfg.get("ke2_seed", 7))  # stand-in KE2 code table until Team 5's model lands
        rows = rng.integers(0, 256, (spec["vocab_size"], 4), dtype=np.uint8)
        lengths = rng.integers(1, 5, spec["vocab_size"], dtype=np.int16)
        return module_policy(LanguageModel(spec, rows, lengths), None, device), None
    raise ValueError(f"unknown model.kind {kind!r}")


def build_task(tcfg: dict, tokenizer, seed: int):
    if tcfg["kind"] == "toy":
        return tasks.ToyTask(**{k: v for k, v in tcfg.items() if k != "kind"}, seed=seed)
    if tcfg["kind"] == "code_tasks":
        if tokenizer is None:
            raise ValueError("code_tasks needs model.kind hf (a tokenizer)")
        return tasks.RewardEnvTask(tokenizer, tasks.code_task_prompts(tcfg["train"]),
                                   tasks.code_task_prompts(tcfg["heldout"]))
    raise ValueError(f"unknown task.kind {tcfg['kind']!r}")


def set_trainable(model, patterns: list[str] | None) -> int:
    """If patterns are given, train only matching parameters (e.g. attention); the router stays frozen regardless."""
    if patterns:
        rx = [re.compile(p) for p in patterns]
        for n, p in model.named_parameters():
            p.requires_grad_(any(r.search(n) for r in rx))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- evaluation and training ----------------------------------------------------------------------

def evaluate(policy, task, prompts, k: int, max_new: int, temperature: float, gen) -> float:
    total = 0.0
    for p in prompts:
        seq, cmask = policy.sample(p, k, max_new, temperature, gen)
        for row, m in zip(seq[:, len(p):], cmask):
            total += task.reward(p, row[m].tolist())
    return total / (len(prompts) * k)


def train(cfg: dict) -> dict:
    seed = int(cfg.get("seed", 1234))
    random.seed(seed)
    torch.manual_seed(seed)
    device = cfg.get("device", "cpu")
    out = Path(cfg.get("out_dir", "out"))
    out.mkdir(parents=True, exist_ok=True)
    policy, tok = build_model(cfg["model"], device)
    task = build_task(cfg["task"], tok, seed)
    t = cfg["train"]
    algo, k, max_new = t["algo"], int(t["k"]), int(t["max_new"])

    model = policy.model
    n_trainable = set_trainable(model, t.get("trainable"))
    frozen = router.freeze_router(model, tuple(cfg.get("router_patterns", router.ROUTER_PATTERNS)))
    router.assert_router_frozen(model)
    checksum_before = router.router_checksum(model)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ref = None
    if float(t.get("kl_coef", 0)) > 0:
        ref = copy.deepcopy(policy)
        ref.model.requires_grad_(False)
        ref.model.eval()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(t["lr"]),
                            weight_decay=float(t.get("weight_decay", 0.0)))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    order = list(range(len(task.train)))
    metrics_f = open(out / "metrics.jsonl", "w")
    start = time.time()
    heldout_curve = []

    def log_heldout(step):
        r = evaluate(policy, task, task.heldout, int(t.get("eval_k", 4)), max_new, float(t["temperature"]), gen)
        heldout_curve.append((step, r))
        return r

    h0 = log_heldout(0)
    for step in range(1, int(t["steps"]) + 1):
        random.shuffle(order)
        batch = [task.train[i] for i in order[: int(t["prompts_per_step"])]]
        groups, rewards_all, dropped = [], [], 0
        for p in batch:
            seq, cmask = policy.sample(p, k, max_new, float(t["temperature"]), gen)
            rewards = torch.tensor([task.reward(p, row[m].tolist()) for row, m in zip(seq[:, len(p):], cmask)])
            rewards_all += rewards.tolist()
            adv = grpo.advantages(rewards, algo)
            if adv is None:
                dropped += 1
                continue
            with torch.no_grad():
                old_lp = completion_logprobs(policy, seq, len(p))
                ref_lp = completion_logprobs(ref, seq, len(p)) if ref is not None else None
            groups.append((seq, cmask, adv.to(device), old_lp, ref_lp, len(p)))
        stats = {"clip_frac": 0.0, "ratio_mean": 1.0, "kl": 0.0}
        loss_val = 0.0
        if groups:
            model.train()
            for _ in range(int(t.get("ppo_epochs", 1))):
                opt.zero_grad()
                for seq, cmask, adv, old_lp, ref_lp, plen in groups:
                    new_lp = completion_logprobs(policy, seq, plen)
                    loss, stats = grpo.policy_loss(new_lp, old_lp, adv, cmask, algo, float(t.get("clip_eps", 0.2)),
                                                   ref_lp, float(t.get("kl_coef", 0)))
                    (loss / len(groups)).backward()
                    loss_val += float(loss.detach()) / len(groups)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                               float(t.get("grad_clip", 1.0)))
                opt.step()
        rec = {"step": step, "train_reward": sum(rewards_all) / len(rewards_all), "groups_used": len(groups),
               "groups_dropped": dropped, "loss": loss_val, **stats, "seconds": round(time.time() - start, 1)}
        if step % int(t.get("eval_every", 10)) == 0 or step == int(t["steps"]):
            rec["heldout_reward"] = log_heldout(step)
        metrics_f.write(json.dumps(rec) + "\n")
        metrics_f.flush()
        if cfg.get("verbose", True):
            print(json.dumps(rec), flush=True)
    metrics_f.close()

    checksum_after = router.router_checksum(model)
    summary = {"algo": algo, "steps": int(t["steps"]), "k": k, "trainable_params": n_trainable,
               "router_params_frozen": frozen, "router_checksum_before": checksum_before,
               "router_checksum_after": checksum_after, "router_unchanged": checksum_before == checksum_after,
               "heldout_reward_start": h0, "heldout_reward_end": heldout_curve[-1][1], "heldout_curve": heldout_curve,
               "seconds": round(time.time() - start, 1),
               "seconds_per_step": round((time.time() - start) / max(int(t["steps"]), 1), 2),
               "peak_gpu_memory_gb": (round(torch.cuda.max_memory_allocated() / 2**30, 2)
                                      if device.startswith("cuda") else None)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    if not summary["router_unchanged"]:
        raise RuntimeError("router checksum changed during RL")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--steps", type=int, help="override train.steps")
    ap.add_argument("--model-path", help="override model.path (a local checkpoint folder)")
    ap.add_argument("--device", help="override device (cpu, cuda, mps)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    if args.out_dir:
        cfg["out_dir"] = str(args.out_dir)
    if args.steps:
        cfg["train"]["steps"] = args.steps
    if args.model_path:
        cfg["model"]["path"] = args.model_path
    if args.device:
        cfg["device"] = args.device
    s = train(cfg)
    print(json.dumps({k: v for k, v in s.items() if k != "heldout_curve"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
