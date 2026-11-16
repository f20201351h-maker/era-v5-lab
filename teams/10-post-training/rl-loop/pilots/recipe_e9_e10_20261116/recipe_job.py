"""E9 (SFT arms -> RL screen) and E10 (fixed cohort vs pool refresh RL) as ONE portable job: the same file runs on
this Mac (MPS/CPU), on Kaggle 2x T4, inside the Modal driver and as a TSAI job request.

    python recipe_job.py e9  --model HuggingFaceTB/SmolLM2-1.7B --revision <sha> --data <dir> --tasks cc.jsonl \
        --rung rung1_s128_ids.txt --rest rest_s32_ids.txt --device cuda --gpus 0,1 --out <dir>
    python recipe_job.py e10 --sft-model <E9 arm model dir> --tasks cc.jsonl --train-ids <ids> --heldout-ids <ids> \
        --device cuda --out <dir>
    python recipe_job.py e9 --dry-run ...   /   python recipe_job.py e10 --dry-run ...   (tiny, CPU-friendly)

E9 per arm (A0 base, A1 Dolly, A2 OCR-2 clean, A3 Dolly + OCR-2): sft/bench/train.py (full FT, equal steps and equal
supervised tokens per update), then sft/bench/rl_screen.py (k samples at temperature 1, failure ladder, mixed groups),
then the bench's forgetting metric (evaluate.py --metrics forget). Every step leaves a done-marker, so a rerun after a
crash or a session limit continues where it stopped. --gpus 0,1 runs arms in parallel, one per GPU.

E10 arms share one config except `train.pool.refresh_below` (0 = the fixed cohort, drawn exactly as the refresh arm
until its first refresh). Both screen the same candidates, start from the same cohort, and are evaluated on the same
unfiltered held-out prompts. The trainer's durable state makes every call resumable; --max-steps-this-call splits a
run across sessions. The model folder gets a chat template equal to the E9 screen prompt and <|im_end|> as EOS
(a tokenizer config shim, not a weight change).

Data (dolly.tok.jsonl, ocr2_clean.tok.jsonl, task records, id lists) is private: it is passed in, never fetched here.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEAM = HERE.parents[2]
BENCH = TEAM / "sft" / "bench"
RLLOOP = TEAM / "rl-loop"
ARMS = {"A1": "dolly.tok.jsonl", "A2": "ocr2_clean.tok.jsonl", "A3": "mix.tok.jsonl",
        "A2v2": "ocr2_clean_v2.tok.jsonl",   # A2 minus the 72 rows that share a problem family with an eval problem
        "A4": "ocr2_short_v2.tok.jsonl"}     # A2v2 with the reasoning block deleted (code-only answers)
CHAT_TEMPLATE = ("<|im_start|>system\nReasoning effort: {effort}<|im_end|>\n{{% for m in messages %}}<|im_start|>"
                 "{{{{ m['role'] }}}}\n{{{{ m['content'] }}}}<|im_end|>\n{{% endfor %}}<|im_start|>assistant{{{{ '\\n' }}}}")
# (Jinja drops one trailing newline of a template, so the header's final newline is an expression)


def log(msg: str, out: Path) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(out / "job.log", "a") as f:
        f.write(line + "\n")


def run(cmd: list[str], logf: Path, env: dict | None = None) -> int:
    with open(logf, "a") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})}).returncode


def runner_env(a) -> dict | None:
    return {"G10_STDIO_RUNNER": a.stdio_runner} if a.stdio_runner else None


def resolve_model(model: str, revision: str | None) -> Path:
    if Path(model).is_dir():
        return Path(model)
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    return Path(snapshot_download(model, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.txt"]))


# ---------------------------------------------------------------- E9

def e9_arm(a, arm: str, base: Path, out: Path, gpu: str | None) -> dict:
    env = {"CUDA_VISIBLE_DEVICES": gpu} if gpu is not None else {}
    if a.stdio_runner:
        env["G10_STDIO_RUNNER"] = a.stdio_runner
    rd = out / arm
    rd.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    res = {"arm": arm}
    model_dir = base
    if arm != "A0":
        model_dir = rd / "model"
        if not (rd / "train.done").exists():
            cmd = [py, str(BENCH / "train.py"), "--train", str(Path(a.data) / ARMS[arm]), "--model", str(base),
                   "--revision", "none", "--tokenizer-dir", str(base), "--method", "full", "--steps", str(a.steps),
                   "--warmup", str(min(10, max(1, a.steps // 10))), "--lr", str(a.lr), "--batch-target-tokens",
                   str(a.batch_target_tokens), "--micro-tokens", str(a.micro_tokens), "--device", a.device,
                   "--seed", str(a.seed), "--log-every", "10", "--save-every", "10", "--out", str(rd)]
            if a.method != "full":  # e.g. the Kaggle T4 variant at 1.7B: LoRA + the chat-marker rows (a method change)
                cmd[cmd.index("--method") + 1] = a.method
                cmd += ["--lora-r", str(a.lora_r), "--marker-lr", str(a.marker_lr), "--train-marker-rows", "--wd", "0"]
            if a.grad_ckpt:
                cmd.append("--grad-checkpointing")
            if a.autocast != "none":
                cmd += ["--autocast", a.autocast]
            log(f"E9 {arm} train start", out)
            if run(cmd, rd / "train.log", env):
                return {**res, "error": "train failed (see train.log)"}
            (rd / "train.done").touch()
        res["train_report"] = str(rd / "train_report.json")
    if not (rd / "screen.done").exists():
        cmd = [py, str(BENCH / "rl_screen.py"), "--model", str(model_dir), "--tasks", a.tasks, "--set", f"rung1={a.rung}",
               "--set", f"rest={a.rest}", "--k", str(a.k), "--temperature", "1", "--max-new", str(a.max_new),
               "--seqs-per-batch", str(a.gen_batch), "--workers", str(a.workers), "--seed", str(a.seed),
               "--device", a.device, "--out", str(rd / "screen")]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        if a.static_cache:
            cmd.append("--static-cache")
        cmd += ["--dtype", a.screen_dtype]
        log(f"E9 {arm} screen start", out)
        if run(cmd, rd / "screen.log", env):
            return {**res, "error": "screen failed (see screen.log)"}
        (rd / "screen.done").touch()
    res["screen_report"] = str(rd / "screen" / "screen_report.json")
    if a.forget and not (rd / "forget.json").exists():
        cmd = [py, str(BENCH / "evaluate.py"), "--model", str(model_dir), "--revision", "none", "--metrics", "forget",
               "--device", a.device, "--token-budget", "16384", "--out", str(rd / "forget.json")]
        log(f"E9 {arm} forget start", out)
        run(cmd, rd / "forget.log", env)
    log(f"E9 {arm} done", out)
    return res


def e9(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.dry_run:  # tiny everything; proves the commands, the files and the resume markers, not a result
        a.steps, a.batch_target_tokens, a.limit, a.k, a.max_new, a.gen_batch, a.forget = 2, 512, 2, 2, 16, 4, False
    base = resolve_model(a.model, a.revision)
    arms = a.arms.split(",")
    gpus = a.gpus.split(",") if a.gpus else [None]
    log(f"E9 start model={base} arms={arms} gpus={gpus} device={a.device}", out)
    t0 = time.time()
    import queue  # noqa: PLC0415
    free: queue.Queue = queue.Queue()
    for g in gpus:
        free.put(g)

    def on_free_gpu(arm):  # an arm takes whichever GPU is free and gives it back: never two arms on one GPU
        g = free.get()
        try:
            return e9_arm(a, arm, base, out, g)
        finally:
            free.put(g)
    with ThreadPoolExecutor(len(gpus)) as ex:
        jobs = [ex.submit(on_free_gpu, arm) for arm in arms]
        results = [j.result() for j in jobs]
    summary = {"arms": results, "seconds": round(time.time() - t0, 1), "dry_run": a.dry_run}
    for r in results:
        p = Path(r.get("screen_report", ""))
        if p.is_file():
            rep = json.loads(p.read_text())
            r["brief"] = {s: {"ladder": {k: v["share"] for k, v in d["ladder"].items()},
                              "mixed": d.get("groups_pass_kept", {}).get("mixed_share")} for s, d in rep["sets"].items()}
    (out / "e9_summary.json").write_text(json.dumps(summary, indent=1))
    log(f"E9 end {json.dumps({r['arm']: r.get('brief', r.get('error')) for r in results})}", out)
    return 0 if all("error" not in r for r in results) else 1


# ---------------------------------------------------------------- E10

def rl_model_dir(sft: Path, dest: Path, effort: str) -> Path:
    """The SFT folder with a chat template equal to the E9 screen prompt, <|im_end|> as EOS (stop token) and the
    tokenizer loaded from tokenizer.json (token-identical to the SFT rows; checked)."""
    dest.mkdir(parents=True, exist_ok=True)
    for f in sft.iterdir():
        if f.name in ("tokenizer_config.json",) or f.is_dir():
            continue
        tgt = dest / f.name
        if not tgt.exists():
            try:
                tgt.symlink_to(f.resolve())
            except OSError:
                shutil.copy(f, tgt)
    cfg = json.loads((sft / "tokenizer_config.json").read_text())
    cfg["chat_template"] = CHAT_TEMPLATE.format(effort=effort)
    cfg["eos_token"] = "<|im_end|>"
    # transformers 5 rebuilds SmolLM2's "GPT2Tokenizer" from vocab/merges and splits "\n\n" into two tokens, unlike
    # tokenizer.json (which built every SFT row and the E9 screen): load tokenizer.json itself
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    (dest / "tokenizer_config.json").write_text(json.dumps(cfg, indent=1))
    check_prompt_identity(dest, effort)
    return dest


def check_prompt_identity(model_dir: Path, effort: str) -> None:
    """The RL trainer's prompt (apply_chat_template) must be token-identical to the E9 screen prompt (rl_screen.py)."""
    import importlib.util  # noqa: PLC0415
    from tokenizers import Tokenizer  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415
    spec = importlib.util.spec_from_file_location("g10_rl_screen_job", BENCH / "rl_screen.py")
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    hf = AutoTokenizer.from_pretrained(model_dir)
    tk = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    samples = ["Read n and print n squared.", "Line one.\nLine two:\n\n1 2 3\n"]
    for st in samples:
        (_, want), = rs.encode_prompts(tk, [{"id": "x", "statement": st}], effort)[0]
        got = hf.apply_chat_template([{"role": "user", "content": st}], add_generation_prompt=True, tokenize=True)
        got = [int(x) for x in (got["input_ids"] if hasattr(got, "keys") else got)]
        if got != want:
            raise RuntimeError(f"chat template differs from the E9 screen prompt: {got[-8:]} vs {want[-8:]}")
    if hf.eos_token_id != 2:
        raise RuntimeError(f"EOS must be <|im_end|> (2), got {hf.eos_token_id}")


def export_checkpoint(model_dir: Path, state_dir: Path, dest: Path) -> dict:
    """Base weights + the trainable tensors of the newest durable state -> a plain HF folder rl_screen.py can load."""
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415
    latest = json.loads((state_dir / "LATEST.json").read_text())
    st = torch.load(state_dir / latest["file"], map_location="cpu", weights_only=False)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    missing = [n for n in st["trainable"] if n not in dict(model.named_parameters())]
    if missing:
        raise RuntimeError(f"state tensors not in the model: {missing[:3]}")
    with torch.no_grad():
        params = dict(model.named_parameters())
        for n, w in st["trainable"].items():
            params[n].copy_(w.to(torch.float32))
    model.save_pretrained(dest, safe_serialization=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (model_dir / f).exists():
            shutil.copy(model_dir / f, dest / f)
    return {"step": int(st["step"]), "tensors": len(st["trainable"]), "state_file": latest["file"]}


def e10_config(a, arm: str, model_dir: Path, rd: Path) -> dict:
    refresh = 0.0 if arm == "fixed" else a.refresh_below
    t = {"algo": "objective_v1", "steps": a.steps, "prompts_per_step": a.prompts_per_step, "k": a.k,
         "max_new": a.max_new, "temperature": 1.0, "lr": a.lr, "ppo_epochs": 1, "kl_coef": 0.0, "grad_clip": 1.0,
         "eval_every": a.eval_every, "eval_k": a.eval_k, "cohort_eval_k": a.eval_k, "micro_rows": a.micro_rows,
         "reward_workers": a.workers, "reward_pool": a.workers, "fp32_master": True, "sampler": a.sampler,
         "state_dir": str(rd / "state"), "state_every": a.eval_every,
         "screen": {"n": a.screen_n, "k": a.k, "min_mixed": a.min_mixed},
         "pool": {"mode": "refresh", "refresh_below": refresh, "baseline_steps": a.baseline_steps, "window": a.window,
                  "cohort_size": a.cohort_size}}
    if a.trainable:
        t["trainable"] = [a.trainable]
    if a.sampler == "vllm":
        t["vllm"] = {"gpu_memory_utilization": 0.3, "max_model_len": 2048}
    if a.deadline_unix:
        t["deadline_unix"] = a.deadline_unix
    if a.max_steps_this_call:
        t["max_steps_this_call"] = a.max_steps_this_call
    return {"seed": a.seed, "device": a.device, "verbose": True, "out_dir": str(rd),
            "model": {"kind": "hf", "path": str(model_dir), "dtype": "float32", "stop_at_eos": True, "dense": True},
            "task": {"kind": "stdio", "tasks": a.tasks, "train_ids": a.train_ids, "heldout_ids": a.heldout_ids},
            "train": t}


def e10(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        a.steps, a.prompts_per_step, a.k, a.max_new, a.eval_every, a.eval_k = 4, 2, 2, 16, 2, 2
        a.screen_n, a.min_mixed, a.baseline_steps, a.window, a.micro_rows, a.cohort_size = 6, 0, 1, 1, 1, 4
        a.checkpoints, a.screen_k, a.screen_max_new = "2,4", 2, 16
        small = out / "dry_heldout_ids.txt"  # the trainer's own held-out evaluation also runs at tiny size
        small.write_text("\n".join(Path(a.heldout_ids).read_text().split()[:4]) + "\n")
        a.heldout_ids = str(small)
    model_dir = rl_model_dir(Path(a.sft_model), out / "rl_model", a.effort)
    import yaml  # noqa: PLC0415
    rcs = {}
    for arm in a.arms.split(","):
        rd = out / arm
        rd.mkdir(parents=True, exist_ok=True)
        # one trainer call per checkpoint: the call pauses at the checkpoint (durable state on disk), the weights are
        # exported and screened on the held-out prompts (k samples, kept + unused tests), and the next call resumes
        points = [int(x) for x in a.checkpoints.split(",") if x] if a.checkpoints else []
        points = sorted(set([0] + [p for p in points if p <= a.steps] + [a.steps]))
        rcs[arm] = 0
        for prev, nxt in zip(points, points[1:]):
            done_mark = rd / f"ckpt_{nxt:04d}.done"
            if not done_mark.exists():
                a2 = argparse.Namespace(**{**vars(a), "max_steps_this_call": nxt - prev if nxt < a.steps else None})
                cfg = e10_config(a2, arm, model_dir, rd)
                (rd / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
                log(f"E10 {arm} train to step {nxt}", out)
                rcs[arm] = run([sys.executable, str(RLLOOP / "train.py"), "--config", str(rd / "config.yaml")],
                               rd / "train.log", runner_env(a))
                if rcs[arm] or (rd / "summary.json").is_file() and json.loads((rd / "summary.json").read_text()).get(
                        "status") == "NO_GO_MIXED_BAND":
                    log(f"E10 {arm} stopped (rc={rcs[arm]}; see train.log / summary.json)", out)
                    rcs[arm] = rcs[arm] or 2
                    break
                ck = rd / f"ckpt_{nxt:04d}"
                info = export_checkpoint(model_dir, rd / "state", ck)
                if info["step"] != nxt:
                    raise RuntimeError(f"exported state is step {info['step']}, expected {nxt}")
                done_mark.write_text(json.dumps(info))
            sc = rd / f"heldout_{nxt:04d}"
            if not (sc / "screen_report.json").exists():
                cmd = [sys.executable, str(BENCH / "rl_screen.py"), "--model", str(rd / f"ckpt_{nxt:04d}"),
                       "--tasks", a.tasks, "--set", f"heldout={a.heldout_ids}", "--k", str(a.screen_k), "--temperature",
                       "1", "--max-new", str(a.screen_max_new), "--seqs-per-batch", str(a.gen_batch), "--workers",
                       str(a.workers), "--seed", str(a.seed + nxt), "--device", a.device, "--out", str(sc)]
                if a.hidden_tests:
                    cmd += ["--hidden-tests", a.hidden_tests]
                if a.dry_run:
                    cmd += ["--limit", "4"]
                run(cmd, rd / "heldout_screen.log", runner_env(a))
        if rcs[arm] == 0 and not (rd / "heldout_0000" / "screen_report.json").exists():  # step 0 = the SFT model
            cmd = [sys.executable, str(BENCH / "rl_screen.py"), "--model", str(model_dir), "--tasks", a.tasks, "--set",
                   f"heldout={a.heldout_ids}", "--k", str(a.screen_k), "--temperature", "1", "--max-new",
                   str(a.screen_max_new), "--seqs-per-batch", str(a.gen_batch), "--workers", str(a.workers),
                   "--seed", str(a.seed), "--device", a.device, "--out", str(rd / "heldout_0000")]
            if a.hidden_tests:
                cmd += ["--hidden-tests", a.hidden_tests]
            if a.dry_run:
                cmd += ["--limit", "4"]
            run(cmd, rd / "heldout_screen.log", runner_env(a))
        summ = rd / "summary.json"
        if summ.is_file():
            s = json.loads(summ.read_text())
            log(f"E10 {arm} heldout {s.get('heldout_curve')} pool {s.get('prompt_pool', {}).get('refreshes')}", out)
    log(f"E10 end rc={rcs}", out)
    return 0 if not any(rcs.values()) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("e9")
    e.add_argument("--model", required=True)
    e.add_argument("--revision")
    e.add_argument("--data", required=True, help="folder with dolly.tok.jsonl, ocr2_clean.tok.jsonl[, mix.tok.jsonl]")
    e.add_argument("--arms", default="A2,A1,A0")
    e.add_argument("--steps", type=int, default=150)
    e.add_argument("--batch-target-tokens", type=int, default=8000)
    e.add_argument("--lr", type=float, default=1e-4)
    e.add_argument("--micro-tokens", type=int, default=2048)
    e.add_argument("--grad-ckpt", action="store_true")
    e.add_argument("--autocast", default="none", choices=("none", "bf16"))
    e.add_argument("--rung", required=True)
    e.add_argument("--rest", required=True)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--k", type=int, default=8)
    e.add_argument("--max-new", type=int, default=1024)
    e.add_argument("--gen-batch", type=int, default=64)
    e.add_argument("--static-cache", action="store_true")
    e.add_argument("--no-forget", dest="forget", action="store_false")
    e.add_argument("--gpus", default="")
    e.add_argument("--method", default="full", choices=("full", "lora"))
    e.add_argument("--lora-r", type=int, default=4)
    e.add_argument("--marker-lr", type=float, default=1e-4)
    e.add_argument("--screen-dtype", default="float32", choices=("float32", "bfloat16"))
    r = sub.add_parser("e10")
    r.add_argument("--sft-model", required=True)
    r.add_argument("--arms", default="fixed,refresh")
    r.add_argument("--train-ids", required=True)
    r.add_argument("--heldout-ids", required=True)
    r.add_argument("--steps", type=int, default=40)
    r.add_argument("--prompts-per-step", type=int, default=16)
    r.add_argument("--k", type=int, default=8)
    r.add_argument("--max-new", type=int, default=1024)
    r.add_argument("--lr", type=float, default=1e-5)
    r.add_argument("--eval-every", type=int, default=10)
    r.add_argument("--eval-k", type=int, default=8)
    r.add_argument("--micro-rows", type=int, default=4)
    r.add_argument("--sampler", default="hf_batch", choices=("hf_batch", "vllm", "policy"))
    r.add_argument("--screen-n", type=int, default=192)
    r.add_argument("--min-mixed", type=int, default=16)
    r.add_argument("--refresh-below", type=float, default=0.5)
    r.add_argument("--cohort-size", type=int, default=16, help="a screened cohort smaller than this is topped up")
    r.add_argument("--baseline-steps", type=int, default=3)
    r.add_argument("--window", type=int, default=3)
    r.add_argument("--effort", default="1k")
    r.add_argument("--deadline-unix", type=float)
    r.add_argument("--max-steps-this-call", type=int)
    r.add_argument("--trainable", default=r"\.self_attn\.", help="regex of trainable parameters ('' = all)")
    r.add_argument("--checkpoints", default="10,20,40", help="steps at which weights are exported and screened")
    r.add_argument("--screen-k", type=int, default=8)
    r.add_argument("--screen-max-new", type=int, default=1024)
    r.add_argument("--gen-batch", type=int, default=64)
    r.add_argument("--hidden-tests", default="", help="private jsonl of unused source tests (strict pass)")
    for p in (e, r):
        p.add_argument("--tasks", required=True)
        p.add_argument("--device", default="cuda")
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--workers", type=int, default=2)
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--stdio-runner", default="", help="G10_STDIO_RUNNER for the reward: docker (default) | local "
                       "(Linux, root: per-uid isolation, no Docker)")
        p.add_argument("--out", required=True)
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--config"]:  # TSAI job-request form: one YAML holds the subcommand and every option
        import yaml  # noqa: PLC0415
        cfg = yaml.safe_load(Path(argv[1]).read_text())
        argv = [cfg.pop("cmd")] + [x for k, v in cfg.items() for x in (
            [f"--{k.replace('_', '-')}"] if v is True else [] if v is False or v is None else [f"--{k.replace('_', '-')}", str(v)])]
    a = ap.parse_args(argv)
    (Path(a.out)).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "job_args.json").write_text(json.dumps(vars(a), indent=1))
    return e9(a) if a.cmd == "e9" else e10(a)


if __name__ == "__main__":
    raise SystemExit(main())
