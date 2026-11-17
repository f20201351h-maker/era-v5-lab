"""Kaggle 2x T4 (fp32): E9 at SmolLM2-1.7B base, or E10 on a post-SFT 360M model. One notebook cell, user-launched.

    !python /kaggle/input/<g10-code>/teams/10-post-training/rl-loop/pilots/recipe_e9_e10_20261116/kaggle_recipe.py \
        e9 --size 360m --data /kaggle/input/<g10-e9-private> --out /kaggle/working/e9_360m
    !python .../kaggle_recipe.py e10 --data /kaggle/input/<g10-e9-private> --sft-model <dir> --out /kaggle/working/e10
    python kaggle_recipe.py e9 --dry-run --data <dir> --out <dir>    # the same script, tiny, on a laptop CPU

Inputs (both PRIVATE Kaggle datasets; never public):
- <g10-code>: this repository's teams/10-post-training tree (sft/, rl-loop/, rl-data/, rl/verifier.py, dt/) at the commit
  that holds this file.
- <g10-e9-private>: dolly.tok.jsonl, ocr2_clean.tok.jsonl (work/recipe-e9-2026-11-16/prep_data.py), cc.jsonl (RL data v1
  code records), rung1_s128_ids.txt, rest_s32_ids.txt, e10_train_ids.txt, e10_heldout_ids.txt. Training and evaluation
  text from admitted human / execution-verified sources only.
- The model comes from the Hub at a pinned revision (internet on): HuggingFaceTB/SmolLM2-1.7B @ effd688a (Apache-2.0).

What fits in 16 GB per T4, fp32 (T4 has no bf16):
- E9 at 1.7B: full fine-tuning does NOT fit (fp32 weights + grads + AdamW moments ~27 GB). This route therefore runs
  LoRA r4 + the chat-marker rows (R3: 81/110 valid at 1.7B vs 93/110 for full FT), which is a METHOD CHANGE against the
  135M / 360M rows (full FT); the A100 routes (Modal, TSAI) keep full FT. Arms A2 and A1 run in parallel (GPU 0, 1), then
  A0. Screen: 128 rung + 32 rest prompts x 8, 1,024 new tokens, 8 sequences per batch (1.7B has no GQA: ~0.4 MB of KV
  cache per token per sequence in fp32).
- E10 at 360M: full fine-tuning fits; arms fixed and refresh run one after the other (rl-loop is single-process).
Estimated time (ESTIMATE from FLOP and bandwidth arithmetic, not measured on a T4): E9 ~1.4 h of training per arm (two in
parallel) + ~1.4 h of screening per arm -> ~4.5-5 h for A2, A1, A0. Stage 2 (all 754 rung prompts) does not fit in the
same 12 h session (~6 h per arm). E10 at 360M, 40 steps x 16 prompts x 8: ~2.5-3.5 h per arm. Everything leaves done-markers
and durable RL state in --out: if a session ends, attach the previous output as a dataset, copy it to --out, rerun.

The reward runs programs with rl-data/stdio_check.py's `local` runner (Kaggle has no Docker; Kaggle runs as root, which
the runner's per-uid isolation needs). The expected outputs never reach the program.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOB = HERE / "recipe_job.py"
MODEL_17B = ("HuggingFaceTB/SmolLM2-1.7B", "effd688a12921b4cc83e3312b6feb579f70f9c71")
MODEL_360M = ("HuggingFaceTB/SmolLM2-360M", "f8027fd0eaeea54caa13c31d31b9fdc459c38b49")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("e9", "e10"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sft-model", help="E10: the post-SFT model folder (an E9 arm's model/)")
    ap.add_argument("--size", choices=("360m", "1.7b"), default="1.7b",
                    help="360m: full fine-tune (fits a T4, same method as the Mac rows); 1.7b: LoRA r4 + marker rows")
    ap.add_argument("--model", help="override the Hub id (or a local folder for --dry-run)")
    ap.add_argument("--revision")
    ap.add_argument("--arms", default="A2,A1,A0")
    ap.add_argument("--method", choices=("full", "lora"), help="override the size default (360m full, 1.7b lora)")
    ap.add_argument("--dry-run", action="store_true", help="tiny run on CPU (no GPU, Docker sandbox on a laptop)")
    a = ap.parse_args(argv)
    d = Path(a.data)
    env = dict(os.environ)
    py = sys.executable
    if a.dry_run:
        device, gpus, runner = "cpu", [], env.get("G10_STDIO_RUNNER", "docker")
    else:
        import torch  # noqa: PLC0415
        n = torch.cuda.device_count()
        print({"cuda_devices": n, "names": [torch.cuda.get_device_name(i) for i in range(n)], "torch": torch.__version__})
        device, gpus, runner = "cuda", [str(i) for i in range(n)], "local"
    env["G10_STDIO_RUNNER"] = runner
    default = MODEL_360M if a.size == "360m" else MODEL_17B
    model, revision = a.model or default[0], a.revision or default[1]
    if a.what == "e9":
        cmd = [py, str(JOB), "e9", "--model", model, "--revision", revision, "--data", str(d), "--tasks",
               str(d / "cc.jsonl"), "--rung", str(d / "rung1_s128_ids.txt"), "--rest", str(d / "rest_s32_ids.txt"),
               "--arms", a.arms, "--grad-ckpt", "--micro-tokens", "2048", "--device", device, "--workers", "2",
               "--out", a.out]
        if (a.method or ("lora" if a.size == "1.7b" else "full")) == "lora":  # 1.7B full FT does not fit 16 GB fp32
            cmd += ["--method", "lora", "--lora-r", "4", "--lr", "1e-3", "--marker-lr", "1e-4",
                    "--gen-batch", "8" if a.size == "1.7b" else "32"]
        else:  # 360M: the Mac recipe unchanged (full FT, lr 1e-4); 32 sequences per batch (~0.08 MB KV per token)
            cmd += ["--method", "full", "--lr", "1e-4", "--gen-batch", "32"]
        if gpus:
            cmd += ["--gpus", ",".join(gpus)]
    else:
        if not a.sft_model:
            raise SystemExit("e10 needs --sft-model")
        cmd = [py, str(JOB), "e10", "--sft-model", a.sft_model, "--tasks", str(d / "cc.jsonl"), "--train-ids",
               str(d / "e10_train_ids.txt"), "--heldout-ids", str(d / "e10_heldout_ids.txt"), "--trainable", "",
               "--micro-rows", "2", "--gen-batch", "32", "--device", device, "--workers", "2", "--out", a.out]
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = gpus[0]
    if a.dry_run:
        cmd.append("--dry-run")
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
