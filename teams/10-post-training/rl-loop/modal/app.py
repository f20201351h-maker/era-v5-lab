"""Modal launcher for the G10 RL experiments (optional; the TSAI job queue is the lab's normal route).

    modal run teams/10-post-training/rl-loop/modal/app.py::weights
    modal run teams/10-post-training/rl-loop/modal/app.py::train --gpu L40S --configs a.yaml,b.yaml --run-id smoke1 --max-minutes 20
    modal run teams/10-post-training/rl-loop/modal/app.py::survey --jobs jobs.json --run-id survey1 --max-minutes 60
    modal volume get g10-rl runs/<run-id> <local folder>

What runs where:
    weights        CPU. Downloads the OLMoE checkpoints once into the Volume g10-rl (/vol/hf/<repo>) and records
                   the exact Hub revisions in /vol/hf/revisions.json.
    code_contests  CPU. Downloads deepmind/code_contests train (39 shards, about 7.5 GB) into the Volume and builds
                   the non-Codeforces stdio tasks with rl-data/build.py; also writes an id + statement file for
                   the decontamination check (run locally with sft/decontam_run.py).
    train          GPU (L40S, H100! or A100-80GB). Runs rl-loop/train.py once per config, each in its own process
                   so vLLM's GPU memory is freed between configs. Outputs: /vol/runs/<run-id>/<config>/.
    survey         GPU (L40S), network blocked. Runs rl-data/survey.py once per job in a JSON file of named argument
                   lists. Output: /vol/runs/<run-id>/<job>.json.

Spend control: every function has a timeout, and the local entry point refuses to start a run whose worst case
(timeout x GPU rate) would take the ledger total past the budget (default $27, --budget). After each run the
measured cost (wall seconds x rate, GPU plus the CPU and memory reservation) is added to the ledger
(G10_SPEND_LEDGER, default research/g10_gpu/spend.jsonl in the capstone folder). Rates are modal.com/pricing,
read 5 Nov 2026; Modal's Usage page is the source of truth.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
# Repo paths exist only on the laptop; inside the container this file is /root/app.py and they are not needed.
TEAM = HERE.parents[1] if modal.is_local() else Path("/nonexistent")  # teams/10-post-training
REMOTE = "/root/g10"

# $/second, modal.com/pricing (5 Nov 2026)
GPU_RATE = {"L40S": 0.000542, "H100!": 0.001097, "H100": 0.001097, "A100-80GB": 0.000694, None: 0.0}
CPU_RATE, MEM_RATE = 0.0000131, 0.00000222  # per physical core-second, per GiB-second
CPU_CORES, MEM_GIB = 6.0, 32.0  # reserved per GPU function (scoring pool, model load); billed on top of the GPU

TIMEOUT = {"weights": 1800, "code_contests": 3600, "train": 3 * 3600, "survey": 5400}

VOL = modal.Volume.from_name("g10-rl", create_if_missing=True, version=2)
IGNORE = ["**/__pycache__/**", "**/out/**", "**/*.pyc", "**/.pytest_cache/**"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install("vllm==0.32.0", "math-verify[antlr4_13_2]==0.9.0", "pyyaml", "pyarrow",
                    "huggingface_hub[hf_xet]")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/vol/hf_cache", "TOKENIZERS_PARALLELISM": "false",
          # A1 (5 Nov): FlashInfer's sampler JIT-compiles on first use and needs nvcc, which debian_slim lacks;
          # vLLM's own PyTorch sampler needs no compiler.
          "VLLM_USE_FLASHINFER_SAMPLER": "0",
          # C1/C2 (6 Nov) ran out of memory with 2 GB reserved but unallocated: let the allocator grow segments
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    # our code is mounted at container start, so editing it never rebuilds the image
    .add_local_dir(TEAM / "rl-loop", f"{REMOTE}/rl-loop", ignore=IGNORE)
    .add_local_dir(TEAM / "rl-data", f"{REMOTE}/rl-data", ignore=IGNORE)
    .add_local_dir(TEAM / "rl", f"{REMOTE}/rl", ignore=IGNORE)
)
app = modal.App("g10-rl", image=image)
function, local_entrypoint = app.function, app.local_entrypoint  # pr-guard reads "+@app.x" as an email

MODELS = ("allenai/OLMoE-1B-7B-0125-DPO", "allenai/OLMoE-1B-7B-0125-Instruct")


def _cost(gpu: str | None, seconds: float, cpu: float = CPU_CORES, mem: float = MEM_GIB) -> float:
    return seconds * (GPU_RATE[gpu] + cpu * CPU_RATE + mem * MEM_RATE)


def _record(run_dir: Path, gpu: str | None, started: float, extra: dict) -> dict:
    secs = time.time() - started
    rec = {"gpu": gpu, "seconds": round(secs, 1), "usd": round(_cost(gpu, secs), 4), **extra}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cost.json").write_text(json.dumps(rec, indent=1))
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            rec["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    VOL.commit()
    return rec


# --- CPU functions ------------------------------------------------------------------------------

@function(volumes={"/vol": VOL}, timeout=TIMEOUT["weights"], cpu=4.0, memory=8192)
def download_weights(repos: list[str]) -> dict:
    from huggingface_hub import HfApi, snapshot_download  # noqa: PLC0415
    started, revs = time.time(), {}
    for repo in repos:
        sha = HfApi().model_info(repo).sha
        path = snapshot_download(repo_id=repo, revision=sha, local_dir=f"/vol/hf/{repo}",
                                 allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.md"])
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        revs[repo] = {"revision": sha, "bytes": size}
    (Path("/vol/hf") / "revisions.json").write_text(json.dumps(revs, indent=1))
    return {"revisions": revs, **_record(Path("/vol/runs/weights"), None, started, {"cpu": 4})}


@function(volumes={"/vol": VOL}, timeout=TIMEOUT["code_contests"], cpu=4.0, memory=16384)
def build_code_contests(judges: str = "aizu,atcoder,codechef,hackerearth") -> dict:
    from huggingface_hub import HfApi, snapshot_download  # noqa: PLC0415
    started = time.time()
    sha = HfApi().dataset_info("deepmind/code_contests").sha
    local = snapshot_download(repo_id="deepmind/code_contests", repo_type="dataset", revision=sha,
                              local_dir="/vol/data/code_contests", allow_patterns=["data/train-*"])
    shards = sorted(str(p) for p in Path(local, "data").glob("train-*.parquet"))
    out = "/vol/data/cc_tasks.jsonl"
    subprocess.run([sys.executable, f"{REMOTE}/rl-data/build.py", "--source", "code_contests", "--sources", judges,
                    "--parquet", *shards, "--out", out], check=True)
    n = 0
    with open(out, encoding="utf-8") as f, open("/vol/data/cc_statements.jsonl", "w", encoding="utf-8") as g:
        for line in f:
            t = json.loads(line)
            g.write(json.dumps({"id": t["id"], "text": t["statement"]}, ensure_ascii=False) + "\n")
            n += 1
    return {"revision": sha, "shards": len(shards), "tasks": n,
            **_record(Path("/vol/runs/code_contests"), None, started, {"cpu": 4})}


@function(volumes={"/vol": VOL}, timeout=1800, cpu=4.0, memory=40960)
def merge_weights(base: str, weights: str, out: str) -> dict:
    """Base checkpoint + saved trainable weights (train.py `save_trainable`) -> a full HF checkpoint for evaluation."""
    import hashlib  # noqa: PLC0415
    import shutil  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415
    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, local_files_only=True)
    state = torch.load(weights, map_location="cpu")
    params = dict(model.named_parameters())
    missing = [n for n in state if n not in params]
    if missing:
        raise ValueError(f"{len(missing)} saved tensors have no matching parameter, e.g. {missing[:3]}")
    with torch.no_grad():
        for n, t in state.items():
            if params[n].shape != t.shape:
                raise ValueError(f"shape mismatch for {n}: {tuple(params[n].shape)} vs {tuple(t.shape)}")
            params[n].copy_(t.to(params[n].dtype))
    model.save_pretrained(out, safe_serialization=True)
    for f in Path(base).iterdir():  # tokenizer and generation files only, unchanged
        # never copy the base's weight index files: they point at the base's shard files, not ours (E1, 6 Nov)
        if f.suffix in (".json", ".txt", ".model") and not f.name.endswith(".index.json") \
                and not (Path(out) / f.name).exists():
            shutil.copy(f, Path(out) / f.name)
    rec = {"base": base, "weights": weights, "out": out, "tensors_replaced": len(state),
           "weights_sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest()}
    return {**rec, **_record(Path("/vol/runs/merge"), None, started, {"cpu": 4})}


# --- GPU functions (one per GPU type: the GPU is fixed when a function is declared) -------------

def _train_body(gpu: str, configs: dict[str, str], run_id: str, max_seconds: float) -> dict:
    started, results = time.time(), {}
    run_dir = Path("/vol/runs") / run_id
    for name, text in configs.items():
        out = run_dir / name
        out.mkdir(parents=True, exist_ok=True)
        cfg_path = out / "config.yaml"
        cfg_path.write_text(text)
        left = max_seconds - (time.time() - started)
        rc = "skipped: time cap reached"
        if left > 30:
            with open(out / "stdout.log", "w") as log:
                try:
                    rc = subprocess.run([sys.executable, f"{REMOTE}/rl-loop/train.py", "--config", str(cfg_path),
                                         "--out-dir", str(out)], stdout=log, stderr=subprocess.STDOUT,
                                        timeout=left).returncode
                except subprocess.TimeoutExpired:
                    rc = "killed: time cap reached"
        summary = out / "summary.json"
        results[name] = {"returncode": rc,
                         "summary": json.loads(summary.read_text()) if summary.exists() else None}
        VOL.commit()
    return {"results": results, **_record(run_dir, gpu, started, {"configs": list(configs)})}


def _survey_body(gpu: str, jobs: dict[str, list[str]], run_id: str, max_seconds: float) -> dict:
    """Runs each named survey job (a survey.py argument list) in turn; outputs /vol/runs/<run-id>/<job>.json."""
    started, results = time.time(), {}
    run_dir = Path("/vol/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, args in jobs.items():
        left = max_seconds - (time.time() - started)
        rc = "skipped: time cap reached"
        if left > 60:
            with open(run_dir / f"{name}.log", "w") as log:
                try:
                    rc = subprocess.run([sys.executable, f"{REMOTE}/rl-data/survey.py", *args,
                                         "--out", str(run_dir / f"{name}.json")], stdout=log,
                                        stderr=subprocess.STDOUT, cwd=f"{REMOTE}/rl-data", timeout=left).returncode
                except subprocess.TimeoutExpired:
                    rc = "killed: time cap reached"
        out = run_dir / f"{name}.json"
        head = None
        if out.exists():
            m = json.loads(out.read_text())
            head = {k: v for k, v in m.items() if k not in ("rows", "by_group")}
        results[name] = {"returncode": rc, "summary": head}
        VOL.commit()
    return {"results": results, **_record(run_dir, gpu, started, {"jobs": list(jobs)})}


GPU_KW = dict(volumes={"/vol": VOL}, cpu=CPU_CORES, memory=int(MEM_GIB * 1024))


@function(gpu="L40S", timeout=TIMEOUT["train"], **GPU_KW)
def train_l40s(configs: dict[str, str], run_id: str, max_seconds: float) -> dict:
    return _train_body("L40S", configs, run_id, max_seconds)


@function(gpu="H100!", timeout=TIMEOUT["train"], **GPU_KW)
def train_h100(configs: dict[str, str], run_id: str, max_seconds: float) -> dict:
    return _train_body("H100!", configs, run_id, max_seconds)


@function(gpu="A100-80GB", timeout=TIMEOUT["train"], **GPU_KW)
def train_a100(configs: dict[str, str], run_id: str, max_seconds: float) -> dict:
    return _train_body("A100-80GB", configs, run_id, max_seconds)


# The survey runs model-written programs: no network in this container, Docker-free local runner, offline HF.
@function(gpu="L40S", timeout=TIMEOUT["survey"], block_network=True,
              env={"G10_STDIO_RUNNER": "local", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}, **GPU_KW)
def survey_l40s(jobs: dict[str, list[str]], run_id: str, max_seconds: float) -> dict:
    return _survey_body("L40S", jobs, run_id, max_seconds)


TRAIN_FNS = {"L40S": train_l40s, "H100!": train_h100, "A100-80GB": train_a100}


# --- local entry points: budget guard and ledger -------------------------------------------------

def _ledger_path() -> Path:
    default = HERE.parents[5] / "g10_gpu" / "spend.jsonl"  # <capstone>/research/g10_gpu, outside the repo
    return Path(os.environ.get("G10_SPEND_LEDGER", default))


def spent(ledger: Path) -> float:
    if not ledger.exists():
        return 0.0
    return sum(json.loads(line)["usd"] for line in ledger.read_text().splitlines() if line.strip())


def guard(gpu: str | None, timeout_s: int, budget: float, ledger: Path) -> float:
    worst = _cost(gpu, timeout_s)
    so_far = spent(ledger)
    if so_far + worst > budget:
        raise SystemExit(f"refused: spent ${so_far:.2f} + worst case ${worst:.2f} ({gpu}, {timeout_s}s) "
                         f"> budget ${budget:.2f}")
    print(f"budget ok: spent ${so_far:.2f}, worst case for this run ${worst:.2f}, budget ${budget:.2f}")
    return worst


def log_spend(ledger: Path, kind: str, run_id: str, result: dict) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    row = {"kind": kind, "run_id": run_id, "gpu": result.get("gpu"), "gpu_name": result.get("gpu_name"),
           "seconds": result.get("seconds"), "usd": result.get("usd", 0.0),
           "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(ledger, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))


@local_entrypoint()
def weights(budget: float = 27.0):
    ledger = _ledger_path()
    guard(None, TIMEOUT["weights"], budget, ledger)
    r = download_weights.remote(list(MODELS))
    log_spend(ledger, "weights", "weights", r)
    print(json.dumps(r["revisions"], indent=1))


@local_entrypoint()
def code_contests(budget: float = 27.0, judges: str = "aizu,atcoder,codechef,hackerearth"):
    ledger = _ledger_path()
    guard(None, TIMEOUT["code_contests"], budget, ledger)
    r = build_code_contests.remote(judges)
    log_spend(ledger, "code_contests", "code_contests", r)
    print(json.dumps({k: v for k, v in r.items()}, indent=1))


@local_entrypoint()
def train(configs: str, run_id: str, gpu: str = "L40S", max_minutes: float = 20.0, budget: float = 27.0):
    ledger = _ledger_path()
    cap = min(max_minutes * 60, TIMEOUT["train"] - 120)  # leave time to write cost.json before the hard timeout
    guard(gpu, int(cap + 600), budget, ledger)  # + 10 min for image pull, model load and teardown
    texts = {Path(c).stem: Path(c).read_text() for c in configs.split(",")}
    r = TRAIN_FNS[gpu].remote(texts, run_id, cap)
    log_spend(ledger, "train", run_id, r)
    for name, res in r["results"].items():
        s = res["summary"] or {}
        print(name, "rc", res["returncode"], json.dumps({k: s.get(k) for k in (
            "sampler", "mean_step_s_after_step1", "mean_times_s_after_step1", "mean_rollout_tokens_per_s",
            "peak_gpu_memory_gb", "end_gpu_used_gb", "router_unchanged", "groups_used_total",
            "mean_mismatch_k3", "vllm_sync_path")}))


@local_entrypoint()
def survey(jobs: str, run_id: str, max_minutes: float = 60.0, budget: float = 27.0):
    """jobs: a JSON file {"job_name": ["--tasks", "...", ...], ...} with paths as seen inside the container."""
    ledger = _ledger_path()
    cap = min(max_minutes * 60, TIMEOUT["survey"] - 120)
    guard("L40S", int(cap + 600), budget, ledger)
    r = survey_l40s.remote(json.loads(Path(jobs).read_text()), run_id, cap)
    log_spend(ledger, "survey", run_id, r)
    print(json.dumps(r, indent=1))


@local_entrypoint()
def merge(base: str, weights: str, out: str, budget: float = 27.0):
    ledger = _ledger_path()
    guard(None, 1800, budget, ledger)
    r = merge_weights.remote(base, weights, out)
    log_spend(ledger, "merge", "merge", r)
    print(json.dumps(r, indent=1))
