"""E9 at SmolLM2-1.7B and E10 on Modal, only if the new $30 credit exists (WORKING_PLAN 12.4 branch b). Nothing here
has been launched; the local dry run (`dryrun_local`) runs the same recipe_job.py commands this driver sends.

    M=~/.venvs/modal312/bin/modal
    G10_RECIPE_WORK=<work/recipe-e9-2026-11-16> $M run modal_app.py::upload --data <private data dir>
    $M run modal_app.py::weights                        # SmolLM2-135M + 1.7B to the volume (network on, CPU)
    $M run modal_app.py::dryrun                         # CPU dry run of e9 AND e10 inside the job image (cents)
    nohup $M run --detach modal_app.py::e9 --run-id e9a > e9a.out 2>&1 &     # then start watch_run.py (below)
    nohup $M run --detach modal_app.py::e10 --run-id e10a --sft-run e9a --arm A2 > e10a.out 2>&1 &
    python3 work/fable-research-2026-11-15/watch_run.py --volume g10-recipe-e9-2026-11-16 \
        --log runs/e9a/job.log --app g10-recipe-e9-2026-11-16 --launched HH:MM:SS --name e9a --stall-s 1800

Cost control (rates as in the E1 driver, `modal billing rates` read 15 Nov 2026; re-read before launching):
- every GPU call has a hard `timeout`; before a call starts, ledger + (timeout x rate) must stay under the
  experiment cap (E9 $7, E10 $10) and under STOP_TOTAL ($22 for the whole branch, so one failed launch can be repeated);
- E9 runs as up to `max_calls` calls of `budget_s`; recipe_job leaves done-markers, so the next call continues;
- E10 passes a deadline to the trainer (durable state, pause and resume) and checkpoints at steps 10/20/40;
- each call appends a ledger line (estimated from wall time) to $G10_RECIPE_WORK/SPEND.jsonl; reconcile with
  `modal billing summary` after every launch and report any overrun.
Order if the credit exists: E9 (gate), then E10 only if E9 passes the 10% mixed-group gate (PREREG_E10).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
TEAM = HERE.parents[2] if modal.is_local() else Path("/root/team")
REPO = TEAM.parents[1] if modal.is_local() else Path("/root/repo")
REMOTE_TEAM, REMOTE_REPO = "/root/team", "/root/repo"
WORK = Path(os.environ.get("G10_RECIPE_WORK", "")) if modal.is_local() else Path("/nonexistent")
MODEL_17B = ("HuggingFaceTB/SmolLM2-1.7B", "effd688a12921b4cc83e3312b6feb579f70f9c71")
MODEL_135M = ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2")
GPU_RATE = 2.50 / 3600      # A100-80GB
CPU_RATE = 0.0473 / 3600    # per core
MEM_RATE = 0.008 / 3600     # per GiB
GPU_CPU, GPU_MEM_GIB = 8.0, 64.0
CAPS = {"e9": 7.0, "e10": 10.0}
STOP_TOTAL = 22.0
LEDGER = WORK / "SPEND.jsonl"

VOL = modal.Volume.from_name("g10-recipe-e9-2026-11-16", create_if_missing=True, version=2)
IGNORE = ["**/__pycache__/**", "**/out/**", "**/*.pyc", "**/.pytest_cache/**", "**/runs/**"]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.1", "transformers==5.22.1", "tokenizers", "huggingface_hub[hf_xet]", "pyyaml", "numpy",
                    "safetensors")
    .env({"HF_HOME": "/vol/hf_cache", "TOKENIZERS_PARALLELISM": "false", "G10_STDIO_RUNNER": "local",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(TEAM / "sft", f"{REMOTE_TEAM}/sft", ignore=IGNORE)
    .add_local_dir(TEAM / "rl-loop", f"{REMOTE_TEAM}/rl-loop", ignore=IGNORE)
    .add_local_dir(TEAM / "rl-data", f"{REMOTE_TEAM}/rl-data", ignore=IGNORE)
    .add_local_dir(TEAM / "chat-format", f"{REMOTE_TEAM}/chat-format", ignore=IGNORE)
    .add_local_file(TEAM / "rl" / "verifier.py", f"{REMOTE_TEAM}/rl/verifier.py")  # extract_code only; never rl/tasks
    .add_local_file(TEAM / "dt" / "mask.py", f"{REMOTE_TEAM}/dt/mask.py")
    .add_local_dir(REPO / "teams" / "05-architecture" / "heldout_v1", f"{REMOTE_REPO}/teams/05-architecture/heldout_v1",
                   ignore=IGNORE)  # forgetting metric (evaluate.py default path)
)
app = modal.App("g10-recipe-e9-2026-11-16", image=image)
function, local_entrypoint = app.function, app.local_entrypoint
JOB = f"{REMOTE_TEAM}/rl-loop/pilots/recipe_e9_e10_20261116/recipe_job.py"
DATA = "/vol/data"


def _cost(seconds: float, gpu: bool, cpu: float, mem: float) -> float:
    return seconds * ((GPU_RATE if gpu else 0.0) + cpu * CPU_RATE + mem * MEM_RATE)


def _model(key) -> str:
    """Weights live on the volume (downloaded once by `weights`); job functions run with the network blocked."""
    repo, rev = key
    path = f"/vol/hf/{repo.split('/')[1]}@{rev[:12]}"
    if not Path(path, "config.json").exists():
        raise RuntimeError(f"{path} missing: run modal_app.py::weights first")
    return path


@function(volumes={"/vol": VOL}, timeout=1800, cpu=2.0, memory=4096)
def download_weights() -> dict:
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    out = {}
    for repo, rev in (MODEL_135M, MODEL_17B):
        dest = f"/vol/hf/{repo.split('/')[1]}@{rev[:12]}"
        snapshot_download(repo, revision=rev, local_dir=dest, allow_patterns=["*.json", "*.safetensors", "*.txt"])
        out[repo] = dest
    VOL.commit()
    return out


def _run_job(args: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stop = {"x": False}

    def committer():  # what is on the volume survives a container that dies mid-call
        while not stop["x"]:
            time.sleep(120)
            VOL.commit()
    import threading  # noqa: PLC0415
    th = threading.Thread(target=committer, daemon=True)
    th.start()
    with open(log_path, "a") as f:
        f.write(f"$ {' '.join(args)}\n")
        f.flush()
        rc = subprocess.run([sys.executable, JOB, *args], stdout=f, stderr=subprocess.STDOUT).returncode
    stop["x"] = True
    VOL.commit()
    return rc


@function(volumes={"/vol": VOL}, timeout=1800, cpu=4.0, memory=8192, block_network=True)
def smoke() -> dict:
    """CPU dry run of BOTH experiments in the job image, before any GPU call (same recipe_job.py, tiny settings)."""
    t0 = time.time()
    base = _model(MODEL_135M)
    out = Path("/vol/runs/_smoke")
    common = ["--tasks", f"{DATA}/cc.jsonl", "--device", "cpu", "--workers", "2", "--dry-run"]
    rc9 = _run_job(["e9", "--model", base, "--data", DATA, "--rung", f"{DATA}/rung1_s128_ids.txt", "--rest",
                    f"{DATA}/rest_s32_ids.txt", "--arms", "A2,A0", "--out", str(out / "e9"), *common], out / "smoke.log")
    rc10 = _run_job(["e10", "--sft-model", str(out / "e9" / "A2" / "model"), "--train-ids", f"{DATA}/e10_train_ids.txt",
                     "--heldout-ids", f"{DATA}/e10_heldout_ids.txt", "--out", str(out / "e10"), *common], out / "smoke.log")
    checks = {"e9_rc0": rc9 == 0, "e10_rc0": rc10 == 0,
              "e9_screen": (out / "e9" / "A2" / "screen" / "screen_report.json").exists(),
              "e10_resumed": all((out / "e10" / arm / "ckpt_0004.done").exists() for arm in ("fixed", "refresh"))}
    return {"ok": all(checks.values()), "checks": checks, "seconds": round(time.time() - t0, 1)}


@function(gpu="A100-80GB", volumes={"/vol": VOL}, timeout=3600, cpu=GPU_CPU, memory=int(GPU_MEM_GIB * 1024),
          block_network=True)
def e9_call(run_id: str) -> dict:
    t0 = time.time()
    base = _model(MODEL_17B)
    rd = Path("/vol/runs") / run_id
    rc = _run_job(["e9", "--model", base, "--data", DATA, "--tasks", f"{DATA}/cc.jsonl", "--rung",
                   f"{DATA}/rung1_s128_ids.txt", "--rest", f"{DATA}/rest_s32_ids.txt", "--arms", "A2,A1,A0",
                   "--autocast", "bf16", "--grad-ckpt", "--micro-tokens", "8192", "--gen-batch", "64",
                   "--device", "cuda", "--workers", "6", "--out", str(rd)], rd / "driver.log")
    summ = rd / "e9_summary.json"
    return {"rc": rc, "seconds": round(time.time() - t0, 1), "summary": json.loads(summ.read_text()) if summ.exists() else None}


@function(gpu="A100-80GB", volumes={"/vol": VOL}, timeout=3600, cpu=GPU_CPU, memory=int(GPU_MEM_GIB * 1024),
          block_network=True)
def e10_call(run_id: str, sft_run: str, arm_dir: str, deadline_unix: float) -> dict:
    t0 = time.time()
    rd = Path("/vol/runs") / run_id
    rc = _run_job(["e10", "--sft-model", f"/vol/runs/{sft_run}/{arm_dir}/model", "--tasks", f"{DATA}/cc.jsonl",
                   "--train-ids", f"{DATA}/e10_train_ids.txt", "--heldout-ids", f"{DATA}/e10_heldout_ids.txt",
                   "--device", "cuda", "--workers", "6", "--gen-batch", "64", "--deadline-unix", str(deadline_unix),
                   "--out", str(rd)], rd / "driver.log")
    return {"rc": rc, "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- local side: ledger and guards

def _spent(kind: str | None = None) -> float:
    if not LEDGER.exists():
        return 0.0
    rows = [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]
    return sum(r["usd_est"] for r in rows if kind is None or r.get("experiment") == kind)


def _guard(kind: str, worst: float) -> None:
    if _spent(kind) + worst > CAPS[kind]:
        raise SystemExit(f"{kind}: ledger {_spent(kind):.2f} + worst case {worst:.2f} > cap {CAPS[kind]}")
    if _spent() + worst > STOP_TOTAL:
        raise SystemExit(f"branch: ledger {_spent():.2f} + worst case {worst:.2f} > stop {STOP_TOTAL}")


def _log(kind: str, run_id: str, started: float, gpu: bool, cpu: float, mem: float, extra: dict) -> None:
    secs = time.time() - started
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "experiment": kind, "run_id": run_id,
           "seconds": round(secs, 1), "usd_est": round(_cost(secs, gpu, cpu, mem), 4), **extra,
           "note": "estimate from client wall time; reconcile with `modal billing summary`"}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _smoke_or_exit() -> None:
    t0 = time.time()
    r = smoke.remote()
    _log("smoke", "cpu-dry-run", t0, False, 4.0, 8.0, {"ok": r.get("ok")})
    print(json.dumps(r, indent=1))
    if not r.get("ok"):
        raise SystemExit("CPU dry run in the job image failed; no GPU call was made")


@local_entrypoint()
def upload(data: str):
    """Put the private inputs on the volume (once). Nothing else ever reads them."""
    need = ["dolly.tok.jsonl", "ocr2_clean.tok.jsonl", "cc.jsonl", "rung1_s128_ids.txt", "rest_s32_ids.txt",
            "e10_train_ids.txt", "e10_heldout_ids.txt"]
    missing = [n for n in need if not (Path(data) / n).exists()]
    if missing:
        raise SystemExit(f"missing inputs: {missing}")
    with VOL.batch_upload(force=True) as b:
        for n in need:
            b.put_file(str(Path(data) / n), f"/data/{n}")
    print({"uploaded": need})


@local_entrypoint()
def weights():
    t0 = time.time()
    print(download_weights.remote())
    _log("weights", "download", t0, False, 2.0, 4.0, {})


@local_entrypoint()
def dryrun():
    _smoke_or_exit()


@local_entrypoint()
def e9(run_id: str = "e9a", max_calls: int = 3):
    _smoke_or_exit()
    worst = _cost(3600, True, GPU_CPU, GPU_MEM_GIB)
    for call in range(max_calls):
        _guard("e9", worst)
        t0 = time.time()
        r = e9_call.remote(run_id)
        _log("e9", run_id, t0, True, GPU_CPU, GPU_MEM_GIB, {"call": call, "rc": r["rc"]})
        print(json.dumps({"call": call, "rc": r["rc"], "seconds": r["seconds"]}))
        if r["rc"] == 0 and r["summary"]:
            print(json.dumps(r["summary"], indent=1))
            return
    raise SystemExit(f"e9 not finished after {max_calls} calls; the volume holds the progress")


@local_entrypoint()
def e10(run_id: str = "e10a", sft_run: str = "e9a", arm: str = "A2", budget_s: int = 3000, max_calls: int = 4):
    _smoke_or_exit()
    worst = _cost(3600, True, GPU_CPU, GPU_MEM_GIB)
    for call in range(max_calls):
        _guard("e10", worst)
        t0 = time.time()
        r = e10_call.remote(run_id, sft_run, arm, time.time() + budget_s)
        _log("e10", run_id, t0, True, GPU_CPU, GPU_MEM_GIB, {"call": call, "rc": r["rc"]})
        print(json.dumps({"call": call, **r}))
        if r["rc"] == 0:
            return
    raise SystemExit(f"e10 not finished after {max_calls} calls; the volume holds the state")


def dryrun_local(data: str, model: str, out: str) -> int:
    """No Modal call: the exact argument lists of smoke() with this machine's python (proof of the commands).
    `python modal_app.py dryrun-local <data> <model dir> <out>` (any python that has `modal` installed)."""
    py = os.environ.get("G10_PY", sys.executable)
    job = str(HERE / "recipe_job.py")
    common = ["--tasks", f"{data}/cc.jsonl", "--device", "cpu", "--workers", "2", "--dry-run"]
    rc9 = subprocess.run([py, job, "e9", "--model", model, "--data", data, "--rung", f"{data}/rung1_s128_ids.txt",
                          "--rest", f"{data}/rest_s32_ids.txt", "--arms", "A2,A0", "--out", f"{out}/e9", *common]).returncode
    rc10 = subprocess.run([py, job, "e10", "--sft-model", f"{out}/e9/A2/model", "--train-ids", f"{data}/e10_train_ids.txt",
                           "--heldout-ids", f"{data}/e10_heldout_ids.txt", "--out", f"{out}/e10", *common]).returncode
    print({"e9_rc": rc9, "e10_rc": rc10})
    return int(bool(rc9 or rc10))


if __name__ == "__main__" and sys.argv[1:2] == ["dryrun-local"]:
    raise SystemExit(dryrun_local(*sys.argv[2:5]))
