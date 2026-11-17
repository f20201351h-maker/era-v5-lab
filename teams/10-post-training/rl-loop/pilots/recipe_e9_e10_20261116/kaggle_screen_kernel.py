"""Kaggle kernel: re-screen already trained E9 models (outputs of earlier kernels, mounted as kernel sources) with
sft/bench/rl_screen.py at a larger token budget, one model per GPU in parallel. No training.

SCREENS lists (tag, path fragment that identifies the model folder under /kaggle/input). Output: /kaggle/working/out.
"""
SCREENS = [("A2_full_2048", "g10-recipe-e9-360m/"), ("A2_lora_2048", "g10-recipe-e9-360m-lora/")]
MAX_NEW, K, SETS = 2048, 8, ("rung1=rung1_s128_ids.txt",)

import os
import subprocess
import sys
import time
from pathlib import Path

t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.22.1"], check=True)
inp = Path("/kaggle/input")
found = [p.parent.parent for p in inp.rglob("10-post-training") if p.parent.name == "teams"]
if not found:
    import zipfile
    zipfile.ZipFile(next(inp.rglob("teams.zip"))).extractall("/kaggle/working/code/teams")
    found = [Path("/kaggle/working/code")]
team = found[0] / "teams" / "10-post-training"
data = next(p.parent for p in inp.rglob("dolly.tok.jsonl"))
models = {}
for tag, frag in SCREENS:
    cands = [p for p in inp.rglob("A2/model") if frag in str(p) + "/" and (p / "model.safetensors").exists()]
    cands = [p for p in cands if ("lora" in frag) == ("lora" in str(p))]
    if not cands:
        raise SystemExit(f"model for {tag} not found under {inp}")
    models[tag] = cands[0]
print({"models": {k: str(v) for k, v in models.items()}, "data": str(data)}, flush=True)
os.environ["G10_STDIO_RUNNER"] = "local"
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("rs", team / "sft" / "bench" / "rl_screen.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)
print({"sandbox_preflight": rs.sandbox_preflight(2)}, flush=True)
out = Path("/kaggle/working/out")
procs = []
for gpu, (tag, md) in enumerate(models.items()):
    od = out / tag
    od.mkdir(parents=True, exist_ok=True)
    tok = md / "tokenizer.json"
    cmd = [sys.executable, str(team / "sft/bench/rl_screen.py"), "--model", str(md), "--tasks", str(data / "cc.jsonl"),
           *[x for s in SETS for x in ("--set", f"{s.split('=')[0]}={data / s.split('=')[1]}")], "--k", str(K),
           "--temperature", "1", "--max-new", str(MAX_NEW), "--seqs-per-batch", "32", "--kv-cache-gb", "6",
           "--workers", "2", "--seed", "1", "--device", "cuda", "--out", str(od)]
    if not tok.exists():  # a LoRA run saves the merged model; its tokenizer is the base model's (same file)
        cmd += ["--tokenizer", str(next(m for m in models.values() if (m / "tokenizer.json").exists()) / "tokenizer.json")]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu % 2)}
    print("$ " + " ".join(cmd), flush=True)
    procs.append((tag, subprocess.Popen(cmd, env=env, stdout=open(od / "screen.log", "w"), stderr=subprocess.STDOUT)))
rcs = {tag: p.wait() for tag, p in procs}
for tag, _ in procs:
    rep = out / tag / "screen_report.json"
    if rep.exists():
        import json
        r = json.loads(rep.read_text())
        print(tag, {s: {"ladder": {k: v["share"] for k, v in d["ladder"].items()},
                        "mixed": d.get("groups_pass_kept", {}).get("mixed"), "truncated": d["truncated_share"]}
                    for s, d in r["sets"].items()}, flush=True)
print({"rc": rcs, "hours": round((time.time() - t0) / 3600, 2)}, flush=True)
sys.exit(max(rcs.values()))
