"""Kaggle kernel: speed of rl_screen generation on a T4, one change per arm (see work PREREG_SPEED.md). No training.

Each run (kernel version) takes ARMS (one arm per GPU, started together, so both see the same VM load, as the
original two-screen kernel did), first prints a decode-step micro-profile on GPU 0 (step time vs rows and dtype, top ops
by GPU time), then runs rl_screen.py on the first 32 ids of the rung-1 set, k 8, seed 1, 2,048 new tokens, temperature 1.
Output: /kaggle/working/out/<tag>/{screen_report.json, screen_meta.json, generations.jsonl, screen.log}, profile.json.
"""
RUN = "R2"  # which block of RUNS this kernel version executes (edited per push)
# RUN: (arms, profile configs, profile 1.7B). An arm is (tag, extra rl_screen.py arguments); one arm per GPU.
BASE = ["--dtype", "float32", "--kv-cache-gb", "6", "--seqs-per-batch", "32"]
RUNS = {
    "R1": ([("base_fp32_hf_kv6", BASE), ("fp16_hf_kv6", ["--dtype", "float16", "--kv-cache-gb", "6", "--seqs-per-batch", "32"])],
           [("float32", "sdpa"), ("float16", "sdpa")], False),
    "R2": ([("drop_sdpa_kv6", [*BASE, "--decode-loop", "custom"]), ("gqa_hf_kv6", [*BASE, "--attn", "gqa_decode"])],
           [("float32", "gqa_decode")], False),
    "R3": ([("drop_gqa_kv6", [*BASE, "--decode-loop", "custom", "--attn", "gqa_decode"]),
            ("drop_gqa_kv10_b64", ["--dtype", "float32", "--kv-cache-gb", "10", "--seqs-per-batch", "64",
                                   "--decode-loop", "custom", "--attn", "gqa_decode"])],
           [], False),
    # R5: the custom loop now draws its random numbers over all rows (exact_rng), so its sampled ids should equal HF generate's
    "R5": ([("exact_gqa_kv6", [*BASE, "--decode-loop", "custom", "--attn", "gqa_decode"]),
            ("exact_gqa_kv10_b64", ["--dtype", "float32", "--kv-cache-gb", "10", "--seqs-per-batch", "64",
                                    "--decode-loop", "custom", "--attn", "gqa_decode"])],
           [], False),
    # R4: seed control of the fast config on 360M (GPU 1) and SmolLM2-1.7B base fp16 measured directly on a T4 (GPU 0);
    # "@B17" is replaced by the downloaded 1.7B snapshot folder
    "R4": ([("b17_fp16_fast_8prompts", ["--model", "@B17", "--tokenizer", "@B17/tokenizer.json", "--dtype", "float16",
                                        "--kv-cache-gb", "8", "--seqs-per-batch", "64", "--decode-loop", "custom",
                                        "--attn", "gqa_decode", "--limit", "8"]),
            ("drop_gqa_kv10_b64_seed2", ["--dtype", "float32", "--kv-cache-gb", "10", "--seqs-per-batch", "64",
                                         "--decode-loop", "custom", "--attn", "gqa_decode", "--seed", "2"])],
           [], True),
}
ARMS, PROFILES, PROFILE_17B = RUNS[RUN]
LIMIT, MAX_NEW, K = 32, 2048, 8

import json
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
cands = [p for p in inp.rglob("A2/model") if "g10-recipe-e9-360m/" in str(p) + "/" and "lora" not in str(p)
         and (p / "model.safetensors").exists()]
if not cands:
    raise SystemExit(f"360M model not found under {inp}")
model_dir = cands[0]
print({"model": str(model_dir), "data": str(data), "code": str(team)}, flush=True)
os.environ["G10_STDIO_RUNNER"] = "local"
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("rs", team / "sft" / "bench" / "rl_screen.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)
print({"sandbox_preflight": rs.sandbox_preflight(2)}, flush=True)
out = Path("/kaggle/working/out")
out.mkdir(parents=True, exist_ok=True)


def step_profile(path, dtype_name, rows_list, ctx=1500, steps=20, prof_rows=24, attn="sdpa"):
    """Time decode steps on random KV of length ctx (dynamic cache, sdpa), plus a torch profiler table."""
    import torch
    from transformers import AutoModelForCausalLM, DynamicCache
    dt = getattr(torch, dtype_name)
    kw = {"attn_implementation": rs.register_gqa_decode_attention()} if attn == "gqa_decode" else {}
    m = AutoModelForCausalLM.from_pretrained(str(path), dtype=dt, **kw).to("cuda").eval()
    cfg = m.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    kvh = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    res = {}
    for B in rows_list:
        def fresh():
            c = DynamicCache(config=cfg)
            for li in range(cfg.num_hidden_layers):
                c.update(torch.randn(B, kvh, ctx, hd, device="cuda", dtype=dt), torch.randn(B, kvh, ctx, hd, device="cuda", dtype=dt), li)
            return c
        cache = None
        try:
            cache = fresh()
            attn = torch.ones(B, ctx + 1, dtype=torch.long, device="cuda")
            tok = torch.randint(100, 40000, (B, 1), device="cuda")
            pos = torch.full((B, 1), ctx, device="cuda")

            def one(cache, attn, pos):
                with torch.no_grad():
                    m(input_ids=tok, attention_mask=attn, position_ids=pos, past_key_values=cache, use_cache=True)
                return torch.cat([attn, attn[:, :1]], 1), pos + 1
            for _ in range(3):
                attn, pos = one(cache, attn, pos)
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(steps):
                attn, pos = one(cache, attn, pos)
            torch.cuda.synchronize()
            ms = (time.time() - t) / steps * 1000
            res[f"rows{B}"] = {"ms_per_step": round(ms, 1), "tok_per_s": round(B / ms * 1000, 1)}
            if B == prof_rows:
                from torch.profiler import ProfilerActivity, profile
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
                    for _ in range(5):
                        attn, pos = one(cache, attn, pos)
                    torch.cuda.synchronize()
                res["profile_table"] = pr.key_averages().table(sort_by="cuda_time_total", row_limit=10, max_name_column_width=40)
        except torch.OutOfMemoryError:
            res[f"rows{B}"] = "OOM"
        del cache
        torch.cuda.empty_cache()
    res["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    torch.cuda.reset_peak_memory_stats()
    del m
    torch.cuda.empty_cache()
    return res


prof = {}
for dtype_name, attn in PROFILES:
    key = f"{dtype_name}_{attn}"
    prof[key] = step_profile(model_dir, dtype_name, [8, 24, 48], attn=attn)
    print("PROFILE", key, {k: v for k, v in prof[key].items() if k != "profile_table"}, flush=True)
    if "profile_table" in prof[key]:
        print(prof[key]["profile_table"], flush=True)
p17 = None
if PROFILE_17B:
    from huggingface_hub import snapshot_download
    p17 = snapshot_download("HuggingFaceTB/SmolLM2-1.7B", revision="effd688a12921b4cc83e3312b6feb579f70f9c71",
                           allow_patterns=["*.json", "*.safetensors", "merges.txt"])
    for attn in ("sdpa", "gqa_decode"):
        key = f"smollm2_1.7b_fp16_{attn}"
        prof[key] = step_profile(p17, "float16", [4, 8, 12], ctx=1500, steps=10, prof_rows=-1, attn=attn)
        print("PROFILE", key, prof[key], flush=True)
(out / "profile.json").write_text(json.dumps(prof, indent=1))

ids_file = data / "rung1_s128_ids.txt"
tok = model_dir / "tokenizer.json"
procs = []
for gpu, (tag, extra) in enumerate(ARMS):
    od = out / tag
    od.mkdir(parents=True, exist_ok=True)
    extra = [x.replace("@B17", str(p17)) for x in extra]
    cmd = [sys.executable, str(team / "sft/bench/rl_screen.py"), "--model", str(model_dir), "--tasks", str(data / "cc.jsonl"),
           "--set", f"rung1={ids_file}", "--limit", str(LIMIT), "--k", str(K), "--temperature", "1", "--max-new", str(MAX_NEW),
           "--workers", "2", "--seed", "1", "--device", "cuda", "--out", str(od), *extra]
    if not tok.exists() and "--tokenizer" not in extra:
        cmd += ["--tokenizer", str(next(p for p in inp.rglob("tokenizer.json") if "g10-recipe-e9-360m/" in str(p) + "/"))]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu % 2)}
    print("$ " + " ".join(cmd), flush=True)
    procs.append((tag, subprocess.Popen(cmd, env=env, stdout=open(od / "screen.log", "w"), stderr=subprocess.STDOUT)))
rcs = {tag: p.wait() for tag, p in procs}
for tag, _ in procs:
    od = out / tag
    meta = od / "screen_meta.json"
    if meta.exists():
        g = json.loads(meta.read_text()).get("generation_runs", [])
        print(tag, g, flush=True)
    rep = od / "screen_report.json"
    if rep.exists():
        r = json.loads(rep.read_text())
        print(tag, {s: {"ladder": {k: v["share"] for k, v in d["ladder"].items()},
                        "mixed": d.get("groups_pass_kept", {}).get("mixed"), "truncated": d["truncated_share"]}
                    for s, d in r["sets"].items()}, flush=True)
print({"rc": rcs, "hours": round((time.time() - t0) / 3600, 2)}, flush=True)
sys.exit(max(rcs.values()))
