"""Kaggle kernel entry (pushed with `kaggle kernels push`; GPU T4 x2, internet on for pip and the pinned model).

Finds the two private datasets under /kaggle/input (code: has teams/10-post-training; data: has dolly.tok.jsonl),
pins transformers to the version the Mac runs used, checks the program sandbox BEFORE any GPU work (positive and
negative control through the `local` runner), then runs kaggle_recipe.py. Arguments come from KAGGLE_RECIPE_ARGS in
this file's header line below so the same file serves E9 at 360M, E9 at 1.7B or E10. Output: /kaggle/working/out.
"""
KAGGLE_RECIPE_ARGS = "e9 --size 360m"

import os
import subprocess
import sys
import time
from pathlib import Path

t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.22.1"], check=True)
inp = Path("/kaggle/input")
found = [p.parent.parent for p in inp.rglob("10-post-training") if p.parent.name == "teams"]
if not found:  # the dataset kept teams.zip as an archive: unpack it into the working area
    import zipfile
    z = next(inp.rglob("teams.zip"))
    zipfile.ZipFile(z).extractall("/kaggle/working/code/teams")
    found = [Path("/kaggle/working/code")]
code = found[0]
data = next(p.parent for p in inp.rglob("dolly.tok.jsonl"))
team = code / "teams" / "10-post-training"
print({"code": str(code), "data": str(data), "python": sys.version.split()[0]}, flush=True)
import torch  # noqa: E402
import transformers  # noqa: E402
print({"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.cuda.is_available(),
       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "uid": os.getuid()}, flush=True)
os.environ["G10_STDIO_RUNNER"] = "local"
sys.path.insert(0, str(team / "sft" / "bench"))
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("rs", team / "sft" / "bench" / "rl_screen.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)
print({"sandbox_preflight": rs.sandbox_preflight(2)}, flush=True)  # exits here if the sandbox cannot isolate programs
out = Path("/kaggle/working/out")
args = KAGGLE_RECIPE_ARGS.split()
if args[0] == "e10" and "--sft-model" not in args:
    args += ["--sft-model", str(next(inp.rglob("A2/model")))]
cmd = [sys.executable, str(team / "rl-loop/pilots/recipe_e9_e10_20261116/kaggle_recipe.py"), *args, "--data", str(data),
       "--out", str(out / args[0])]
print("$ " + " ".join(cmd), flush=True)
rc = subprocess.run(cmd).returncode
print({"rc": rc, "hours": round((time.time() - t0) / 3600, 2)}, flush=True)
sys.exit(rc)
