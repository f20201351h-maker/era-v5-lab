"""Build the two PRIVATE Kaggle datasets for kaggle_recipe.py: the code (repo layout) and the inputs.

    python make_kaggle_bundle.py --work <work/recipe-e9-2026-11-16> --out <dir> --owner <kaggle user>

code/  teams/10-post-training (without rl/ except rl/verifier.py: rl/tasks are assistant-written evaluation tasks and
       never go to an RL job; without results/ and swe-envs/) + teams/05-architecture/heldout_v1 (forgetting metric).
data/  dolly.tok.jsonl, ocr2_clean.tok.jsonl, cc.jsonl restricted to the tasks the runs use (rung 1, rest sample),
       the id lists. Training / evaluation text from admitted sources only; keep the dataset private.
Each folder gets a dataset-metadata.json (private by default with `kaggle datasets create`). A MANIFEST.json lists
every file with its sha256 so the run can be tied to these exact inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEAM = HERE.parents[2]
REPO = TEAM.parents[1]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def manifest(root: Path) -> None:
    files = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"}
    (root / "MANIFEST.json").write_text(json.dumps(files, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--owner", default="OWNER")
    a = ap.parse_args(argv)
    code, data = a.out / "code", a.out / "data"
    for d in (code, data):
        if d.exists():
            shutil.rmtree(d)
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "out", "runs", "results", "swe-envs")
    shutil.copytree(TEAM, code / "teams" / "10-post-training", ignore=ign)
    rl = code / "teams" / "10-post-training" / "rl"
    shutil.rmtree(rl)
    rl.mkdir()
    shutil.copy(TEAM / "rl" / "verifier.py", rl / "verifier.py")
    shutil.copytree(REPO / "teams" / "05-architecture" / "heldout_v1", code / "teams" / "05-architecture" / "heldout_v1",
                    ignore=ign)
    data.mkdir(parents=True)
    w = a.work
    for src, name in ((w / "private/sft/dolly.tok.jsonl", "dolly.tok.jsonl"),
                      (w / "private/sft/ocr2_clean.tok.jsonl", "ocr2_clean.tok.jsonl"),
                      (w / "private/sft/ocr2_clean_v2.tok.jsonl", "ocr2_clean_v2.tok.jsonl"),
                      (w / "private/sft/ocr2_short_v2.tok.jsonl", "ocr2_short_v2.tok.jsonl")):
        shutil.copy(src, data / name)
    ids = {}
    for f in ("rung1_ids.txt", "rest_sample_ids.txt", "rung1_s128_ids.txt", "rest_s32_ids.txt", "rung1_s64_ids.txt",
              "e10_train_ids.txt", "e10_heldout_ids.txt"):
        shutil.copy(w / "private/rung" / f, data / f)
        ids[f] = (data / f).read_text().split()
    want = set().union(*ids.values())
    tasks_src = w.parent / "rl-data-v1-2026-11-11/out/v1/cc.jsonl"
    n = 0
    with open(tasks_src, encoding="utf-8") as fi, open(data / "cc.jsonl", "w", encoding="utf-8") as fo:
        for line in fi:
            if json.loads(line)["id"] in want:
                fo.write(line)
                n += 1
    if n != len(want):
        raise SystemExit(f"{len(want) - n} ids missing from {tasks_src}")
    for root, slug, title in ((code, "g10-recipe-e9-code", "G10 recipe E9 code"),
                              (data, "g10-recipe-e9-data", "G10 recipe E9 data (private)")):
        manifest(root)
        (root / "dataset-metadata.json").write_text(json.dumps(
            {"title": title, "id": f"{a.owner}/{slug}", "licenses": [{"name": "other"}]}, indent=1))
    print(json.dumps({"code_files": sum(1 for _ in code.rglob("*") if _.is_file()), "tasks": n,
                      "data_files": sorted(p.name for p in data.iterdir()), "tasks_source_sha256": sha(tasks_src)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
