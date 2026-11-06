"""Turn the MATH train split (hendrycks MATH) into math task records (JSONL), with source, licence and generator.

    python build_math.py --parquet-dir <folder with <subject>/train-*.parquet> --out out/math_tasks.jsonl

Source: EleutherAI/hendrycks_math (MIT, Hub tag and GitHub LICENSE). Problems and step-by-step solutions are
human-written competition material (AMC, AIME and others; the MATH paper says many come from the AoPS archives).
No model wrote any of it. Only the TRAIN split is used; the test split (and MATH-500, a subset of it) is evaluation
material and is never read. Not yet in licences/licence_table.md; whether it may be used for RL training is
TSAI's call. Records are data and stay out of the repository (out/ is local).

Each record: id "math-<subject>-<row>", problem, answer (content of the solution's last \\boxed{}), level, subject.
Rows whose solution has no \\boxed{} answer are skipped and counted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from math_check import last_boxed

SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
            "prealgebra", "precalculus")
META = {"dataset": "EleutherAI/hendrycks_math", "licence": "MIT",
        "generator": "none (human competition problems and solutions)", "licence_table_set": None}


def math_rows(subject: str, rows: list[dict]) -> tuple[list[dict], int]:
    out, skipped = [], 0
    for i, r in enumerate(rows):
        ans = last_boxed(r.get("solution") or "")
        if not ans or not r.get("problem"):
            skipped += 1
            continue
        out.append({"id": f"math-{subject}-{i}", "kind": "math", "problem": r["problem"], "answer": ans,
                    "level": r.get("level"), "subject": subject,
                    "source": f"{META['dataset']} (train, {subject})", "licence": META["licence"],
                    "generator": META["generator"], "licence_table_set": META["licence_table_set"]})
    return out, skipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    import pyarrow.parquet as pq  # noqa: PLC0415

    tasks, skipped = [], {}
    for s in SUBJECTS:
        files = sorted((args.parquet_dir / s).glob("train-*.parquet"))
        if not files:
            raise FileNotFoundError(f"no train parquet for {s} under {args.parquet_dir}")
        rows = [r for f in files for r in pq.read_table(f, columns=["problem", "solution", "level"]).to_pylist()]
        got, skipped[s] = math_rows(s, rows)
        tasks += got
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(json.dumps({"tasks": len(tasks), "skipped_no_boxed": skipped, "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
