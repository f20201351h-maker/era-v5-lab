"""Turn a local copy of a human-written contest set into stdio task records (JSONL), with source, licence and
generator recorded per task. No network: point it at a parquet file that TSAI (or you) downloaded.

    python build.py --source code_contests --parquet <path/to/train-*.parquet> --out out/tasks.jsonl
                    [--min-rating 800 --max-rating 1200 --max-tests 20]

Sources (the generator rule, TSAI 3 Nov: human-written data is in; record the generator for every set):
    code_contests   deepmind/code_contests, CC-BY-4.0. Problems and solutions are human (Codeforces, AtCoder,
                    CodeChef, Aizu, HackerEarth); extra tests are programmatic mutations of existing inputs.
                    Licence table: `rl-code-code-contests`, TSAI reserves. Only Codeforces rows (source == 2)
                    are kept, so each task has a rating.

Which sets may be used for training is TSAI's call ("TSAI reserves" in licences/licence_table.md). This script
only prepares the records; the records themselves are data and stay out of the repository (out/ is local).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SOURCES = {
    "code_contests": {"dataset": "deepmind/code_contests", "licence": "CC-BY-4.0",
                      "generator": "none (human problems and tests; extra tests are programmatic mutations)",
                      "licence_table_set": "rl-code-code-contests"},
}


def code_contests_rows(rows: list[dict], min_rating: int, max_rating: int, max_tests: int) -> list[dict]:
    meta = SOURCES["code_contests"]
    out = []
    for r in rows:
        if r.get("source") != 2 or not (min_rating <= (r.get("cf_rating") or 0) <= max_rating):
            continue
        if r.get("input_file") or r.get("output_file") or not r.get("description"):
            continue  # file-based IO, or no statement
        tests = []
        for key in ("public_tests", "private_tests", "generated_tests"):
            t = r.get(key) or {}
            tests += [{"input": i, "output": o} for i, o in zip(t.get("input", []), t.get("output", []))]
        if len(tests) < 3:
            continue
        cf = f"{r.get('cf_contest_id')}{r.get('cf_index')}"
        out.append({"id": f"cf{cf}", "statement": r["description"], "tests": tests[:max_tests],
                    "time_limit_s": 4, "rating": r["cf_rating"], "source": f"{meta['dataset']} (Codeforces {cf})",
                    "licence": meta["licence"], "generator": meta["generator"],
                    "licence_table_set": meta["licence_table_set"]})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=SOURCES, required=True)
    ap.add_argument("--parquet", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-rating", type=int, default=800)
    ap.add_argument("--max-rating", type=int, default=1200)
    ap.add_argument("--max-tests", type=int, default=20)
    args = ap.parse_args(argv)
    import pyarrow.parquet as pq  # noqa: PLC0415  (only needed for the CLI)

    cols = ["name", "description", "public_tests", "private_tests", "generated_tests", "source", "cf_rating",
            "cf_contest_id", "cf_index", "input_file", "output_file"]
    tasks = []
    for path in args.parquet:
        tasks += code_contests_rows(pq.read_table(path, columns=cols).to_pylist(), args.min_rating, args.max_rating,
                                    args.max_tests)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(json.dumps({"tasks": len(tasks), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
