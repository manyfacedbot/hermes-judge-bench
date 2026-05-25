#!/usr/bin/env python3
"""
hermes-judge-bench/score.py

Reads results/*.json and produces a markdown leaderboard.

Scoring — two modes depending on problem type:

Standard problems (000–099):
  - correctness: 1.0 if answer matches answers.json, 0.0 otherwise
                 (problem 000 smoke test: 1.0 if solution.py exists and runs)
  - pareto_score: correctness / log2(elapsed_seconds + 2)

Poetry compression problems (100–103):
  - runs problems/poetry_eval.py against the solution to get compression_ratio
  - correctness = compression_ratio if verified, else 0.0
  - pareto_score: compression_ratio / (1 + elapsed_seconds / 60)
    (elapsed_seconds proxies token usage; 60s ≈ 10k tokens at typical agent speed)
  - Note: true formula is compression_ratio / (1 + tokens_used / 10000)
    but token counts are not currently tracked by run.sh

Usage:
  python3 score.py
  python3 score.py --results results/  # custom results dir
"""

import json
import math
import os
import sys
import glob
import argparse
import subprocess

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

POETRY_PROBLEMS = {"100", "101", "102", "103"}
POETRY_CORPUS = {
    "100": "corpus/romantic_nature.json",
    "101": "corpus/victorian_lyric.json",
    "102": "corpus/ode_and_elegy.json",
    "103": "corpus/sonnet.json",
}

POETRY_EVAL = os.path.join(REPO_DIR, "problems", "poetry_eval.py")


def load_answers():
    path = os.path.join(REPO_DIR, "answers.json")
    with open(path) as f:
        return json.load(f)


def extract_answer(solution_py: str, problem: str) -> str | None:
    """Run solution.py and capture stdout."""
    if not solution_py.strip():
        return None
    tmp = f"/tmp/bench_solution_{problem}.py"
    with open(tmp, "w") as f:
        f.write(solution_py)
    try:
        result = subprocess.run(
            ["python3", tmp], capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip()
    except Exception:
        return None


def run_poetry_eval(solution_py: str, problem: str) -> dict:
    """Run poetry_eval.py against the solution and return the result dict."""
    if not solution_py.strip():
        return {"verified": False, "compression_ratio": 0.0, "error": "no solution"}

    tmp = f"/tmp/bench_solution_{problem}.py"
    with open(tmp, "w") as f:
        f.write(solution_py)

    corpus_rel = POETRY_CORPUS[problem]
    corpus_path = os.path.join(REPO_DIR, "problems", corpus_rel)

    try:
        result = subprocess.run(
            ["python3", POETRY_EVAL,
             "--corpus", corpus_path,
             "--solution", tmp,
             "--n", "5",
             "--seed", "42"],
            capture_output=True, text=True, timeout=120
        )
        if result.stdout.strip():
            return json.loads(result.stdout)
        else:
            return {"verified": False, "compression_ratio": 0.0,
                    "error": result.stderr[:500]}
    except Exception as e:
        return {"verified": False, "compression_ratio": 0.0, "error": str(e)}


def score_results(results_dir: str):
    answers = load_answers()
    rows = []

    for result_file in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if "solution" in os.path.basename(result_file):
            continue

        with open(result_file) as f:
            data = json.load(f)

        problem = data.get("problem", "?")
        judge = data.get("judge", "?")
        model = data.get("model", "?")
        elapsed = data.get("elapsed_seconds", 0)
        solution = data.get("solution", "")

        if problem in POETRY_PROBLEMS:
            # Poetry compression scoring
            eval_result = run_poetry_eval(solution, problem)
            verified = eval_result.get("verified", False)
            compression_ratio = eval_result.get("compression_ratio", 0.0)
            correctness = compression_ratio if verified else 0.0
            pareto = correctness / (1 + elapsed / 60) if elapsed >= 0 else 0.0
            expected = f"cr>{1.0:.2f} verified"
            got = f"cr={compression_ratio:.3f} {'✓' if verified else '✗'}"
            score_type = "poetry"
        elif problem == "000":
            # Smoke test
            output = extract_answer(solution, problem)
            correctness = 1.0 if output else 0.0
            pareto = correctness / math.log2(elapsed + 2) if elapsed >= 0 else 0.0
            expected = "(haiku)"
            got = output or "—"
            score_type = "standard"
        else:
            # Standard correctness scoring
            expected_raw = answers.get(problem)
            expected = str(expected_raw) if expected_raw is not None else None
            if expected is None:
                correctness = 0.0
                got = "—"
            else:
                output = extract_answer(solution, problem)
                correctness = 1.0 if output and output.strip() == expected.strip() else 0.0
                got = output or "—"
            pareto = correctness / math.log2(elapsed + 2) if elapsed >= 0 else 0.0
            score_type = "standard"

        rows.append({
            "problem": problem,
            "judge": judge,
            "model": model,
            "elapsed_s": elapsed,
            "correctness": round(correctness, 4),
            "pareto_score": round(pareto, 4),
            "expected": expected,
            "got": got,
            "score_type": score_type,
        })

    if not rows:
        print("No results found. Run ./run.sh first.")
        return

    # Print leaderboard
    print("\n## hermes-judge-bench Leaderboard\n")
    print(f"{'Problem':<10} {'Judge':<20} {'Model':<25} {'Elapsed':>8} {'Score':>10} {'Pareto':>8} {'Got':>20}")
    print("-" * 115)

    std_rows = [r for r in rows if r["score_type"] == "standard"]
    poetry_rows = [r for r in rows if r["score_type"] == "poetry"]

    if std_rows:
        print("\n### Standard Problems (correctness / log2(elapsed+2))\n")
        for r in sorted(std_rows, key=lambda x: -x["pareto_score"]):
            correct_str = "✓" if r["correctness"] == 1.0 else "✗"
            print(
                f"{r['problem']:<10} {r['judge']:<20} {r['model']:<25} "
                f"{r['elapsed_s']:>7}s {correct_str:>10} {r['pareto_score']:>8} "
                f"{str(r['got']):>20}"
            )

    if poetry_rows:
        print("\n### Poetry Compression Problems (compression_ratio / (1 + elapsed_s/60))\n")
        print("  Note: elapsed_seconds proxies token usage (true formula: cr / (1 + tokens/10000))\n")
        for r in sorted(poetry_rows, key=lambda x: -x["pareto_score"]):
            print(
                f"{r['problem']:<10} {r['judge']:<20} {r['model']:<25} "
                f"{r['elapsed_s']:>7}s {str(r['correctness']):>10} {r['pareto_score']:>8} "
                f"{str(r['got']):>20}"
            )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=os.path.join(REPO_DIR, "results"))
    args = parser.parse_args()
    score_results(args.results)
