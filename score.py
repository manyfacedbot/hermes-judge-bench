#!/usr/bin/env python3
"""
hermes-judge-bench/score.py

Reads results/*.json and produces a markdown leaderboard.

Scoring:
  - correctness: 1.0 if answer matches answers.json, 0.0 otherwise
                 (problem 000 smoke test is always scored 1.0 if solution.py exists)
  - pareto_score: correctness / log2(elapsed_seconds + 2)
                  (log2 so short runs aren't insanely penalized; +2 avoids log(0))

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
    except Exception as e:
        return None


def score_results(results_dir: str):
    answers = load_answers()
    rows = []

    for result_file in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        # Skip solution files
        if "solution" in os.path.basename(result_file):
            continue

        with open(result_file) as f:
            data = json.load(f)

        problem = data.get("problem", "?")
        judge = data.get("judge", "?")
        model = data.get("model", "?")
        elapsed = data.get("elapsed_seconds", 0)
        solution = data.get("solution", "")
        response = data.get("response", "")

        # Smoke test (000) — correctness = 1 if solution.py exists and runs
        if problem == "000":
            output = extract_answer(solution, problem)
            correctness = 1.0 if output else 0.0
            expected = "(haiku)"
        else:
            expected_raw = answers.get(problem)
            expected = str(expected_raw) if expected_raw is not None else None
            if expected is None:
                correctness = 0.0
            else:
                output = extract_answer(solution, problem)
                correctness = 1.0 if output and output.strip() == expected.strip() else 0.0

        pareto = correctness / math.log2(elapsed + 2) if elapsed >= 0 else 0.0

        rows.append({
            "problem": problem,
            "judge": judge,
            "model": model,
            "elapsed_s": elapsed,
            "correctness": correctness,
            "pareto_score": round(pareto, 4),
            "expected": expected,
            "got": extract_answer(solution, problem) if solution else "—",
        })

    if not rows:
        print("No results found. Run ./run.sh first.")
        return

    # Print leaderboard
    print("\n## hermes-judge-bench Leaderboard\n")
    print(f"{'Problem':<10} {'Judge':<20} {'Model':<25} {'Elapsed':>8} {'Correct':>8} {'Pareto':>8} {'Expected':>12} {'Got':>12}")
    print("-" * 110)
    for r in sorted(rows, key=lambda x: -x["pareto_score"]):
        correct_str = "✓" if r["correctness"] == 1.0 else "✗"
        print(
            f"{r['problem']:<10} {r['judge']:<20} {r['model']:<25} "
            f"{r['elapsed_s']:>7}s {correct_str:>8} {r['pareto_score']:>8} "
            f"{str(r['expected']):>12} {str(r['got']):>12}"
        )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=os.path.join(REPO_DIR, "results"))
    args = parser.parse_args()
    score_results(args.results)
