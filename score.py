#!/usr/bin/env python3
"""
hermes-judge-bench/score.py

Reads results from heat directories and produces a markdown leaderboard.

Heat structure:
  results/
    heat_<timestamp>/
      <judge>/
        <problem>-result.json
        <problem>-solution.py
        <problem>-response.txt

Scoring:
  Standard problems (000–099):
    correctness / log2(elapsed_seconds + 2)

  Poetry compression problems (100–103):
    compression_ratio / (1 + effective_tokens / 10000)
    where effective_tokens = input + output + cache_read * cache_read_weight
    Falls back to elapsed_seconds proxy for legacy results without token data.

Usage:
  python3 score.py                   # all heats
  python3 score.py --heat heat_123   # specific heat
  python3 score.py --results results/
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
ZKP_PROBLEMS = {"200", "201", "202"}

POETRY_CORPUS = {
    "100": "corpus/romantic_nature.json",
    "101": "corpus/victorian_lyric.json",
    "102": "corpus/ode_and_elegy.json",
    "103": "corpus/sonnet.json",
}

POETRY_EVAL = os.path.join(REPO_DIR, "problems", "poetry_eval.py")
ZKP_EVAL    = os.path.join(REPO_DIR, "problems", "zkp_eval.py")


def load_answers():
    path = os.path.join(REPO_DIR, "answers.json")
    with open(path) as f:
        return json.load(f)


def extract_answer(solution_py: str, problem: str) -> str | None:
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


def find_result_files(results_dir: str, heat_filter: str | None) -> list[str]:
    """Find all *-result.json files, optionally filtered to a specific heat."""
    if heat_filter:
        pattern = os.path.join(results_dir, heat_filter, "*", "*-result.json")
    else:
        # Both new heat structure and legacy flat structure
        pattern_heat = os.path.join(results_dir, "heat_*", "*", "*-result.json")
        pattern_flat = os.path.join(results_dir, "*.json")
        files = glob.glob(pattern_heat) + [
            f for f in glob.glob(pattern_flat)
            if "solution" not in os.path.basename(f)
            and "result" not in os.path.basename(f).split("-")[-1].replace(".json", "")
            # legacy: NNN-judge.json files (no 'result' suffix)
            or (os.path.basename(f).endswith(".json")
                and "solution" not in os.path.basename(f)
                and not os.path.basename(f).startswith("heat_"))
        ]
        return sorted(set(files))
    return sorted(glob.glob(pattern))


def score_results(results_dir: str, heat_filter: str | None):
    answers = load_answers()
    rows = []

    # Find result files — handle both heat structure and legacy flat structure
    if heat_filter:
        result_files = sorted(glob.glob(
            os.path.join(results_dir, heat_filter, "*", "*-result.json")
        ))
    else:
        result_files = sorted(glob.glob(
            os.path.join(results_dir, "heat_*", "*", "*-result.json")
        ))
        # Also pick up legacy flat results (NNN-judge.json, no 'result' or 'solution' in name)
        for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
            bn = os.path.basename(f)
            if "solution" not in bn and not bn.endswith("-result.json"):
                result_files.append(f)

    if not result_files:
        print("No results found. Run ./run.sh first.")
        return

    for result_file in result_files:
        with open(result_file) as f:
            data = json.load(f)

        heat      = data.get("heat", "legacy")
        problem   = data.get("problem", "?")
        judge     = data.get("judge", "?")
        model     = data.get("model", "?")
        elapsed   = data.get("elapsed_seconds", 0)
        solution  = data.get("solution", "")
        killed    = data.get("killed_by_timeout", False)
        eff_tok   = data.get("effective_tokens", 0)
        raw_tok   = data.get("input_tokens", 0) + data.get("output_tokens", 0)
        cache_r   = data.get("cache_read_tokens", 0)
        token_note = ""

        # For heat structure, solution lives in a separate file
        if not solution:
            sol_path = os.path.join(os.path.dirname(result_file),
                                    f"{problem}-solution.py")
            if os.path.exists(sol_path):
                with open(sol_path) as sf:
                    solution = sf.read()

        if problem in ZKP_PROBLEMS:
            # ZKP scoring — run zkp_eval.py, use total_score (mechanical + qualitative)
            # qualitative_score may be None if judge hasn't scored it yet
            tmp = f"/tmp/zkp_solution_{problem}.py"
            with open(tmp, "w") as f:
                f.write(solution)
            try:
                r = subprocess.run(
                    ["python3", ZKP_EVAL, "--solution", tmp],
                    capture_output=True, text=True, timeout=180
                )
                if r.stdout.strip():
                    res = json.loads(r.stdout)
                    total = res.get("total_score", 0.0)
                    mech  = res.get("mechanical_score", 0.0)
                    qual  = res.get("qualitative_score")
                    correctness = total / 6.0  # normalise to [0,1] range for pareto
                    qual_str = f"{qual:.1f}" if qual is not None else "pending"
                    got = f"mech={mech:.1f} qual={qual_str} tot={total:.1f}/6"
                else:
                    correctness = 0.0
                    got = "eval failed"
            except Exception as e:
                correctness = 0.0
                got = f"error: {e}"
            if total_tokens > 0:
                pareto = correctness / (1 + total_tokens / 10000)
                token_note = f"{raw_tok}+{cache_r}cr"
            else:
                pareto = correctness / (1 + elapsed / 60)
                token_note = f"{elapsed}s~"
            expected = "mech+qual/6"
            score_type = "zkp"

        elif problem in POETRY_PROBLEMS:
            eval_result = run_poetry_eval(solution, problem)
            verified = eval_result.get("verified", False)
            compression_ratio = eval_result.get("compression_ratio", 0.0)
            correctness = compression_ratio if verified else 0.0
            if eff_tok > 0:
                pareto = correctness / (1 + eff_tok / 10000)
                token_note = f"{raw_tok}+{cache_r}cr"
            else:
                pareto = correctness / (1 + elapsed / 60)
                token_note = f"{elapsed}s~"
            expected = "cr>1.0 verified"
            got = f"cr={compression_ratio:.3f} {'✓' if verified else '✗'}"
            score_type = "poetry"

        elif problem == "000":
            output = extract_answer(solution, problem)
            correctness = 1.0 if output else 0.0
            pareto = correctness / math.log2(elapsed + 2)
            expected = "(haiku)"
            got = output or "—"
            score_type = "standard"

        else:
            expected_raw = answers.get(problem)
            expected = str(expected_raw) if expected_raw is not None else None
            if expected is None:
                correctness = 0.0
                got = "—"
            else:
                output = extract_answer(solution, problem)
                correctness = 1.0 if output and output.strip() == expected.strip() else 0.0
                got = output or "—"
            pareto = correctness / math.log2(elapsed + 2)
            score_type = "standard"

        rows.append({
            "heat":         heat,
            "problem":      problem,
            "judge":        judge,
            "model":        model,
            "elapsed_s":    elapsed,
            "eff_tokens":   eff_tok,
            "token_note":   token_note,
            "correctness":  round(correctness, 4),
            "pareto_score": round(pareto, 4),
            "expected":     expected,
            "got":          got,
            "score_type":   score_type,
            "killed":       killed,
        })

    if not rows:
        print("No scorable results found.")
        return

    # Group by heat for display
    heats = sorted(set(r["heat"] for r in rows))
    std_rows    = [r for r in rows if r["score_type"] == "standard"]
    poetry_rows = [r for r in rows if r["score_type"] == "poetry"]

    title = f"Heat: {heat_filter}" if heat_filter else f"All heats ({len(heats)})"
    print(f"\n## hermes-judge-bench Leaderboard — {title}\n")

    if std_rows:
        print("### Standard Problems  (correctness / log2(elapsed+2))\n")
        print(f"{'Heat':<22} {'Problem':<10} {'Judge':<20} {'Model':<25} {'Elapsed':>8} {'Score':>8} {'Pareto':>8} {'Got':>14} {'':>8}")
        print("-" * 130)
        for r in sorted(std_rows, key=lambda x: -x["pareto_score"]):
            c = "✓" if r["correctness"] == 1.0 else "✗"
            t = "⏱" if r.get("killed") else ""
            print(f"{r['heat']:<22} {r['problem']:<10} {r['judge']:<20} {r['model']:<25} "
                  f"{r['elapsed_s']:>7}s {c:>8} {r['pareto_score']:>8} {str(r['got']):>14} {t:>8}")

    if poetry_rows:
        print("\n### Poetry Compression  (compression_ratio / (1 + effective_tokens/10000))\n")
        print("  Tokens column: raw_input+output + cache_reads (cr). '~' = elapsed proxy (legacy).\n")
        print(f"{'Heat':<22} {'Problem':<10} {'Judge':<20} {'Model':<25} {'Tokens':>14} {'cr':>10} {'Pareto':>8} {'Got':>20} {'':>8}")
        print("-" * 140)
        for r in sorted(poetry_rows, key=lambda x: -x["pareto_score"]):
            t = "⏱" if r.get("killed") else ""
            print(f"{r['heat']:<22} {r['problem']:<10} {r['judge']:<20} {r['model']:<25} "
                  f"{str(r['token_note']):>14} {str(r['correctness']):>10} {r['pareto_score']:>8} "
                  f"{str(r['got']):>20} {t:>8}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=os.path.join(REPO_DIR, "results"))
    parser.add_argument("--heat", default=None, help="Score only this heat (e.g. heat_1716900000)")
    args = parser.parse_args()
    score_results(args.results, args.heat)
