#!/usr/bin/env python3
"""
zkp_eval.py — Fully automated evaluation harness for problem 200 (ZKP: Subset-Sum Partition).

Scoring (max 4.0, normalised to 1.0):
  1. Completeness (1pt):  honest prover passes all rounds
  2. Soundness   (2pt):  empirical cheat success rate ≤ 0.10 (+1) or ≤ 0.01 (+2)
  3. Bound check (1pt):  soundness_bound(rounds) is a valid formula satisfying:
                          - bound >= empirical soundness_error (not over-optimistic)
                          - bound <= 1.0 (non-trivial)
                          - bound(40) < bound(20) (decreasing with rounds — it's a formula)

All scoring is deterministic and requires no human input.

Returns JSON to stdout:
  {
    "completeness_ok": bool,
    "soundness_error": float,
    "soundness_bound_20": float,
    "bound_correct": bool,
    "bound_reason": str,
    "mechanical_score": float,   # 0-4
    "normalised": float,         # 0.0-1.0 (mechanical_score / 4.0)
    "errors": [str]
  }

Usage:
    python3 zkp_eval.py --solution /tmp/solution.py
    python3 zkp_eval.py --solution /tmp/solution.py --rounds 20 --cheat-trials 500
"""

import argparse
import importlib.util
import json
import os
import sys
import traceback

CHEAT_TRIALS_DEFAULT = 500
HONEST_TRIALS = 10
MAX_SCORE = 4.0


def load_solution(path: str):
    spec = importlib.util.spec_from_file_location("solution", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def run_stdout(path: str) -> dict:
    """Run solution as subprocess and parse its JSON stdout."""
    import subprocess
    try:
        r = subprocess.run(
            ["python3", path],
            capture_output=True, text=True, timeout=120
        )
        if not r.stdout.strip():
            return {"error": f"No stdout. stderr: {r.stderr[:500]}"}
        return json.loads(r.stdout)
    except subprocess.TimeoutExpired:
        return {"error": "Timed out after 120s"}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON output: {e}"}
    except Exception as e:
        return {"error": str(e)}


def evaluate(solution_path: str, rounds: int, cheat_trials: int) -> dict:
    result = {
        "completeness_ok": False,
        "soundness_error": 1.0,
        "soundness_bound_20": None,
        "bound_correct": False,
        "bound_reason": "",
        "mechanical_score": 0.0,
        "normalised": 0.0,
        "errors": [],
    }

    # --- Step 1: run solution to get self-reported output + test_case ---
    stdout_data = run_stdout(solution_path)
    if "error" in stdout_data:
        result["errors"].append(f"stdout error: {stdout_data['error']}")
        return result

    tc = stdout_data.get("test_case", {})
    S = tc.get("S", [])
    A = tc.get("A", [])
    B = tc.get("B", [])

    if not S or not A or not B:
        result["errors"].append("test_case missing S, A, or B in stdout JSON")
        return result

    if len(S) < 10:
        result["errors"].append(f"|S| = {len(S)} < 10")

    if sorted(A + B) != sorted(S):
        result["errors"].append("A ∪ B ≠ S — invalid partition")
        return result

    if sum(A) != sum(B):
        result["errors"].append(f"sum(A)={sum(A)} ≠ sum(B)={sum(B)}")
        return result

    # --- Step 2: import module, check required functions ---
    try:
        mod = load_solution(solution_path)
    except Exception:
        result["errors"].append(f"import failed:\n{traceback.format_exc()}")
        return result

    for fn in ["prove", "cheat_attempt", "soundness_bound"]:
        if not hasattr(mod, fn):
            result["errors"].append(f"missing function: {fn}()")
            return result

    # --- Step 3: completeness ---
    completeness_passes = 0
    for _ in range(HONEST_TRIALS):
        try:
            r = mod.prove(S, (A, B), rounds=rounds)
            if r:
                completeness_passes += 1
        except Exception as e:
            result["errors"].append(f"prove() exception: {e}")
            break
    result["completeness_ok"] = (completeness_passes == HONEST_TRIALS)

    # --- Step 4: soundness (empirical) ---
    cheat_passes = 0
    for _ in range(cheat_trials):
        try:
            r = mod.cheat_attempt(S, rounds=rounds)
            if isinstance(r, bool):
                if r:
                    cheat_passes += 1
            elif isinstance(r, (int, float)):
                # If the function returned a rate directly, use it and break
                result["soundness_error"] = float(r)
                cheat_passes = -1  # sentinel
                break
            else:
                result["errors"].append(f"cheat_attempt() returned unexpected type: {type(r)}")
                break
        except Exception as e:
            result["errors"].append(f"cheat_attempt() exception: {e}")
            break

    if cheat_passes != -1:
        result["soundness_error"] = cheat_passes / cheat_trials

    se = result["soundness_error"]

    # --- Step 5: soundness_bound check ---
    try:
        b20 = float(mod.soundness_bound(rounds))
        b40 = float(mod.soundness_bound(rounds * 2))
        result["soundness_bound_20"] = b20

        reasons = []
        ok = True

        if b20 < se:
            ok = False
            reasons.append(
                f"bound({rounds})={b20:.6f} < empirical error {se:.6f} — over-optimistic"
            )
        if b20 >= 1.0:
            ok = False
            reasons.append(f"bound({rounds})={b20:.6f} ≥ 1.0 — trivial/non-informative")
        if b40 >= b20:
            ok = False
            reasons.append(
                f"bound({rounds*2})={b40:.6f} ≥ bound({rounds})={b20:.6f} — not decreasing with rounds"
            )

        result["bound_correct"] = ok
        result["bound_reason"] = "; ".join(reasons) if reasons else "valid"

    except Exception as e:
        result["errors"].append(f"soundness_bound() exception: {e}")
        result["bound_reason"] = f"exception: {e}"

    # --- Step 6: compute total score ---
    score = 0.0
    if result["completeness_ok"]:
        score += 1.0
    if se <= 0.01:
        score += 2.0
    elif se <= 0.10:
        score += 1.0
    if result["bound_correct"]:
        score += 1.0

    result["mechanical_score"] = score
    result["normalised"] = score / MAX_SCORE
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--cheat-trials", type=int, default=CHEAT_TRIALS_DEFAULT)
    args = parser.parse_args()

    if not os.path.exists(args.solution):
        print(json.dumps({"error": f"not found: {args.solution}"}))
        sys.exit(1)

    result = evaluate(args.solution, args.rounds, args.cheat_trials)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["completeness_ok"] else 1)


if __name__ == "__main__":
    main()
