#!/usr/bin/env python3
"""
zkp_eval.py — Evaluation harness for problem 200 (ZKP: Subset-Sum Partition).

Runs the agent's solution.py, evaluates:
  1. Mechanical correctness (completeness + soundness empirically)
  2. Qualitative ZK argument (scored 0-3 by the judge)

Returns JSON:
  {
    "mechanical_score": float,   # 0-3 (completeness 1pt + soundness 1-2pt)
    "qualitative_score": float,  # 0-3 (judge-assigned)
    "total_score": float,        # mechanical + qualitative (max 6, typical good = 4-5)
    "completeness_ok": bool,
    "soundness_error": float,
    "zk_argument": str,
    "errors": [str],
    "notes": str
  }

Usage:
    python3 zkp_eval.py --solution /tmp/solution.py
    python3 zkp_eval.py --solution /tmp/solution.py --judge-score 2.5
"""

import argparse
import importlib.util
import json
import os
import random
import subprocess
import sys
import traceback

CHEAT_TRIALS = 500   # number of cheat attempts for soundness estimation
HONEST_TRIALS = 10   # number of honest runs to verify completeness


def load_solution(path: str):
    spec = importlib.util.spec_from_file_location("solution", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def run_solution_stdout(path: str) -> dict:
    """Run solution as subprocess and parse its JSON stdout."""
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


def evaluate_mechanical(solution_path: str) -> dict:
    """
    Evaluate mechanical correctness:
    - Load and parse the solution's stdout JSON
    - Re-run completeness test using the solution's own prove() function
    - Run soundness test using the solution's own cheat_attempt() function
    """
    result = {
        "completeness_ok": False,
        "soundness_error": 1.0,
        "zk_argument": "",
        "test_case": {},
        "errors": [],
        "mechanical_score": 0.0,
    }

    # Step 1: get the solution's self-reported output
    stdout_data = run_solution_stdout(solution_path)
    if "error" in stdout_data:
        result["errors"].append(f"Solution stdout error: {stdout_data['error']}")
        return result

    result["zk_argument"] = stdout_data.get("zk_argument", "")
    result["test_case"] = stdout_data.get("test_case", {})
    self_reported_completeness = stdout_data.get("completeness_ok", False)
    self_reported_soundness = stdout_data.get("soundness_error", 1.0)

    # Step 2: independently verify by importing and calling prove() / cheat_attempt()
    try:
        mod = load_solution(solution_path)
    except Exception as e:
        result["errors"].append(f"Failed to import solution: {traceback.format_exc()}")
        return result

    # Check required functions exist
    for fn in ["prove", "cheat_attempt"]:
        if not hasattr(mod, fn):
            result["errors"].append(f"Missing function: {fn}()")
            return result

    # Extract test case
    tc = result["test_case"]
    S = tc.get("S", [])
    A = tc.get("A", [])
    B = tc.get("B", [])

    if not S or not A or not B:
        result["errors"].append("test_case missing S, A, or B")
        return result

    if len(S) < 10:
        result["errors"].append(f"S has only {len(S)} elements — need ≥ 10")

    # Verify the partition is actually valid
    if sorted(A + B) != sorted(S):
        result["errors"].append("A ∪ B ≠ S — partition is invalid")
        return result
    if sum(A) != sum(B):
        result["errors"].append(f"sum(A)={sum(A)} ≠ sum(B)={sum(B)} — not a valid partition")
        return result

    # Step 3: completeness — run prove() HONEST_TRIALS times, all must pass
    completeness_passes = 0
    for _ in range(HONEST_TRIALS):
        try:
            r = mod.prove(S, (A, B), rounds=20)
            if r:
                completeness_passes += 1
        except Exception as e:
            result["errors"].append(f"prove() raised exception: {e}")
            break

    result["completeness_ok"] = (completeness_passes == HONEST_TRIALS)

    # Step 4: soundness — run cheat_attempt() CHEAT_TRIALS times
    cheat_passes = 0
    try:
        # Try batch first (faster if solution supports it)
        cheat_result = mod.cheat_attempt(S, rounds=20)
        if isinstance(cheat_result, float):
            # Solution returned a fraction directly
            result["soundness_error"] = cheat_result
        elif isinstance(cheat_result, bool):
            # Single trial — run many
            passes = 1 if cheat_result else 0
            for _ in range(CHEAT_TRIALS - 1):
                try:
                    if mod.cheat_attempt(S, rounds=20):
                        passes += 1
                except Exception:
                    pass
            result["soundness_error"] = passes / CHEAT_TRIALS
        else:
            result["soundness_error"] = float(cheat_result)
    except Exception as e:
        result["errors"].append(f"cheat_attempt() raised exception: {e}")
        result["soundness_error"] = 1.0

    # Step 5: compute mechanical score
    score = 0.0
    if result["completeness_ok"]:
        score += 1.0
    se = result["soundness_error"]
    if se <= 0.10:
        score += 1.0
    if se <= 0.01:
        score += 1.0  # bonus for tight soundness

    # Penalise missing or too-short ZK argument
    zk_arg = result["zk_argument"]
    if not zk_arg or len(zk_arg.split()) < 50:
        result["errors"].append("zk_argument too short (< 50 words) — qualitative score will be 0")

    result["mechanical_score"] = score
    return result


def qualitative_rubric() -> str:
    return """
JUDGE EVALUATION RUBRIC — ZKP Zero-Knowledge Argument (0–3 points)

Read the agent's `zk_argument` field carefully. Score it 0–3:

3 points — GENUINE ZK:
  - Describes a concrete simulator that produces valid transcripts without the witness
  - Explains WHY simulator transcripts are indistinguishable from real transcripts
  - Identifies the specific hiding property of the commitment scheme used
  - Notes any caveats (e.g. computational vs statistical ZK, random oracle assumptions)

2 points — PARTIAL ZK:
  - Correct intuition but missing the simulator description
  - OR: correctly describes what the verifier sees but doesn't argue indistinguishability
  - OR: valid argument but with one identifiable gap or unstated assumption

1 point — WEAK / HAND-WAVY:
  - Claims ZK because "verifier only sees commitments" without explaining hiding property
  - OR: describes the protocol correctly but makes no ZK argument
  - OR: confuses ZK with soundness or completeness

0 points — WRONG or ABSENT:
  - No ZK argument provided
  - Argument is demonstrably incorrect (e.g. commitment scheme is not hiding)
  - Circular reasoning ("it's ZK because the prover doesn't reveal A and B")

IMPORTANT: A commitment scheme using hash(value) with no random blinding is NOT hiding.
If the agent uses such a scheme, the ZK claim is false regardless of how it's argued.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True)
    parser.add_argument("--judge-score", type=float, default=None,
                        help="Qualitative judge score 0-3 (if omitted, prints rubric and exits)")
    args = parser.parse_args()

    if not os.path.exists(args.solution):
        print(json.dumps({"error": f"Solution not found: {args.solution}"}))
        sys.exit(1)

    print("Evaluating mechanical correctness...", file=sys.stderr)
    mech = evaluate_mechanical(args.solution)

    if args.judge_score is None:
        # Print mechanical results + rubric for human/judge evaluation
        print("\n" + "="*60, file=sys.stderr)
        print("MECHANICAL RESULTS", file=sys.stderr)
        print("="*60, file=sys.stderr)
        print(f"  Completeness : {'✓' if mech['completeness_ok'] else '✗'}", file=sys.stderr)
        print(f"  Soundness err: {mech['soundness_error']:.4f}", file=sys.stderr)
        print(f"  Mech score   : {mech['mechanical_score']:.1f} / 3.0", file=sys.stderr)
        if mech["errors"]:
            print(f"  Errors       : {mech['errors']}", file=sys.stderr)
        print(f"\nZK ARGUMENT:\n{mech['zk_argument']}", file=sys.stderr)
        print(qualitative_rubric(), file=sys.stderr)
        print("\nRe-run with --judge-score N to record final result.", file=sys.stderr)

        # Still output partial JSON so score.py can use mechanical score
        result = {
            "mechanical_score": mech["mechanical_score"],
            "qualitative_score": None,
            "total_score": mech["mechanical_score"],
            "completeness_ok": mech["completeness_ok"],
            "soundness_error": mech["soundness_error"],
            "zk_argument": mech["zk_argument"],
            "errors": mech["errors"],
            "notes": "qualitative_score pending judge evaluation",
        }
        print(json.dumps(result, indent=2))
        sys.exit(0)

    # Judge score provided — compute final result
    qual_score = max(0.0, min(3.0, args.judge_score))
    total = mech["mechanical_score"] + qual_score

    result = {
        "mechanical_score": mech["mechanical_score"],
        "qualitative_score": qual_score,
        "total_score": total,
        "completeness_ok": mech["completeness_ok"],
        "soundness_error": mech["soundness_error"],
        "zk_argument": mech["zk_argument"],
        "errors": mech["errors"],
        "notes": f"max possible: 6.0 (3 mechanical + 3 qualitative)",
    }
    print(json.dumps(result, indent=2))
    sys.exit(0 if mech["completeness_ok"] else 1)


if __name__ == "__main__":
    main()
