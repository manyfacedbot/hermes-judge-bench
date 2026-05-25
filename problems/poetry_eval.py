#!/usr/bin/env python3
"""
poetry_eval.py — Evaluation harness for poetry compression problems (100–103).

Usage:
    python3 poetry_eval.py --corpus corpus/romantic_nature.json --solution /tmp/solution.py
    python3 poetry_eval.py --corpus corpus/sonnet.json --solution /tmp/solution.py --n 8 --seed 7

Output (JSON to stdout):
    {
        "compression_ratio": 1.84,
        "verified": true,
        "failures": 0,
        "poems_tested": 5,
        "details": [
            {"id": "rn_01", "title": "...", "author": "...",
             "original_chars": 512, "encoded_chars": 278, "ratio": 1.84, "roundtrip_ok": true}
        ]
    }

Scoring note:
    The benchmark runner uses elapsed_seconds as a proxy for tokens_used.
    True score = compression_ratio / (1 + tokens_used / 10000)
    Proxy score = compression_ratio / (1 + elapsed_seconds / 60)
"""

import argparse
import importlib.util
import json
import os
import random
import sys
import traceback


def load_solution(solution_path: str):
    """Import the agent's solution.py and return the module."""
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    assert spec is not None and spec.loader is not None, "Could not load spec"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def load_corpus(corpus_path: str):
    """Load poems from a JSON corpus file."""
    with open(corpus_path) as f:
        return json.load(f)


def run_eval(corpus_path: str, solution_path: str, n: int = 5, seed: int = 42) -> dict:
    result = {
        "compression_ratio": 0.0,
        "verified": False,
        "failures": 0,
        "poems_tested": 0,
        "details": [],
        "error": None,
    }

    # Load corpus
    try:
        poems = load_corpus(corpus_path)
    except Exception as e:
        result["error"] = f"Failed to load corpus: {e}"
        return result

    # Sample poems — never fragments, always full poems
    rng = random.Random(seed)
    sample = rng.sample(poems, min(n, len(poems)))
    result["poems_tested"] = len(sample)

    # Load agent solution
    try:
        mod = load_solution(solution_path)
    except Exception as e:
        result["error"] = f"Failed to load solution: {traceback.format_exc()}"
        return result

    # Verify encode and decode exist
    if not hasattr(mod, "encode") or not hasattr(mod, "decode"):
        result["error"] = "solution.py must define encode(text: str) -> str and decode(encoded: str) -> str"
        return result

    # Run round-trip on each sampled poem
    ratios = []
    details = []
    failures = 0

    for poem in sample:
        poem_id = poem.get("id", "?")
        title = poem.get("title", "?")
        author = poem.get("author", "?")
        original_text = poem["text"]
        original_chars = len(original_text.encode("utf-8"))

        detail = {
            "id": poem_id,
            "title": title,
            "author": author,
            "original_chars": original_chars,
            "encoded_chars": None,
            "ratio": None,
            "roundtrip_ok": False,
            "error": None,
        }

        try:
            encoded = mod.encode(original_text)
            decoded = mod.decode(encoded)
            encoded_chars = len(encoded.encode("utf-8")) if isinstance(encoded, str) else len(encoded)

            roundtrip_ok = (decoded == original_text)
            ratio = original_chars / encoded_chars if encoded_chars > 0 else 0.0

            detail["encoded_chars"] = encoded_chars
            detail["ratio"] = round(ratio, 4)
            detail["roundtrip_ok"] = roundtrip_ok

            if not roundtrip_ok:
                failures += 1
                detail["error"] = (
                    f"Round-trip failed: first diff at char "
                    f"{next((i for i,(a,b) in enumerate(zip(original_text,decoded)) if a!=b), len(min(original_text,decoded,key=len)))}"
                )
            else:
                ratios.append(ratio)

        except Exception as e:
            failures += 1
            detail["error"] = traceback.format_exc()

        details.append(detail)

    result["failures"] = failures
    result["details"] = details
    result["verified"] = (failures == 0)
    result["compression_ratio"] = round(sum(ratios) / len(ratios), 4) if ratios else 0.0

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate a poetry compression solution")
    parser.add_argument("--corpus", required=True, help="Path to corpus JSON file")
    parser.add_argument("--solution", required=True, help="Path to solution.py")
    parser.add_argument("--n", type=int, default=5, help="Number of poems to sample (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    # Resolve relative paths from this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = args.corpus if os.path.isabs(args.corpus) else os.path.join(script_dir, args.corpus)
    solution_path = args.solution if os.path.isabs(args.solution) else os.path.abspath(args.solution)

    result = run_eval(corpus_path, solution_path, n=args.n, seed=args.seed)
    print(json.dumps(result, indent=2))

    # Exit code: 0 if verified, 1 if failures
    sys.exit(0 if result["verified"] else 1)


if __name__ == "__main__":
    main()
