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
ARCH_PROBLEMS = {"300"}

# Archipelago head-to-head config. Bots may be deterministic, so the only
# variation between games of a fixed pair is who moves first. We play
# ARCH_GAMES_PER_SIDE games with each judge's bot moving first, seeding the
# RNG per game so stochastic bots vary but the result stays reproducible.
ARCH_GAMES_PER_SIDE = 3
ARCH_BASE_SEED = 42

# Poetry corpora ship WITH the repo (operators who clone it need them to score).
# The poems are kept secret from the AGENT UNDER TEST a different way: run.py runs
# each heat in an isolated working directory that doesn't contain the corpus, and
# never tells the agent the poems exist. So score.py just reads them in-repo.
# Override the location with HJB_CORPUS_DIR if you keep the poems elsewhere.
CORPUS_DIR = os.environ.get(
    "HJB_CORPUS_DIR", os.path.join(REPO_DIR, "problems", "corpus")
)
POETRY_CORPUS = {
    "100": "romantic_nature.json",
    "101": "victorian_lyric.json",
    "102": "ode_and_elegy.json",
    "103": "sonnet.json",
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


def run_poetry_eval(solution_py: str, problem: str, seed: int | None = None) -> dict:
    if not solution_py.strip():
        return {"verified": False, "compression_ratio": 0.0, "error": "no solution"}

    tmp = f"/tmp/bench_solution_{problem}.py"
    with open(tmp, "w") as f:
        f.write(solution_py)

    corpus_path = os.path.join(CORPUS_DIR, POETRY_CORPUS[problem])
    if not os.path.exists(corpus_path):
        return {"verified": False, "compression_ratio": 0.0,
                "error": (f"corpus not found at {corpus_path}. It should ship with the "
                          f"repo under problems/corpus/ — or set HJB_CORPUS_DIR. See README.")}

    # Fixed seed by default so scores are reproducible and comparable across
    # operators (the agent can't see the poems, so the sample needn't be secret).
    # Pass --poetry-seed to vary the held-out sample.
    if seed is None:
        seed = 42

    try:
        result = subprocess.run(
            ["python3", POETRY_EVAL,
             "--corpus", corpus_path,
             "--solution", tmp,
             "--n", "5",
             "--seed", str(seed)],
            capture_output=True, text=True, timeout=120
        )
        if result.stdout.strip():
            return json.loads(result.stdout)
        else:
            return {"verified": False, "compression_ratio": 0.0,
                    "error": result.stderr[:500]}
    except Exception as e:
        return {"verified": False, "compression_ratio": 0.0, "error": str(e)}


def _load_arch_bot(solution_path: str):
    """Load an Archipelago bot's choose_move from a solution file. None on failure."""
    if not solution_path or not os.path.exists(solution_path):
        return None
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("arch_bot", solution_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "choose_move", None)
    except Exception:
        return None


def run_archipelago_headtohead(judge_to_solution: dict) -> dict:
    """Play each pair of judges' bots head-to-head and return per-judge quality.

    judge_to_solution: {judge_name: solution_path}

    Returns {judge_name: {"quality": float, "got": str, "playable": bool}}.

    quality = (wins + 0.5*draws) / total_games over a symmetric set of games
    (each judge's bot moves first an equal number of times). With exactly two
    judges this is a single matchup; with more it's a round-robin and quality
    is averaged across opponents.
    """
    import random as _random
    sys.path.insert(0, os.path.join(REPO_DIR, "problems"))
    from archipelago import play_game, RED, BLUE  # noqa: E402

    bots = {j: _load_arch_bot(p) for j, p in judge_to_solution.items()}
    judges = sorted(judge_to_solution)

    out = {}
    for j in judges:
        if bots[j] is None:
            out[j] = {"quality": 0.0, "got": "no valid bot", "playable": False}
        else:
            out[j] = {"quality": None, "got": "", "playable": True,
                      "_pts": 0.0, "_games": 0, "_record": [0, 0, 0]}  # W,L,D

    playable = [j for j in judges if out[j]["playable"]]
    if len(playable) < 2:
        for j in playable:
            out[j].update(quality=0.0, got="no opponent in heat")
        return {j: {k: out[j][k] for k in ("quality", "got", "playable")} for j in judges}

    # Round-robin over playable judges.
    for ai in range(len(playable)):
        for bi in range(ai + 1, len(playable)):
            ja, jb = playable[ai], playable[bi]
            bot_a, bot_b = bots[ja], bots[jb]
            game_idx = 0
            # Each bot moves first ARCH_GAMES_PER_SIDE times.
            for a_is_red in (True, False):
                red_bot = bot_a if a_is_red else bot_b
                blue_bot = bot_b if a_is_red else bot_a
                for _ in range(ARCH_GAMES_PER_SIDE):
                    _random.seed(ARCH_BASE_SEED + game_idx)
                    game_idx += 1
                    res = play_game(red_bot, blue_bot, red_first=True)
                    w = res["winner"]
                    if w == "draw":
                        pa = pb = 0.5
                        out[ja]["_record"][2] += 1
                        out[jb]["_record"][2] += 1
                    else:
                        a_won = (w == RED and a_is_red) or (w == BLUE and not a_is_red)
                        pa, pb = (1.0, 0.0) if a_won else (0.0, 1.0)
                        out[ja]["_record"][0 if a_won else 1] += 1
                        out[jb]["_record"][1 if a_won else 0] += 1
                    out[ja]["_pts"] += pa; out[ja]["_games"] += 1
                    out[jb]["_pts"] += pb; out[jb]["_games"] += 1

    for j in playable:
        g = out[j]["_games"] or 1
        out[j]["quality"] = round(out[j]["_pts"] / g, 4)
        wlr = out[j]["_record"]
        out[j]["got"] = f"{wlr[0]}W-{wlr[1]}L-{wlr[2]}D"

    return {j: {k: out[j][k] for k in ("quality", "got", "playable")} for j in judges}


def score_results(results_dir: str, heat_filter: str | None, poetry_seed: int | None = None):
    answers = load_answers()
    rows = []

    if heat_filter:
        pattern = os.path.join(results_dir, heat_filter, "*", "*-result.json")
    else:
        pattern = os.path.join(results_dir, "heat_*", "*", "*-result.json")
    result_files = sorted(glob.glob(pattern))

    if not result_files:
        print("No results found. Run ./run.sh first.")
        return

    # --- Pre-compute Archipelago (300) head-to-head per heat ---------------
    # 300 is competitive: a judge's quality depends on the OTHER judge's bot,
    # so it can't be scored one file at a time like the rest. Group the 300
    # result files by heat, play each heat's judges head-to-head, and stash
    # the per-judge quality for the main loop to pick up.
    arch_by_heat: dict[str, dict[str, str]] = {}
    for rf in result_files:
        with open(rf) as f:
            d = json.load(f)
        if d.get("problem") in ARCH_PROBLEMS:
            h = d.get("heat", "legacy")
            sol = os.path.join(os.path.dirname(rf), f"{d.get('problem')}-solution.py")
            arch_by_heat.setdefault(h, {})[d.get("judge", "?")] = sol
    arch_scores: dict[tuple[str, str], dict] = {}
    for h, j2s in arch_by_heat.items():
        for j, res in run_archipelago_headtohead(j2s).items():
            arch_scores[(h, j)] = res

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
            if eff_tok > 0:
                pareto = correctness / (1 + eff_tok / 10000)
                token_note = f"{raw_tok}+{cache_r}cr"
            else:
                pareto = correctness / (1 + elapsed / 60)
                token_note = f"{elapsed}s~"
            expected = "mech+qual/6"
            score_type = "zkp"

        elif problem in POETRY_PROBLEMS:
            eval_result = run_poetry_eval(solution, problem, seed=poetry_seed)
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

        elif problem in ARCH_PROBLEMS:
            arch = arch_scores.get((heat, judge), {"quality": 0.0, "got": "not scored", "playable": False})
            correctness = arch.get("quality") or 0.0
            if eff_tok > 0:
                pareto = correctness / (1 + eff_tok / 10000)
                token_note = f"{raw_tok}+{cache_r}cr"
            else:
                pareto = correctness / (1 + elapsed / 60)
                token_note = f"{elapsed}s~"
            expected = "win H2H"
            got = arch.get("got", "—")
            score_type = "archipelago"

        elif problem == "000":
            output = extract_answer(solution, problem)
            correctness = 1.0 if output else 0.0
            if eff_tok > 0:
                pareto = correctness / (1 + eff_tok / 10000)
                token_note = f"{raw_tok}+{cache_r}cr"
            else:
                pareto = correctness / (1 + elapsed / 60)
                token_note = f"{elapsed}s~"
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
            if eff_tok > 0:
                pareto = correctness / (1 + eff_tok / 10000)
                token_note = f"{raw_tok}+{cache_r}cr"
            else:
                pareto = correctness / (1 + elapsed / 60)
                token_note = f"{elapsed}s~"
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
    arch_rows   = [r for r in rows if r["score_type"] == "archipelago"]

    title = f"Heat: {heat_filter}" if heat_filter else f"All heats ({len(heats)})"
    print(f"\n## hermes-judge-bench Leaderboard — {title}\n")

    if std_rows:
        print("### Standard Problems  (correctness / (1 + effective_tokens/10000))\n")
        print("  Tokens column: raw_input+output + cache_reads (cr). '~' = elapsed proxy (legacy).\n")
        print(f"{'Heat':<22} {'Problem':<10} {'Judge':<20} {'Model':<25} {'Tokens':>14} {'Score':>8} {'Pareto':>8} {'Got':>14} {'':>8}")
        print("-" * 140)
        for r in sorted(std_rows, key=lambda x: -x["pareto_score"]):
            c = "✓" if r["correctness"] == 1.0 else "✗"
            t = "⏱" if r.get("killed") else ""
            print(f"{r['heat']:<22} {r['problem']:<10} {r['judge']:<20} {r['model']:<25} "
                  f"{str(r['token_note']):>14} {c:>8} {r['pareto_score']:>8} {str(r['got']):>14} {t:>8}")

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

    if arch_rows:
        print("\n### Archipelago Head-to-Head  (win_rate / (1 + effective_tokens/10000))\n")
        print("  win_rate = (wins + 0.5·draws) / games, bots played head-to-head per heat.")
        print(f"  {ARCH_GAMES_PER_SIDE} games per first-mover side. Got column: W-L-D record.\n")
        print(f"{'Heat':<22} {'Problem':<10} {'Judge':<20} {'Model':<25} {'Tokens':>14} {'WinRate':>8} {'Pareto':>8} {'Got':>14} {'':>8}")
        print("-" * 140)
        for r in sorted(arch_rows, key=lambda x: -x["pareto_score"]):
            t = "⏱" if r.get("killed") else ""
            print(f"{r['heat']:<22} {r['problem']:<10} {r['judge']:<20} {r['model']:<25} "
                  f"{str(r['token_note']):>14} {str(r['correctness']):>8} {r['pareto_score']:>8} "
                  f"{str(r['got']):>14} {t:>8}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=os.path.join(REPO_DIR, "results"))
    parser.add_argument("--heat", default=None, help="Score only this heat (e.g. heat_1716900000)")
    parser.add_argument("--poetry-seed", type=int, default=None,
                        help="Fix the held-out poem sample for reproducible poetry scores "
                             "(default: a fresh random sample each run).")
    args = parser.parse_args()
    score_results(args.results, args.heat, poetry_seed=args.poetry_seed)
