#!/usr/bin/env python3
"""
archipelago_tournament.py — Tournament runner for Problem 300 (Archipelago).

Runs a 20-round tournament between two judge configurations (e.g. bare vs judge-face).
Each round:
  1. Each agent (independently, sequentially) may spend tokens to rewrite its bot.
     Token budget is tracked — total across all rounds is capped at TOKEN_BUDGET.
  2. Agent that spent fewer tokens this round plays Red (goes first).
     Tiebreak: agent that went second last round goes first.
  3. Best-of-3 games played between the two bots.
  4. Full results (move histories, scores, round winner) written to disk.
  5. Each agent receives a results summary for the next round's prompt.

Usage:
    python3 archipelago_tournament.py \\
        --judge-a bare \\
        --judge-b judge-face \\
        --rounds 20 \\
        --token-budget 20000 \\
        --heat heat_XYZ \\
        --results-dir ~/hermes-judge-bench/results

Output:
    results/<heat>/<judge>/300-result.json    — final result (for score.py)
    results/<heat>/archipelago_tournament.json — full tournament log
"""

import argparse
import importlib.util
import json
import os
import random
import subprocess
import sys
import time

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)

from archipelago import play_best_of_3, board_to_str, MOVES_PER_PLAYER, random_bot

PROBLEM_FILE = os.path.join(os.path.dirname(REPO_DIR), "problems", "300.md")
TOKEN_BUDGET_DEFAULT = 40000
ROUNDS_DEFAULT = 20
MOVE_TIMEOUT = 5


def load_bot(path: str):
    """Load a bot from solution.py — returns the choose_move callable or None."""
    if not path or not os.path.exists(path):
        return None
    try:
        spec = importlib.util.spec_from_file_location("bot", path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        if hasattr(mod, "choose_move"):
            return mod.choose_move
        return None
    except Exception:
        return None


def random_bot(board, colour, move_number):
    """Fallback bot — plays randomly."""
    moves = [(r, c) for r in range(10) for c in range(10) if board[r][c] == ""]
    return random.choice(moves) if moves else (0, 0)


def read_judge_config(judges_file: str, judge_name: str) -> dict:
    """Parse judges.yaml for model/provider."""
    model = "claude-sonnet-4-6"
    provider = "anthropic"
    in_judge = False
    with open(judges_file) as f:
        for line in f:
            if line.strip().startswith(f"{judge_name}:"):
                in_judge = True
                continue
            if in_judge:
                if line.startswith("  ") and not line.startswith("   "):
                    # new top-level judge
                    if not line.strip().startswith("model:") and not line.strip().startswith("provider:") and not line.strip().startswith("face:"):
                        break
                if "model:" in line:
                    model = line.split("model:")[1].strip()
                if "provider:" in line:
                    provider = line.split("provider:")[1].strip()
    return {"model": model, "provider": provider}


def run_hermes(prompt: str, model: str, provider: str, solution_path: str, token_limit: int = 0) -> dict:
    """
    Run hermes -z with the given prompt. Returns token counts.
    solution_path: where to copy /tmp/solution.py after the run.
    token_limit: if > 0, passed as wall-budget approximation via prompt only (hermes has no hard token cutoff flag).
    """
    env = os.environ.copy()

    # Pass remaining budget as a hard wall-clock timeout proxy:
    # approximate 1000 tokens ≈ 10 seconds of inference; floor at 30s
    wall_timeout = max(30, (token_limit // 100)) if token_limit > 0 else 300

    cmd = [
        "timeout", str(wall_timeout),
        "hermes", "-z", prompt,
        "-m", model,
        "--provider", provider,
        "-t", "terminal,file",
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = int(time.time() - t0)

    # Copy solution if written
    if os.path.exists("/tmp/solution.py"):
        import shutil
        shutil.copy("/tmp/solution.py", solution_path)
        os.remove("/tmp/solution.py")

    # Read token counts from most recent session
    input_tokens = output_tokens = cache_read_tokens = 0
    try:
        sess_result = subprocess.run(
            ["hermes", "sessions", "list", "--limit", "1"],
            capture_output=True, text=True
        )
        session_id = sess_result.stdout.strip().split("\n")[-1].split()[-1] if sess_result.stdout.strip() else ""
        if session_id:
            export_result = subprocess.run(
                ["hermes", "sessions", "export", "--session-id", session_id, "-"],
                capture_output=True, text=True
            )
            lines = [l for l in export_result.stdout.strip().split("\n") if l.strip()]
            if lines:
                data = json.loads(lines[0])
                input_tokens = data.get("input_tokens", 0)
                output_tokens = data.get("output_tokens", 0)
                cache_read_tokens = data.get("cache_read_tokens", 0)
    except Exception:
        pass

    raw_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "raw_tokens": raw_tokens,
        "elapsed_seconds": elapsed,
        "response": result.stdout,
    }


def format_round_results(round_results: list[dict], my_colour_history: list[str]) -> str:
    """Format previous rounds for inclusion in the next prompt."""
    lines = ["## Previous round results\n"]
    for i, rr in enumerate(round_results):
        rw = rr["round_winner"]
        wr = rr["wins_r"]
        wb = rr["wins_b"]
        my_c = my_colour_history[i]
        opp_c = "B" if my_c == "R" else "R"
        my_wins = wr if my_c == "R" else wb
        opp_wins = wb if my_c == "R" else wr

        lines.append(f"### Round {i+1}: {'YOU WON' if rw == my_c else 'YOU LOST' if rw == opp_c else 'DRAW'} "
                     f"(you={my_c}, you won {my_wins}/3 games)")

        for j, g in enumerate(rr["games"]):
            my_score = g["score_r"] if my_c == "R" else g["score_b"]
            opp_score = g["score_b"] if my_c == "R" else g["score_r"]
            gw = g["winner"]
            outcome = "WIN" if gw == my_c else ("LOSS" if gw == opp_c else "DRAW")
            lines.append(f"  Game {j+1}: {outcome} — you {my_score} pts, opponent {opp_score} pts")
            # Show last 10 moves
            moves = g["moves"][-10:]
            move_strs = [f"{'you' if m[0]==my_c else 'opp'}({m[1]},{m[2]})" for m in moves]
            lines.append(f"  Last moves: {', '.join(move_strs)}")
            lines.append(f"  Final board:\n```\n{board_to_str(g['final_board'])}\n```")
        lines.append("")
    return "\n".join(lines)


def build_prompt(
    problem_text: str,
    round_num: int,
    budget_remaining: int,
    total_budget: int,
    rounds_remaining: int,
    previous_results: str,
    my_colour_hint: str,
    went_first_last: bool | None,
) -> str:
    budget_note = (
        f"**Token budget:** You have {budget_remaining:,} tokens remaining out of {total_budget:,} total "
        f"across all {20} rounds. This is round {round_num}/20 — you have {rounds_remaining} rounds left after this. "
        f"Spending 0 tokens this round preserves your budget but keeps your current bot unchanged. "
        f"The agent that spends fewer tokens this round goes first (Red)."
    )

    base = f"""You are playing a strategy game called Archipelago against another agent.

{problem_text}

---

{budget_note}

{previous_results if previous_results else "This is round 1 — no previous results yet."}

---

Your task: write (or rewrite) your bot to `/tmp/solution.py`.
The file must define `choose_move(board, my_colour, move_number) -> (row, col)`.

If you are satisfied with your current bot and want to spend 0 tokens on revision,
simply say DONE without writing any file.

Rules:
- No web search.
- No external libraries beyond Python stdlib.
- Your bot must return a move in under 5 seconds.
- Write a complete, self-contained `/tmp/solution.py` every time you revise.
- Say DONE when you are finished (or immediately if spending 0 tokens).
"""
    return base


def run_tournament(
    judge_a: str,
    judge_b: str,
    rounds: int,
    token_budget: int,
    heat_id: str,
    results_dir: str,
    judges_file: str,
    problem_file: str,
):
    os.makedirs(results_dir, exist_ok=True)
    heat_dir = os.path.join(results_dir, heat_id)
    dir_a = os.path.join(heat_dir, judge_a)
    dir_b = os.path.join(heat_dir, judge_b)
    os.makedirs(dir_a, exist_ok=True)
    os.makedirs(dir_b, exist_ok=True)

    config_a = read_judge_config(judges_file, judge_a)
    config_b = read_judge_config(judges_file, judge_b)

    with open(problem_file) as f:
        problem_text = f.read()

    budget_a = token_budget
    budget_b = token_budget

    bot_a_path = os.path.join(dir_a, "300-bot-current.py")
    bot_b_path = os.path.join(dir_b, "300-bot-current.py")

    bot_a = random_bot
    bot_b = random_bot

    round_results = []
    colour_history_a = []  # what colour A played each round
    colour_history_b = []

    # Track who went first last round (None = first round)
    a_went_first_last: bool | None = None

    tournament_log = {
        "heat": heat_id,
        "judge_a": judge_a,
        "judge_b": judge_b,
        "token_budget": token_budget,
        "rounds": [],
    }

    for rnd in range(1, rounds + 1):
        print(f"\n{'='*60}")
        print(f"ROUND {rnd}/{rounds}  |  budget_a={budget_a:,}  budget_b={budget_b:,}")
        print(f"{'='*60}")

        prev_str_a = format_round_results(round_results, colour_history_a) if round_results else ""
        prev_str_b = format_round_results(round_results, colour_history_b) if round_results else ""

        # --- Agent A writes its bot ---
        prompt_a = build_prompt(
            problem_text, rnd, budget_a, token_budget,
            rounds - rnd, prev_str_a, "?", a_went_first_last
        )
        sol_a_path = os.path.join(dir_a, f"300-round{rnd:02d}-solution.py")
        print(f"  Running agent A ({judge_a})...")
        if budget_a > 0:
            tok_a = run_hermes(prompt_a, config_a["model"], config_a["provider"], sol_a_path, token_limit=budget_a)
            spent_a = tok_a["raw_tokens"]
            budget_a = max(0, budget_a - spent_a)
            if os.path.exists(sol_a_path):
                new_bot = load_bot(sol_a_path)
                if new_bot:
                    bot_a = new_bot
                    print(f"    Agent A revised bot ({spent_a:,} tokens spent, {budget_a:,} remaining)")
                else:
                    print(f"    Agent A wrote file but no valid choose_move — keeping previous bot")
            else:
                print(f"    Agent A spent {spent_a:,} tokens but wrote no bot — keeping previous")
        else:
            spent_a = 0
            tok_a = {"raw_tokens": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "elapsed_seconds": 0}
            print(f"    Agent A budget exhausted — keeping previous bot")

        # --- Agent B writes its bot ---
        prompt_b = build_prompt(
            problem_text, rnd, budget_b, token_budget,
            rounds - rnd, prev_str_b, "?", not a_went_first_last if a_went_first_last is not None else None
        )
        sol_b_path = os.path.join(dir_b, f"300-round{rnd:02d}-solution.py")
        print(f"  Running agent B ({judge_b})...")
        if budget_b > 0:
            tok_b = run_hermes(prompt_b, config_b["model"], config_b["provider"], sol_b_path, token_limit=budget_b)
            spent_b = tok_b["raw_tokens"]
            budget_b = max(0, budget_b - spent_b)
            if os.path.exists(sol_b_path):
                new_bot = load_bot(sol_b_path)
                if new_bot:
                    bot_b = new_bot
                    print(f"    Agent B revised bot ({spent_b:,} tokens spent, {budget_b:,} remaining)")
                else:
                    print(f"    Agent B wrote file but no valid choose_move — keeping previous bot")
            else:
                print(f"    Agent B spent {spent_b:,} tokens but wrote no bot — keeping previous")
        else:
            spent_b = 0
            tok_b = {"raw_tokens": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "elapsed_seconds": 0}
            print(f"    Agent B budget exhausted — keeping previous bot")

        # --- Determine who goes first (Red) ---
        if spent_a < spent_b:
            a_is_red = True
            print(f"  Handicap: A goes first (spent {spent_a:,} < {spent_b:,})")
        elif spent_b < spent_a:
            a_is_red = False
            print(f"  Handicap: B goes first (spent {spent_b:,} < {spent_a:,})")
        else:
            # Tiebreak: agent that went second last round goes first
            if a_went_first_last is None:
                a_is_red = True  # A goes first in round 1 by default
            else:
                a_is_red = not a_went_first_last
            print(f"  Handicap: tied ({spent_a:,} each) — tiebreak: {'A' if a_is_red else 'B'} goes first")

        a_went_first_last = a_is_red

        red_bot = bot_a if a_is_red else bot_b
        blue_bot = bot_b if a_is_red else bot_a
        colour_a = "R" if a_is_red else "B"
        colour_b = "B" if a_is_red else "R"
        colour_history_a.append(colour_a)
        colour_history_b.append(colour_b)

        # --- Play best-of-3 ---
        print(f"  Playing best-of-3 (A={colour_a}, B={colour_b})...")
        bo3 = play_best_of_3(red_bot, blue_bot, red_first=True)

        # Translate R/B winner back to A/B
        if bo3["round_winner"] == colour_a:
            round_winner_agent = "A"
        elif bo3["round_winner"] == colour_b:
            round_winner_agent = "B"
        else:
            round_winner_agent = "draw"

        print(f"  Round {rnd} winner: {round_winner_agent} "
              f"(A wins {bo3['wins_r'] if a_is_red else bo3['wins_b']}/3 games)")

        round_log = {
            "round": rnd,
            "spent_a": spent_a,
            "spent_b": spent_b,
            "budget_remaining_a": budget_a,
            "budget_remaining_b": budget_b,
            "colour_a": colour_a,
            "colour_b": colour_b,
            "round_winner": round_winner_agent,
            "wins_a": bo3["wins_r"] if a_is_red else bo3["wins_b"],
            "wins_b": bo3["wins_b"] if a_is_red else bo3["wins_r"],
            "games": bo3["games"],
            "tokens_a": tok_a,
            "tokens_b": tok_b,
        }
        round_results.append(bo3)
        tournament_log["rounds"].append(round_log)

    # --- Compute final scores ---
    rounds_won_a = sum(1 for r in tournament_log["rounds"] if r["round_winner"] == "A")
    rounds_won_b = sum(1 for r in tournament_log["rounds"] if r["round_winner"] == "B")
    total_tokens_a = token_budget - budget_a
    total_tokens_b = token_budget - budget_b

    quality_a = rounds_won_a / rounds
    quality_b = rounds_won_b / rounds
    pareto_a = quality_a / (1 + total_tokens_a / 10000)
    pareto_b = quality_b / (1 + total_tokens_b / 10000)

    tournament_log["final"] = {
        "rounds_won_a": rounds_won_a,
        "rounds_won_b": rounds_won_b,
        "total_tokens_a": total_tokens_a,
        "total_tokens_b": total_tokens_b,
        "quality_a": quality_a,
        "quality_b": quality_b,
        "pareto_a": pareto_a,
        "pareto_b": pareto_b,
        "tournament_winner": "A" if rounds_won_a > rounds_won_b else ("B" if rounds_won_b > rounds_won_a else "draw"),
    }

    # Write full tournament log
    log_path = os.path.join(heat_dir, "archipelago_tournament.json")
    with open(log_path, "w") as f:
        json.dump(tournament_log, f, indent=2)
    print(f"\nTournament log: {log_path}")

    # Write per-agent result JSONs (for score.py compatibility)
    for judge, quality, total_tok, pareto, rounds_won in [
        (judge_a, quality_a, total_tokens_a, pareto_a, rounds_won_a),
        (judge_b, quality_b, total_tokens_b, pareto_b, rounds_won_b),
    ]:
        result = {
            "heat": heat_id,
            "problem": "300",
            "judge": judge,
            "model": read_judge_config(judges_file, judge)["model"],
            "rounds_won": rounds_won,
            "total_rounds": rounds,
            "quality": quality,
            "total_tokens": total_tok,
            "effective_tokens": total_tok,
            "input_tokens": total_tok,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "pareto_score": pareto,
            "tournament_log": log_path,
        }
        result_path = os.path.join(heat_dir, judge, "300-result.json")
        os.makedirs(os.path.join(heat_dir, judge), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result ({judge}): {result_path}")

    print(f"\n{'='*60}")
    print(f"TOURNAMENT COMPLETE")
    print(f"  A ({judge_a}): {rounds_won_a}/{rounds} rounds, {total_tokens_a:,} tokens, pareto={pareto_a:.4f}")
    print(f"  B ({judge_b}): {rounds_won_b}/{rounds} rounds, {total_tokens_b:,} tokens, pareto={pareto_b:.4f}")
    print(f"  Winner: {tournament_log['final']['tournament_winner']}")

    return tournament_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judges", required=True,
                        help="Exactly two judge names, comma-separated (e.g. bare,judge-face)")
    parser.add_argument("--rounds", type=int, default=ROUNDS_DEFAULT)
    parser.add_argument("--token-budget", type=int, default=TOKEN_BUDGET_DEFAULT)
    parser.add_argument("--results-dir", default=os.path.join(os.path.dirname(REPO_DIR), "results"))
    parser.add_argument("--judges-file", default=os.path.join(os.path.dirname(REPO_DIR), "judges.yaml"))
    parser.add_argument("--problem-file", default=PROBLEM_FILE)
    args = parser.parse_args()

    judges = [j.strip() for j in args.judges.split(",")]
    if len(judges) != 2:
        print(f"Error: --judges requires exactly 2 comma-separated judges, got {len(judges)}: {judges}")
        sys.exit(1)

    heat_id = f"heat_{int(time.time())}"

    run_tournament(
        judge_a=judges[0],
        judge_b=judges[1],
        rounds=args.rounds,
        token_budget=args.token_budget,
        heat_id=heat_id,
        results_dir=args.results_dir,
        judges_file=args.judges_file,
        problem_file=args.problem_file,
    )


if __name__ == "__main__":
    main()
