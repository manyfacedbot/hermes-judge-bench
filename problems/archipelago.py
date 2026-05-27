"""
archipelago.py — Game engine for Archipelago (Problem 300).

Board: 10×10 grid. Players: 'R' (Red) and 'B' (Blue).
Each player places 30 stones. 60 total moves.

Scoring:
  - 1 point per stone on the board (island size)
  - 2 bonus points per empty cell completely enclosed by a single player's stones
    (flood-fill from outside: empty cells NOT reachable from the border without
     crossing that player's stones)

Engine is deliberately simple — no numpy, no external deps.
"""

import copy
import random
from typing import Callable

ROWS = 10
COLS = 10
MOVES_PER_PLAYER = 30
EMPTY = ""
RED = "R"
BLUE = "B"


def empty_board() -> list[list[str]]:
    return [[EMPTY] * COLS for _ in range(ROWS)]


def valid_moves(board: list[list[str]]) -> list[tuple[int, int]]:
    return [(r, c) for r in range(ROWS) for c in range(COLS) if board[r][c] == EMPTY]


def apply_move(board: list[list[str]], row: int, col: int, colour: str) -> list[list[str]]:
    b = copy.deepcopy(board)
    b[row][col] = colour
    return b


def _flood_fill_from_outside(board: list[list[str]], blocked_colour: str) -> set[tuple[int, int]]:
    """
    Returns all empty cells reachable from outside the board
    without crossing a cell belonging to blocked_colour.
    These are NOT enclosed by blocked_colour.
    """
    visited: set[tuple[int, int]] = set()
    queue: list[tuple[int, int]] = []

    # Seed from all border-adjacent empty cells
    for r in range(ROWS):
        for c in range(COLS):
            if (r == 0 or r == ROWS - 1 or c == 0 or c == COLS - 1):
                if board[r][c] != blocked_colour and (r, c) not in visited:
                    visited.add((r, c))
                    queue.append((r, c))

    while queue:
        r, c = queue.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                if (nr, nc) not in visited and board[nr][nc] != blocked_colour:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    return visited


def score(board: list[list[str]]) -> dict[str, int]:
    """
    Returns {"R": int, "B": int} — total scores for each player.

    Bonus: an empty cell scores +2 for player C only when reaching it from
    outside the board REQUIRES crossing C's stones AND a path exists that
    crosses no stones of the other colour. Cells walled off by both colours
    award no bonus to either (no player alone is doing the enclosing).
    """
    scores = {RED: 0, BLUE: 0}

    # Stone counts
    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] in (RED, BLUE):
                scores[board[r][c]] += 1

    reachable_blocking_red  = _flood_fill_from_outside(board, RED)
    reachable_blocking_blue = _flood_fill_from_outside(board, BLUE)

    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] != EMPTY:
                continue
            walled_by_red  = (r, c) not in reachable_blocking_red
            walled_by_blue = (r, c) not in reachable_blocking_blue
            if walled_by_red and not walled_by_blue:
                scores[RED] += 2
            elif walled_by_blue and not walled_by_red:
                scores[BLUE] += 2
            # else: open, or walled by both → no bonus

    return scores


def play_game(
    red_bot: Callable,
    blue_bot: Callable,
    red_first: bool = True,
) -> dict:
    """
    Play one game. Bots are callables:
        move = bot(board, my_colour, move_number)  # returns (row, col)

    Returns:
    {
        "winner":       "R" | "B" | "draw",
        "score_r":      int,
        "score_b":      int,
        "moves":        [(colour, row, col), ...],
        "final_board":  list[list[str]],
        "errors":       [str],
    }
    """
    board = empty_board()
    moves = []
    errors = []
    move_counts = {RED: 0, BLUE: 0}

    order = [RED, BLUE] if red_first else [BLUE, RED]

    for turn in range(MOVES_PER_PLAYER * 2):
        colour = order[turn % 2]
        move_counts[colour] += 1
        mv = valid_moves(board)
        if not mv:
            break

        try:
            import signal

            def _timeout_handler(signum, frame):
                raise TimeoutError()

            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(5)  # 5s per move
            try:
                bot = red_bot if colour == RED else blue_bot
                row, col = bot(board, colour, move_counts[colour])
            finally:
                signal.alarm(0)

            if (row, col) not in mv:
                errors.append(f"{colour} move {move_counts[colour]}: ({row},{col}) invalid — random fallback")
                row, col = random.choice(mv)
        except TimeoutError:
            errors.append(f"{colour} move {move_counts[colour]}: timeout — random fallback")
            row, col = random.choice(mv)
        except Exception as e:
            errors.append(f"{colour} move {move_counts[colour]}: exception {e} — random fallback")
            row, col = random.choice(mv)

        board = apply_move(board, row, col, colour)
        moves.append((colour, row, col))

    final_scores = score(board)
    sr, sb = final_scores[RED], final_scores[BLUE]

    if sr > sb:
        winner = RED
    elif sb > sr:
        winner = BLUE
    else:
        winner = "draw"

    return {
        "winner": winner,
        "score_r": sr,
        "score_b": sb,
        "moves": moves,
        "final_board": board,
        "errors": errors,
    }


def board_to_str(board: list[list[str]]) -> str:
    lines = []
    for r in range(ROWS):
        row = ""
        for c in range(COLS):
            cell = board[r][c]
            row += (cell if cell else ".")
        lines.append(row)
    return "\n".join(lines)


def random_bot(board: list[list[str]], colour: str, move_number: int) -> tuple[int, int]:
    """Fallback bot — plays randomly."""
    moves = valid_moves(board)
    return random.choice(moves) if moves else (0, 0)


def play_best_of_3(red_bot, blue_bot, red_first: bool = True, alternate_first_move: bool = True) -> dict:
    """
    Play best-of-3.

    red_first              — whether Red moves first in game 1.
    alternate_first_move   — when True (default), first-move alternates each
                             game for standalone fairness. When False, the
                             same player moves first in every game (used by
                             the tournament when the lower-spender handicap
                             must apply to all 3 games).

    Returns:
    {
        "round_winner": "R" | "B" | "draw",
        "games": [game_result, game_result, game_result?],
        "wins_r": int,
        "wins_b": int,
    }
    """
    wins = {RED: 0, BLUE: 0}
    games = []
    rf = red_first

    for g in range(3):
        result = play_game(red_bot, blue_bot, red_first=rf)
        games.append(result)
        if result["winner"] in (RED, BLUE):
            wins[result["winner"]] += 1
        if alternate_first_move:
            rf = not rf
        if wins[RED] == 2 or wins[BLUE] == 2:
            break  # early exit — best of 3 decided

    if wins[RED] > wins[BLUE]:
        round_winner = RED
    elif wins[BLUE] > wins[RED]:
        round_winner = BLUE
    else:
        round_winner = "draw"

    return {
        "round_winner": round_winner,
        "games": games,
        "wins_r": wins[RED],
        "wins_b": wins[BLUE],
    }
