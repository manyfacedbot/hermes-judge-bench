#!/usr/bin/env python3
"""
pareto_plot.py — Generate interactive Pareto-Bench visualisation.

Reads all heat result JSONs from results/, runs the poetry eval harness
to get quality scores, then plots:

  X axis: effective_tokens (log scale)
  Y axis: quality score (compression_ratio or correctness)
  - One point per (heat, problem, judge) run
  - Colour: judge configuration
  - Shape: problem category
  - Pareto frontier: step-line across the Pareto-optimal set
  - Per-judge frontier: dashed lines showing each judge's frontier

Output: /home/hermes/pareto-bench-www/pareto.html (live at http://135.181.89.192:8437/pareto.html)

Usage:
    python3 pareto_plot.py                    # all heats
    python3 pareto_plot.py --heat heat_123    # specific heat
    python3 pareto_plot.py --out /tmp/out.html
"""

import argparse
import glob
import json
import math
import os
import subprocess
import sys

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
WWW_DIR  = "/home/hermes/pareto-bench-www"
POETRY_EVAL = os.path.join(REPO_DIR, "problems", "poetry_eval.py")

POETRY_PROBLEMS = {"100", "101", "102", "103"}
POETRY_CORPUS = {
    "100": "corpus/romantic_nature.json",
    "101": "corpus/victorian_lyric.json",
    "102": "corpus/ode_and_elegy.json",
    "103": "corpus/sonnet.json",
}
ANSWERS_FILE = os.path.join(REPO_DIR, "answers.json")

PROBLEM_LABELS = {
    "000": "smoke test",
    "001": "Euler #1",
    "002": "Euler #2",
    "003": "Euler #3",
    "004": "Euler #4",
    "005": "Euler #5",
    "054": "Euler #54 (poker)",
    "100": "poetry: romantic nature",
    "101": "poetry: victorian lyric",
    "102": "poetry: ode & elegy",
    "103": "poetry: sonnet",
}

JUDGE_COLORS = {
    "bare":       "#7eb8f7",
    "judge-face": "#7effa0",
    "poet-face":  "#ffd97d",
    "bad-face":   "#ff7eb8",
}

CATEGORY_SHAPES = {
    "euler":   "circle",
    "poetry":  "diamond",
    "smoke":   "square",
    "unknown": "cross",
}

def problem_category(pid):
    if pid == "000":              return "smoke"
    if pid.startswith("0"):       return "euler"
    if pid in POETRY_PROBLEMS:    return "poetry"
    return "unknown"


def load_answers():
    with open(ANSWERS_FILE) as f:
        return json.load(f)


def eval_solution(solution: str, problem: str) -> float:
    """Return quality score in [0, ∞). 0 = wrong/failed."""
    if not solution.strip():
        return 0.0

    if problem in POETRY_PROBLEMS:
        tmp = f"/tmp/plot_solution_{problem}.py"
        with open(tmp, "w") as f:
            f.write(solution)
        corpus_path = os.path.join(REPO_DIR, "problems", POETRY_CORPUS[problem])
        try:
            r = subprocess.run(
                ["python3", POETRY_EVAL, "--corpus", corpus_path,
                 "--solution", tmp, "--n", "5", "--seed", "42"],
                capture_output=True, text=True, timeout=120
            )
            if r.stdout.strip():
                res = json.loads(r.stdout)
                return res.get("compression_ratio", 0.0) if res.get("verified") else 0.0
        except Exception:
            pass
        return 0.0

    else:
        answers = load_answers()
        expected = answers.get(problem)
        if expected is None:
            # smoke test — 1.0 if any output
            tmp = f"/tmp/plot_solution_{problem}.py"
            with open(tmp, "w") as f:
                f.write(solution)
            try:
                r = subprocess.run(["python3", tmp], capture_output=True,
                                   text=True, timeout=30)
                return 1.0 if r.stdout.strip() else 0.0
            except Exception:
                return 0.0
        else:
            tmp = f"/tmp/plot_solution_{problem}.py"
            with open(tmp, "w") as f:
                f.write(solution)
            try:
                r = subprocess.run(["python3", tmp], capture_output=True,
                                   text=True, timeout=30)
                return 1.0 if r.stdout.strip() == str(expected).strip() else 0.0
            except Exception:
                return 0.0


def load_runs(results_dir: str, heat_filter: str | None) -> list[dict]:
    pattern = os.path.join(results_dir, "heat_*", "*", "*-result.json")
    files = sorted(glob.glob(pattern))
    if heat_filter:
        files = [f for f in files if heat_filter in f]

    runs = []
    for path in files:
        with open(path) as f:
            data = json.load(f)

        # Load solution from adjacent file if not embedded
        solution = data.get("solution", "")
        if not solution:
            sol_path = os.path.join(os.path.dirname(path),
                                    f"{data['problem']}-solution.py")
            if os.path.exists(sol_path):
                with open(sol_path) as sf:
                    solution = sf.read()

        eff_tok = data.get("effective_tokens", 0)
        raw_tok = data.get("input_tokens", 0) + data.get("output_tokens", 0)
        cache_r = data.get("cache_read_tokens", 0)

        quality = eval_solution(solution, data["problem"])

        runs.append({
            "heat":      data.get("heat", "?"),
            "problem":   data["problem"],
            "judge":     data.get("judge", "?"),
            "model":     data.get("model", "?"),
            "elapsed_s": data.get("elapsed_seconds", 0),
            "eff_tok":   eff_tok,
            "raw_tok":   raw_tok,
            "cache_r":   cache_r,
            "quality":   quality,
            "killed":    data.get("killed_by_timeout", False),
            "category":  problem_category(data["problem"]),
            "label":     PROBLEM_LABELS.get(data["problem"], data["problem"]),
            "solution_snippet": solution[:300].replace("`", "'") if solution else "(none)",
        })

    return runs


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Given list of (tokens, quality) points, return the Pareto-optimal frontier
    as a sorted list of (tokens, quality) points where no point is dominated.
    A point A dominates B if A has fewer tokens AND equal/higher quality,
    or equal tokens AND higher quality.
    We want the lower-left envelope: minimum tokens for each quality level.
    """
    if not points:
        return []
    # Sort by tokens ascending
    sorted_pts = sorted(points, key=lambda p: p[0])
    frontier = []
    max_quality = -1
    for tok, qual in sorted_pts:
        if qual > max_quality:
            frontier.append((tok, qual))
            max_quality = qual
    return frontier


def make_step_line(frontier: list[tuple[float, float]]) -> tuple[list, list]:
    """Convert frontier points to step-line x/y lists for plotting."""
    if not frontier:
        return [], []
    xs, ys = [], []
    for i, (tok, qual) in enumerate(frontier):
        if i == 0:
            xs.append(tok)
            ys.append(qual)
        else:
            # step: horizontal then vertical
            xs.append(tok)
            ys.append(frontier[i-1][1])
            xs.append(tok)
            ys.append(qual)
    # Extend rightward
    xs.append(xs[-1] * 10)
    ys.append(ys[-1])
    return xs, ys


def build_html(runs: list[dict], heat_filter: str | None) -> str:
    """Build self-contained Plotly HTML."""

    if not runs:
        return "<html><body><h2>No data yet — run the benchmark first.</h2></body></html>"

    # Collect all judges for colour mapping
    judges = sorted(set(r["judge"] for r in runs))

    # Build traces — one per (judge, category)
    traces = []
    judge_points: dict[str, list] = {j: [] for j in judges}

    for judge in judges:
        for cat in ["euler", "poetry", "smoke", "unknown"]:
            pts = [r for r in runs if r["judge"] == judge and r["category"] == cat]
            if not pts:
                continue

            color = JUDGE_COLORS.get(judge, "#cccccc")
            symbol = CATEGORY_SHAPES.get(cat, "cross")

            xs = [r["eff_tok"] for r in pts]
            ys = [r["quality"] for r in pts]
            texts = [
                f"<b>{r['label']}</b><br>"
                f"judge: {r['judge']}<br>"
                f"model: {r['model']}<br>"
                f"heat: {r['heat']}<br>"
                f"quality: {r['quality']:.4f}<br>"
                f"effective tokens: {r['eff_tok']:,}<br>"
                f"raw tokens: {r['raw_tok']:,}<br>"
                f"cache reads: {r['cache_r']:,}<br>"
                f"elapsed: {r['elapsed_s']}s<br>"
                f"{'⏱ TIMED OUT' if r['killed'] else ''}"
                f"<br><br><i>solution preview:</i><br><pre>{r['solution_snippet']}</pre>"
                for r in pts
            ]

            traces.append({
                "type": "scatter",
                "x": xs,
                "y": ys,
                "mode": "markers",
                "name": f"{judge} / {cat}",
                "marker": {
                    "color": color,
                    "symbol": symbol,
                    "size": 12,
                    "line": {"width": 1, "color": "#333"},
                    "opacity": 0.85,
                },
                "text": texts,
                "hovertemplate": "%{text}<extra></extra>",
            })

            for r in pts:
                judge_points[judge].append((r["eff_tok"], r["quality"]))

    # Global Pareto frontier (all runs)
    all_points = [(r["eff_tok"], r["quality"]) for r in runs if r["quality"] > 0]
    global_frontier = pareto_frontier(all_points)
    gfx, gfy = make_step_line(global_frontier)
    if gfx:
        traces.append({
            "type": "scatter",
            "x": gfx,
            "y": gfy,
            "mode": "lines",
            "name": "▶ Pareto frontier (global)",
            "line": {"color": "#ffffff", "width": 2.5, "dash": "solid"},
            "hoverinfo": "skip",
        })

    # Per-judge frontiers
    dash_styles = ["dash", "dot", "dashdot", "longdash"]
    for i, judge in enumerate(judges):
        pts = [(t, q) for t, q in judge_points[judge] if q > 0]
        frontier = pareto_frontier(pts)
        fx, fy = make_step_line(frontier)
        if fx:
            color = JUDGE_COLORS.get(judge, "#cccccc")
            traces.append({
                "type": "scatter",
                "x": fx,
                "y": fy,
                "mode": "lines",
                "name": f"frontier: {judge}",
                "line": {
                    "color": color,
                    "width": 1.5,
                    "dash": dash_styles[i % len(dash_styles)],
                },
                "hoverinfo": "skip",
            })

    # Compute frontier distance per judge
    stats = []
    for judge in judges:
        pts = judge_points[judge]
        good = [(t, q) for t, q in pts if q > 0]
        zero = [(t, q) for t, q in pts if q == 0]
        if good:
            # Distance from global frontier: for each point, find nearest frontier segment
            # Simple approximation: for each run point, find the frontier point with
            # closest tokens, compute euclidean distance in log-token / quality space
            distances = []
            for tok, qual in good:
                min_d = float("inf")
                for ft, fq in global_frontier:
                    lt = math.log10(max(tok, 1)) - math.log10(max(ft, 1))
                    lq = qual - fq
                    d = math.sqrt(lt**2 + lq**2)
                    min_d = min(min_d, d)
                distances.append(min_d)
            avg_dist = sum(distances) / len(distances)
        else:
            avg_dist = float("inf")

        stats.append({
            "judge": judge,
            "n_runs": len(pts),
            "n_correct": len(good),
            "n_zero": len(zero),
            "avg_frontier_dist": round(avg_dist, 4) if avg_dist != float("inf") else "∞",
            "color": JUDGE_COLORS.get(judge, "#ccc"),
        })

    stats.sort(key=lambda s: s["avg_frontier_dist"] if isinstance(s["avg_frontier_dist"], float) else 1e9)

    # Build stats table HTML
    stats_rows = ""
    for s in stats:
        dist_str = str(s["avg_frontier_dist"])
        best = stats[0]["judge"] if stats else ""
        bold = "font-weight:bold;" if s["judge"] == best else ""
        stats_rows += f"""
        <tr>
          <td><span style="color:{s['color']}">■</span> {s['judge']}</td>
          <td>{s['n_runs']}</td>
          <td>{s['n_correct']}</td>
          <td>{s['n_zero']}</td>
          <td style="{bold}">{dist_str}</td>
        </tr>"""

    title = f"Pareto-Bench — {heat_filter or 'all heats'}"
    n_runs = len(runs)
    n_problems = len(set(r["problem"] for r in runs))
    n_judges = len(judges)

    layout = {
        "title": {
            "text": title,
            "font": {"color": "#e0e0e0", "size": 18},
        },
        "paper_bgcolor": "#0f0f0f",
        "plot_bgcolor": "#1a1a1a",
        "xaxis": {
            "title": "Effective tokens (log scale)",
            "type": "log",
            "color": "#aaa",
            "gridcolor": "#2a2a2a",
            "zerolinecolor": "#333",
        },
        "yaxis": {
            "title": "Quality score",
            "color": "#aaa",
            "gridcolor": "#2a2a2a",
            "zerolinecolor": "#333",
        },
        "legend": {
            "bgcolor": "#1a1a1a",
            "bordercolor": "#333",
            "font": {"color": "#ccc"},
        },
        "hovermode": "closest",
        "margin": {"t": 60, "b": 80, "l": 70, "r": 20},
    }

    traces_json = json.dumps(traces)
    layout_json = json.dumps(layout)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #0f0f0f; color: #e0e0e0; font-family: 'Courier New', monospace; }}
    #header {{ padding: 1.5rem 2rem 0.5rem; border-bottom: 1px solid #222; }}
    #header h1 {{ color: #7eb8f7; font-size: 1.4rem; }}
    #header .meta {{ color: #666; font-size: 0.8rem; margin-top: 0.3rem; }}
    #plot {{ width: 100%; height: 65vh; min-height: 400px; }}
    #stats {{ padding: 1.5rem 2rem; }}
    #stats h2 {{ color: #aaa; font-size: 1rem; margin-bottom: 1rem; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 700px; font-size: 0.85rem; }}
    th {{ color: #666; text-align: left; padding: 0.3rem 1rem 0.3rem 0; border-bottom: 1px solid #222; }}
    td {{ padding: 0.3rem 1rem 0.3rem 0; color: #ccc; }}
    tr:hover td {{ background: #1a1a1a; }}
    .legend-note {{ color: #555; font-size: 0.75rem; margin-top: 1.5rem; }}
    .legend-note span {{ margin-right: 1.5rem; }}
  </style>
</head>
<body>
  <div id="header">
    <h1>Pareto-Bench</h1>
    <div class="meta">
      {n_runs} runs &nbsp;·&nbsp; {n_problems} problems &nbsp;·&nbsp; {n_judges} judge configs
      &nbsp;·&nbsp; {heat_filter or "all heats"}
    </div>
  </div>

  <div id="plot"></div>

  <div id="stats">
    <h2>Frontier distance (lower = better — closer to Pareto-optimal)</h2>
    <table>
      <thead>
        <tr>
          <th>Judge</th>
          <th>Runs</th>
          <th>Correct</th>
          <th>Zero score</th>
          <th>Avg distance from frontier</th>
        </tr>
      </thead>
      <tbody>{stats_rows}</tbody>
    </table>
    <div class="legend-note">
      <span>● circle = euler problem</span>
      <span>◆ diamond = poetry compression</span>
      <span>■ square = smoke test</span>
      <span>── white line = global Pareto frontier</span>
      <span>- - dashed = per-judge frontier</span>
    </div>
  </div>

  <script>
    Plotly.newPlot('plot', {traces_json}, {layout_json}, {{
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    }});
  </script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=os.path.join(REPO_DIR, "results"))
    parser.add_argument("--heat", default=None)
    parser.add_argument("--out", default=os.path.join(WWW_DIR, "pareto.html"))
    args = parser.parse_args()

    print(f"Loading runs from {args.results}...")
    runs = load_runs(args.results, args.heat)
    print(f"  {len(runs)} runs found")

    if not runs:
        print("No runs found. Run ./run.sh first.")
        sys.exit(0)

    for r in runs:
        print(f"  [{r['heat']}] problem={r['problem']} judge={r['judge']} "
              f"eff_tok={r['eff_tok']:,} quality={r['quality']:.4f}")

    print(f"\nBuilding plot...")
    html = build_html(runs, args.heat)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)

    print(f"  Written to {args.out}")
    print(f"  Live at: https://bench.headwaters.ai/{os.path.basename(args.out)}")


if __name__ == "__main__":
    main()
