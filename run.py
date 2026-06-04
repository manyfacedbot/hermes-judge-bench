#!/usr/bin/env python3
"""
hermes-judge-bench/run.py — orchestrator + per-heat runner.

Each (problem × judge) pair is one heat. For each heat we:
  1. Build an isolated HERMES_HOME under results/<heat>/hermes_home/<judge>/
     with its own .env and config.yaml (the latter sets auxiliary.goal_judge
     to the row under test).
  2. Re-exec this file with the "_heat" subcommand inside that HERMES_HOME
     so hermes_cli reads the isolated config, not the user's global one.
  3. The child runs a real /goal-style loop using the same AIAgent and
     same goal_judge auxiliary client the CLI's `/goal` slash command uses.
     Agent tokens come from AIAgent's session counters; judge tokens come
     from a thin wrapper around the same auxiliary client judge_goal uses.
  4. Result JSON is written to results/<heat>/<judge>/<problem>-result.json,
     compatible with score.py.

Usage:
    python3 run.py --problems 000 --judges bare
    python3 run.py --problems 000,001 --judges bare,judge-face
    python3 run.py                              # all problems × all judges
    python3 run.py --max-turns 15

The bench tests *the judge*, not the agent. The agent's model/provider are
the same across rows of judges.yaml; only auxiliary.goal_judge varies.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml


# Captured at import time — these point at the original terminal even when
# we redirect_stdout / redirect_stderr to /dev/null during agent.chat().
_REAL_STDOUT = sys.__stdout__


def _term_print(*args, **kwargs):
    """Print to the original terminal, ignoring any active redirect context."""
    kwargs.setdefault("file", _REAL_STDOUT)
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


@contextlib.contextmanager
def _silence_hermes_ui():
    """Suppress hermes' Rich spinner / status output during agent.chat().

    quiet_mode=True is NOT enough — Rich draws its progress UI through a
    Console instantiated separately from AIAgent's quiet flag, writing to
    stderr regardless. When stdout/stderr isn't a TTY (piped to a file) each
    spinner frame becomes its own line and the log is unreadable. This
    mirrors what hermes_cli.oneshot.run_oneshot does for the same reason.
    """
    devnull = open(os.devnull, "w", encoding="utf-8")
    try:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
    finally:
        devnull.close()

REPO_DIR = Path(__file__).resolve().parent
PROBLEMS_DIR = REPO_DIR / "problems"
RESULTS_DIR = REPO_DIR / "results"
JUDGES_FILE = REPO_DIR / "judges.yaml"
ENV_FILE = REPO_DIR / ".env"

# Cache read tokens are billed at ~10% of full price by Anthropic — we count
# them at that weight so effective_tokens reflects spend, not raw window size.
CACHE_READ_WEIGHT = 0.1

DEFAULT_MAX_TURNS = 20
DEFAULT_JUDGE_TIMEOUT = 30.0
# Hermes' /goal judge is fail-OPEN — a judge error returns "continue" so a
# transient network blip doesn't wedge a real session. For a benchmark with a
# misconfigured judge that's a disaster (agent burns 20 full turns on dead
# verdicts). We bail when the judge errors this many times in a row.
MAX_CONSECUTIVE_JUDGE_ERRORS = 3

# Toolsets the agent gets each heat. Restricting to these two keeps the agent
# surface consistent across machines and dodges optional-dep noise from
# hermes' bundled tools (browser, computer-use, etc.). Override with
# --toolsets if a problem needs something more.
DEFAULT_TOOLSETS = "terminal,file"


# ──────────────────────────────────────────────────────────────────────
# .env helpers — load the repo-local .env into the parent process so
# the subprocess can inherit ANTHROPIC_API_KEY / FACES_API_KEY.
# ──────────────────────────────────────────────────────────────────────

def load_env_file(path: Path) -> dict[str, str]:
    """Parse a .env file. Returns dict of key → value. Missing file → {}."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip().strip('"').strip("'")
        if v:
            out[k.strip()] = v
    return out


def write_env_file(path: Path, kvs: dict[str, str]) -> None:
    path.write_text("\n".join(f"{k}={v}" for k, v in kvs.items() if v) + "\n")


# ──────────────────────────────────────────────────────────────────────
# judges.yaml + per-heat HERMES_HOME setup
# ──────────────────────────────────────────────────────────────────────

def load_judges() -> dict[str, dict]:
    data = yaml.safe_load(JUDGES_FILE.read_text())
    return data.get("judges") or {}


def build_judge_config(judge_entry: dict, env_kvs: dict[str, str], agent_model: str, agent_provider: str) -> dict:
    """Return the dict to dump into HERMES_HOME/config.yaml for this heat.

    auxiliary.goal_judge is what we're varying — it's what the /goal loop
    consults to decide done/continue. The agent's main model stays constant
    across heats so the only thing changing is the judge.

    Each judges.yaml entry may carry:
      model        — model slug. For a face judge this is the BASE model
                     (used for clarity/grouping); the actual call uses `face`.
      provider     — hermes provider name (anthropic, openai, custom, …)
      base_url     — optional explicit endpoint (e.g. Faces). When set, that
                     URL receives the judge calls regardless of provider.
      api_key_env  — optional name of the env var to read the API key from.
      face         — optional Faces alias; when set, the judge call uses the
                     face's compiled model name instead of `model`.
    """
    judge_model    = judge_entry.get("model")    or agent_model
    judge_provider = judge_entry.get("provider") or agent_provider
    base_url       = judge_entry.get("base_url")
    api_key_env    = judge_entry.get("api_key_env")
    face           = judge_entry.get("face")

    cfg = {
        "model": {
            "default":  agent_model,
            "provider": agent_provider,
        },
        "auxiliary": {
            "goal_judge": {
                "model":    judge_model,
                "provider": judge_provider,
                "timeout":  DEFAULT_JUDGE_TIMEOUT,
            },
        },
        "goals": {
            "max_turns": DEFAULT_MAX_TURNS,
        },
        "telemetry": {"enabled": False},
    }

    # When the entry specifies a face, the call model is the face alias.
    if face:
        cfg["auxiliary"]["goal_judge"]["model"] = face

    if base_url:
        cfg["auxiliary"]["goal_judge"]["base_url"] = base_url
    if api_key_env:
        api_key = env_kvs.get(api_key_env, "")
        cfg["auxiliary"]["goal_judge"]["api_key"] = api_key
    return cfg


def setup_heat_home(home: Path, judge_entry: dict,
                    env_kvs: dict[str, str], agent_model: str, agent_provider: str) -> Path:
    """Write an isolated HERMES_HOME (config.yaml + .env) at `home`.

    `home` lives OUTSIDE the repo (a per-heat temp dir) so an agent that
    inspects $HERMES_HOME can't learn the repo path and walk to the secret
    poetry corpus. config.yaml / .env carry no repo paths themselves.
    """
    home.mkdir(parents=True, exist_ok=True)

    # .env — pass through API keys
    keep = {k: v for k, v in env_kvs.items() if k in {
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
        "FACES_API_KEY", "FACES_BASE_URL",
    }}
    write_env_file(home / ".env", keep)

    # config.yaml
    cfg = build_judge_config(judge_entry, env_kvs, agent_model, agent_provider)
    (home / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return home


# ──────────────────────────────────────────────────────────────────────
# Per-heat child: drives the goal loop. Run with `python3 run.py _heat ...`
# in a HERMES_HOME-isolated subprocess.
# ──────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = (
    "You are a strict judge evaluating whether an autonomous agent has "
    "achieved a user's stated goal. You receive the goal text and the "
    "agent's most recent response. Your only job is to decide whether "
    "the goal is fully satisfied based on that response.\n\n"
    "A goal is DONE only when:\n"
    "- The response explicitly confirms the goal was completed, OR\n"
    "- The response clearly shows the final deliverable was produced, OR\n"
    "- The response explains the goal is unachievable / blocked / needs "
    "user input (treat this as DONE with reason describing the block).\n\n"
    "Otherwise the goal is NOT done — CONTINUE.\n\n"
    "Reply ONLY with a single JSON object on one line:\n"
    '{"done": <true|false>, "reason": "<one-sentence rationale>"}'
)

JUDGE_USER_TEMPLATE = (
    "Goal:\n{goal}\n\n"
    "Agent's most recent response:\n{response}\n\n"
    "Is the goal satisfied?"
)

CONTINUATION_PROMPT = (
    "[Continuing toward your standing goal]\n"
    "Goal: {goal}\n\n"
    "Continue working toward this goal. Take the next concrete step. "
    "If you believe the goal is complete, state so explicitly and stop. "
    "If you are blocked and need input from the user, say so clearly and stop."
)


POETRY_PROBLEM_IDS = {"100", "101", "102", "103"}


def _build_system_prompt(problem_id: str) -> str:
    base = (
        "You are an autonomous coding agent competing against other agents to solve a "
        "problem efficiently. Total tokens (yours + the judge's) are penalised linearly — "
        "each 10,000 tokens halves your score. Stopping early with a correct answer beats "
        "a slightly better answer that costs twice as many tokens.\n\n"
    )

    if problem_id in POETRY_PROBLEM_IDS:
        # SECRET held-out eval: the agent must NOT be given the poems. It only
        # gets the category description in the problem statement, and is scored
        # on poems it never sees. The heat runs in an isolated cwd that does not
        # contain the corpus, so don't mention or hint at any corpus file.
        data_note = (
            "No data files are provided for this problem. You will be scored on a "
            "held-out set of poems from the stated category that you do NOT get to "
            "see — so your encode/decode must generalise to any poem of that kind, "
            "not memorise specific texts. Do not go looking for a poem corpus on "
            "disk; the scoring poems are not in your working directory. "
        )
    elif problem_id == "054":
        # poker.txt is public test data (not secret like the poetry corpus), so
        # an absolute path is fine — and it keeps working when score.py re-runs
        # the solution from a different cwd.
        data_note = f"The data file for this problem is at {PROBLEMS_DIR}/poker.txt. "
    else:
        data_note = "This problem needs no external data files. "

    return (
        base + data_note +
        "Write your Python solution to /tmp/solution.py and run it with `python3 /tmp/solution.py`. "
        "IMPORTANT: You are scored ONLY by running /tmp/solution.py. If you do not save your "
        "solution to /tmp/solution.py, you will receive a score of ZERO no matter what you say in "
        "your reply — stating the answer in chat is not enough. "
        "When you are confident your answer is correct AND saved to /tmp/solution.py, say DONE and stop."
    )


def _judge_call(goal: str, response_text: str, *, timeout: float = DEFAULT_JUDGE_TIMEOUT) -> dict:
    """Mirror hermes_cli.goals.judge_goal() — same auxiliary client, same prompts,
    but expose token usage so we can attribute spend to the judge.
    """
    from agent.auxiliary_client import get_auxiliary_extra_body, get_text_auxiliary_client

    client, model = get_text_auxiliary_client("goal_judge")
    if client is None or not model:
        return {
            "verdict": "continue", "reason": "no auxiliary client configured",
            "parse_failed": False, "input_tokens": 0, "output_tokens": 0,
        }

    prompt = JUDGE_USER_TEMPLATE.format(goal=goal[:2000], response=response_text[:4000])
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            max_tokens=4096,
            timeout=timeout,
            extra_body=get_auxiliary_extra_body() or None,
        )
    except Exception as exc:
        return {
            "verdict": "continue", "reason": f"judge error: {type(exc).__name__}: {exc}",
            "parse_failed": False, "input_tokens": 0, "output_tokens": 0,
        }

    try:
        raw = resp.choices[0].message.content or ""
    except Exception:
        raw = ""

    usage = getattr(resp, "usage", None)
    in_tok  = getattr(usage, "prompt_tokens", 0) or 0
    out_tok = getattr(usage, "completion_tokens", 0) or 0

    # Parse — same logic as goals._parse_judge_response, condensed.
    done, reason, parse_failed = False, "no reason provided", False
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
    data = None
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None
    if isinstance(data, dict):
        done_val = data.get("done")
        if isinstance(done_val, str):
            done = done_val.strip().lower() in {"true", "yes", "1", "done"}
        else:
            done = bool(done_val)
        reason = str(data.get("reason") or "").strip() or "no reason provided"
    else:
        parse_failed = True
        reason = f"judge reply was not JSON: {(raw or '')[:200]!r}"

    return {
        "verdict": "done" if done else "continue", "reason": reason,
        "parse_failed": parse_failed, "input_tokens": in_tok, "output_tokens": out_tok,
    }


def run_heat_child(*, problem_id: str, judge_name: str, agent_model: str, agent_provider: str,
                   max_turns: int, judge_dir: Path, toolsets: str = DEFAULT_TOOLSETS,
                   verbose: bool = False) -> dict:
    """Run one heat. Returns the full result dict (also written to disk)."""
    problem_md = PROBLEMS_DIR / f"{problem_id}.md"
    if not problem_md.exists():
        raise FileNotFoundError(f"problem file missing: {problem_md}")
    problem_text = problem_md.read_text()

    goal_text = problem_text.strip()
    system_prompt = _build_system_prompt(problem_id)

    judge_dir.mkdir(parents=True, exist_ok=True)
    response_file = judge_dir / f"{problem_id}-response.txt"
    solution_file = judge_dir / f"{problem_id}-solution.py"
    result_file   = judge_dir / f"{problem_id}-result.json"

    # Wipe any prior /tmp/solution.py to detect whether the agent wrote one.
    if os.path.exists("/tmp/solution.py"):
        os.remove("/tmp/solution.py")

    # Lazy import — keep startup cheap when the orchestrator doesn't need it.
    from run_agent import AIAgent
    from hermes_cli.runtime_provider import resolve_runtime_provider

    runtime = resolve_runtime_provider(requested=agent_provider, target_model=agent_model)
    toolset_list = [t.strip() for t in (toolsets or "").split(",") if t.strip()]

    # Always quiet_mode=True — even with --verbose. Rich's spinner writes one
    # animation frame per line when stdout isn't a TTY (e.g. tee'd to a file),
    # which makes the log unreadable. --verbose adds clean tool-call lines via
    # our own callback instead of relying on hermes' UI.
    agent_kwargs = dict(
        api_key=runtime.get("api_key"),
        base_url=runtime.get("base_url"),
        provider=runtime.get("provider"),
        api_mode=runtime.get("api_mode"),
        model=agent_model,
        enabled_toolsets=toolset_list,
        quiet_mode=True,
        platform="cli",
        credential_pool=runtime.get("credential_pool"),
        ephemeral_system_prompt=system_prompt,
    )
    if verbose:
        def _tool_start_cb(tool_name, args=None, **_kw):
            try:
                arg_preview = (json.dumps(args, default=str)[:140] if args else "")
            except Exception:
                arg_preview = str(args)[:140]
            # Uses _term_print so it bypasses _silence_hermes_ui's redirect.
            _term_print(f"      [tool] {tool_name}({arg_preview})")
        agent_kwargs["tool_start_callback"] = _tool_start_cb

    agent = AIAgent(**agent_kwargs)
    agent.suppress_status_output = True
    agent.stream_delta_callback = None
    agent.tool_gen_callback = None

    t0 = time.time()
    transcript: list[dict] = []
    judge_in_total = 0
    judge_out_total = 0
    judge_calls = 0
    last_response = ""
    last_verdict = "unknown"
    last_reason  = ""

    user_msg = goal_text
    consecutive_judge_errors = 0
    for turn in range(1, max_turns + 1):
        prev_agent_total = int(getattr(agent, "session_input_tokens", 0) or 0) + \
                           int(getattr(agent, "session_output_tokens", 0) or 0)
        turn_t0 = time.time()
        _term_print(f"    [turn {turn}/{max_turns}] agent thinking…")
        try:
            with _silence_hermes_ui():
                # Use run_conversation, not chat(): chat() does
                # result["final_response"] and raises KeyError when hermes ends a
                # turn on its internal tool-call ceiling (the result dict lacks
                # that key). run_conversation returns the same dict — we pull the
                # reply out defensively so a maxed-out turn doesn't crash the heat
                # and the real token spend below still gets recorded.
                conv = agent.run_conversation(user_msg)
            if isinstance(conv, dict):
                last_response = (conv.get("final_response")
                                 or conv.get("response")
                                 or conv.get("final_text") or "") or ""
            else:
                last_response = str(conv or "")
        except Exception as exc:
            last_response = f"[agent error: {type(exc).__name__}: {exc}]"
            transcript.append({"turn": turn, "role": "agent_error", "text": last_response})
            _term_print(f"    [turn {turn}] agent ERROR: {last_response[:200]}")
            break

        agent_now_total = int(getattr(agent, "session_input_tokens", 0) or 0) + \
                          int(getattr(agent, "session_output_tokens", 0) or 0)
        turn_agent_tok = agent_now_total - prev_agent_total
        turn_elapsed   = int(time.time() - turn_t0)
        reply_preview  = last_response.replace("\n", " ⏎ ")[:120]
        _term_print(
            f"    [turn {turn}] agent done — {turn_agent_tok} tok in {turn_elapsed}s | "
            f"reply: {reply_preview!r}"
        )

        transcript.append({"turn": turn, "role": "agent", "text": last_response[:4000]})

        with _silence_hermes_ui():
            verdict = _judge_call(goal_text, last_response)
        judge_in_total  += verdict["input_tokens"]
        judge_out_total += verdict["output_tokens"]
        judge_calls += 1
        last_verdict = verdict["verdict"]
        last_reason  = verdict["reason"]

        # Distinguish a real "continue" verdict from a judge error/parse-fail
        # that fell open to "continue". The reason text starts with "judge
        # error:" or "judge reply was not JSON" / "no auxiliary client" /
        # "auxiliary client unavailable" in the failure paths.
        is_judge_failure = verdict["parse_failed"] or any(
            last_reason.startswith(prefix) for prefix in (
                "judge error:", "no auxiliary client", "auxiliary client unavailable",
            )
        )
        if is_judge_failure:
            consecutive_judge_errors += 1
        else:
            consecutive_judge_errors = 0

        transcript.append({
            "turn": turn, "role": "judge",
            "verdict": last_verdict, "reason": last_reason,
            "input_tokens": verdict["input_tokens"], "output_tokens": verdict["output_tokens"],
            "is_judge_failure": is_judge_failure,
        })
        _term_print(
            f"    [turn {turn}] judge → {last_verdict} "
            f"({verdict['input_tokens']+verdict['output_tokens']} tok) — {last_reason[:140]}"
        )

        if last_verdict == "done":
            break
        if consecutive_judge_errors >= MAX_CONSECUTIVE_JUDGE_ERRORS:
            last_verdict = "judge_dead"
            last_reason = (
                f"aborted after {consecutive_judge_errors} consecutive judge failures; "
                f"last reason: {last_reason}"
            )
            _term_print(f"    [turn {turn}] ⛔ {last_reason}")
            break
        user_msg = CONTINUATION_PROMPT.format(goal=goal_text)

    elapsed = int(time.time() - t0)

    # Pull agent token counts off AIAgent
    agent_in   = int(getattr(agent, "session_input_tokens", 0) or 0)
    agent_out  = int(getattr(agent, "session_output_tokens", 0) or 0)
    agent_crd  = int(getattr(agent, "session_cache_read_tokens", 0) or 0)
    agent_cwr  = int(getattr(agent, "session_cache_write_tokens", 0) or 0)

    agent_effective = agent_in + agent_out + round(agent_crd * CACHE_READ_WEIGHT)
    judge_effective = judge_in_total + judge_out_total
    effective_tokens = agent_effective + judge_effective

    # Persist response + solution if the agent wrote one.
    response_file.write_text(last_response)
    if os.path.exists("/tmp/solution.py"):
        shutil.copy("/tmp/solution.py", solution_file)
        # leave /tmp/solution.py in place — score.py reads it via the file we copied

    result = {
        "heat":               judge_dir.parent.name,
        "problem":            problem_id,
        "judge":              judge_name,
        "model":              agent_model,
        "elapsed_seconds":    elapsed,
        "max_turns":          max_turns,
        "turns_used":         (turn if last_response else 0),
        "killed_by_timeout":  False,
        "final_verdict":      last_verdict,
        "final_reason":       last_reason,

        # Agent tokens (the solver)
        "input_tokens":        agent_in,
        "output_tokens":       agent_out,
        "cache_read_tokens":   agent_crd,
        "cache_write_tokens":  agent_cwr,
        "cache_read_weight":   CACHE_READ_WEIGHT,

        # Judge tokens (the goal_judge auxiliary)
        "judge_input_tokens":  judge_in_total,
        "judge_output_tokens": judge_out_total,
        "judge_calls":         judge_calls,

        # Combined for pareto scoring
        "effective_tokens":    effective_tokens,

        "response_file": response_file.name,
        "solution_file": solution_file.name if solution_file.exists() else None,
    }

    result_file.write_text(json.dumps(result, indent=2))
    (judge_dir / f"{problem_id}-transcript.json").write_text(json.dumps(transcript, indent=2))
    return result


# ──────────────────────────────────────────────────────────────────────
# Orchestrator entry point
# ──────────────────────────────────────────────────────────────────────

def orchestrate(args) -> int:
    judges = load_judges()
    if not judges:
        print(f"No judges found in {JUDGES_FILE}", file=sys.stderr)
        return 2

    if args.judges:
        sel = [j.strip() for j in args.judges.split(",")]
        missing = [j for j in sel if j not in judges]
        if missing:
            print(f"Unknown judge(s): {missing}. Available: {list(judges)}", file=sys.stderr)
            return 2
        judges = {k: judges[k] for k in sel}

    if args.problems:
        problem_ids = [p.strip() for p in args.problems.split(",")]
    else:
        problem_ids = sorted(p.stem for p in PROBLEMS_DIR.glob("*.md"))

    env_kvs = load_env_file(ENV_FILE)
    if not env_kvs.get("ANTHROPIC_API_KEY") and not env_kvs.get("OPENAI_API_KEY"):
        print(f"⚠  {ENV_FILE} has no ANTHROPIC_API_KEY (or OPENAI_API_KEY). "
              "Copy .env.example → .env and fill in your key.", file=sys.stderr)

    heat_id = f"heat_{int(time.time())}"
    heat_dir = RESULTS_DIR / heat_id
    heat_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== hermes-judge-bench ===", flush=True)
    print(f"Heat       : {heat_id}", flush=True)
    print(f"Problems   : {problem_ids}", flush=True)
    print(f"Judges     : {list(judges)}", flush=True)
    print(f"Max turns  : {args.max_turns}", flush=True)
    print(f"Agent model: {args.agent_model} (provider={args.agent_provider})", flush=True)
    print(flush=True)

    # Up-front: skip any judge whose required API key is missing, so we don't
    # burn agent turns on a judge that's going to PermissionDenied every call.
    usable_judges = {}
    for name, entry in judges.items():
        api_key_env = entry.get("api_key_env")
        if api_key_env and not env_kvs.get(api_key_env):
            print(f"⚠  Skipping judge '{name}': api_key_env={api_key_env} but that var is not "
                  f"set in {ENV_FILE}. Add it to .env to enable this judge.", flush=True)
            continue
        if entry.get("provider") == "anthropic" and not env_kvs.get("ANTHROPIC_API_KEY"):
            print(f"⚠  Skipping judge '{name}': provider=anthropic but ANTHROPIC_API_KEY is not set.",
                  flush=True)
            continue
        usable_judges[name] = entry
    if not usable_judges:
        print("No runnable judges — fix .env and retry.", file=sys.stderr)
        return 2

    for problem_id in problem_ids:
        for judge_name, judge_entry in usable_judges.items():
            print(f"→ problem={problem_id}  judge={judge_name}", flush=True)
            judge_dir = heat_dir / judge_name
            judge_dir.mkdir(parents=True, exist_ok=True)

            # Per-heat sandbox OUTSIDE the repo. It holds the isolated HERMES_HOME
            # and the agent's working directory. Keeping both off the repo tree
            # means an agent that inspects $HERMES_HOME, $PWD, or runs `ls`/`pwd`
            # never learns the repo path, so it can't walk to the secret poetry
            # corpus at problems/corpus/. (A determined `find /` could still reach
            # it — that needs real container isolation; see README.) Result
            # artifacts still go to judge_dir (an absolute path under the repo).
            sandbox = tempfile.mkdtemp(prefix=f"hjb_{problem_id}_{judge_name}_")
            home = setup_heat_home(Path(sandbox) / "hermes_home", judge_entry,
                                   env_kvs, args.agent_model, args.agent_provider)
            work = Path(sandbox) / "work"
            work.mkdir(parents=True, exist_ok=True)

            child_env = os.environ.copy()
            child_env["HERMES_HOME"] = str(home)
            child_env["PWD"] = str(work)   # keep $PWD consistent with the real cwd
            child_env["HERMES_YOLO_MODE"] = "1"
            child_env["HERMES_ACCEPT_HOOKS"] = "1"
            # Kill ANSI / Rich spinner output — we want clean log-friendly text.
            child_env["NO_COLOR"] = "1"
            child_env["TERM"] = "dumb"
            child_env["FORCE_COLOR"] = "0"
            child_env["RICH_FORCE_TERMINAL"] = "0"
            child_env["PYTHONUNBUFFERED"] = "1"
            for k, v in env_kvs.items():
                child_env.setdefault(k, v)
            # Never expose the corpus location (or any HJB_* scoring config) to
            # the agent subprocess — that's score.py's business.
            for k in [k for k in child_env if k.startswith("HJB_")]:
                child_env.pop(k, None)

            cmd = [
                sys.executable, str(Path(__file__).resolve()), "_heat",
                "--problem-id",     problem_id,
                "--judge-name",     judge_name,
                "--agent-model",    args.agent_model,
                "--agent-provider", args.agent_provider,
                "--max-turns",      str(args.max_turns),
                "--judge-dir",      str(judge_dir),
                "--toolsets",       args.toolsets,
            ]
            if args.verbose:
                cmd.append("--verbose")
            # Don't capture — let the child's per-turn progress lines stream
            # straight through to the user's terminal in real time.
            try:
                proc = subprocess.run(cmd, env=child_env, cwd=str(work))
            finally:
                shutil.rmtree(sandbox, ignore_errors=True)
            if proc.returncode != 0:
                print(f"  ⚠ heat exited non-zero ({proc.returncode}) — see traceback above")

    print(f"\n=== Done. Heat: {heat_id} ===")
    print(f"  Results: {heat_dir}")
    print(f"  Score:   python3 score.py --heat {heat_id}")
    return 0


def heat_main(args) -> int:
    """Subcommand `_heat`: do one (problem × judge) heat in this process.

    HERMES_HOME is already set by the orchestrator before exec'ing us.
    """
    result = run_heat_child(
        problem_id=args.problem_id,
        judge_name=args.judge_name,
        agent_model=args.agent_model,
        agent_provider=args.agent_provider,
        max_turns=args.max_turns,
        judge_dir=Path(args.judge_dir),
        toolsets=args.toolsets,
        verbose=args.verbose,
    )
    summary = (
        f"  ✓ {result['final_verdict']} in {result['turns_used']}/{result['max_turns']} turns | "
        f"agent_tok={result['input_tokens']+result['output_tokens']} "
        f"judge_tok={result['judge_input_tokens']+result['judge_output_tokens']} "
        f"effective={result['effective_tokens']} elapsed={result['elapsed_seconds']}s"
    )
    print(summary, flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    sub = parser.add_subparsers(dest="command")

    # Default = orchestrate
    parser.add_argument("--problems", default="", help="Comma-separated problem IDs (default: all)")
    parser.add_argument("--judges",   default="", help="Comma-separated judge names from judges.yaml (default: all)")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS,
                        help=f"Max /goal turns per heat (default: {DEFAULT_MAX_TURNS})")
    parser.add_argument("--agent-model",    default="claude-sonnet-4-6",
                        help="Agent's main model — held constant across heats so only the judge varies.")
    parser.add_argument("--agent-provider", default="anthropic")
    parser.add_argument("--toolsets", default=DEFAULT_TOOLSETS,
                        help=f"Comma-separated toolset names enabled for the agent "
                             f"(default: '{DEFAULT_TOOLSETS}'). Keep this narrow — wider toolsets "
                             f"pull in optional hermes deps (websockets, browser, etc.) and add noise.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Add a clean one-line echo for every tool call the agent makes, on top "
                             "of the default per-turn progress lines. Does NOT enable hermes' Rich "
                             "spinner UI — that breaks log files.")

    heat = sub.add_parser("_heat", help=argparse.SUPPRESS)
    heat.add_argument("--problem-id",     required=True)
    heat.add_argument("--judge-name",     required=True)
    heat.add_argument("--agent-model",    required=True)
    heat.add_argument("--agent-provider", required=True)
    heat.add_argument("--max-turns",      type=int, required=True)
    heat.add_argument("--judge-dir",      required=True)
    heat.add_argument("--toolsets",       default=DEFAULT_TOOLSETS)
    heat.add_argument("--verbose",        action="store_true")

    args = parser.parse_args()
    if args.command == "_heat":
        sys.exit(heat_main(args))
    sys.exit(orchestrate(args))


if __name__ == "__main__":
    main()
