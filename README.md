# hermes-judge-bench

A benchmark for testing whether a Faces-compiled judge outperforms a bare model as the `/goal` judge in [Hermes Agent](https://hermes-agent.nousresearch.com).

## Hypothesis

A judge face compiled from sources that exemplify rigorous, conservative, evidence-based evaluation will sit closer to the Pareto frontier of **task quality vs. token cost** than a bare model used as judge.

## Architecture

Hermes' `/goal` slash command runs a Ralph-style loop: after every agent turn, an auxiliary judge LLM (configured at `auxiliary.goal_judge` in `~/.hermes/config.yaml`) decides done/continue. This benchmark *holds the agent constant* and varies only the judge, then measures pareto = `correctness / (1 + total_tokens / 10000)` across a set of problems.

The runner spawns one subprocess per (problem × judge) heat. Each subprocess gets its own isolated `HERMES_HOME` with a per-judge `config.yaml`, so judges don't share state and the host's hermes config isn't touched.

```
results/heat_<unix>/
  hermes_home/<judge>/        # isolated ~/.hermes/ for this heat (gitignored)
    config.yaml               #   auxiliary.goal_judge = the judge under test
    .env                      #   API keys
  <judge>/                    # per-judge outputs for scoring
    <problem>-result.json     #   tokens, verdict, elapsed, agent/judge attribution
    <problem>-response.txt    #   the agent's final reply
    <problem>-solution.py     #   /tmp/solution.py the agent wrote (if any)
    <problem>-transcript.json #   per-turn agent reply + judge verdict
```

## Setup (one time)

```bash
pip install hermes-agent          # CLI + Python modules
cp .env.example .env              # then fill in ANTHROPIC_API_KEY (and FACES_API_KEY if testing a face)
```

## Quick start

```bash
# Smoke test — problem 000 (haiku), bare judge only
python3 run.py --problems 000 --judges bare

# Full grid: every problem × every judge
python3 run.py

# Specific problems / judges
python3 run.py --problems 100,101,102,103 --judges bare,judge-face

# Override agent model (held constant across heats — only the judge varies)
python3 run.py --agent-model claude-sonnet-4-6 --agent-provider anthropic

# Tighter or looser goal-loop budget
python3 run.py --max-turns 15

# Score everything (or one heat)
python3 score.py
python3 score.py --heat heat_1779902817
```

`./run.sh` is a thin shim around `run.py` — use either.

## How it works (and why it's structured this way)

`hermes -z` is one-shot mode and doesn't dispatch slash commands; `hermes chat` is interactive and doesn't pipe cleanly. To exercise the real `/goal` judge code path on a per-heat basis, `run.py`:

1. **Isolates** — writes a fresh `HERMES_HOME` with a config.yaml whose `auxiliary.goal_judge` block is the row under test (bare model, or a Faces alias pointed at `https://api.faces.sh/v1`).
2. **Drives the loop** — uses the same `AIAgent` the CLI uses for agent turns, and calls `agent.auxiliary_client.get_text_auxiliary_client("goal_judge")` for the judge turn (this is the same client `hermes_cli.goals.judge_goal` uses, with the same `JUDGE_SYSTEM_PROMPT`).
3. **Attributes tokens** — agent tokens come off `AIAgent.session_*_tokens`; judge tokens come from the auxiliary client's `usage` field. They're summed in `effective_tokens` and reported separately in the result JSON.

If you'd rather inspect the goal loop logic directly, see `_judge_call` and `run_heat_child` in `run.py`.

## Adding a Faces judge

1. Compile your face via `/facemake` in Hermes.
2. Add to `judges.yaml`:

```yaml
judges:
  my-judge:
    description: "Custom judge persona"
    model: claude-sonnet-4-6
    provider: anthropic
    face: "myjudge@claude-sonnet-4-6"
```

3. Set `FACES_API_KEY` (and optionally `FACES_BASE_URL`) in `.env`.
4. Run: `python3 run.py --judges bare,my-judge`

The runner sees `face:` is set and writes the Faces endpoint into the per-heat `auxiliary.goal_judge` config automatically.

---

## Problem sets

### Standard problems (000–099) — correctness benchmarks

| Problem | Description | Answer |
|---|---|---|
| 000 | Smoke test: print a haiku | any output |
| 001 | Project Euler #1: sum of multiples of 3 or 5 below 1000 | 233168 |
| 002 | Project Euler #2: sum of even Fibonacci terms below 4M | 4613732 |
| 003 | Project Euler #3: largest prime factor of 600851475143 | 6857 |
| 004 | Project Euler #4: largest palindrome product of two 3-digit numbers | 906609 |
| 005 | Project Euler #5: smallest number divisible by 1–20 | 232792560 |
| 054 | Project Euler #54: how many poker hands does Player 1 win? | 473 |

Scoring: `correctness / (1 + effective_tokens / 10000)`.

### Poetry compression (100–103) — open-ended agentic benchmark

Agents design a losslessly invertible encoding for a *category* of 19th-century English poetry, competing to maximise compression ratio while minimising token spend.

| Problem | Category |
|---|---|
| 100 | Romantic nature (Wordsworth, Keats, Shelley, Coleridge) |
| 101 | Victorian lyric (Tennyson, Browning, Arnold, Rossetti) |
| 102 | Ode & elegy (Keats's odes, Shelley, Arnold, *In Memoriam*) |
| 103 | Sonnet (Keats, E.B. Browning, C. Rossetti) |

**The test poems are secret — by design.** The whole point of these problems is that the agent (and the judge moderating it) gets *only the category description* and must infer what poems of that kind look like — which poets, which diction, which archaic spellings, what line structure — and build an encoder that generalises to poems it never sees. Handing over the poems would collapse that fuzziness into a memorisation exercise (an agent that can read the corpus just embeds it and "compresses" by table lookup).

So the corpora are **not shipped in this repo**. They live outside the working tree at `$HJB_CORPUS_DIR` (default `~/.hjb-private/corpus/`), referenced only by `score.py`. `run.py` strips `HJB_*` from the agent subprocess env, the problem statements name no file, and the agent's system prompt explicitly tells it the poems aren't on the machine during its run.

**To score poetry problems**, place the four corpus JSON files (each a list of `{id,title,author,text}` objects) at `~/.hjb-private/corpus/`:
`romantic_nature.json`, `victorian_lyric.json`, `ode_and_elegy.json`, `sonnet.json` — or point `HJB_CORPUS_DIR` elsewhere.

Scoring: `compression_ratio / (1 + effective_tokens / 10000)`, computed by `score.py` on a **random held-out sample** of the secret poems (fresh sample each run; pass `--poetry-seed N` to reproduce a score). Round-trip failure on any sampled poem scores 0.

> **Secrecy caveat:** the agent runs as a normal subprocess with full filesystem access, so moving the corpus out of the repo defends against honest agents and accidental leakage but not against an agent that actively crawls the filesystem. For an adversarial / public deployment, sandbox the agent (run each heat in a container or jailed cwd where the corpus isn't on any reachable path). Also note: the poems that were committed earlier remain in this repo's **git history** — if this repo is ever made public, treat those specific poems as already exposed and rotate to fresh ones.

Direct invocation of the eval harness against your private corpus:

```bash
python3 problems/poetry_eval.py \
  --corpus ~/.hjb-private/corpus/romantic_nature.json \
  --solution /tmp/solution.py \
  --n 5 --seed 42
```

### ZKP (200) — stubbed for now
### Archipelago tournament (300) — 20-round head-to-head game between two judges

Engine: `problems/archipelago.py`. Tournament harness: `problems/archipelago_tournament.py`. (The tournament harness currently still uses the legacy `hermes -z` invocation per round; it'll be migrated to the new `run.py` heat-runner in a follow-up.)

---

## Scoring reference

| Problem type | Correctness | Pareto formula |
|---|---|---|
| Standard (000–099) | 1.0 if exact stdout match, 0.0 otherwise | `correctness / (1 + effective_tokens / 10000)` |
| Poetry (100–103)   | `compression_ratio` if round-trip verified, 0.0 otherwise | `compression_ratio / (1 + effective_tokens / 10000)` |

`effective_tokens = agent_input + agent_output + round(agent_cache_read * 0.1) + judge_input + judge_output`. The 0.1 weight on cache-read tokens reflects Anthropic's ~90% cache discount.

## Results are gitignored

Add specific heat files manually if you want to commit them:
```bash
git add -f results/heat_1779902817/bare/000-result.json
```
