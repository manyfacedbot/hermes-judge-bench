# hermes-judge-bench

A benchmark for testing whether a Faces-compiled judge model outperforms a bare model as the `/goal` judge in Hermes Agent.

## Hypothesis

A judge face compiled from sources that exemplify rigorous, conservative, evidence-based evaluation will sit closer to the Pareto frontier of **quality vs. token efficiency** than a bare model used as judge.

## Structure

```
problems/               # problem statements (.md) and supporting data
  corpus/               # poetry corpora for problems 100–103
    romantic_nature.json
    victorian_lyric.json
    ode_and_elegy.json
    sonnet.json
  poetry_eval.py        # evaluation harness for poetry compression problems
results/                # output from run.sh (gitignored)
answers.json            # ground truth for standard problems
judges.yaml             # judge configurations (model + optional Faces alias)
run.sh                  # main benchmark runner
score.py                # pareto scorer → markdown leaderboard
```

## Quick start

```bash
# Smoke test (problem 000 — haiku)
./run.sh --problems 000 --judges bare

# Poetry compression problems (all four categories, both judges)
./run.sh --problems 100,101,102,103 --judges bare,judge-face

# With custom budgets
./run.sh --problems 100,101,102,103 --judges bare,judge-face \
  --token-budget 50000 --wall-budget 600

# Score everything
python3 score.py

# Full run (all problems × all judges)
./run.sh
```

**Budget defaults** (set in `judges.yaml`, overridable per-run):
| Budget | Default | Flag | Enforcement |
|---|---|---|---|
| Token budget | 20,000 tokens | `--token-budget N` | Informational — told to agent in prompt |
| Wall-clock | 300 seconds | `--wall-budget N` | Hard — `timeout` kills the agent process |

Timed-out runs are marked `⏱TIMEOUT` in the leaderboard. The solution file is still saved if the agent wrote it before the cutoff.

---

## Problem Sets

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

**Scoring:** `correctness / log2(elapsed_seconds + 2)`

---

### Poetry compression problems (100–103) — open-ended agentic benchmark

Agents design a losslessly invertible encoding for a category of 19th century English poetry, competing to maximise compression ratio while minimising token spend.

| Problem | Category | Corpus | Poets |
|---|---|---|---|
| 100 | Romantic nature | `romantic_nature.json` (9 poems) | Wordsworth, Keats, Shelley, Clare |
| 101 | Victorian lyric | `victorian_lyric.json` (8 poems) | Tennyson, Browning, D.G. Rossetti, C. Rossetti, Arnold |
| 102 | Ode & elegy | `ode_and_elegy.json` (8 poems) | Keats, Shelley, Tennyson, Arnold |
| 103 | Sonnet | `sonnet.json` (8 poems) | Keats, E.B. Browning, C. Rossetti |

**Task:** Write a Python script that defines `encode(text: str) -> str` and `decode(encoded: str) -> str` such that `decode(encode(poem)) == poem` exactly (lossless round-trip), maximising `mean(len(original) / len(encoded))` across the corpus.

**Scoring:** `compression_ratio / (1 + tokens_used / 10000)`
- Linear token penalty — no cherry-picked inflection point
- Agents are told they compete against each other; score is absolute, ranking relative
- Eval samples complete poems at random (never fragments); seed is fixed at 42

**Proxy scoring** (since `run.sh` tracks wall time, not tokens):
`compression_ratio / (1 + elapsed_seconds / 60)`

**Running the eval harness directly:**
```bash
python3 problems/poetry_eval.py \
  --corpus problems/corpus/romantic_nature.json \
  --solution /tmp/solution.py \
  --n 5 --seed 42
```

---

## Adding a Faces judge

1. Compile your face via `/facemake` in Hermes
2. Get the Faces API endpoint and your alias (e.g. `judge`)
3. Add to `judges.yaml`:

```yaml
judges:
  judge-face:
    description: "Judge face compiled from rigorous evaluator sources"
    model: claude-sonnet-4-6
    provider: anthropic
    face: "judge@claude-sonnet-4-6"
```

4. Configure the Faces auxiliary endpoint in `~/.hermes/config.yaml`:

```yaml
auxiliary:
  goal_judge:
    base_url: https://api.faces.sh/v1
    model: judge@claude-sonnet-4-6
    api_key: YOUR_FACES_API_KEY
```

5. Run: `./run.sh --judges judge-face`

---

## Scoring reference

| Problem type | Correctness | Pareto formula |
|---|---|---|
| Standard (000–099) | 1.0 if exact match, 0.0 otherwise | `correctness / log2(elapsed_s + 2)` |
| Poetry (100–103) | compression_ratio if verified, 0.0 if round-trip fails | `compression_ratio / (1 + elapsed_s / 60)` |

---

## Results are gitignored

Add specific result files manually if you want to commit them:
```bash
git add -f results/100-bare.json
```
