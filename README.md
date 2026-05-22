# hermes-judge-bench

A minimal benchmark for testing whether a Faces-compiled judge model outperforms a bare model as the `/goal` judge in Hermes Agent.

## Hypothesis

A judge face compiled from sources that exemplify rigorous, conservative, evidence-based evaluation (Feynman, Kahneman, Gawande, etc.) will sit closer to the Pareto frontier of **correctness vs. token efficiency** than a bare model used as judge.

## Structure

```
problems/       # problem statements (plain text, no answers)
results/        # output from run.sh (gitignored)
answers.json    # ground truth (gitignore if publishing problems)
judges.yaml     # judge configurations (model + optional Faces alias)
run.sh          # main benchmark runner
score.py        # pareto scorer → markdown leaderboard
```

## Quick start

```bash
# Smoke test (problem 000 — haiku)
./run.sh --problems 000 --judges bare

# Score it
python3 score.py

# Full run
./run.sh
```

## Adding a Faces judge

1. Compile your face via `/facemake` in Hermes
2. Get the Faces API endpoint and your alias (e.g. `judge`)
3. Add to `judges.yaml`:

```yaml
judges:
  judge-feynman:
    description: "Judge face compiled from Feynman + Kahneman sources"
    model: claude-sonnet-4-6
    provider: anthropic
    face: "judge@claude-sonnet-4-6"
```

4. Configure the Faces auxiliary endpoint in `~/.hermes/config.yaml`:

```yaml
auxiliary:
  goal_judge:
    provider: main
    base_url: https://api.faces.sh/v1
    model: judge@claude-sonnet-4-6
    api_key: YOUR_FACES_API_KEY
```

5. Run: `./run.sh --judges judge-feynman`

## Scoring

- **Correctness**: 1.0 if `solution.py` prints the right answer, 0.0 otherwise
- **Pareto score**: `correctness / log2(elapsed_seconds + 2)`
- Higher pareto score = correct answer reached faster

## Adding problems

Add a `problems/NNN.md` file with the problem statement. Add the answer to `answers.json`. Problems with `null` answers are scored on existence of a runnable `solution.py` only (useful for smoke tests and open-ended tasks).

## Results are gitignored

Add specific result files manually if you want to commit them:
```bash
git add -f results/001-bare.json
```
