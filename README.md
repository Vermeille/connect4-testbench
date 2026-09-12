# selfplay-rl-bench

A small, reproducible benchmark harness for adversarial self-play RL on board games, currently centered on Connect Four.

This repository packages the **common benchmark code** used while iterating on stable self-play recipes: the environment, PPO harness, exact late-game probe, reset-state distributions, two-sided self-play data collection, and optimizer schedules. It intentionally excludes one-off scratch experiments and checkpoint-league machinery.

## Current benchmark recipe

The current default is deliberately simple:

- current policy plays itself;
- train on **both players' trajectories** from each self-play game;
- reset with a **geometric random legal prefix**: keep playing random legal moves while `U(0,1) < 0.8`, then discard the prefix from the training batch;
- undiscounted episodic game, `gamma = 1`;
- separate traces: `lambda_GAE = 0`, `lambda_V = 0.5`;
- one PPO pass over each fresh batch;
- advantage normalization;
- value coefficient `1.0`, entropy coefficient `0.002`;
- AdamW with no weight decay, `beta2 = 0.999`, `eps = 1e-8`;
- cosine `beta1: 0.5 -> 0.9` and LR `5e-4 -> 5e-5` over **policy-generated training transitions**;
- no opponent archive, NFSP, MCTS, exploiter league, target critic, or required EMA policy.

The historical depth-reservoir reset is retained as a benchmark baseline.

## Why the random-prefix reset?

For continuation probability `p`, the untruncated prefix length is geometric:

`P(D=d) = (1-p) p^d`, with `E[D] = p/(1-p)`.

At `p=0.8`, most games remain close to the real opening while a long tail injects unusual but reachable states. Random-prefix moves are **not** included in the PPO batch and, by default, do not advance the optimizer schedule.

## Exact probe

`selfplay_rl_bench.exact_probe` generates reachable late-game positions (roughly 6–12 plies from terminal), solves every legal action exactly with cached negamax, and reports:

- `opt_argmax` — fraction where policy argmax is minimax-optimal;
- `argmax_regret` — minimax regret of argmax action;
- `expected_regret` — regret under the full policy distribution;
- `optimal_prob` — probability mass assigned to optimal actions.

This is not full-game exploitability, but it is far less discontinuous than win rate against a scripted deterministic bot.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e .[dev]
pytest
```

## Run the current benchmark

```bash
selfplay-bench --seed 0 --steps 480000 --reset random-prefix --continue-prob 0.8
```

Or compare against the historical-state reset:

```bash
selfplay-bench --seed 0 --steps 480000 --reset historical --max-depth 12
```

For a cheap smoke test:

```bash
selfplay-bench --seed 0 --steps 20000 --probe-size 50
```

The CLI prints one JSON object containing the full config, final training metrics, exact-probe metrics, and wall-clock time.

## Layout

```text
src/selfplay_rl_bench/
  connect4.py      # environment, policy/value net, batch structures
  resets.py        # root, geometric-prefix, historical-depth reset samplers
  rollout.py       # two-sided current/current rollout + split lambda returns
  ppo.py           # PPO update used by the benchmark
  trainer.py       # schedules and current training recipe
  exact_probe.py   # exact late-game minimax probe
  cli.py           # benchmark CLI

tests/
```

## Reproducibility notes

The benchmark is intentionally CPU-friendly and small enough for fast ablations. Report results over multiple seeds: adversarial self-play remains high variance, and a single trajectory is not evidence of convergence.

When comparing reset strategies, distinguish three budgets:

1. policy-generated transitions used for training;
2. raw environment transitions, including random reset moves;
3. wall-clock / neural compute.

Geometric random-prefix moves are cheap environment transitions with no network inference, so equal-policy-compute and equal-raw-transition comparisons answer different questions.

## Status

This is a research benchmark, not a claim of a solved self-play algorithm. Its purpose is to make ablations small, comparable, and difficult to accidentally confound.
