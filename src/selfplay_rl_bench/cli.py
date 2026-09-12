from __future__ import annotations

import argparse
import json
import time

from .exact_probe import evaluate_exact, generate_exact_probe
from .trainer import TrainConfig, config_dict, train


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Connect Four adversarial self-play benchmark")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=480_000)
    p.add_argument("--reset", choices=["root", "random-prefix", "historical"], default="random-prefix")
    p.add_argument("--continue-prob", type=float, default=0.8)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--lambda-v", type=float, default=0.5)
    p.add_argument("--lambda-gae", type=float, default=0.0)
    p.add_argument("--beta1-start", type=float, default=0.5)
    p.add_argument("--beta1-end", type=float, default=0.9)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--probe-size", type=int, default=300)
    return p


def main() -> None:
    args = build_parser().parse_args()
    config = TrainConfig(
        seed=args.seed,
        total_policy_steps=args.steps,
        reset=args.reset,
        continue_prob=args.continue_prob,
        max_depth=args.max_depth,
        lambda_v=args.lambda_v,
        lambda_gae=args.lambda_gae,
        beta1_start=args.beta1_start,
        beta1_end=args.beta1_end,
        eps=args.eps,
    )
    probe = generate_exact_probe(args.probe_size, seed=20260910)
    started = time.time()
    net, train_metrics = train(config)
    exact = evaluate_exact(net, probe)
    print(json.dumps({"config": config_dict(config), "train": train_metrics, "exact": exact, "seconds": time.time() - started}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
