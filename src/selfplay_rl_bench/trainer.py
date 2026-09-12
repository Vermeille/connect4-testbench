from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from . import connect4 as c4
from .ppo import ppo_update
from .resets import GeometricRandomPrefix, HistoricalDepthReplay, ResetSampler, RootReset
from .rollout import collect_two_sided


@dataclass
class TrainConfig:
    seed: int = 0
    total_policy_steps: int = 480_000
    policy_steps_per_update: int = 4096
    n_envs: int = 64
    reset: str = "random-prefix"
    continue_prob: float = 0.8
    max_depth: int = 12
    recent_prob: float = 0.8
    lambda_gae: float = 0.0
    lambda_v: float = 0.5
    ppo_epochs: int = 1
    minibatch: int = 256
    clip: float = 0.2
    vf_coef: float = 1.0
    ent_coef: float = 0.002
    max_grad: float = 0.5
    target_kl: float = 0.04
    lr_start: float = 5e-4
    lr_end: float = 5e-5
    beta1_start: float = 0.5
    beta1_end: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0


def cosine(x: float, total: float, start: float, end: float) -> float:
    z = min(max(x / float(total), 0.0), 1.0)
    return end + (start - end) * 0.5 * (1 + math.cos(math.pi * z))


def build_reset(config: TrainConfig) -> ResetSampler:
    if config.reset == "root":
        return RootReset()
    if config.reset == "random-prefix":
        return GeometricRandomPrefix(config.continue_prob)
    if config.reset == "historical":
        return HistoricalDepthReplay(
            max_depth=config.max_depth,
            recent_prob=config.recent_prob,
            seed=config.seed + 456,
        )
    raise ValueError(f"unknown reset strategy: {config.reset}")


def train(config: TrainConfig):
    c4.seed_all(config.seed)
    net = c4.PolicyNet()
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=config.lr_start,
        betas=(config.beta1_start, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    reset_sampler = build_reset(config)

    policy_steps = 0
    updates = 0
    last_metrics: dict[str, float] = {}
    cumulative_prefix_moves = 0.0

    while policy_steps < config.total_policy_steps:
        lr = cosine(policy_steps, config.total_policy_steps, config.lr_start, config.lr_end)
        beta1 = cosine(
            policy_steps,
            config.total_policy_steps,
            config.beta1_start,
            config.beta1_end,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
            group["betas"] = (beta1, config.beta2)

        batch, rollout_stats = collect_two_sided(
            net,
            reset_sampler,
            target_policy_steps=config.policy_steps_per_update,
            n_envs=config.n_envs,
            seed=config.seed * 100_000 + updates + 1,
            lambda_gae=config.lambda_gae,
            lambda_v=config.lambda_v,
        )
        metrics = ppo_update(
            net,
            optimizer,
            batch,
            epochs=config.ppo_epochs,
            minibatch=config.minibatch,
            clip=config.clip,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad=config.max_grad,
            target_kl=config.target_kl,
        )
        step_count = int(rollout_stats["policy_steps"])
        policy_steps += step_count
        cumulative_prefix_moves += rollout_stats["prefix_moves"]
        updates += 1
        last_metrics = {
            **metrics,
            **rollout_stats,
            "lr": lr,
            "beta1": beta1,
            "policy_steps_total": float(policy_steps),
            "prefix_moves_total": cumulative_prefix_moves,
            "updates": float(updates),
        }

    return net, last_metrics


def config_dict(config: TrainConfig) -> dict:
    return asdict(config)
