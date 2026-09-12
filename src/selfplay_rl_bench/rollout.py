from __future__ import annotations

import numpy as np

from . import connect4 as c4
from .resets import ResetSampler


def finish_split(
    traj: list[c4.Transition],
    outcome: float,
    gamma: float = 1.0,
    lambda_gae: float = 0.0,
    lambda_v: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate policy-credit and critic-return trace parameters.

    Each transition spans one decision by a player to that same player's next decision,
    so gamma=1 is natural for undiscounted episodic board games.
    """

    n = len(traj)
    rewards = np.zeros(n, np.float32)
    if n:
        rewards[-1] = outcome
    values = np.asarray([t.value for t in traj], np.float32)

    adv = np.zeros(n, np.float32)
    gae = 0.0
    for t in range(n - 1, -1, -1):
        next_v = 0.0 if t == n - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lambda_gae * gae
        adv[t] = gae

    ret = np.zeros(n, np.float32)
    g = 0.0
    for t in range(n - 1, -1, -1):
        next_v = 0.0 if t == n - 1 else values[t + 1]
        g = rewards[t] + gamma * ((1 - lambda_v) * next_v + lambda_v * g)
        ret[t] = g

    return adv, ret


def collect_two_sided(
    net: c4.PolicyNet,
    reset_sampler: ResetSampler,
    target_policy_steps: int = 4096,
    n_envs: int = 64,
    seed: int = 0,
    lambda_gae: float = 0.0,
    lambda_v: float = 0.5,
) -> tuple[c4.Batch, dict[str, float]]:
    """Current-vs-current self-play, training on both players' on-policy trajectories."""

    rng = np.random.default_rng(seed)
    rows_all = []
    policy_steps = 0
    prefix_moves = 0
    games = 0
    prefix_depths: list[int] = []

    while policy_steps < target_policy_steps:
        boards, current, start_depth, prefix_cost = reset_sampler.sample_batch(n_envs, rng)
        prefix_moves += prefix_cost
        prefix_depths.extend(start_depth.tolist())
        plies = start_depth.copy()
        active = np.ones(n_envs, bool)
        traj = [[[], []] for _ in range(n_envs)]

        while np.any(active):
            idx = np.flatnonzero(active)
            colors = current[idx]

            for i in idx:
                reset_sampler.observe(boards[i], int(current[i]))

            obs = c4.canonical_obs(boards[idx], colors)
            mask = c4.legal_mask(boards[idx])
            actions, logp, values = net.act_np(obs, mask, False)

            for j, i in enumerate(idx):
                p = int(colors[j]) - 1
                traj[i][p].append(
                    c4.Transition(
                        obs[j].copy(),
                        mask[j].copy(),
                        int(actions[j]),
                        float(logp[j]),
                        float(values[j]),
                    )
                )

            landing = c4.apply_actions(boards, idx, actions, colors)
            plies[idx] += 1
            policy_steps += len(idx)

            for j, i in enumerate(idx):
                terminal = False
                winner = 0
                if c4.is_win_from(
                    boards, i, int(landing[j]), int(actions[j]), int(colors[j])
                ):
                    terminal = True
                    winner = int(colors[j])
                elif plies[i] >= 42:
                    terminal = True

                if terminal:
                    for p in (1, 2):
                        outcome = 0.0 if winner == 0 else (1.0 if p == winner else -1.0)
                        player_traj = traj[i][p - 1]
                        adv, ret = finish_split(
                            player_traj,
                            outcome,
                            gamma=1.0,
                            lambda_gae=lambda_gae,
                            lambda_v=lambda_v,
                        )
                        rows_all.append((player_traj, adv, ret))
                    games += 1
                    active[i] = False

            still = idx[active[idx]]
            current[still] = 3 - current[still]

    obs, masks, actions, logps, old_values, advs, rets = [], [], [], [], [], [], []
    for player_traj, adv, ret in rows_all:
        for k, transition in enumerate(player_traj):
            obs.append(transition.obs)
            masks.append(transition.mask)
            actions.append(transition.action)
            logps.append(transition.logp)
            old_values.append(transition.value)
            advs.append(adv[k])
            rets.append(ret[k])

    batch = c4.Batch(
        np.asarray(obs, np.float32),
        np.asarray(masks, bool),
        np.asarray(actions, np.int64),
        np.asarray(logps, np.float32),
        np.asarray(old_values, np.float32),
        np.asarray(advs, np.float32),
        np.asarray(rets, np.float32),
        games,
        np.empty((0,), np.float32),
        0.0,
    )
    stats = {
        "policy_steps": float(policy_steps),
        "prefix_moves": float(prefix_moves),
        "games": float(games),
        "mean_prefix_depth": float(np.mean(prefix_depths)) if prefix_depths else 0.0,
        "train_samples": float(len(batch.obs)),
    }
    return batch, stats
