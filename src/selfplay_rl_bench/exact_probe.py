from __future__ import annotations

import functools

import numpy as np
import torch

from . import connect4 as c4


@functools.lru_cache(maxsize=2_000_000)
def solve_bytes(board_bytes: bytes, player: int):
    board = np.frombuffer(board_bytes, dtype=np.int8).reshape(c4.ROWS, c4.COLS)
    legal = np.flatnonzero(c4.legal_mask(board))
    if not len(legal):
        return 0, ()

    best = -2
    values = []
    for action in sorted(map(int, legal), key=lambda a: abs(a - 3)):
        child = board.copy()
        row = int(np.flatnonzero(child[:, action] == 0)[0])
        child[row, action] = player
        if c4.is_win_from(child[None], 0, row, action, player):
            value = 1
        else:
            child_value, _ = solve_bytes(child.tobytes(), 3 - player)
            value = -child_value
        values.append((action, value))
        best = max(best, value)
    return best, tuple(values)


def generate_exact_probe(
    n: int = 300,
    seed: int = 123,
    min_depth: int = 30,
    max_depth: int = 36,
):
    rng = np.random.default_rng(seed)
    states, qvals = [], []
    attempts = 0
    while len(states) < n and attempts < n * 100:
        attempts += 1
        board = np.zeros((c4.ROWS, c4.COLS), np.int8)
        player = 1
        terminal = False
        depth = int(rng.integers(min_depth, max_depth + 1))
        for _ in range(depth):
            legal = np.flatnonzero(c4.legal_mask(board))
            if not len(legal):
                terminal = True
                break
            action = int(rng.choice(legal))
            row = int(np.flatnonzero(board[:, action] == 0)[0])
            board[row, action] = player
            if c4.is_win_from(board[None], 0, row, action, player):
                terminal = True
                break
            player = 3 - player
        if terminal:
            continue

        _, pairs = solve_bytes(board.tobytes(), player)
        q = np.full(c4.COLS, np.nan, np.float32)
        for action, value in pairs:
            q[action] = value
        legal = np.flatnonzero(~np.isnan(q))
        if len(legal) < 2 or np.all(q[legal] == q[legal][0]):
            continue
        states.append((board.copy(), player))
        qvals.append(q)

    boards = np.stack([state[0] for state in states])
    colors = np.asarray([state[1] for state in states], np.int8)
    return boards, colors, np.stack(qvals)


def evaluate_exact(net: c4.PolicyNet, probe) -> dict[str, float]:
    boards, colors, q = probe
    obs = c4.canonical_obs(boards, colors)
    mask = c4.legal_mask(boards)
    with torch.no_grad():
        dist, _ = net.dist_value(torch.from_numpy(obs).float(), torch.from_numpy(mask).bool())
        probs = dist.probs.numpy()
        actions = probs.argmax(1)

    qsafe = np.nan_to_num(q, nan=0.0)
    best = np.nanmax(q, axis=1)
    expected_q = (probs * qsafe).sum(1)
    chosen = q[np.arange(len(q)), actions]
    optmask = q == best[:, None]
    return {
        "n": float(len(q)),
        "opt_argmax": float((chosen == best).mean()),
        "argmax_regret": float((best - chosen).mean()),
        "expected_regret": float((best - expected_q).mean()),
        "optimal_prob": float((probs * optmask).sum(1).mean()),
    }
