from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from . import connect4 as c4


class Opponent(Protocol):
    name: str

    def act_batch(
        self,
        boards: np.ndarray,
        colors: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        ...


def _winning_action(board: np.ndarray, color: int) -> int | None:
    for action in np.flatnonzero(c4.legal_mask(board)):
        b = board.copy()
        row = int(np.flatnonzero(b[:, action] == 0)[0])
        b[row, action] = color
        if c4.is_win_from(b[None], 0, row, int(action), color):
            return int(action)
    return None


def tactical_action_single(
    board: np.ndarray,
    color: int,
    rng: np.random.Generator,
    center_bias: bool = True,
) -> int:
    legal = np.flatnonzero(c4.legal_mask(board))
    action = _winning_action(board, color)
    if action is not None:
        return action

    action = _winning_action(board, 3 - color)
    if action is not None:
        return action

    if center_bias:
        distance = np.abs(legal - (c4.COLS // 2))
        best = legal[distance == distance.min()]
        return int(rng.choice(best))
    return int(rng.choice(legal))


class RandomBot:
    name = "random"

    def act_batch(self, boards, colors, rng):
        del colors
        return np.asarray(
            [int(rng.choice(np.flatnonzero(c4.legal_mask(board)))) for board in boards],
            dtype=np.int64,
        )


class WinBot:
    name = "win"

    def act_batch(self, boards, colors, rng):
        out = []
        for board, color in zip(boards, colors):
            legal = np.flatnonzero(c4.legal_mask(board))
            action = _winning_action(board, int(color))
            out.append(action if action is not None else int(rng.choice(legal)))
        return np.asarray(out, dtype=np.int64)


class WinBlockBot:
    name = "win-block"

    def act_batch(self, boards, colors, rng):
        out = []
        for board, color in zip(boards, colors):
            legal = np.flatnonzero(c4.legal_mask(board))
            color = int(color)
            action = _winning_action(board, color)
            if action is None:
                action = _winning_action(board, 3 - color)
            out.append(action if action is not None else int(rng.choice(legal)))
        return np.asarray(out, dtype=np.int64)


class TacticalBot:
    name = "tactical"

    def act_batch(self, boards, colors, rng):
        return np.asarray(
            [
                tactical_action_single(board, int(color), rng, center_bias=True)
                for board, color in zip(boards, colors)
            ],
            dtype=np.int64,
        )


@dataclass
class NetOpponent:
    net: c4.PolicyNet
    name: str = "network"
    deterministic: bool = False

    def act_batch(self, boards, colors, rng):
        del rng
        obs = c4.canonical_obs(boards, colors)
        mask = c4.legal_mask(boards)
        action, _, _ = self.net.act_np(obs, mask, deterministic=self.deterministic)
        return action.astype(np.int64)


def _summarize(results: np.ndarray) -> dict[str, float]:
    return {
        "score": float(np.mean((results + 1.0) / 2.0)),
        "win": float(np.mean(results == 1.0)),
        "draw": float(np.mean(results == 0.0)),
        "loss": float(np.mean(results == -1.0)),
        "payoff": float(np.mean(results)),
    }


def evaluate_vs_opponent(
    net: c4.PolicyNet,
    opponent: Opponent,
    games_per_seat: int = 256,
    seed: int = 1234,
    deterministic: bool = True,
) -> dict[str, object]:
    """Evaluate from both seats to avoid first-player bias."""
    rng = np.random.default_rng(seed)
    seat_results = []

    for learner_color_fixed in (1, 2):
        n = games_per_seat
        boards = np.zeros((n, c4.ROWS, c4.COLS), dtype=np.int8)
        active = np.ones(n, dtype=bool)
        current = np.ones(n, dtype=np.int8)
        plies = np.zeros(n, dtype=np.int16)
        learner_color = np.full(n, learner_color_fixed, dtype=np.int8)
        result = np.zeros(n, dtype=np.float32)

        while np.any(active):
            learner_idx = np.flatnonzero(active & (current == learner_color))
            if len(learner_idx):
                obs = c4.canonical_obs(boards[learner_idx], learner_color[learner_idx])
                mask = c4.legal_mask(boards[learner_idx])
                action, _, _ = net.act_np(obs, mask, deterministic=deterministic)
                rows = c4.apply_actions(
                    boards, learner_idx, action, learner_color[learner_idx]
                )
                plies[learner_idx] += 1
                for j, i in enumerate(learner_idx):
                    if c4.is_win_from(
                        boards,
                        int(i),
                        int(rows[j]),
                        int(action[j]),
                        int(learner_color[i]),
                    ):
                        result[i] = 1.0
                        active[i] = False
                    elif plies[i] >= c4.ROWS * c4.COLS:
                        active[i] = False
                still = learner_idx[active[learner_idx]]
                current[still] = 3 - current[still]

            opponent_idx = np.flatnonzero(active & (current != learner_color))
            if len(opponent_idx):
                colors = current[opponent_idx]
                action = opponent.act_batch(boards[opponent_idx], colors, rng)
                rows = c4.apply_actions(boards, opponent_idx, action, colors)
                plies[opponent_idx] += 1
                for j, i in enumerate(opponent_idx):
                    if c4.is_win_from(
                        boards,
                        int(i),
                        int(rows[j]),
                        int(action[j]),
                        int(colors[j]),
                    ):
                        result[i] = -1.0
                        active[i] = False
                    elif plies[i] >= c4.ROWS * c4.COLS:
                        active[i] = False
                still = opponent_idx[active[opponent_idx]]
                current[still] = 3 - current[still]

        seat_results.append(result)

    p1 = _summarize(seat_results[0])
    p2 = _summarize(seat_results[1])
    return {
        "opponent": opponent.name,
        "p1": p1,
        "p2": p2,
        "balanced_score": float((p1["score"] + p2["score"]) / 2.0),
    }


def generate_tactical_probe(n: int = 2000, seed: int = 0):
    """Reachable states requiring an immediate win or immediate block."""
    rng = np.random.default_rng(seed)
    states = []
    labels = []
    attempts = 0

    while len(states) < n and attempts < n * 100:
        attempts += 1
        board = np.zeros((c4.ROWS, c4.COLS), np.int8)
        current = 1
        depth = int(rng.integers(4, 30))
        terminal = False

        for _ in range(depth):
            legal = np.flatnonzero(c4.legal_mask(board))
            if not len(legal):
                terminal = True
                break
            action = int(rng.choice(legal))
            row = int(np.flatnonzero(board[:, action] == 0)[0])
            board[row, action] = current
            if c4.is_win_from(board[None], 0, row, action, current):
                terminal = True
                break
            current = 3 - current

        if terminal:
            continue

        legal = np.flatnonzero(c4.legal_mask(board))
        wins = []
        blocks = []
        for action in legal:
            b = board.copy()
            row = int(np.flatnonzero(b[:, action] == 0)[0])
            b[row, action] = current
            if c4.is_win_from(b[None], 0, row, int(action), current):
                wins.append(int(action))

        if not wins:
            other = 3 - current
            for action in legal:
                b = board.copy()
                row = int(np.flatnonzero(b[:, action] == 0)[0])
                b[row, action] = other
                if c4.is_win_from(b[None], 0, row, int(action), other):
                    blocks.append(int(action))

        if wins:
            states.append((board.copy(), current))
            labels.append(("win", set(wins)))
        elif blocks:
            states.append((board.copy(), current))
            labels.append(("block", set(blocks)))

    return states, labels


def tactical_accuracy(net: c4.PolicyNet, probe, deterministic: bool = True):
    states, labels = probe
    if not states:
        return {"win": np.nan, "block": np.nan, "all": np.nan, "n": 0}

    boards = np.stack([x[0] for x in states])
    colors = np.asarray([x[1] for x in states], np.int8)
    obs = c4.canonical_obs(boards, colors)
    mask = c4.legal_mask(boards)
    action, _, _ = net.act_np(obs, mask, deterministic=deterministic)

    ok = np.asarray([int(a) in good for a, (_, good) in zip(action, labels)])
    kinds = np.asarray([kind for kind, _ in labels])
    return {
        "win": float(ok[kinds == "win"].mean()) if np.any(kinds == "win") else np.nan,
        "block": float(ok[kinds == "block"].mean()) if np.any(kinds == "block") else np.nan,
        "all": float(ok.mean()),
        "n": len(ok),
    }
