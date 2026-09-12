from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from . import connect4 as c4


class ResetSampler(Protocol):
    def sample_batch(
        self, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]: ...

    def observe(self, board: np.ndarray, current: int) -> None: ...


@dataclass
class RootReset:
    def sample_batch(self, n: int, rng: np.random.Generator):
        del rng
        return (
            np.zeros((n, c4.ROWS, c4.COLS), np.int8),
            np.ones(n, np.int8),
            np.zeros(n, np.int16),
            0,
        )

    def observe(self, board: np.ndarray, current: int) -> None:
        del board, current


@dataclass
class GeometricRandomPrefix:
    """Reachable random reset states using P(continue)=continue_prob.

    Prefix transitions are intentionally not added to the training batch. By default,
    they also do not advance the optimizer schedule because they require no policy
    inference or backward pass.
    """

    continue_prob: float = 0.8

    def sample_batch(self, n: int, rng: np.random.Generator):
        boards = np.zeros((n, c4.ROWS, c4.COLS), np.int8)
        current = np.ones(n, np.int8)
        depths = np.zeros(n, np.int16)
        prefix_moves = 0

        for i in range(n):
            # Resample from root whenever a random prefix becomes terminal.
            while True:
                restarted = False
                while rng.random() < self.continue_prob:
                    legal = np.flatnonzero(c4.legal_mask(boards[i]))
                    action = int(rng.choice(legal))
                    row = int(np.flatnonzero(boards[i, :, action] == 0)[0])
                    boards[i, row, action] = current[i]
                    depths[i] += 1
                    prefix_moves += 1
                    if (
                        c4.is_win_from(
                            boards[i : i + 1], 0, row, action, int(current[i])
                        )
                        or depths[i] >= 42
                    ):
                        boards[i].fill(0)
                        current[i] = 1
                        depths[i] = 0
                        restarted = True
                        break
                    current[i] = 3 - current[i]
                if not restarted:
                    break

        return boards, current, depths, prefix_moves

    def observe(self, board: np.ndarray, current: int) -> None:
        del board, current


class HistoricalDepthReplay:
    """Legacy comparison reset: uniform requested depth + recent/lifetime reservoirs."""

    def __init__(
        self,
        max_depth: int = 12,
        lifetime_cap: int = 5000,
        recent_cap: int = 1000,
        recent_prob: float = 0.8,
        seed: int = 0,
    ):
        self.max_depth = max_depth
        self.lifetime_cap = lifetime_cap
        self.recent_prob = recent_prob
        self.rng = np.random.default_rng(seed)
        self.life: list[list[np.ndarray]] = [[] for _ in range(max_depth + 1)]
        self.recent = [deque(maxlen=recent_cap) for _ in range(max_depth + 1)]
        self.seen = np.zeros(max_depth + 1, dtype=np.int64)
        zero = np.zeros((c4.ROWS, c4.COLS), np.int8)
        self.life[0].append(zero.copy())
        self.recent[0].append(zero.copy())
        self.seen[0] = 1

    def observe(self, board: np.ndarray, current: int) -> None:
        depth = int(np.count_nonzero(board))
        if depth > self.max_depth:
            return
        canonical = (
            board.copy()
            if current == 1
            else np.where(board == 0, 0, np.where(board == current, 1, 2)).astype(np.int8)
        )
        self.recent[depth].append(canonical.copy())
        self.seen[depth] += 1
        bucket = self.life[depth]
        if len(bucket) < self.lifetime_cap:
            bucket.append(canonical.copy())
        else:
            j = int(self.rng.integers(0, self.seen[depth]))
            if j < self.lifetime_cap:
                bucket[j] = canonical.copy()

    def sample_batch(self, n: int, rng: np.random.Generator):
        del rng  # this sampler has its own RNG so results are seed-stable.
        boards = np.zeros((n, c4.ROWS, c4.COLS), np.int8)
        depths = np.zeros(n, np.int16)
        requested = self.rng.integers(0, self.max_depth + 1, size=n)
        for i, depth in enumerate(requested):
            d = int(depth)
            while d > 0 and not self.life[d]:
                d -= 1
            use_recent = len(self.recent[d]) > 0 and self.rng.random() < self.recent_prob
            bucket = self.recent[d] if use_recent else self.life[d]
            boards[i] = bucket[int(self.rng.integers(0, len(bucket)))]
            depths[i] = d
        return boards, np.ones(n, np.int8), depths, 0
