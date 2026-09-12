from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

ROWS, COLS = 6, 7
DEVICE = torch.device("cpu")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def legal_mask(board: np.ndarray) -> np.ndarray:
    """Legal columns for a [6,7] board or a batch [N,6,7]. Row 0 is bottom."""
    return board[..., ROWS - 1, :] == 0


def canonical_obs(board: np.ndarray, color: np.ndarray | int) -> np.ndarray:
    """Flattened own/opponent binary planes from the current player's perspective."""
    b = np.asarray(board)
    if b.ndim == 2:
        c = int(color)
        return np.stack((b == c, b == (3 - c)), axis=0).astype(np.float32).reshape(-1)
    c = np.asarray(color).reshape(-1, 1, 1)
    return np.stack((b == c, b == (3 - c)), axis=1).astype(np.float32).reshape(len(b), -1)


def apply_actions(
    board: np.ndarray, idx: np.ndarray, actions: np.ndarray, colors: np.ndarray
) -> np.ndarray:
    """Apply legal actions in place to batched boards and return landing rows."""
    rows = np.empty(len(idx), dtype=np.int64)
    for k, (i, a, color) in enumerate(zip(idx, actions, colors)):
        col = board[i, :, a]
        empties = np.flatnonzero(col == 0)
        if not len(empties):
            raise RuntimeError(f"illegal action env={i} col={a}")
        r = int(empties[0])
        board[i, r, a] = int(color)
        rows[k] = r
    return rows


def is_win_from(board: np.ndarray, i: int, r: int, col: int, color: int) -> bool:
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        n = 1
        rr, cc = r + dr, col + dc
        while 0 <= rr < ROWS and 0 <= cc < COLS and board[i, rr, cc] == color:
            n += 1
            rr += dr
            cc += dc
        rr, cc = r - dr, col - dc
        while 0 <= rr < ROWS and 0 <= cc < COLS and board[i, rr, cc] == color:
            n += 1
            rr -= dr
            cc -= dc
        if n >= 4:
            return True
    return False


class PolicyNet(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(2 * ROWS * COLS, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, COLS)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, gain=1.0)
        nn.init.zeros_(self.v.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.pi(h), self.v(h).squeeze(-1)

    def dist_value(self, obs: torch.Tensor, mask: torch.Tensor):
        logits, value = self(obs)
        logits = logits.masked_fill(~mask.bool(), -1e9)
        return Categorical(logits=logits), value

    @torch.no_grad()
    def act_np(self, obs: np.ndarray, mask: np.ndarray, deterministic: bool = False):
        o = torch.from_numpy(obs).float().to(DEVICE)
        m = torch.from_numpy(mask).bool().to(DEVICE)
        dist, value = self.dist_value(o, m)
        action = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        logp = dist.log_prob(action)
        return action.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()


@dataclass
class Transition:
    obs: np.ndarray
    mask: np.ndarray
    action: int
    logp: float
    value: float


@dataclass
class Batch:
    obs: np.ndarray
    mask: np.ndarray
    action: np.ndarray
    old_logp: np.ndarray
    old_value: np.ndarray
    adv: np.ndarray
    ret: np.ndarray
    episodes: int
    outcomes: np.ndarray
    mean_len: float
