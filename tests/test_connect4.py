import numpy as np

from selfplay_rl_bench import connect4 as c4
from selfplay_rl_bench.rollout import finish_split


def test_wins_and_mask():
    board = np.zeros((1, c4.ROWS, c4.COLS), np.int8)
    board[0, 0, :4] = 1
    assert c4.is_win_from(board, 0, 0, 3, 1)

    board = np.zeros((1, c4.ROWS, c4.COLS), np.int8)
    board[0, :4, 2] = 2
    assert c4.is_win_from(board, 0, 3, 2, 2)

    board = np.zeros((c4.ROWS, c4.COLS), np.int8)
    board[:, 0] = 1
    mask = c4.legal_mask(board)
    assert not mask[0]
    assert mask[1:].all()


def test_canonical_observation_swaps_players():
    board = np.zeros((c4.ROWS, c4.COLS), np.int8)
    board[0, 0] = 1
    board[0, 1] = 2
    p1 = c4.canonical_obs(board, 1).reshape(2, c4.ROWS, c4.COLS)
    p2 = c4.canonical_obs(board, 2).reshape(2, c4.ROWS, c4.COLS)
    assert np.array_equal(p1[0], p2[1])
    assert np.array_equal(p1[1], p2[0])


def test_split_lambda_terminal_target():
    traj = [
        c4.Transition(np.zeros(84, np.float32), np.ones(7, bool), 0, 0.0, 0.0)
        for _ in range(3)
    ]
    adv, ret = finish_split(traj, -1.0, lambda_gae=0.0, lambda_v=1.0)
    assert adv[-1] == -1.0
    assert np.allclose(ret, -1.0)
