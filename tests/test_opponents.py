import numpy as np

from selfplay_rl_bench import connect4 as c4
from selfplay_rl_bench.opponents import (
    RandomBot,
    TacticalBot,
    WinBlockBot,
    WinBot,
    evaluate_vs_opponent,
    generate_tactical_probe,
    tactical_accuracy,
)


def _board_with_three(color=1):
    b = np.zeros((c4.ROWS, c4.COLS), np.int8)
    b[0, :3] = color
    return b


def test_win_bot_takes_immediate_win():
    rng = np.random.default_rng(0)
    bot = WinBot()
    board = _board_with_three(1)
    action = bot.act_batch(board[None], np.asarray([1], np.int8), rng)[0]
    assert action == 3


def test_winblock_bot_blocks_immediate_loss():
    rng = np.random.default_rng(0)
    bot = WinBlockBot()
    board = _board_with_three(2)
    action = bot.act_batch(board[None], np.asarray([1], np.int8), rng)[0]
    assert action == 3


def test_tactical_prefers_center_without_tactic():
    rng = np.random.default_rng(0)
    bot = TacticalBot()
    board = np.zeros((c4.ROWS, c4.COLS), np.int8)
    action = bot.act_batch(board[None], np.asarray([1], np.int8), rng)[0]
    assert action == 3


def test_opponent_evaluation_smoke():
    c4.seed_all(0)
    net = c4.PolicyNet()
    result = evaluate_vs_opponent(net, RandomBot(), games_per_seat=4, seed=0)
    assert 0.0 <= result["balanced_score"] <= 1.0
    assert result["opponent"] == "random"


def test_tactical_probe_smoke():
    c4.seed_all(0)
    net = c4.PolicyNet()
    probe = generate_tactical_probe(n=16, seed=0)
    metrics = tactical_accuracy(net, probe)
    assert metrics["n"] > 0
    assert 0.0 <= metrics["all"] <= 1.0
