import numpy as np

from selfplay_rl_bench import connect4 as c4
from selfplay_rl_bench.resets import GeometricRandomPrefix, HistoricalDepthReplay


def test_geometric_prefix_is_reachable_and_nonterminal():
    reset = GeometricRandomPrefix(0.8)
    rng = np.random.default_rng(123)
    boards, current, depths, prefix_moves = reset.sample_batch(256, rng)
    assert prefix_moves >= int(depths.sum())
    assert np.all(depths == np.count_nonzero(boards, axis=(1, 2)))
    assert np.all(c4.legal_mask(boards).sum(axis=1) >= 1)
    assert set(np.unique(current)).issubset({1, 2})


def test_geometric_depth_mean_is_close_to_p_over_one_minus_p():
    p = 0.8
    reset = GeometricRandomPrefix(p)
    rng = np.random.default_rng(7)
    _, _, depths, _ = reset.sample_batch(10_000, rng)
    # Terminal-prefix resampling shifts this slightly, so keep the tolerance broad.
    assert abs(float(depths.mean()) - p / (1 - p)) < 0.8


def test_historical_replay_starts_at_root_before_observations():
    reset = HistoricalDepthReplay(max_depth=12, seed=0)
    boards, current, depths, _ = reset.sample_batch(32, np.random.default_rng(0))
    assert np.all(boards == 0)
    assert np.all(current == 1)
    assert np.all(depths == 0)
