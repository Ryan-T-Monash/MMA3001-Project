"""Tests for splits.py. Run with: pytest test_splits.py"""
import numpy as np
import pytest

from splits import block_split, interleaved_split, time_split

T = np.arange(984) / 6  # test 2: 984 snapshots, 10 minutes apart


@pytest.mark.parametrize("ratios", [(0.6, 0.2, 0.2), (0.7, 0.15, 0.15), (0.5, 0.25, 0.25)])
def test_time_split_is_in_order_with_right_sizes(ratios):
    roles = time_split(T.size, ratios)
    order = {"train": 0, "val": 1, "test": 2}
    assert np.all(np.diff([order[r] for r in roles]) >= 0)
    assert np.mean(roles == "test") == pytest.approx(ratios[2], abs=0.01)


def test_interleaved_split_repeats_pattern():
    roles = interleaved_split(30, (0.6, 0.2, 0.2), period=10)
    assert list(roles[:10]) == ["train"] * 6 + ["val"] * 2 + ["test"] * 2
    assert np.array_equal(roles[:10], roles[10:20])
    assert np.sum(interleaved_split(40, (0.7, 0.15, 0.15), period=20) == "val") == 6


def test_block_split_keeps_blocks_whole_and_gaps_changes():
    roles = block_split(T, block_hours=5, gap_hours=1, seed=0)
    block = (T // 5).astype(int)
    for b in np.unique(block):
        used = set(roles[block == b]) - {"gap"}
        assert len(used) <= 1
    for r in ("train", "val", "test"):          # gaps mean no snapshot within
        other = (roles != r) & (roles != "gap")  # 1 h of another role
        if np.any(roles == r) and np.any(other):
            d = np.abs(T[roles == r][:, None] - T[other][None, :]).min()
            assert d >= 1 - 1e-9


def test_block_split_seed_changes_assignment():
    a, b = block_split(T, seed=0), block_split(T, seed=1)
    assert np.array_equal(a, block_split(T, seed=0))
    assert not np.array_equal(a, b)
