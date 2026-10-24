"""Tests for trends.py. Run with: pytest test_trends.py"""
import numpy as np
import pytest

from trends import (choose_k, health_stages, knn_smooth, moving_average,
                    remaining_life, spline_smooth)

rng = np.random.default_rng(0)
T = np.linspace(0, 160, 900)
TRUE = 0.07 + 0.002 * np.exp((T - 120) / 10).clip(max=50)  # flat, then rising
Y = TRUE + rng.normal(0, 0.005, T.size)


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def test_moving_average_of_constant_is_constant():
    assert np.allclose(moving_average(np.full(20, 2.0), 5), 2.0)


@pytest.mark.parametrize("smoother", [
    lambda: moving_average(Y, 11),
    lambda: spline_smooth(T, Y),
    lambda: knn_smooth(T, Y, 11),
])
def test_smoothers_get_closer_to_true_trend(smoother):
    assert rmse(smoother(), TRUE) < rmse(Y, TRUE)


def test_knn_with_k1_reproduces_data():
    assert np.allclose(knn_smooth(T, Y, 1), Y)


def test_choose_k_avoids_extremes():
    k, scores = choose_k(T, Y, ks=range(1, 41))
    assert 1 < k < 40
    assert scores["train"][0] == 0  # k = 1 fits training points exactly


def test_health_stages_rise_and_never_drop():
    stage = health_stages(TRUE, fail_factor=2)
    assert stage[0] == 0 and stage[-1] == 2
    assert np.all(np.diff(stage) >= 0)


def test_remaining_life_ends_at_zero():
    rul = remaining_life(T)
    assert rul[0] == 160 and rul[-1] == 0
