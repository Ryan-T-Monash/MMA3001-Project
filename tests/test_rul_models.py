"""Tests for rul_models.py. Run with: pytest test_rul_models.py

The test data are exact (noise-free) curves with known answers.
"""
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from rul_models import (candidate_models, catalogue_life_hours, catalogue_rul,
                        compare_models, complexity_curve, extrapolate_rul,
                        extrapolate_rul_fast, fixed_validation_curve, validation_curve,
                        leave_one_out_by_group, fixed_leave_one_out_by_group,
                        safety_margin, apply_margin, inner_validation_predictions,
                        failure_probability, life_at_reliability, rul_scores,
                        tune_and_test, tune_fixed,
                        trailing_mean)

T = np.arange(0, 100, 1 / 6)  # 10-minute snapshots, hours


def test_catalogue_life_hand_calculation():
    # C = P: one million revolutions at 2000 rpm = 8.33 h
    assert catalogue_life_hours(C=1, P=1, rpm=2000) == pytest.approx(1e6 / 120000)
    # halving the load multiplies roller-bearing life by 2^(10/3)
    ratio = catalogue_life_hours(P=3000) / catalogue_life_hours(P=6000)
    assert ratio == pytest.approx(2 ** (10 / 3))


def test_weibull_conversions_are_consistent():
    assert life_at_reliability(1000, 0.9) == pytest.approx(1000)
    assert life_at_reliability(1000, 0.5) > 1000
    assert failure_probability(1000, 1000) == pytest.approx(0.1)


def test_catalogue_rul_never_negative():
    assert np.array_equal(catalogue_rul([0, 50, 150], 100), [100, 50, 0])


def test_trailing_mean_uses_no_future_data():
    y = np.arange(10.0)
    assert np.array_equal(trailing_mean(y, 1), y)
    changed = y.copy()
    changed[-1] = 1e6
    assert np.array_equal(trailing_mean(y, 3)[:-1], trailing_mean(changed, 3)[:-1])
    assert trailing_mean(y, 3)[5] == pytest.approx(4.0)  # mean of 3, 4, 5


def test_extrapolation_exact_for_straight_line():
    y = 0.1 + 0.01 * T          # reaches 1.0 at t = 90 h
    rul = extrapolate_rul(T, y, threshold=1.0, window=5)
    late = T > 1
    assert np.allclose(rul[late], np.clip(90 - T[late], 0, None))


def test_extrapolation_exact_for_exponential_with_log():
    y = 0.1 * np.exp(0.05 * T)  # reaches 1.0 at t = 20 ln(10)
    rul = extrapolate_rul(T, y, threshold=1.0, window=5, log=True)
    expected = np.clip(20 * np.log(10) - T, 0, None)
    assert np.allclose(rul[T > 1], expected[T > 1])


def test_extrapolation_flat_trend_gives_cap():
    rul = extrapolate_rul(T, np.full(T.size, 0.1), threshold=1.0, max_rul=500)
    assert np.all(rul == 500)


def test_rul_scores():
    s = rul_scores([10, 20, 30], [12, 20, 25])
    assert s["MAE_h"] == pytest.approx(7 / 3)
    assert s["late_%"] == pytest.approx(100 / 3)


def test_models_beat_mean_baseline_on_learnable_data():
    X = np.column_stack([np.exp(T / 30), T])
    rul = T[-1] - T
    train = T < 70
    table, pred = compare_models(candidate_models(), X[train], rul[train],
                                 X[~train], rul[~train])
    assert len(table) == 4 and set(pred) == set(table.index)
    assert table.loc["linear", "MAE_h"] < table.loc["mean (Dummy)", "MAE_h"]


def test_deeper_trees_fit_training_data_better():
    X = np.sin(T / 7).reshape(-1, 1) + T.reshape(-1, 1) / 50
    y = T[-1] - T
    curve = complexity_curve(lambda d: DecisionTreeRegressor(max_depth=d, random_state=0),
                             [1, 2, 4, 8], X[::2], y[::2], X[1::2], y[1::2])
    assert np.all(np.diff(curve["train_MAE_h"]) <= 1e-12)


def test_tune_and_test_picks_setting_on_validation_only():
    X = T.reshape(-1, 1)
    y = T[-1] - T
    roles = np.where(np.arange(T.size) % 10 < 6, "train",
                     np.where(np.arange(T.size) % 10 < 8, "val", "test"))
    grid = [{"smooth": 1, "max_depth": d} for d in (1, 3, 8)]
    row, pred = tune_and_test(lambda max_depth: DecisionTreeRegressor(max_depth=max_depth),
                              grid, {1: X}, y, roles)
    assert row["max_depth"] == 8          # deepest tree interpolates a straight line best
    assert pred.size == np.sum(roles == "test")
    assert row["val_MAE_h"] < 1


def test_tune_fixed_chooses_lowest_validation_error_and_caps():
    y = T[-1] - T
    roles = np.repeat(["train", "val", "test"], [300, 150, T.size - 450])
    preds = {"exact": y.copy(), "too long": y + 1000}
    row, pred = tune_fixed(preds, y, roles)
    assert row["settings"] == "exact" and row["MAE_h"] == 0
    capped, _ = tune_fixed({"too long": y + 1000}, y, roles)
    assert capped["late_%"] == 100 and capped["MAE_h"] <= y[:300].max()


def test_tune_and_test_clips_to_physical_range():
    X = T.reshape(-1, 1)
    y = T[-1] - T
    roles = np.repeat(["train", "val", "test"], [300, 150, T.size - 450])
    grid = [{"smooth": 1}]
    make = lambda: make_pipeline(StandardScaler(), LinearRegression())
    _, pred = tune_and_test(make, grid, {1: X + 0.01 * X ** 2}, y, roles)
    assert pred.min() >= 0 and pred.max() <= y[:300].max()
    _, raw = tune_and_test(make, grid, {1: X + 0.01 * X ** 2}, y, roles, clip=False)
    assert raw.min() < 0                  # unclipped straight line overshoots


@pytest.mark.parametrize("log", [False, True])
def test_fast_extrapolation_matches_loop_version(log):
    # rising and falling trend so every branch (crossing, no crossing, past threshold) is used
    y = 1 + 0.02 * T + 0.3 * np.sin(T / 3)
    for window in (2, 10, 40):
        slow = extrapolate_rul(T, y, threshold=2.5, window=window, log=log, max_rul=300)
        fast = extrapolate_rul_fast(T, y, threshold=2.5, window=window, log=log, max_rul=300)
        assert np.allclose(fast, slow, atol=1e-6)


def test_validation_curves_report_every_setting():
    X = T.reshape(-1, 1)
    y = T[-1] - T
    roles = np.repeat(["train", "val", "test"], [300, 150, T.size - 450])
    grid = [{"smooth": 1, "max_depth": d} for d in (1, 3)]
    curve = validation_curve(lambda max_depth: DecisionTreeRegressor(max_depth=max_depth),
                             grid, {1: X}, y, roles)
    assert list(curve["max_depth"]) == [1, 3] and np.all(curve["fit_ms"] >= 0)
    fixed = fixed_validation_curve({(1, 5): y, (1, 9): y + 10}, y, roles, ["smooth", "window_h"])
    assert list(fixed["val_MAE_h"]) == [0, 10]


def test_leave_one_out_never_uses_test_group_to_choose():
    # three "bearings" sharing the same exact RUL = 100 - 10 * feature relation
    groups = {g: np.linspace(0, 10, 50) + 0.01 * i for i, g in enumerate("abc")}
    features = {g: {1: x.reshape(-1, 1)} for g, x in groups.items()}
    y = {g: 100 - 10 * x for g, x in groups.items()}
    grid = [{"smooth": 1, "max_depth": d} for d in (1, 8)]
    row, pred = leave_one_out_by_group(lambda max_depth: DecisionTreeRegressor(max_depth=max_depth),
                                       grid, features, y, "c")
    assert row["max_depth"] == 8 and pred.size == 50 and row["MAE_h"] < 2
    fixed, _ = fixed_leave_one_out_by_group({"exact": y, "late": {g: v + 5 for g, v in y.items()}}, y, "c")
    assert fixed["settings"] == "exact" and fixed["MAE_h"] == 0


def test_safety_margin_meets_late_target():
    y = np.arange(100.0)
    pred = y + np.linspace(-20, 20, 100)       # half late before the margin
    m = safety_margin(y, pred, late_target=0.1)
    assert np.mean(apply_margin(pred, m) > y) <= 0.1
    assert safety_margin(y, y - 5) == 0         # already early: no margin
    assert apply_margin([3.0], 10)[0] == 0      # never negative


def test_inner_validation_skips_test_group():
    groups = {g: np.linspace(0, 10, 20) for g in "abc"}
    features = {g: {1: x.reshape(-1, 1)} for g, x in groups.items()}
    y = {g: 100 - 10 * x for g, x in groups.items()}
    true, pred = inner_validation_predictions(lambda: LinearRegression(), {"smooth": 1},
                                              features, y, "c")
    assert true.size == 40 and np.allclose(pred, true)
