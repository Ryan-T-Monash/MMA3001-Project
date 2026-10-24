"""
Remaining useful life (RUL) models for bearing 1 of IMS test 2, and the
baseline they are all compared against.

Engineering question: *how many hours does this bearing have left?*

Baseline (no sensor data)
    The catalogue rating life, as a maintenance planner would use it:
    $L_{10} = (C/P)^p \\times 10^6$ revolutions, with $p = 10/3$ for roller
    bearings (3 for ball bearings), converted to hours at the shaft speed.
    For the Rexnord ZA-2115, $C = 28\\,890$ lb. $L_{10}$ is a population
    statistic (90 % of bearings survive it), so a Weibull distribution
    converts it to other reliabilities, e.g. the median life $L_{50}$.
    `sklearn.dummy.DummyRegressor` (predict the mean training RUL, week 5.5)
    is a second, data-only baseline.

Candidate models (use the vibration data)
    * Trend extrapolation (week 3.1, least squares as in week 5.1): fit a
      straight line to the last `window` hours of the RMS trend and
      extrapolate it to a failure threshold.
    * Linear regression, decision tree and SVR (weeks 5.1 to 5.3) mapping
      smoothed RMS and kurtosis to RUL.

Tuning knobs (week 3.2, overfitting and noise)
    No noise is added to the data. Instead the real measurement noise is
    filtered by `trailing_mean` with an adjustable window (1 = raw), and
    flexibility is set by the extrapolation `window`, tree `max_depth` or
    SVR `C`. Too little filtering or too much flexibility follows the noise
    (low training error, high test error); too much makes the model sluggish.
    The smoother only looks backwards, because in service the future
    snapshots do not exist yet: a centred smoother would leak future
    information into the features (week 5.6, target leakage).

Evaluation (week 5.5)
    MAE and RMSE in hours, plus the percentage of *late* predictions
    (predicted RUL longer than the true RUL), which are the dangerous ones
    because maintenance would be scheduled after the failure. Settings are
    chosen on validation snapshots and the test snapshots are scored once
    (`tune_and_test`, `tune_fixed`); the splits themselves are in `splits`.
"""
import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

ZA2115_C_LB = 28890.0
"""Catalogue dynamic load rating of the Rexnord ZA-2115 (lb)."""


def catalogue_life_hours(C=ZA2115_C_LB, P=6000.0, rpm=2000.0, p=10 / 3):
    """Basic rating life L10 in hours.

    Parameters
    ----------
    C : float, optional
        Dynamic load rating (same units as `P`).
    P : float, optional
        Equivalent dynamic load on the bearing. IMS tests apply 6000 lb.
    rpm : float, optional
        Shaft speed in rev/min.
    p : float, optional
        Life exponent: 10/3 for roller bearings, 3 for ball bearings.

    Returns
    -------
    float
        Hours that 90 % of identical bearings are expected to exceed.
    """
    return (C / P) ** p * 1e6 / (60 * rpm)


def life_at_reliability(l10, reliability=0.5, beta=1.5):
    """Convert L10 to the life reached with another survival probability.

    Assumes a two-parameter Weibull life distribution,
    $L_R = L_{10} (\\ln(1/R) / \\ln(1/0.9))^{1/\\beta}$.

    Parameters
    ----------
    l10 : float
        Rating life (hours).
    reliability : float, optional
        Survival probability, 0.5 gives the median life L50.
    beta : float, optional
        Weibull slope, typically 1.1 to 1.5 for rolling bearings.

    Returns
    -------
    float
        Life in hours.
    """
    return l10 * (np.log(1 / reliability) / np.log(1 / 0.9)) ** (1 / beta)


def failure_probability(t, l10, beta=1.5):
    """Weibull probability that a bearing has failed by time `t`.

    Parameters
    ----------
    t : float or array_like
        Running time (hours).
    l10, beta : float
        As in `life_at_reliability`.

    Returns
    -------
    float or numpy.ndarray
        Probability between 0 and 1 (0.1 at ``t = l10``).
    """
    eta = l10 / np.log(1 / 0.9) ** (1 / beta)
    return 1 - np.exp(-(np.asarray(t, dtype=float) / eta) ** beta)


def catalogue_rul(t, life):
    """Baseline RUL prediction: rated life minus hours run, never negative.

    Parameters
    ----------
    t : array_like
        Hours run at each snapshot.
    life : float
        Rated life (e.g. L10 or L50) in hours.

    Returns
    -------
    numpy.ndarray
        Predicted RUL in hours.
    """
    return np.clip(life - np.asarray(t, dtype=float), 0, None)


def trailing_mean(y, window=6):
    """Mean of the current and previous ``window - 1`` values.

    Unlike `trends.moving_average` it uses no future snapshots, so it can be
    applied in service. ``window = 1`` returns the raw data.

    Parameters
    ----------
    y : array_like
        Indicator values in time order.
    window : int, optional
        Number of snapshots averaged (fewer at the start).

    Returns
    -------
    numpy.ndarray
        Filtered values, same length as `y`.
    """
    y = np.asarray(y, dtype=float)
    total = np.concatenate([[0.0], np.cumsum(y)])
    i = np.arange(1, y.size + 1)
    lo = np.maximum(i - window, 0)
    return (total[i] - total[lo]) / (i - lo)


def extrapolate_rul(t, y, threshold, window=10.0, log=False, max_rul=np.inf):
    """Predict RUL by extrapolating the recent trend to a failure threshold.

    At each snapshot a least-squares line is fitted to the values in the
    last `window` hours (only past data). Times are measured from the
    current snapshot, which keeps the fit well conditioned (week 2.2).

    Parameters
    ----------
    t : array_like
        Hours, increasing.
    y : array_like
        Health indicator, usually RMS (g).
    threshold : float
        Indicator value treated as failure.
    window : float, optional
        Hours of history fitted. Short follows noise, long reacts slowly.
    log : bool, optional
        Fit ``log(y)`` instead, i.e. exponential growth.
    max_rul : float, optional
        Prediction used when the trend is flat or falling (no crossing).

    Returns
    -------
    numpy.ndarray
        Predicted RUL in hours, between 0 and `max_rul`.
    """
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    if log:
        y, threshold = np.log(y), np.log(threshold)
    rul = np.full(t.size, float(max_rul))
    for i in range(t.size):
        past = (t <= t[i]) & (t > t[i] - window)
        if past.sum() < 3:
            continue
        slope, now = np.polyfit(t[past] - t[i], y[past], 1)
        if slope > 0:
            rul[i] = np.clip((threshold - now) / slope, 0, max_rul)
        elif now >= threshold:
            rul[i] = 0.0
    return rul


def extrapolate_rul_fast(t, y, threshold, window=10.0, log=False, max_rul=np.inf):
    """Same result as `extrapolate_rul`, in O(n) instead of O(n^2) time.

    `extrapolate_rul` runs a Python loop that builds a boolean mask over
    all n snapshots and calls `numpy.polyfit` at every snapshot, so its cost
    grows as n^2 (in practice the ~50 us overhead of each `polyfit` call
    dominates at IMS sizes). Here the least-squares sums come from running
    (cumulative) sums, vectorised over all snapshots, so each window costs
    a few subtractions:
    $b = S_{ty} / S_{tt}$ with $S_{tt} = \\sum t^2 - (\\sum t)^2/m$ and
    $S_{ty} = \\sum ty - \\sum t \\sum y / m$ over the m points in the window.

    Subtracting large running sums loses digits (catastrophic cancellation,
    week 2.2), so times are first centred on their mean, which keeps the
    sums small. `test_rul_models` checks the two versions agree to 1e-6 h.

    Parameters and return value are as in `extrapolate_rul`; `t` must be
    strictly increasing.
    """
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    if log:
        y, threshold = np.log(y), np.log(threshold)
    u = t - t.mean()                        # centred times, smaller sums
    zero = [0.0]
    S1, Su, Sy, Suu, Suy = (np.concatenate([zero, np.cumsum(v)])
                            for v in (np.ones_like(u), u, y, u * u, u * y))
    hi = np.arange(1, t.size + 1)           # window is snapshots lo .. hi-1
    lo = np.searchsorted(t, t - window, side="right")
    m, su, sy = S1[hi] - S1[lo], Su[hi] - Su[lo], Sy[hi] - Sy[lo]
    suu, suy = Suu[hi] - Suu[lo], Suy[hi] - Suy[lo]
    rul = np.full(t.size, float(max_rul))
    ok = m >= 3
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (suy - su * sy / m) / (suu - su * su / m)
        now = sy / m + slope * (u - su / m)  # fitted line at the current time
        rising = ok & (slope > 0)
        rul[rising] = np.clip((threshold - now[rising]) / slope[rising], 0, max_rul)
    rul[ok & (slope <= 0) & (now >= threshold)] = 0.0
    return rul


def rul_scores(y_true, y_pred):
    """Error metrics for RUL predictions.

    Parameters
    ----------
    y_true, y_pred : array_like
        True and predicted RUL (hours).

    Returns
    -------
    dict
        ``MAE_h`` and ``RMSE_h`` in hours, and ``late_%``: share of
        predictions longer than the true RUL (unsafe direction).
    """
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return {"MAE_h": np.mean(np.abs(err)),
            "RMSE_h": np.sqrt(np.mean(err ** 2)),
            "late_%": 100 * np.mean(err > 0)}


def candidate_models(depth=4, C=100.0):
    """The week 5 regression models, with their flexibility settings.

    Parameters
    ----------
    depth : int, optional
        Decision-tree maximum depth.
    C : float, optional
        SVR penalty (larger = more flexible).

    Returns
    -------
    dict
        Name -> unfitted scikit-learn estimator.
    """
    return {
        "mean (Dummy)": DummyRegressor(strategy="mean"),
        "linear": make_pipeline(StandardScaler(), LinearRegression()),
        f"tree (depth {depth})": DecisionTreeRegressor(max_depth=depth, random_state=0),
        f"SVR (C {C:g})": make_pipeline(StandardScaler(), SVR(C=C, epsilon=1.0)),
    }


def compare_models(models, X_train, y_train, X_test, y_test):
    """Fit each model and score it on training and test data.

    Parameters
    ----------
    models : dict
        Name -> unfitted estimator (see `candidate_models`).
    X_train, y_train, X_test, y_test : array_like
        Features and true RUL (hours).

    Returns
    -------
    pandas.DataFrame
        One row per model: training MAE, test scores from `rul_scores` and
        fitting time in milliseconds.
    predictions : dict
        Name -> predicted RUL on the test data.
    """
    rows, predictions = {}, {}
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_ms = 1000 * (time.perf_counter() - start)
        predictions[name] = model.predict(X_test)
        train_mae = np.mean(np.abs(model.predict(X_train) - np.asarray(y_train)))
        rows[name] = {"train_MAE_h": train_mae,
                      **rul_scores(y_test, predictions[name]),
                      "fit_ms": fit_ms}
    return pd.DataFrame(rows).T, predictions


def complexity_curve(make_model, values, X_train, y_train, X_test, y_test):
    """Training and test MAE as one setting is varied (overfitting plot).

    Parameters
    ----------
    make_model : callable
        ``make_model(value)`` returns an unfitted estimator.
    values : iterable
        Settings to try, e.g. tree depths.
    X_train, y_train, X_test, y_test : array_like
        As in `compare_models`.

    Returns
    -------
    pandas.DataFrame
        Indexed by setting, columns ``train_MAE_h`` and ``test_MAE_h``.
    """
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    rows = {}
    for v in values:
        model = make_model(v).fit(X_train, y_train)
        rows[v] = {"train_MAE_h": np.mean(np.abs(model.predict(X_train) - y_train)),
                   "test_MAE_h": np.mean(np.abs(model.predict(X_test) - y_test))}
    return pd.DataFrame(rows).T


def _masks(roles):
    roles = np.asarray(roles)
    return roles == "train", roles == "val", roles == "test"


def tune_and_test(make_model, grid, features, y, roles, clip=True):
    """Choose settings on validation data, then score once on test data.

    Each candidate is fitted on the ``"train"`` snapshots and its validation
    MAE recorded. Only the best candidate is applied to ``"test"``, so the
    test score played no part in the choice (week 5.5).

    Parameters
    ----------
    make_model : callable
        ``make_model(**settings)`` returns an unfitted estimator.
    grid : list of dict
        Candidate settings. Each has a ``"smooth"`` key (key of `features`)
        plus any keyword arguments for `make_model`.
    features : dict
        Smoothing window -> feature matrix for all snapshots.
    y : array_like
        True RUL (hours) for all snapshots.
    roles : array_like of str
        ``"train"``, ``"val"``, ``"test"`` or ``"gap"`` per snapshot
        (see `splits`).
    clip : bool, optional
        Physical constraint: keep predictions between 0 (RUL cannot be
        negative) and the longest RUL seen in training. Applied when
        choosing settings as well as when scoring.

    Returns
    -------
    dict
        Chosen settings, ``train_MAE_h``, ``val_MAE_h`` and the test scores
        from `rul_scores`.
    numpy.ndarray
        Predicted RUL for the test snapshots.
    """
    y = np.asarray(y, dtype=float)
    tr, va, te = _masks(roles)
    hi = y[tr].max() if clip else np.inf
    lo = 0.0 if clip else -np.inf

    def predict(model, X):
        return np.clip(model.predict(X), lo, hi)

    best = None
    for settings in grid:
        kwargs = {k: v for k, v in settings.items() if k != "smooth"}
        X = features[settings["smooth"]]
        model = make_model(**kwargs).fit(X[tr], y[tr])
        val_mae = np.mean(np.abs(predict(model, X[va]) - y[va]))
        if best is None or val_mae < best[0]:
            best = (val_mae, settings, model, X)
    val_mae, settings, model, X = best
    pred = predict(model, X[te])
    train_mae = np.mean(np.abs(predict(model, X[tr]) - y[tr]))
    return {**settings, "train_MAE_h": train_mae, "val_MAE_h": val_mae,
            **rul_scores(y[te], pred)}, pred


def tune_fixed(predictions, y, roles, cap=True):
    """`tune_and_test` for methods that need no fitting (catalogue, trend
    extrapolation): pick the precomputed prediction with the lowest
    validation MAE and score it on test.

    Parameters
    ----------
    predictions : dict
        Settings (tuple or label) -> predicted RUL for all snapshots.
    y : array_like
        True RUL (hours).
    roles : array_like of str
        Role per snapshot.
    cap : bool, optional
        Limit predictions to the longest RUL seen in training, so a flat
        trend (no predicted crossing) falls back to "as new".

    Returns
    -------
    dict
        ``settings``, ``val_MAE_h`` and the test scores.
    numpy.ndarray
        Predicted RUL for the test snapshots.
    """
    y = np.asarray(y, dtype=float)
    tr, va, te = _masks(roles)
    limit = y[tr].max() if cap else np.inf
    val_mae = {k: np.mean(np.abs(np.minimum(p[va], limit) - y[va]))
               for k, p in predictions.items()}
    key = min(val_mae, key=val_mae.get)
    pred = np.minimum(predictions[key][te], limit)
    return {"settings": key, "val_MAE_h": val_mae[key], **rul_scores(y[te], pred)}, pred


def validation_curve(make_model, grid, features, y, roles, clip=True):
    """Training and validation MAE for every candidate setting.

    The same fitting as `tune_and_test`, but every setting is reported, so
    the sensitivity of the error to each setting can be plotted. The test
    snapshots are not used.

    Parameters
    ----------
    make_model, grid, features, y, roles, clip
        As in `tune_and_test`.

    Returns
    -------
    pandas.DataFrame
        One row per setting: the setting values, ``train_MAE_h``,
        ``val_MAE_h`` and ``fit_ms`` (fitting time in milliseconds).
    """
    y = np.asarray(y, dtype=float)
    tr, va, _ = _masks(roles)
    hi = y[tr].max() if clip else np.inf
    lo = 0.0 if clip else -np.inf
    rows = []
    for settings in grid:
        kwargs = {k: v for k, v in settings.items() if k != "smooth"}
        X = features[settings["smooth"]]
        start = time.perf_counter()
        model = make_model(**kwargs).fit(X[tr], y[tr])
        fit_ms = 1000 * (time.perf_counter() - start)
        rows.append({**settings,
                     "train_MAE_h": np.mean(np.abs(np.clip(model.predict(X[tr]), lo, hi) - y[tr])),
                     "val_MAE_h": np.mean(np.abs(np.clip(model.predict(X[va]), lo, hi) - y[va])),
                     "fit_ms": fit_ms})
    return pd.DataFrame(rows)


def fixed_validation_curve(predictions, y, roles, names, cap=True):
    """Validation MAE for every precomputed prediction (see `tune_fixed`).

    Parameters
    ----------
    predictions : dict
        Settings tuple -> predicted RUL for all snapshots.
    y, roles, cap
        As in `tune_fixed`.
    names : list of str
        Column name for each element of the settings tuples,
        e.g. ``["smooth", "window_h", "fail_factor"]``.

    Returns
    -------
    pandas.DataFrame
        One row per setting with ``val_MAE_h``.
    """
    y = np.asarray(y, dtype=float)
    tr, va, _ = _masks(roles)
    limit = y[tr].max() if cap else np.inf
    return pd.DataFrame([{**dict(zip(names, k)),
                          "val_MAE_h": np.mean(np.abs(np.minimum(p[va], limit) - y[va]))}
                         for k, p in predictions.items()])
