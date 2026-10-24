"""
Smoothing and labelling of condition-indicator trends (RMS, kurtosis).

Features computed once per snapshot are noisy: two snapshots ten minutes
apart can differ a lot even when the bearing has not changed. Following the
week 3 notes, we treat each trend as *measurement = true trend + noise* and
compare three ways of estimating the true trend:

* Moving average - the simplest local smoother (baseline).
* Smoothing spline - minimises
  $\\sum_i (y_i - f(t_i))^2 + \\lambda \\int f''(t)^2\\,dt$,
  a penalised cubic spline. $\\lambda = 0$ is exact interpolation
  (overfits the noise), large $\\lambda$ tends to a straight line (underfits).
* K-nearest neighbours regression - predicts each point as the mean of the
  $k$ nearest snapshots in time, i.e. local interpolation. $k = 1$ passes
  through every point (overfits), large $k$ blurs the failure (underfits).

The smoothing level ($\\lambda$ or $k$) is chosen by validation error on
held-out snapshots, and the smoothed RMS is then turned into health-stage
labels and remaining useful life for the week 5 regression models.

Typical use::

    from trends import knn_smooth, choose_k, health_stages
    hours = features.index * 10 / 60
    k, scores = choose_k(hours, features["rms_1"])
    smooth = knn_smooth(hours, features["rms_1"], k)
    stage = health_stages(smooth)
"""
import numpy as np
from scipy.interpolate import make_smoothing_spline
from sklearn.neighbors import KNeighborsRegressor


def moving_average(y, window=11):
    """Centred moving average.

    Parameters
    ----------
    y : array_like
        Trend values, one per snapshot.
    window : int, optional
        Number of points averaged (odd keeps it centred). The ends use as
        many points as are available.

    Returns
    -------
    numpy.ndarray
        Smoothed values, same length as `y`.
    """
    y = np.asarray(y, dtype=float)
    kernel = np.ones(window)
    total = np.convolve(y, kernel, mode="same")
    count = np.convolve(np.ones_like(y), kernel, mode="same")
    return total / count


def spline_smooth(t, y, lam=None):
    """Fit a smoothing spline and evaluate it at the data times.

    Parameters
    ----------
    t : array_like
        Strictly increasing times (e.g. hours).
    y : array_like
        Trend values.
    lam : float, optional
        Smoothness penalty. ``None`` lets SciPy choose it by generalised
        cross-validation (GCV).

    Returns
    -------
    numpy.ndarray
        Smoothed values at `t`.
    """
    t = np.asarray(t, dtype=float)
    return make_smoothing_spline(t, np.asarray(y, dtype=float), lam=lam)(t)


def knn_smooth(t, y, k=15, t_new=None):
    """K-nearest neighbours regression with time as the only feature.

    Parameters
    ----------
    t, y : array_like
        Training times and trend values.
    k : int, optional
        Number of neighbours averaged.
    t_new : array_like, optional
        Times to predict at. Defaults to `t` (smooth the data itself).

    Returns
    -------
    numpy.ndarray
        Predicted values at `t_new`.
    """
    t = np.asarray(t, dtype=float).reshape(-1, 1)
    t_new = t if t_new is None else np.asarray(t_new, dtype=float).reshape(-1, 1)
    model = KNeighborsRegressor(n_neighbors=k).fit(t, np.asarray(y, dtype=float))
    return model.predict(t_new)


def choose_k(t, y, ks=range(1, 61), every=5):
    """Pick `k` by training vs validation error (week 3.2 style).

    Every `every`-th snapshot is held out for validation and the rest are
    used to fit. Because held-out points sit between training points, this
    tests *interpolation* of the trend, not forecasting.

    Parameters
    ----------
    t, y : array_like
        Times and trend values.
    ks : iterable of int, optional
        Values of k to try.
    every : int, optional
        Hold out one point in every `every`.

    Returns
    -------
    best_k : int
        The k with the lowest validation RMSE.
    scores : dict
        ``{"k": [...], "train": [...], "val": [...]}`` RMSE for plotting.
    """
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    val = np.arange(len(t)) % every == 0
    scores = {"k": [], "train": [], "val": []}
    for k in ks:
        pred_tr = knn_smooth(t[~val], y[~val], k)
        pred_val = knn_smooth(t[~val], y[~val], k, t_new=t[val])
        scores["k"].append(k)
        scores["train"].append(np.sqrt(np.mean((pred_tr - y[~val]) ** 2)))
        scores["val"].append(np.sqrt(np.mean((pred_val - y[val]) ** 2)))
    return scores["k"][int(np.argmin(scores["val"]))], scores


def health_stages(smooth, baseline=100, n_sigma=3, fail_factor=3):
    """Label each snapshot healthy (0), degrading (1) or failing (2).

    Thresholds come from the first `baseline` snapshots, assumed healthy:

    * degrading once the value exceeds baseline mean + `n_sigma` std,
    * failing once it exceeds `fail_factor` x baseline mean.

    Stages never go back down (damage does not heal), so a brief dip after
    the threshold is crossed keeps the higher label.

    Parameters
    ----------
    smooth : array_like
        Smoothed indicator, usually RMS.
    baseline : int, optional
        Number of initial snapshots used as the healthy reference.
    n_sigma, fail_factor : float, optional
        Threshold settings described above.

    Returns
    -------
    numpy.ndarray of int
        Stage label per snapshot.
    """
    smooth = np.asarray(smooth, dtype=float)
    ref = smooth[:baseline]
    stage = np.zeros(len(smooth), dtype=int)
    stage[smooth > ref.mean() + n_sigma * ref.std()] = 1
    stage[smooth > fail_factor * ref.mean()] = 2
    return np.maximum.accumulate(stage)


def remaining_life(t):
    """Remaining useful life: time left until the last snapshot.

    Parameters
    ----------
    t : array_like
        Snapshot times (e.g. hours). The test ends at failure.

    Returns
    -------
    numpy.ndarray
        ``t[-1] - t``, in the units of `t`.
    """
    t = np.asarray(t, dtype=float)
    return t[-1] - t
