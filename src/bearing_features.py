"""
Feature extraction for the NASA IMS bearing vibration dataset.

Each IMS test folder (``1st_test``, ``2nd_test``, ``3rd_test``) holds one ASCII
file per snapshot, named by its timestamp (e.g. ``2004.02.12.10.32.39``). A
snapshot is a 1 second recording of 20,480 samples, with one column per
accelerometer channel (4 channels in tests 2 and 3, 8 in test 1).

This module reduces each snapshot to a few condition indicators per channel:

* RMS, $\\sqrt{\\frac{1}{N}\\sum x_i^2}$ - overall vibration energy.
* Kurtosis, $\\frac{E[(x-\\mu)^4]}{\\sigma^4}$ - impulsiveness (about 3 when healthy).
* Peak, $\\max |x_i|$ - largest single impact.

Typical use::

    from bearing_features import list_snapshot_files, extract_features
    files = list_snapshot_files("/content/drive/MyDrive/.../IMS/2nd_test")
    features = extract_features(files)
"""
import os

import numpy as np
import pandas as pd


def list_snapshot_files(folder):
    """List the snapshot files in an IMS test folder in time order.

    Duplicate copies created by Google Drive (names containing " (") are skipped.

    Parameters
    ----------
    folder : str
        Path to a test folder, e.g. ``.../IMS/2nd_test``.

    Returns
    -------
    list of str
        Full paths to the snapshot files, sorted by timestamp.

    Raises
    ------
    FileNotFoundError
        If `folder` does not exist or contains no files.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    names = sorted(f for f in os.listdir(folder) if " (" not in f)  # timestamp names sort in time order
    if not names:
        raise FileNotFoundError(f"No snapshot files in: {folder}")
    return [os.path.join(folder, f) for f in names]


def load_snapshot(path):
    """Load one IMS snapshot file.

    Parameters
    ----------
    path : str
        Path to a tab-separated snapshot file.

    Returns
    -------
    numpy.ndarray
        2-D array of shape (n_samples, n_channels) of acceleration values (g).

    Raises
    ------
    ValueError
        If the file is empty or not numeric.
    """
    data = np.loadtxt(path, ndmin=2)
    if data.size == 0:
        raise ValueError(f"Empty snapshot file: {path}")
    return data


def rms(x):
    """Root-mean-square of a signal.

    Parameters
    ----------
    x : array_like
        1-D signal.

    Returns
    -------
    float
        RMS value; same units as `x`.
    """
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x**2)))


def kurtosis(x):
    """Kurtosis (non-excess) of a signal.

    A Gaussian signal gives 3; impulsive signals from bearing defects give more.

    Parameters
    ----------
    x : array_like
        1-D signal.

    Returns
    -------
    float
        Kurtosis value (dimensionless). NaN if the signal is constant.
    """
    x = np.asarray(x, dtype=float)
    var = np.var(x)
    if var == 0:
        return float("nan")  # undefined for a constant signal
    return float(np.mean((x - x.mean())**4) / var**2)


def peak(x):
    """Peak absolute value of a signal.

    Parameters
    ----------
    x : array_like
        1-D signal.

    Returns
    -------
    float
        Largest absolute sample; same units as `x`.
    """
    return float(np.max(np.abs(np.asarray(x, dtype=float))))


def snapshot_features(data):
    """Compute RMS, kurtosis and peak for every channel of one snapshot.

    Parameters
    ----------
    data : numpy.ndarray
        2-D array of shape (n_samples, n_channels), as from `load_snapshot`.

    Returns
    -------
    dict
        Keys ``rms_k``, ``kurtosis_k``, ``peak_k`` for channel k = 1..n_channels.
    """
    row = {}
    for ch in range(data.shape[1]):
        x = data[:, ch]
        row[f"rms_{ch+1}"] = rms(x)
        row[f"kurtosis_{ch+1}"] = kurtosis(x)
        row[f"peak_{ch+1}"] = peak(x)
    return row


def extract_features(files, save_to=None):
    """Build a feature table from a list of snapshot files.

    Parameters
    ----------
    files : list of str
        Snapshot paths in time order, e.g. from `list_snapshot_files`.
    save_to : str, optional
        If given, the table is also written to this CSV path.

    Returns
    -------
    pandas.DataFrame
        One row per snapshot, indexed by ``sample`` (0, 1, 2, ...), with a
        ``file`` column and the feature columns from `snapshot_features`.

    Raises
    ------
    ValueError
        If `files` is empty.
    """
    if not files:
        raise ValueError("No files given")
    rows = []
    for i, path in enumerate(files):
        try:
            data = load_snapshot(path)
        except ValueError as err:  # truncated or corrupt file: skip it, keep going
            print(f"Skipped {os.path.basename(path)}: {err}")
            continue
        row = {"sample": i, "file": os.path.basename(path)}
        row.update(snapshot_features(data))
        rows.append(row)
        
    features = pd.DataFrame(rows).set_index("sample")
    if save_to:
        features.to_csv(save_to)
    return features
