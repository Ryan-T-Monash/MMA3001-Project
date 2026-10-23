"""
Data quality checks for the NASA IMS bearing vibration dataset.

Run these before feature extraction or modelling to find problems in the raw
snapshots:

* time gaps - missing snapshots, or the test being stopped and restarted;
* wrong length - files without the expected 20,480 samples (1 s at 20 kHz);
* missing values - NaNs in the signal;
* flat channels - a sensor that stopped responding (zero variance);
* clipping - many samples stuck at the peak value, suggesting the sensor or
  data acquisition hit its range limit.

Typical use::

    from bearing_features import list_snapshot_files
    from data_quality import file_times, time_gaps, quality_report
    files = list_snapshot_files(".../IMS/2nd_test")
    gaps = time_gaps(file_times(files))
    report = quality_report(files, step=10)
"""
import os

import numpy as np
import pandas as pd

from bearing_features import load_snapshot

SAMPLES_PER_SNAPSHOT = 20480  # 1 s at 20 kHz


def file_times(files):
    """Parse snapshot timestamps from IMS file names.

    Parameters
    ----------
    files : list of str
        Snapshot paths or names such as ``2004.02.12.10.32.39``.

    Returns
    -------
    pandas.Series
        Datetimes, one per file, in the same order as `files`.

    Raises
    ------
    ValueError
        If a file name is not a timestamp in the IMS format.
    """
    names = [os.path.basename(f) for f in files]
    return pd.Series(pd.to_datetime(names, format="%Y.%m.%d.%H.%M.%S"))


def time_gaps(times, factor=1.5):
    """Find gaps between snapshots that are longer than usual.

    Parameters
    ----------
    times : pandas.Series
        Snapshot datetimes in time order, e.g. from `file_times`.
    factor : float, optional
        A gap is flagged when it exceeds `factor` times the median interval.

    Returns
    -------
    pandas.DataFrame
        One row per gap with columns ``after_sample`` (index of the snapshot
        before the gap), ``start``, ``end`` and ``gap_minutes``. Empty if
        there are no gaps.
    """
    dt = times.diff().dt.total_seconds() / 60
    usual = dt.median()
    idx = dt.index[dt > factor * usual]
    return pd.DataFrame({
        "after_sample": idx - 1,
        "start": times[idx - 1].values,
        "end": times[idx].values,
        "gap_minutes": dt[idx].values,
    })


def snapshot_quality(data, expected_samples=SAMPLES_PER_SNAPSHOT):
    """Check one snapshot for common data problems.

    Parameters
    ----------
    data : numpy.ndarray
        2-D array of shape (n_samples, n_channels), as from `load_snapshot`.
    expected_samples : int, optional
        Number of rows a complete snapshot should have.

    Returns
    -------
    dict
        ``n_samples``, ``n_channels``, ``wrong_length`` (bool), ``n_nan``
        (count of NaN values), ``flat_channels`` (count of channels with zero
        variance) and ``max_at_peak`` (largest number of samples sitting at a
        channel's peak absolute value; a high count suggests clipping).
    """
    absx = np.abs(data)
    peaks = np.nanmax(absx, axis=0)
    return {
        "n_samples": data.shape[0],
        "n_channels": data.shape[1],
        "wrong_length": data.shape[0] != expected_samples,
        "n_nan": int(np.isnan(data).sum()),
        "flat_channels": int((np.nanstd(data, axis=0) == 0).sum()),
        "max_at_peak": int((absx == peaks).sum(axis=0).max()),
    }


def quality_report(files, step=1):
    """Run `snapshot_quality` over a test's files.

    Parameters
    ----------
    files : list of str
        Snapshot paths in time order, e.g. from `list_snapshot_files`.
    step : int, optional
        Check every `step`-th file (the last file is always checked).
        Use e.g. 10 for a quick look; 1 checks everything.

    Returns
    -------
    pandas.DataFrame
        One row per checked file, indexed by ``sample``, with a ``file``
        column and the `snapshot_quality` columns. Files that cannot be
        read have ``unreadable`` set to True and the other checks blank.
    """
    picks = sorted(set(range(0, len(files), step)) | {len(files) - 1})
    rows = []
    for i in picks:
        row = {"sample": i, "file": os.path.basename(files[i]), "unreadable": False}
        try:
            row.update(snapshot_quality(load_snapshot(files[i])))
        except ValueError:  # truncated or corrupt file
            row["unreadable"] = True
        rows.append(row)
    return pd.DataFrame(rows).set_index("sample")
