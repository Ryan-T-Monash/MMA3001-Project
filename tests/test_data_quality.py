"""Tests for data_quality.py. Run with: pytest test_data_quality.py"""
import numpy as np

from data_quality import file_times, quality_report, snapshot_quality, time_gaps


def test_time_gaps_finds_one_long_gap():
    names = ["2004.02.12.10.00.00", "2004.02.12.10.10.00", "2004.02.12.10.20.00",
             "2004.02.12.12.20.00", "2004.02.12.12.30.00"]
    gaps = time_gaps(file_times(names))
    assert len(gaps) == 1
    assert gaps["after_sample"].iloc[0] == 2
    assert gaps["gap_minutes"].iloc[0] == 120


def test_clean_snapshot_passes():
    data = np.random.default_rng(0).normal(size=(20480, 4))
    q = snapshot_quality(data)
    assert not q["wrong_length"]
    assert q["n_nan"] == 0 and q["flat_channels"] == 0
    assert q["max_at_peak"] == 1


def test_problems_are_detected():
    data = np.random.default_rng(0).normal(size=(100, 4))
    data[:, 1] = 0.5            # flat sensor
    data[5, 2] = np.nan         # missing value
    data[:20, 3] = 10.0         # clipped at +10
    q = snapshot_quality(data)
    assert q["wrong_length"]
    assert q["flat_channels"] == 1
    assert q["n_nan"] == 1
    assert q["max_at_peak"] == 100   # the flat channel sits at its peak everywhere


def test_quality_report_flags_unreadable_file(tmp_path):
    good = tmp_path / "2004.02.12.10.32.39"
    np.savetxt(good, np.ones((10, 4)), delimiter="\t")
    bad = tmp_path / "2004.02.12.10.42.39"
    bad.write_text("1\t2\t3\t4\n1\t2\n")  # cut off mid-row
    report = quality_report([str(good), str(bad)])
    assert list(report["unreadable"]) == [False, True]
