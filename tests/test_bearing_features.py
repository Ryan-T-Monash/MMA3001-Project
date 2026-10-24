"""Tests for bearing_features.py. Run with: pytest test_bearing_features.py"""
import numpy as np
import pytest

from bearing_features import (extract_features, kurtosis, list_snapshot_files, peak, rms,
                              snapshot_features, snapshot_features_fast)


def test_rms_of_sine_is_amplitude_over_root2():
    t = np.linspace(0, 1, 20480, endpoint=False)
    assert rms(np.sin(2 * np.pi * 50 * t)) == pytest.approx(1 / np.sqrt(2), rel=1e-6)


def test_kurtosis_of_gaussian_noise_is_about_3():
    x = np.random.default_rng(0).normal(size=100_000)
    assert kurtosis(x) == pytest.approx(3, abs=0.05)


def test_kurtosis_of_constant_signal_is_nan():
    assert np.isnan(kurtosis(np.ones(10)))


def test_peak_uses_absolute_value():
    assert peak([0.1, -2.5, 1.0]) == 2.5


def test_extract_features_on_fake_folder(tmp_path):
    for name in ["2004.02.12.10.32.39", "2004.02.12.10.42.39", "2004.02.12.10.42.39 (1)"]:
        np.savetxt(tmp_path / name, np.random.default_rng(1).normal(size=(100, 4)), delimiter="\t")
    files = list_snapshot_files(str(tmp_path))
    features = extract_features(files)
    assert len(features) == 2  # Drive duplicate skipped
    assert list(features.columns[:4]) == ["file", "rms_1", "kurtosis_1", "peak_1"]


def test_missing_folder_raises():
    with pytest.raises(FileNotFoundError):
        list_snapshot_files("/no/such/folder")


def test_truncated_file_is_skipped(tmp_path):
    good = tmp_path / "2004.02.12.10.32.39"
    np.savetxt(good, np.ones((10, 8)), delimiter="\t")
    bad = tmp_path / "2004.02.12.10.42.39"
    bad.write_text("1\t2\t3\t4\t5\t6\t7\t8\n1\t2\n")  # cut off mid-row
    features = extract_features(list_snapshot_files(str(tmp_path)))
    assert list(features["file"]) == [good.name]



def test_fast_features_match_reference():
    data = np.random.default_rng(3).normal(size=(2048, 4))
    data[:, 3] = 0.5                       # constant channel: kurtosis undefined
    slow, fast = snapshot_features(data), snapshot_features_fast(data)
    assert slow.keys() == fast.keys()
    for k in slow:
        assert fast[k] == pytest.approx(slow[k], rel=1e-12, nan_ok=True)
