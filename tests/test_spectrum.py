"""Tests for spectrum.py. Run with: pytest test_spectrum.py"""
import numpy as np
import pytest

from spectrum import amplitude_spectrum, band_peak, defect_frequencies


def test_defect_frequencies_match_ims_values():
    f = defect_frequencies()
    assert f["BPFO"] == pytest.approx(236.4, abs=0.5)
    assert f["BPFI"] == pytest.approx(296.9, abs=0.5)
    assert f["BSF"] == pytest.approx(139.9, abs=0.5)
    assert f["FTF"] == pytest.approx(14.8, abs=0.1)


def test_sine_gives_peak_at_its_frequency_and_amplitude():
    fs = 20000
    t = np.arange(20480) / fs
    x = 0.5 * np.sin(2 * np.pi * 236.4 * t) + 3.0   # offset should be removed
    freqs, amp = amplitude_spectrum(x, fs)
    assert freqs[np.argmax(amp)] == pytest.approx(236.4, abs=fs / len(x))
    assert band_peak(freqs, amp, 236.4) == pytest.approx(0.5, rel=0.05)


def test_band_peak_ignores_energy_outside_band():
    freqs = np.arange(0, 1000, 1.0)
    amp = np.zeros_like(freqs)
    amp[100], amp[500] = 1.0, 9.0
    assert band_peak(freqs, amp, 100) == 1.0
