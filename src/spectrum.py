"""
Frequency analysis for the NASA IMS bearing vibration dataset.

A local defect on a bearing produces an impact every time a rolling element
passes it, so the vibration contains energy at a characteristic *defect
frequency* that depends only on the bearing geometry and shaft speed:

* BPFO, ball pass frequency outer race,
  $\\frac{n f_r}{2}\\left(1 - \\frac{d}{D}\\cos\\phi\\right)$
* BPFI, ball pass frequency inner race,
  $\\frac{n f_r}{2}\\left(1 + \\frac{d}{D}\\cos\\phi\\right)$
* BSF, ball spin frequency,
  $\\frac{D f_r}{2 d}\\left(1 - \\left(\\frac{d}{D}\\cos\\phi\\right)^2\\right)$
* FTF, cage (fundamental train) frequency,
  $\\frac{f_r}{2}\\left(1 - \\frac{d}{D}\\cos\\phi\\right)$

where $n$ is the number of rolling elements, $f_r$ the shaft frequency (Hz),
$d$ the roller diameter, $D$ the pitch diameter and $\\phi$ the contact angle.
Defaults are the IMS test rig: Rexnord ZA-2115 bearings at 2000 rpm.

Typical use::

    from bearing_features import load_snapshot
    from spectrum import amplitude_spectrum, defect_frequencies
    freqs, amp = amplitude_spectrum(load_snapshot(path)[:, 0])
    defect_frequencies()   # {'BPFO': 236.4, 'BPFI': 296.9, ...}
"""
import numpy as np

# The IMS readme says "20 kHz", but each file holds 20,480 samples per 1 s
# snapshot. With 20,480 Hz the measured outer race peak (and its harmonics)
# lands on the theoretical BPFO; with 20,000 Hz it sits 2.4 % low.
SAMPLE_RATE = 20480  # Hz


def defect_frequencies(rpm=2000, n_rollers=16, roller_d=0.331, pitch_d=2.815, contact_deg=15.17):
    """Theoretical bearing defect frequencies.

    Parameters
    ----------
    rpm : float, optional
        Shaft speed in revolutions per minute.
    n_rollers : int, optional
        Number of rolling elements per row.
    roller_d, pitch_d : float, optional
        Roller and pitch diameters (any length unit, only the ratio matters).
    contact_deg : float, optional
        Contact angle in degrees.

    Returns
    -------
    dict
        ``BPFO``, ``BPFI``, ``BSF`` and ``FTF`` in Hz.
    """
    fr = rpm / 60
    r = roller_d / pitch_d * np.cos(np.radians(contact_deg))
    return {
        "BPFO": n_rollers * fr / 2 * (1 - r),
        "BPFI": n_rollers * fr / 2 * (1 + r),
        "BSF": pitch_d * fr / (2 * roller_d) * (1 - r**2),
        "FTF": fr / 2 * (1 - r),
    }


def amplitude_spectrum(x, fs=SAMPLE_RATE):
    """One-sided amplitude spectrum of a signal using a Hann window.

    The mean is removed first, and the window reduces spectral leakage
    (energy smearing into neighbouring bins because the record does not
    contain a whole number of cycles).

    Parameters
    ----------
    x : array_like
        1-D signal, e.g. one channel of a snapshot (g).
    fs : float, optional
        Sampling rate in Hz.

    Returns
    -------
    freqs : numpy.ndarray
        Frequencies in Hz, from 0 to fs/2. Resolution is fs / len(x).
    amp : numpy.ndarray
        Amplitude at each frequency, same units as `x`. A sine of amplitude
        A gives a peak of about A.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    w = np.hanning(len(x))
    amp = 2 * np.abs(np.fft.rfft(x * w)) / w.sum()
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)
    return freqs, amp


def band_peak(freqs, amp, f0, width=5):
    """Largest amplitude within a band around a target frequency.

    Useful for tracking a defect frequency over time, since the measured
    peak can sit a few Hz from the theoretical value (slip, speed changes).

    Parameters
    ----------
    freqs, amp : numpy.ndarray
        Output of `amplitude_spectrum`.
    f0 : float
        Centre of the band in Hz, e.g. BPFO.
    width : float, optional
        Half-width of the band in Hz.

    Returns
    -------
    float
        Peak amplitude in the band ``[f0 - width, f0 + width]``.
    """
    band = (freqs >= f0 - width) & (freqs <= f0 + width)
    return float(amp[band].max())
