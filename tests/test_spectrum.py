import numpy as np
import pytest

from numax_proxies.data import LightCurveData
from numax_proxies.spectrum import periodogram

DT_S = 1765.5   # Kepler long cadence


def white_noise_lc(rng, sigma=100.0, days=400.0, gaps=False):
    t = np.arange(0, days, DT_S / 86400)
    if gaps:   # remove 20 % of the points in blocks
        t = t[(t % 50) < 40]
    return LightCurveData(t, rng.normal(0, sigma, t.size), np.full(t.size, sigma))


@pytest.mark.parametrize("gaps", [False, True])
def test_psd_is_in_ppm2_per_muhz(rng, gaps):
    """White noise: one-sided PSD level 2 sigma^2 dt; Parseval: sum(P) df = variance."""
    sigma = 100.0
    lc = white_noise_lc(rng, sigma, gaps=gaps)
    psd = periodogram(lc)
    df = psd.frequency[1] - psd.frequency[0]
    assert np.sum(psd.psd) * df == pytest.approx(np.var(lc.flux), rel=1e-10)
    duty = lc.time.size * DT_S / ((lc.time[-1] - lc.time[0]) * 86400)
    expected = 2 * sigma ** 2 * DT_S * 1e-6 / duty
    assert np.mean(psd.psd) == pytest.approx(expected, rel=0.05)


def test_weighted_periodogram_is_old_normalisation(rng):
    psd = periodogram(white_noise_lc(rng), weighted=True)
    assert np.mean(psd.psd) == pytest.approx(1.0, rel=0.05)   # (flux / flux_err)^2


def test_frequency_grid_in_muhz(rng):
    psd = periodogram(white_noise_lc(rng))
    nyquist_muhz = 1e6 / (2 * DT_S)
    assert psd.frequency[-1] < nyquist_muhz and psd.frequency[-1] > 0.99 * nyquist_muhz
