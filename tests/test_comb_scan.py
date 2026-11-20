"""Comb scan: a comb gives a positive Gaussian at the envelope, noise does not."""
import numpy as np

from numax_proxies.comb_scan import comb_scan, relative_power
from numax_proxies.synthetic import synthetic_psd, white_noise_psd

GRID = np.arange(0.0079, 283.2, 0.0079)          # Kepler LC, 4 years


def scan(psd, numax, seed=0):
    rel = relative_power(psd.frequency, psd.psd)
    return comb_scan(psd.frequency, rel, numax)


def test_comb_found_at_envelope():
    psd = synthetic_psd(100.0, 3.0, frequency=GRID, rng=np.random.default_rng(1))
    r = scan(psd, 100.0)
    assert r["peak_z"] > 10 and abs(r["offset"]) < 0.15


def test_wrong_numax_points_to_envelope():
    psd = synthetic_psd(100.0, 3.0, frequency=GRID, rng=np.random.default_rng(1))
    r = scan(psd, 80.0)
    assert r["peak_z"] > 10 and abs(r["numax_comb"].n / 100.0 - 1) < 0.15


def test_white_noise_no_strong_peak():
    psd = white_noise_psd(GRID, np.random.default_rng(2))
    r = scan(psd, 100.0)
    assert np.isnan(r["peak_z"]) or r["peak_z"] < 6


def test_recentre_moves_to_envelope():
    from numax_proxies.comb_scan import recentre
    psd = synthetic_psd(100.0, 3.0, frequency=GRID, rng=np.random.default_rng(1))
    rel = relative_power(psd.frequency, psd.psd)
    r = recentre(psd.frequency, rel, 75.0)
    assert r["status"] in ("converged", "not_converged")
    assert abs(r["numax"].n / 100.0 - 1) < 0.15


def test_recentre_rejects_white_noise():
    from numax_proxies.comb_scan import recentre
    psd = white_noise_psd(GRID, np.random.default_rng(2))
    rel = relative_power(psd.frequency, psd.psd)
    r = recentre(psd.frequency, rel, 100.0)
    assert r["status"] == "rejected" and np.isnan(r["numax"].n)


def test_null_is_calibrated():
    """z of white-noise windows (line removed) has mean ~0 and sd ~1."""
    from numax_proxies.comb_scan import HALF, _autocorr, _collapse, null_mean_sd
    rng = np.random.default_rng(5)
    n = 1000
    d_pts = np.array([250, 330, 420])
    lo = np.array([np.ceil((m - HALF) * d_pts) for m in (0.5, 1.0)]).astype(int)
    hi = np.array([np.floor((m + HALF) * d_pts) for m in (0.5, 1.0)]).astype(int)
    mu, sd = null_mean_sd(n, d_pts, lo, hi)
    z = np.array([(_collapse(_autocorr(rng.exponential(size=n)), lo, hi) - mu) / sd
                  for _ in range(1500)])
    assert np.all(np.abs(z.mean(0)) < 0.1) and np.all(np.abs(z.std(0) - 1) < 0.07)
