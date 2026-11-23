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
    # the raw comb-scan centre sits ~5-15 % above numax (envelope/background bias; removed by
    # CONFIG.background_correction in the pipeline)
    assert r["peak_z"] > 10 and abs(r["numax_comb"].n / 100.0 - 1) < 0.2


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
    """z of white-noise windows (line removed) has mean ~0 and sd ~1, for every profile and
    for a narrow band half-width."""
    from numax_proxies.comb_scan import PROFILES, _autocorr, _collapse, null_mean_sd, profile_bands
    rng = np.random.default_rng(5)
    n = 1000
    d_pts = np.array([180, 250, 330])
    bands = {(name, hw): profile_bands(d_pts, name, hw) for name in PROFILES for hw in (0.1, 0.04)}
    null = {key: null_mean_sd(n, d_pts, lo, hi, key[0], w, key[1]) for key, (lo, hi, w) in bands.items()}
    z = {key: [] for key in bands}
    for _ in range(1500):
        r = _autocorr(rng.exponential(size=n))
        for key, (lo, hi, w) in bands.items():
            z[key].append((_collapse(r, lo, hi, w) - null[key][0]) / null[key][1])
    for name, v in z.items():
        v = np.array(v)
        assert np.all(np.abs(v.mean(0)) < 0.1) and np.all(np.abs(v.std(0) - 1) < 0.07), name


def test_fit_takes_peak_not_dip():
    """A dip at the tested numax and a peak elsewhere: the fit must take the peak."""
    from numax_proxies.comb_scan import _fit
    from numax_proxies.seismic import envelope_fwhm
    x = np.linspace(250, 1000, 120)
    z = 8 * np.exp(-0.5 * ((x - 550) / 40) ** 2) - 3 * np.exp(-0.5 * ((x - 720) / 50) ** 2)
    p, _ = _fit(x, z, 640.0, float(envelope_fwhm(640.0)))
    assert p is not None and p[0] > 0 and abs(p[2] - 550) < 15


def test_excess_peak_finds_the_envelope():
    """The smoothed-power maximum lands on the envelope, also from a wrong start."""
    from numax_proxies.comb_scan import excess_peak
    psd = synthetic_psd(100.0, 3.0, frequency=GRID, rng=np.random.default_rng(1))
    rel = relative_power(psd.frequency, psd.psd)
    for start in (100.0, 80.0):
        assert abs(excess_peak(psd.frequency, rel, start, 2.0) / 100.0 - 1) < 0.15
