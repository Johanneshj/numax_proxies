"""CoV scan: a power excess is significant at the envelope; white noise is not."""
import numpy as np

from numax_proxies.comb_scan import relative_power
from numax_proxies.cov_scan import cov_scan, significant
from numax_proxies.synthetic import synthetic_psd, white_noise_psd

GRID = np.arange(0.0079, 283.2, 0.0079)          # Kepler LC, 4 years


def test_excess_found():
    psd = synthetic_psd(100.0, 3.0, frequency=GRID, rng=np.random.default_rng(1))
    r = cov_scan(psd.frequency, relative_power(psd.frequency, psd.psd), 80.0)
    assert significant(r, 3.6) and abs(r["numax_cov"].n / 100.0 - 1) < 0.2


def test_white_noise_not_significant():
    psd = white_noise_psd(GRID, np.random.default_rng(2))
    r = cov_scan(psd.frequency, relative_power(psd.frequency, psd.psd), 100.0)
    assert not significant(r, 3.6)


def test_noise_level_of_the_cov():
    """z of the CoV of exponential noise has mean ~0 and sd ~1."""
    rng = np.random.default_rng(3)
    n = 500
    x = rng.exponential(size=(3000, n))
    z = (x.std(axis=1, ddof=1) / x.mean(axis=1) - 1 + 1 / n) * np.sqrt(n)
    assert abs(z.mean()) < 0.1 and abs(z.std() - 1) < 0.07


def test_cov_scan_does_not_change_the_methods():
    """The CoV scan (with its refits) is report-only: the estimates are the same with and without."""
    from numax_proxies import NumaxProxies
    psd = synthetic_psd(50.0, 3.0, rng=np.random.default_rng(4))
    est = {}
    for on in (False, True):
        p = NumaxProxies.from_psd(psd.frequency, psd.psd, {"CONFIG": {"cov_scan": on}})
        est[on] = (p.compute_numax_from_acf().n, p.compute_numax_from_CoV().n)
        if on:
            assert set(p.cov_scan_results) == {"ACF", "CoV"}
    assert est[True] == est[False]


def test_combined_estimate_is_the_mean():
    """numax_combined = mean of the finite seismic estimates, error = their spread."""
    from numax_proxies import NumaxProxies
    psd = synthetic_psd(50.0, 3.0, rng=np.random.default_rng(4))
    p = NumaxProxies.from_psd(psd.frequency, psd.psd, {"CONFIG": {"cov_scan": True}})
    p.compute_numax_from_acf()
    p.compute_numax_from_CoV()
    est = p.numax_estimates
    keys = [k for k in p.COMBINED_FROM if np.isfinite(est[k].n)]
    x = np.array([est[k].n for k in keys])
    assert len(keys) >= 2 and list(est)[-1] == "numax_combined"
    assert np.isclose(est["numax_combined"].n, x.mean()) and np.isclose(est["numax_combined"].s, x.std(ddof=1))
