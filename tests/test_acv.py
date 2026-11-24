"""ACV method: mean |autocovariance| per bin minus its white-noise mean."""
import numpy as np

from numax_proxies.significance import acf_noise_mean, load_tables, table_threshold
from numax_proxies.synthetic import synthetic_psd


def test_acv_table():
    t = load_tables()
    assert t["ACV"]["95"].shape == t["N"].shape == t["ACV"]["mu0"].shape
    assert np.all(t["ACV"]["99"] > t["ACV"]["95"])
    assert np.all(np.diff(t["ACV"]["mu0"]) < 0)


def test_acv_white_noise_calibration():
    """mean|ACV| - mu0 has mean ~0 on white noise and exceeds the 99 % threshold ~1 % of the time."""
    from numax_proxies.methods.acf import acv_in_windows

    rng = np.random.default_rng(8)
    for N in (12, 50, 700):
        n_bins = 20000
        x = rng.exponential(size=N * n_bins) / np.log(2) - 1
        i0 = np.arange(n_bins) * N
        stat = acv_in_windows(x, i0, i0 + N) - acf_noise_mean(np.full(n_bins, N), "ACV")
        assert abs(np.mean(stat)) < 0.05 * acf_noise_mean(N, "ACV")
        rate = np.mean(stat > table_threshold("ACV", np.full(n_bins, N), "99p0"))
        assert 0.005 < rate < 0.02


def test_acv_finds_the_envelope():
    from numax_proxies import NumaxProxies
    psd = synthetic_psd(50.0, 3.0, rng=np.random.default_rng(4))
    p = NumaxProxies.from_psd(psd.frequency, psd.psd)
    numax = p.compute_numax_from_acv()
    assert abs(numax.n / 50.0 - 1) < 0.15        # like the ACF / CoV (+8-10 % without background correction)
    assert "numax_ACV" in p.numax_estimates and "numax_ACV" not in p.COMBINED_FROM
