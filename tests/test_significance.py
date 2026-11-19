"""FAP tables and the significance-at-numax check."""
import numpy as np
import pytest

from numax_proxies.config import ACFConfig
from numax_proxies.significance import load_tables, significant_near, table_threshold


def test_tables_shape_and_order():
    t = load_tables()
    for m in ("ACF", "CoV"):
        assert t[m]["95"].shape == t["N"].shape == t[m]["99"].shape
        assert np.all(t[m]["99"] > t[m]["95"])                 # 99th above 95th


def test_acf_threshold_decreases_with_n():
    t = load_tables()
    assert np.all(np.diff(t["ACF"]["99"]) < 0)


def test_interpolation_hits_table_nodes():
    t = load_tables()
    assert table_threshold("CoV", [t["N"][3]], "99p0")[0] == pytest.approx(t["CoV"]["99"][3])
    assert np.isnan(table_threshold("ACF", [3], "95p0")[0])  # below the table: no threshold
    assert table_threshold("ACF", [1e6], "95p0")[0] == pytest.approx(t["ACF"]["95"][-1])


def test_significant_near():
    x = np.linspace(10, 200, 100)
    raw = np.zeros_like(x)
    raw[np.argmin(np.abs(x - 50))] = 2.0
    assert significant_near(x, raw, 1.0, 52.0)
    assert not significant_near(x, raw, 1.0, 150.0)
    assert not significant_near(x, raw, 1.0, np.nan)


def test_config_levels():
    ACFConfig(FAP_threshold="99p0", significance_at_numax=True)
    with pytest.raises(ValueError):
        ACFConfig(FAP_threshold="99p9")


def test_all_lags_white_noise_calibration():
    """MCC - mu0 has mean ~0 on white noise, and exceeds the 99 % threshold ~1 % of the time."""
    from numax_proxies.methods.acf import cacf_in_windows
    from numax_proxies.significance import acf_noise_mean

    rng = np.random.default_rng(7)
    for N in (12, 50, 700):         # N = 5-7 lie between the first table nodes (less exact)
        n_bins = 20000
        x = rng.exponential(size=N * n_bins) / np.log(2) - 1
        i0 = np.arange(n_bins) * N
        stat = cacf_in_windows(x, i0, i0 + N) - acf_noise_mean(np.full(n_bins, N))
        assert abs(np.mean(stat)) < 0.05 * acf_noise_mean(N)
        rate = np.mean(stat > table_threshold("ACF", np.full(n_bins, N), "99p0"))
        assert 0.005 < rate < 0.02
