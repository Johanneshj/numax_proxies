"""ACF collapse 'delta_nu': white-noise calibration and response to a Delta-nu comb."""
import warnings

import numpy as np
import pytest

from numax_proxies import NumaxProxies, config_from_dict
from numax_proxies.binning import log_numax_bins
from numax_proxies.methods.acf import cacf_delta_nu_in_windows, delta_nu_lag_ranges
from numax_proxies.synthetic import kepler_lc_grid, synthetic_psd


@pytest.mark.parametrize("width_factor", [2.0, 5.0, 15.0])
def test_white_noise_calibration(rng, width_factor):
    """Per bin, (stat - null_mean) / null_sd should be ~ N(0, 1) for chi^2_2 noise."""
    f = kepler_lc_grid()
    z_all = []
    for _ in range(100):
        x = rng.exponential(size=f.size) / np.log(2) - 1       # relative power of white noise
        centres, i0, i1 = log_numax_bins(f, 5.0, None, width_factor, 1.0)   # non-overlapping
        k_lo, k_hi = delta_nu_lag_ranges(centres, f, 0.1, [1.0])
        stat, mu, sd = cacf_delta_nu_in_windows(x, i0, i1, k_lo, k_hi)
        ok = np.isfinite(stat)
        z_all.append((stat[ok] - mu[ok]) / sd[ok])
    z = np.concatenate(z_all)
    assert abs(np.mean(z)) < 0.1
    assert abs(np.std(z) - 1) < 0.1
    assert np.mean(z > 2.326) == pytest.approx(0.01, abs=0.01)  # 99 % per-bin threshold


def test_responds_to_delta_nu_comb():
    psd = synthetic_psd(60.0, 5.0, rng=np.random.default_rng(5))
    grid = {"overlap_scale": [5], "width_factor": [5], "smoothing_factor": [1.0],
            "collapse": "delta_nu", "FAP_threshold": "99p0"}
    p = NumaxProxies.from_psd(psd.frequency, psd.psd, {"ACF_CONFIG": grid})
    p.compute_numax_from_acf()
    d = next(iter(p.acf.results.values()))
    peak = d["bin_centers"][np.nanargmax(d["cacf_smoothed"])]
    assert 40 < peak < 90                                    # near the injected numax = 60
    assert np.nanmax(d["cacf_raw"] - d["fap"]) > 0           # significant somewhere
    assert np.isfinite(p.acf.numax_estimate.n)


def test_config_checks():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config_from_dict({"ACF_CONFIG": {"collapse": "delta_nu", "width_factor": [1, 5]}})
    assert any("narrower than the first lag band" in str(x.message) for x in w)
    with pytest.raises(ValueError):
        config_from_dict({"ACF_CONFIG": {"collapse": "sum"}})


def test_plot_with_2dacf(tmp_path):
    psd = synthetic_psd(60.0, 5.0, rng=np.random.default_rng(5))
    grid = {"overlap_scale": [5], "width_factor": [5], "smoothing_factor": [1.0],
            "collapse": "delta_nu", "plot_2dacf": [5, 5]}
    p = NumaxProxies.from_psd(psd.frequency, psd.psd, {"ACF_CONFIG": grid,
                                                       "CONFIG": {"results_directory": str(tmp_path)}})
    p.compute_numax_from_acf()
    p.acf.plot()
    assert list(tmp_path.rglob("ACF.png"))
    with pytest.raises(ValueError):
        config_from_dict({"ACF_CONFIG": {"plot_2dacf": [5]}})


def test_dnu_bands_expand_and_do_not_overlap(kepler_grid):
    cfg = config_from_dict({"ACF_CONFIG": {"collapse": "delta_nu", "dnu_bands": 8,
                                           "width_factor": [5]}}).acf_config
    assert cfg.dnu_multiples == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    centres, _, _ = log_numax_bins(kepler_grid, 20.0, None, 5.0, 5.0)
    k_lo, k_hi = delta_nu_lag_ranges(centres, kepler_grid, 0.1, cfg.dnu_multiples)
    assert np.all(k_hi[:, :-1] < k_lo[:, 1:])                  # no lag in two bands


def test_white_noise_calibration_with_bands(rng):
    """With several bands, (stat - null_mean) / null_sd is still ~ N(0, 1)."""
    f = kepler_lc_grid()
    z_all = []
    for _ in range(60):
        x = rng.exponential(size=f.size) / np.log(2) - 1
        centres, i0, i1 = log_numax_bins(f, 5.0, None, 5.0, 1.0)
        k_lo, k_hi = delta_nu_lag_ranges(centres, f, 0.1, [0.5 * b for b in range(1, 9)])
        stat, mu, sd = cacf_delta_nu_in_windows(x, i0, i1, k_lo, k_hi)
        ok = np.isfinite(stat)
        z_all.append((stat[ok] - mu[ok]) / sd[ok])
    z = np.concatenate(z_all)
    assert abs(np.mean(z)) < 0.1
    assert abs(np.std(z) - 1) < 0.1


def test_fixed_bands_have_equal_width(kepler_grid):
    """dnu_window_mode 'fixed': every band is m +- window Delta-nu wide, never overlapping."""
    from numax_proxies.methods.acf import band_edges

    lo, hi = band_edges([0.5, 1.0, 1.5, 4.0], 0.1, "fixed")
    np.testing.assert_allclose(hi - lo, 0.2)
    np.testing.assert_allclose(lo, [0.4, 0.9, 1.4, 3.9])
    centres, _, _ = log_numax_bins(kepler_grid, 20.0, None, 5.0, 5.0)
    k_lo, k_hi = delta_nu_lag_ranges(centres, kepler_grid, 0.1, [0.5 * b for b in range(1, 9)], "fixed")
    width = k_hi - k_lo
    assert np.all(np.abs(width - width[:, :1]) <= 1)            # same width for every band
    assert np.all(k_hi[:, :-1] < k_lo[:, 1:])
    with pytest.raises(ValueError):
        config_from_dict({"ACF_CONFIG": {"dnu_window_mode": "absolute"}})
    with pytest.raises(ValueError):                              # window wider than the first band
        config_from_dict({"ACF_CONFIG": {"dnu_window_mode": "fixed", "dnu_window": 0.6,
                                         "dnu_bands": 2}})
