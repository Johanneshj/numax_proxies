import numpy as np

from numax_proxies.binning import (log_numax_bins, replace_below_min_freq, running_median,
                                   smooth_over_envelope)
from numax_proxies.seismic import delta_nu, envelope_fwhm


def test_bins_follow_delta_nu(kepler_grid):
    centres, i0, i1 = log_numax_bins(kepler_grid, 1.0, None, width_factor=2.0, overlap_scale=4.0)
    assert centres[0] == 1.0 and centres[-1] <= kepler_grid[-1]
    np.testing.assert_allclose(np.diff(centres), delta_nu(centres[:-1], 2.0) / 4.0)
    width = (i1 - i0) * (kepler_grid[1] - kepler_grid[0])
    full = centres + delta_nu(centres, 2.0) / 2 < kepler_grid[-1]
    np.testing.assert_allclose(width[full], delta_nu(centres[full], 2.0), rtol=0.05, atol=0.02)


def test_smoothing_equals_naive_nanmean(rng):
    centres = np.cumsum(rng.uniform(0.5, 1.5, 300)) + 10
    values = rng.normal(size=300)
    values[::17] = np.nan
    smoothed = smooth_over_envelope(centres, values, [1.0])[1.0]
    half = envelope_fwhm(centres) / 2
    naive = [np.nanmean(values[(centres >= c - h) & (centres <= c + h)]) for c, h in zip(centres, half)]
    np.testing.assert_allclose(smoothed, naive, rtol=1e-12)


def test_running_median_of_white_noise(kepler_grid, rng):
    power = rng.exponential(size=kepler_grid.size)
    med = running_median(kepler_grid, power)
    inner = kepler_grid > 20
    assert abs(np.median(med[inner]) - np.log(2)) < 0.02   # median of chi^2_2 with mean 1


def test_replace_below_min_freq(kepler_grid, rng):
    power = np.full(kepler_grid.size, 100.0)
    out = replace_below_min_freq(kepler_grid, power, 1.0, seed=3)
    low = kepler_grid <= 1.0
    assert not np.allclose(out[low], 100.0) and np.all(out[~low] == 100.0)
    assert np.all(power == 100.0)                            # input untouched
    np.testing.assert_array_equal(out, replace_below_min_freq(kepler_grid, power, 1.0, seed=3))
