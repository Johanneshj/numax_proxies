"""Kepler short-cadence checks: detection-based cleaning and the PSPS comb test."""
import logging

import numpy as np

from numax_proxies import NumaxProxies
from numax_proxies.artefacts import (CombChecker, clean_detected_artefacts, replace_regions,
                                     max_exceedance_threshold, near_artefact, psps)
from numax_proxies.binning import running_median
from numax_proxies.synthetic import synthetic_psd

GRID = np.arange(100, 8496, 0.38)                    # KASOC-like short-cadence grid [muHz]


def rel(P):
    return P / running_median(GRID, P) - 1


def test_threshold_is_exact():
    u = max_exceedance_threshold(10, 0.01)
    assert 1 - (1 - np.exp(-u)) ** 10 == np.float64(0.01) or abs(1 - (1 - np.exp(-u)) ** 10 - 0.01) < 1e-12


def test_detected_spike_is_cleaned_narrowly(rng):
    P = rng.exponential(size=GRID.size)
    j = np.argmin(np.abs(GRID - 4531.1))
    P[j - 1:j + 2] += [20, 80, 20]
    cleaned, report = clean_detected_artefacts(GRID, P, rng)
    hit = [e for e in report if e["detected"]]
    assert [e["frequency"] for e in hit] == [4531.1]
    assert hit[0]["hi"] - hit[0]["lo"] < 2.0                    # a few bins, not +-5 %
    changed = np.flatnonzero(cleaned != P)
    assert GRID[changed].min() >= hit[0]["lo"] and GRID[changed].max() <= hit[0]["hi"]


def test_pure_noise_is_not_cleaned(rng):
    P = rng.exponential(size=GRID.size)
    cleaned, report = clean_detected_artefacts(GRID, P, rng, fap=0.001)
    assert not any(e["detected"] for e in report)
    np.testing.assert_array_equal(cleaned, P)


def comb(numax, P, fap=0.01):
    r = psps(GRID, rel(P), numax, fap=fap)
    return r["comb"], r["spacing"]


def test_comb_test_finds_comb_not_spike_or_broad_excess(rng):
    psd = synthetic_psd(1000.0, 1.0, frequency=GRID, linewidth=1.5, rng=rng)
    ok, spacing = comb(1000.0, psd.psd)
    assert ok and abs(spacing - 52.0) < 3
    P = rng.exponential(size=GRID.size)
    j = np.argmin(np.abs(GRID - 4531.1))
    P[j - 1:j + 2] += [20, 80, 20]
    assert not comb(4531.1, P)[0]                                 # single spike
    smooth = 1 + 2 * np.exp(-0.5 * ((GRID - 1500) / 150) ** 2)  # broad hump, no comb
    assert not comb(1500.0, rng.exponential(size=GRID.size) * smooth)[0]


def test_comb_test_false_alarm_rate(rng):
    hits = sum(comb(1500.0, rng.exponential(size=GRID.size), fap=0.05)[0] for _ in range(300))
    assert 0.02 < hits / 300 < 0.09                             # 5 % expected


def test_checker_modes():
    assert near_artefact(4520.0) and not near_artefact(1000.0)
    assert near_artefact(1000.0, distance=1.0)                  # 1132.8 within one FWHM
    c = CombChecker(GRID, np.zeros(GRID.size), mode="near_artefacts")
    assert c(1000.0)                                            # far from artefacts: not tested
    assert not c(4531.0)                                        # tested: flat power, no comb
    assert not CombChecker(GRID, np.zeros(GRID.size), mode="always")(1000.0)


def test_only_used_for_kepler_sc(rng, caplog):
    psd = synthetic_psd(1000.0, 3.0, frequency=GRID, rng=rng)
    settings = {"CONFIG": {"artefact_cleaning": True, "comb_check": "always"},
                "COV_CONFIG": {"width_factor": [3], "overlap_scale": [5], "smoothing_factor": [1.0]}}
    with caplog.at_level(logging.WARNING):
        p = NumaxProxies.from_psd(GRID, psd.psd, settings)
        p.compute_numax_from_CoV()
    assert "only used for STAR.cadence 'Kepler_SC'" in caplog.text
    assert p.cov.comb_check_mode is None


def test_psps_oversampling_refines_spacing(rng):
    psd = synthetic_psd(1000.0, 3.0, frequency=GRID, rng=rng)
    a = psps(GRID, rel(psd.psd), 1000.0, oversampling=1)
    b = psps(GRID, rel(psd.psd), 1000.0, oversampling=10)
    assert a["z_max"] == b["z_max"]                                      # same points
    np.testing.assert_allclose(b["z_fine"][::10][:a["z"].size], a["z"], atol=1e-9)
    assert abs(b["spacing"] - 52.0) <= abs(a["spacing"] - 52.0) + 0.5


def test_psps_figure(tmp_path, rng):
    psd = synthetic_psd(60.0, 5.0, rng=rng)
    grid = {"width_factor": [3], "overlap_scale": [5], "smoothing_factor": [1.0]}
    p = NumaxProxies.from_psd(psd.frequency, psd.psd,
                              {"ACF_CONFIG": grid, "COV_CONFIG": grid,
                               "CONFIG": {"plot_psps": True, "results_directory": str(tmp_path)}})
    p.compute_numax_from_acf()
    p.compute_numax_from_CoV()
    p.plotting()
    assert list(tmp_path.rglob("PSPS.png"))
    assert abs(p.dnu_estimates["dnu_CoV"] - 6.1) < 0.6           # Delta-nu(60) = 6.1 muHz


def test_replace_regions(rng):
    P = rng.exponential(size=GRID.size)
    P[(GRID > 362) & (GRID < 369)] *= 10                        # broad excess
    out = replace_regions(GRID, P, [[359, 372]], rng)
    inside = (GRID >= 359) & (GRID <= 372)
    np.testing.assert_array_equal(out[~inside], P[~inside])
    assert 0.8 < out[inside].mean() < 1.25
