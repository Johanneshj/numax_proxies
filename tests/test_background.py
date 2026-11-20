"""Background correction: recover numax from the peak of envelope / background."""
import numpy as np

from numax_proxies import NumaxProxies
from numax_proxies.background import background_slope, correct_numax
from numax_proxies.fitting import FWHM_TO_SIGMA
from numax_proxies.seismic import envelope_fwhm
from numax_proxies.synthetic import synthetic_psd

NU = np.arange(1.0, 283.0, 0.01)


def test_slope_of_power_law():
    assert abs(background_slope(NU, NU ** -2.0, 80.0) - 2.0) < 1e-6


def test_peak_of_ratio_is_corrected():
    numax, alpha = 60.0, 2.5
    sigma = envelope_fwhm(numax) / FWHM_TO_SIGMA
    ratio = np.exp(-0.5 * ((NU - numax) / sigma) ** 2) * NU ** alpha    # H / B, B ~ nu^-alpha
    nu_p = NU[np.argmax(ratio)]
    assert nu_p > 1.05 * numax                                           # biased high
    assert abs(correct_numax(NU, NU ** -alpha, nu_p) / numax - 1) < 0.01


def test_pipeline_option():
    psd = synthetic_psd(50.0, 3.0, rng=np.random.default_rng(4))
    est = {}
    for on in (False, True):
        p = NumaxProxies.from_psd(psd.frequency, psd.psd, {"CONFIG": {"background_correction": on}})
        est[on] = p.compute_numax_from_acf().n
        if on:
            d = next(iter(p.acf.results.values()))
            assert "fit_numax_raw" in d and d["fit_numax"].n < d["fit_numax_raw"].n
    assert est[True] < est[False]
