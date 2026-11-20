"""CoV Gaussian fit: the centre stays within the fitted bins."""
import numpy as np

from numax_proxies.methods.cov import fit_gaussian


def bump(x, centre, sigma=10.0, amp=0.5):
    return 1 + amp * np.exp(-0.5 * ((x - centre) / sigma) ** 2)


def test_peak_inside_window_is_fitted():
    x = np.linspace(100, 136, 40)
    numax, popt, status = fit_gaussian(x, bump(x, 115.0), 110.0)
    assert status == "fit" and abs(numax.n - 115.0) < 0.1


def test_peak_outside_window_is_rejected():
    x = np.linspace(100, 136, 40)                       # spectrum starts at 100 muHz
    numax, popt, status = fit_gaussian(x, bump(x, 85.0), 100.0)
    assert status == "rejected" and np.isnan(numax.n)
