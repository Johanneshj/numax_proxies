import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
from scipy import integrate
import warnings
from numpy.typing import NDArray
from uncertainties import ufloat, UFloat
import matplotlib.pyplot as plt
from typing import Dict, Tuple


import copy
from typing import Dict, Tuple
import numpy as np

def fit_gauss_global(
    cacf_results            :   Dict[Tuple[float, ...], dict],
    initial_numax           :   float,
    max_acf_fit_iterations  :   int,
    n_sigma_numax_acf       :   float,
    inplace                 :   bool = True
) -> Dict[Tuple[float, ...], dict]:
    """
    Performs global Gaussian fitting to collapsed ACFs across all parameter combinations
    and updates cacf_results with the fitting outputs.

    Parameters:
    -----------
    cacf_results : Dict[Tuple[float, ...], dict]
        Dictionary containing 'bin_centers', 'fap', and smoothed CACF arrays.
    initial_numax : float
        Initial estimate for numax (microHz).
    max_acf_fit_iterations : int
        Maximum fitting iterations for fit_gauss_to_collapsed_acf.
    n_sigma_numax_acf : float
        Sigma multiplier defining the fitting window width around initial_numax.
    use_normalized_cacf : bool
        If True, uses 'cacf_norm_smoothed'; if False, uses 'cacf_smoothed'.
    inplace : bool
        If True, updates input dictionary directly. If False, updates a copy.

    Returns:
    --------
    Dict[Tuple[float, ...], dict]
        The updated cacf_results dictionary where each sub-dictionary contains:
        - All previous keys ('bin_centers', 'cacf_raw', 'cacf_smoothed', 'cacf_norm_smoothed', 'fap')
        - "fit_numax": Fitted numax object/float
        - "fit_params": Dict of fitted parameters ('amp', 'sigma', 'numax', 'y', 'numax_err')
    """
    results = cacf_results if inplace else copy.deepcopy(cacf_results)
    cacf_key = "cacf_smoothed"

    for param_key, data in results.items():
        bcs = np.asarray(data["bin_centers"])
        scacf = np.asarray(data[cacf_key])
        fap = np.asarray(data["fap"])

        # Perform Gaussian fit on current parameter combination
        numax, popt = fit_gauss_to_collapsed_acf(
            smoothed_acf=scacf,
            freq_centers=bcs,
            FAP=fap,
            initial_numax=initial_numax,
            max_acf_fit_iterations=max_acf_fit_iterations,
            n_sigma_numax_acf=n_sigma_numax_acf
        )

        numax_err = getattr(numax, "s", getattr(numax, "std_dev", np.nan))

        fit_params = {
            "amp": popt[0],
            "sigma": popt[1],
            "numax": popt[2],
            "y": popt[3],
            "numax_err": numax_err
        }

        # Update sub-dictionary in-place
        data["fit_numax"] = numax
        data["fit_params"] = fit_params

    return results



def fit_gauss_to_collapsed_acf(
    smoothed_acf: NDArray,
    freq_centers: NDArray,
    FAP: NDArray,
    initial_numax: float | None,
    max_acf_fit_iterations: int,
    n_sigma_numax_acf: float,
) -> tuple[UFloat, NDArray]:
    """
    Fit a Gaussian envelope to a collapsed autocorrelation function (CACF)
    to estimate numax.

    The fit is repeated up to ``max_acf_fit_iterations`` times. After each
    fit, the region within ``n_sigma_numax_acf`` Gaussian widths of the fitted
    centre is removed from the data, so the next iteration searches for the
    next-highest peak. Among all successful fits, the one whose integral of
    the CACF over its own window is largest is returned. This guards against
    the first iteration locking onto a spurious peak.

    Iterative maximisation of the integral under the envelope was implemented
    by Enrico Corsaro, 2026, INAF - Catania.

    Parameters
    ----------
    smoothed_acf : ndarray
        Smoothed collapsed 1D ACF, one value per frequency window.
    freq_centers : ndarray
        Median frequency of each window, in muHz. Same length as
        ``smoothed_acf``.
    FAP : ndarray
        False-alarm-probability threshold for each frequency window. Must be
        the same length as ``freq_centers``. If you have a scalar threshold,
        broadcast it first with ``np.broadcast_to(FAP, freq_centers.shape)``.
    initial_numax : float or None
        Initial guess for numax in muHz. It is used only for the first
        iteration. Later iterations start from the highest remaining point.
        If None or 0, every iteration starts from the highest remaining point.
    max_acf_fit_iterations : int
        Maximum number of Gaussian fits to attempt.
    n_sigma_numax_acf : float
        Half-width, in units of the fitted Gaussian sigma, of the window used
        to (a) integrate the CACF under each fitted peak and (b) mask the
        peak out before the next iteration.

    Returns
    -------
    numax : uncertainties.UFloat
        numax estimate in muHz. The nominal value is the fitted Gaussian
        centre. The standard deviation is the fitted Gaussian **width**
        (sigma), not the statistical uncertainty on the centre. It is NaN
        if no fit succeeded.
    popt : ndarray, shape (4,)
        Best-fit parameters of ``gaussian``, in the order
        ``[amplitude, sigma, centre, offset]``. All NaN if no fit succeeded.

    Notes
    -----
    - Points with non-finite frequency or ACF values are discarded before
      fitting.
    - If no point of the ACF reaches its FAP threshold, a message is printed
      and NaNs are returned.
    - A fit that raises ``RuntimeError`` or ``ValueError`` (e.g. non-
      convergence or an infeasible initial guess) ends the iteration loop.
      Fits completed before that point are still considered.
    - The Gaussian width is bounded below by 1e-6 to avoid division by zero
      in ``gaussian``.
    - The initial width guess is the empirical relation
      ``w0 = (2/3) * numax0**(22/25)``, and the initial amplitude is 0.8 times
      the maximum of the remaining ACF.
    """
    nan_result = (ufloat(np.nan, np.nan), np.full(4, np.nan))

    x = np.asarray(freq_centers, float)
    y = np.asarray(smoothed_acf, float)
    FAP = np.asarray(FAP, float)

    # Drop non-finite points, applying the same mask to the FAP array
    valid = np.isfinite(x) & np.isfinite(y)
    x, y, FAP = x[valid], y[valid], FAP[valid]

    # Require at least one point above the significance threshold
    if not np.any(y >= FAP):
        print('No significant CACF values.')
        return nan_result

    # Residual data: shrinks as fitted peaks are masked out
    x_res, y_res = x.copy(), y.copy()
    results = []  # one (numax, sigma, integral, popt) tuple per successful fit

    for it in range(int(max_acf_fit_iterations)):
        # Too few points left for a 4-parameter fit
        if x_res.size < 5:
            break
        try:
            # Starting centre: user guess on the first pass, otherwise the
            # highest remaining point
            if it == 0 and initial_numax:
                numax0 = initial_numax
            else:
                numax0 = x_res[np.argmax(y_res)]
            numax0 = np.clip(numax0, x_res.min(), x_res.max())

            # Initial guesses and bounds: [amplitude, sigma, centre, offset]
            amp0 = 0.8 * np.max(y_res)
            w0 = (2 / 3) * numax0 ** (22 / 25)
            p0 = [amp0, w0, numax0, np.median(y_res)]
            lower = [0, 1e-6, x_res.min(), y_res.min()]
            upper = [1.5 * amp0, 2 * w0, x_res.max(), y_res.max()]

            popt, _ = curve_fit(gaussian, x_res, y_res, p0=p0, bounds=(lower, upper))
            numax, sig = popt[2], popt[1]

            # Window of +/- n_sigma around the fitted centre. Used both for
            # the integral and for masking the peak in the next iteration.
            window = np.abs(x_res - numax) <= n_sigma_numax_acf * sig
            if window.any():
                y_int = integrate.trapezoid(y_res[window], x_res[window])
            else:
                y_int = integrate.trapezoid(y_res, x_res)

            results.append((numax, sig, y_int, popt))

            # Remove the fitted peak before the next iteration
            x_res, y_res = x_res[~window], y_res[~window]

        except (RuntimeError, ValueError) as e:
            print(f'Iteration {it + 1} failed due to {type(e).__name__}')
            break

    # All fits failed
    if not results:
        return nan_result

    # Keep the fit with the largest integral under its envelope
    numax, sig, _, popt = max(results, key=lambda r: r[2])
    return ufloat(numax, abs(sig)), popt

def gaussian(x : NDArray, A : float, sigma : float, mu : float, y : float):
    """Gaussian function"""
    return y + A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))   