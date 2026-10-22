import numpy as np
import warnings
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
from numpy.typing import NDArray
from typing import Dict, Tuple
from uncertainties import ufloat, UFloat
import copy

def global_fitting(
    CoV_results: Dict[Tuple[float, ...], dict],
    initial_numax: float = None,
    inplace: bool = True
) -> Dict[Tuple[float, ...], dict]:
    """
    Fit for numax estimates across all parameter combinations in cov_results.

    Parameters:
    -----------
    cov_results : Dict[Tuple[float, ...], dict]
        Dictionary keyed by (overlap_scale, width_factor, smoothing_factor) 
        containing 'bin_centers' and 'cov_smoothed'.
    initial_numax : float, optional
        Initial estimate for numax. If None, computes initial estimates using 
        get_initial_numax(cov_results).
    inplace : bool
        If True, updates input dictionary directly. If False, operates on a copy.

    Returns:
    --------
    Dict[Tuple[float, ...], dict]
        The updated dictionary where each entry contains:
        - "fit_numax": ufloat object representing the fitted/estimated numax and error
        - "fit_params": Dict with 'amp', 'sigma', 'numax', 'numax_err'
    """
    results = CoV_results if inplace else copy.deepcopy(CoV_results)

    # Determine initial guesses for each entry
    if initial_numax is None:
        numax_inits = get_initial_numax(results)
    else:
        numax_inits = initial_numax

    for param_key, data in results.items():
        bcs = np.asarray(data["bin_centers"])
        scov = np.asarray(data["cov_smoothed"])

        # Retrieve initial guess for this specific parameter key
        numax_init = numax_inits.get(param_key, np.nan) if isinstance(numax_inits, dict) else numax_inits

        # Handle NaN initial guess
        if np.isnan(numax_init) or numax_init is None:
            data["fit_numax"] = ufloat(np.nan, np.nan)
            data["fit_params"] = {
                "amp": np.nan,
                "sigma": np.nan,
                "numax": np.nan,
                "numax_err": np.nan
            }
            continue

        # Define frequency window around initial numax
        width = 0.66 * (numax_init ** 0.88)
        lower = numax_init - width
        upper = numax_init + width
        indices = np.where((bcs >= lower) & (bcs <= upper))[0]

        # Check if window contains valid data
        if len(indices) == 0:
            data["fit_numax"] = ufloat(np.nan, np.nan)
            data["fit_params"] = {
                "amp": np.nan,
                "sigma": np.nan,
                "numax": np.nan,
                "numax_err": np.nan
            }
            continue

        # Attempt fit
        numax, popt, successful_fit = fit_for_numax(
            bcs[indices],
            scov[indices],
            numax_init
        )

        if successful_fit and popt is not None:
            numax_err = getattr(numax, "s", getattr(numax, "std_dev", np.nan))
            
            data["fit_numax"] = numax
            data["fit_params"] = {
                "amp": popt[0],
                "sigma": popt[1],
                "numax": popt[2],
                "numax_err": numax_err
            }

        # Fallback method (Viani+ 2019): Weighted sum of powers
        else:
            numerator = np.nansum(bcs[indices] * scov[indices])
            denominator = np.nansum(scov[indices])

            if denominator == 0 or np.isnan(denominator) or denominator is None:
                data["fit_numax"] = ufloat(np.nan, np.nan)
            else:
                numax_estimate = numerator / denominator
                numax_err_estimate = np.abs(np.nanstd(bcs[indices], ddof=1))
                data["fit_numax"] = ufloat(numax_estimate, numax_err_estimate)

            data["fit_params"] = {
                "amp": np.nan,
                "sigma": np.nan,
                "numax": np.nan,
                "numax_err": np.nan
            }

    return results
        
    
def get_initial_numax(
    cov_results: Dict[Tuple[float, ...], dict]
) -> Dict[Tuple[float, ...], float]:
    """
    Determine which bin center should serve as initial numax guess for each 
    parameter combination in cov_results.

    Parameters:
    -----------
    cov_results : Dict[Tuple[float, ...], dict]
        Dictionary keyed by parameter tuples containing 'bin_centers', 'cov',
        'cov_smoothed', and 'fap'.

    Returns:
    --------
    Dict[Tuple[float, ...], float]
        Dictionary mapping each parameter key to its initial numax estimate (in microHz).
    """
    numax_inits: Dict[Tuple[float, ...], float] = {}

    for param_key, data in cov_results.items():
        bcs = np.asarray(data["bin_centers"])
        cov = np.asarray(data["cov"])
        scov = np.asarray(data["cov_smoothed"])

        # Check if FAP exists in dictionary entry
        if "fap" in data and data["fap"] is not None:
            fap = np.asarray(data["fap"])
            fap_mask = cov > fap
        else:
            # Fallback if no FAP filter applied: treat all bins as valid candidates
            fap_mask = np.ones_like(cov, dtype=bool)

        # In case no points exceed the FAP threshold
        if not np.any(fap_mask):
            numax_inits[param_key] = np.nan
            continue

        # Upper detection threshold to avoid spurious peaks
        upper_limit = 2.69 * (bcs ** 0.154)
        spurious_mask = fap_mask & (cov < upper_limit)

        # Define final mask
        final_mask = spurious_mask if np.any(spurious_mask) else fap_mask
        final_bcs = bcs[final_mask]
        valid_scov = scov[final_mask]

        # Find frequency corresponding to max smoothed CoV within valid range
        if len(valid_scov) > 0:
            best_idx = np.argmax(valid_scov)
            numax_inits[param_key] = float(final_bcs[best_idx])
        else:
            numax_inits[param_key] = np.nan

    return numax_inits


def cv_max_limit(nu: float) -> float:
    """
    Upper limit on the coefficient of variation (CV) expected from
    solar-like oscillations at frequency ``nu`` (muHz):
    CV_max ~= 2.69 * (nu / muHz)**0.154.
    """
    return 2.69 * nu ** 0.154


def _peak_cov(x: NDArray, y: NDArray, centre: float, half_width: float) -> float:
    """Highest observed CoV within ``half_width`` of ``centre``."""
    in_peak = np.abs(x - centre) <= half_width
    if np.any(in_peak):
        return float(np.max(y[in_peak]))
    # Peak narrower than the sampling: use the nearest point
    return float(y[np.argmin(np.abs(x - centre))])


def fit_for_numax(
    x: NDArray,
    y: NDArray,
    numax_init: float,
    window: float | None = None,
) -> tuple[UFloat, NDArray, bool]:
    """
    Estimate numax by fitting a Gaussian to the CoV spectrum, starting from
    an initial guess.

    If the fit fails, numax falls back to the location of the highest CoV
    value within a window around ``numax_init``. Any estimate whose peak CoV
    exceeds the solar-like limit ``2.69 * (nu/muHz)**0.154`` is discarded,
    since such values are not expected from solar-like oscillations.

    Parameters
    ----------
    x : ndarray
        Frequencies in muHz.
    y : ndarray
        CoV values, same length as ``x``.
    numax_init : float
        Initial numax guess in muHz. Must be positive.
    window : float or None, optional
        Half-width in muHz of the fallback search window around
        ``numax_init``. If None (default), ``4 * w0`` is used, with
        ``w0 = 0.66 * numax_init**0.88``.

    Returns
    -------
    numax : uncertainties.UFloat
        numax estimate in muHz.
        - Successful fit: nominal value is the fitted centre and the
          standard deviation is the fitted Gaussian width (sigma), not the
          statistical error on the centre.
        - Fallback: frequency of the highest CoV value in the window, with
          NaN standard deviation.
        - NaN (both) if the fit and fallback fail, or if the estimate is
          discarded by the CV limit.
    popt : ndarray, shape (3,)
        Fitted parameters ``[amplitude, sigma, centre]``. All NaN if the fit
        failed or the estimate was discarded.
    successful_fit : bool
        True only if the Gaussian fit converged and the estimate passed the
        CV limit.

    Notes
    -----
    - Points with non-finite ``x`` or ``y`` are discarded first.
    - ``OptimizeWarning`` is treated as a failure and triggers the fallback.
    - The peak CoV used for the limit check is the maximum observed ``y``
      within +/- 1 fitted sigma of the fit centre (successful fit), or the
      CoV at the selected maximum (fallback).
    """
    nan_popt = np.full(3, np.nan)
    nan_result = (ufloat(np.nan, np.nan), nan_popt, False)

    # Drop non-finite points
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    w0 = 0.66 * numax_init ** 0.88

    try:
        # Initial guesses: [amplitude, sigma, centre]
        p0 = [np.max(y), w0, numax_init]

        # Turn OptimizeWarning into an exception so a fit with an
        # unreliable covariance triggers the fallback
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            popt, _ = curve_fit(gaussian_with_offset, x, y, p0=p0)

        centre, sigma = popt[2], np.abs(popt[1])

        # Discard if the peak CoV exceeds the solar-like limit
        if not np.isfinite(centre) or centre <= 0:
            print('Discarding estimate: fitted numax is not a positive frequency.')
            return nan_result
        peak = _peak_cov(x, y, centre, sigma)
        if peak > cv_max_limit(centre):
            print(f'Discarding estimate: peak CoV {peak:.2f} exceeds limit '
                  f'{cv_max_limit(centre):.2f} at {centre:.1f} muHz.')
            return nan_result

        return ufloat(centre, sigma), popt, True

    except (RuntimeError, OptimizeWarning, ValueError, TypeError) as e:
        print(f'Gaussian fit for CoV method failed because of {e}, '
              'falling back to location of maximum CoV.')

        # Fallback: highest CoV value within a window around numax_init
        half_width = window if window is not None else 4 * w0
        mask = (x >= numax_init - half_width) & (x <= numax_init + half_width)
        if not np.any(mask):
            print('No valid CoV points in the fallback window.')
            return nan_result

        i_max = np.argmax(y[mask])
        numax_peak, peak = x[mask][i_max], y[mask][i_max]

        # Same CV limit applies to the fallback estimate
        if numax_peak <= 0 or peak > cv_max_limit(numax_peak):
            print(f'Discarding fallback estimate: peak CoV {peak:.2f} exceeds '
                  f'limit at {numax_peak:.1f} muHz.')
            return nan_result

        return ufloat(numax_peak, np.nan), nan_popt, False

def gaussian_with_offset(x, A, sigma, mu):
    """Gaussian function"""
    return 1 + A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))