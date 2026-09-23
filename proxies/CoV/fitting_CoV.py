import numpy as np
import warnings
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
from numpy.typing import NDArray
from typing import Dict, Tuple
from uncertainties import ufloat
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


def fit_for_numax(x, y, numax_init, window=500):
    """
        Fit Gaussian to CoV spectrum if initial guess is specified

        Inputs:
            x           : x values (frequencies)
            y           : y values (CoV)
            numax_init  : initial numax guess
            window      : in case fits fails we take numax as most significant 
                            peak in window around numax_init
        
        Outputs:
            numax           : numax estimate from fitting
            popt            : fit values
            successful_fit  : flag indicating if fit was successful
    """

    successful_fit = False # Flag to check if fit was good
    # We do a "try" here in case fits fails or is underresolved
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)

        # Safety check
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

        # Initial guesses
        amp0 = np.max(y)
        w0 = 0.66*numax_init**0.88 
        p0s = [amp0, w0, numax_init]

        # Fit
        popt, pcov = curve_fit(
            gaussian_with_offset,
            x,
            y,
            p0=p0s,
        )
        numax = popt[2]
        successful_fit = True

        numax_final = ufloat(
            nominal_value = numax, 
            std_dev = np.abs(popt[1])
        )

        return numax_final, popt, successful_fit

    except (RuntimeError, OptimizeWarning, ValueError) as e:
        # If fit fails return numax as location of highest peak in a region around numax_init
        print(f'Gaussian fit for CoV method failed because of {e}, falling back to location of maximum CoV.')
        w0 = 0.66 * numax_init ** 0.88
        window = 4 * w0
        mask = (x >= numax_init - window) & (x <= numax_init + window)
        return x[mask][np.argmax(y[mask])], np.nan, successful_fit

def gaussian_with_offset(x, A, sigma, mu):
    """Gaussian function"""
    return 1 + A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))