import numpy as np
import warnings
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
from numpy.typing import NDArray
from uncertainties import ufloat

def global_fitting(
        bin_centers     :   list[NDArray],
        CoVs            :   list[NDArray],
        smoothed_CoVs   :   list[NDArray],
        FAPs            :   list[NDArray],
        initial_numax   :   float = None
):
    """Fit for numax estimates"""

    # Check if initial estimate of numax is provided, otherwise find it based on location of max CoV value.
    if initial_numax:
        numax_inits = [initial_numax]
    else:
        numax_inits = get_initial_numax(bin_centers, CoVs, smoothed_CoVs, FAPs)

    # Information we might want later
    res_list = []
    fit_vals = []

    # Fit for numax
    idx = 0
    for bcs, scovs in zip(bin_centers, smoothed_CoVs):
        bcs = np.asarray(bcs)
        for scov in scovs:
            scov = np.asarray(scov)
            # Define initial numax
            if len(numax_inits) == 1:
                numax_init = numax_inits[0]
            else:
                numax_init = numax_inits[idx]

            # Check if initial numax is nan
            if np.isnan(numax_init):
                # Append nan to res
                res_list.append(ufloat(np.nan, np.nan))
                # Append nans to fit vals
                fit_vals.append({
                    "amp": np.nan,
                    "sigma": np.nan,
                    "numax": np.nan,
                    "numax_err": np.nan
                })
                continue

            # Define region for evaluating power
            width = 0.66 * (numax_init ** 0.88)
            lower = numax_init - width
            upper = numax_init + width
            indices = np.where((bcs >= lower) & (bcs <= upper))[0]

            # Fit for numax
            numax, popt, succesful_fit = fit_for_numax(
                bcs[indices],
                scov[indices],
                numax_init
            )

            # If fit was succesful append numax, numax_err and True
            if succesful_fit and popt is not None:
                # Append ufloat numax to res_list
                res_list.append(numax)

                # Append to fit_vals
                fit_vals.append({
                    "amp": popt[0],
                    "sigma": popt[1],
                    "numax": popt[2],
                    "numax_err": popt[1]
                })

            # If fit failed default to method from Viani+ 2019:
            # Evaluate sum of powers
            else:
                numerator = np.nansum(bcs[indices] * scov[indices])
                denominator = np.nansum(scov[indices])
                if denominator == 0 or np.isnan(denominator) or denominator is None:
                    res_list.append(ufloat(np.nan, np.nan))
                else:
                    numax_estimate = numerator / denominator
                    numax_err_estimate = np.abs(np.nanstd(bcs[indices], ddof=1))
                    res_list.append(ufloat(numax_estimate, numax_err_estimate))

                # Append nans to fit vals
                fit_vals.append({
                    "amp": np.nan,
                    "sigma": np.nan,
                    "numax": np.nan,
                    "numax_err": np.nan
                })
            idx += 1
    # Return numax estimates, error estimates, and succesful fit bools
    return res_list, fit_vals
        
    
def get_initial_numax(
        bin_centers     :   list[NDArray],
        CoVs            :   list[NDArray],
        smoothed_CoVs   :   list[NDArray],
        FAPs            :   list[NDArray]
) -> list[float]:
    """Determine which bin center should serve as initial numax guess"""

    # Master initial numax list
    numax_inits: list[float] = []

    # First loop of bcs, cov, and fap to determine good indices
    for bcs, cov, fap, scovs in zip(bin_centers, CoVs, FAPs, smoothed_CoVs):
        bcs = np.asarray(bcs)
        cov = np.asarray(cov)
        fap = np.asarray(fap)

        # Check good indices where CoV exceeds FAP
        fap_mask = cov > fap

        # In case we are below CoV threshold we append np.nans for each smoothing factor
        if not np.any(fap_mask):
            numax_inits.extend([np.nan] * len(scovs))
            continue

        # Upper detection threshold to avoid spurious peaks
        upper_limit = 2.69 * (bcs ** 0.154)
        spurious_mask = fap_mask & (cov < upper_limit)

        # Define final mask
        final_mask = spurious_mask if np.any(spurious_mask) else fap_mask
        final_bcs = bcs[final_mask]

        # Append best numax init for each smoothing factor 
        for scov in scovs:
            scov = np.asarray(scov)
            valid_scov = scov[final_mask]
            
            best_idx = np.argmax(valid_scov)
            numax_inits.append(float(final_bcs[best_idx]))

    # Return numax inits list
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