import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
from scipy import integrate
import warnings
from numpy.typing import NDArray
from uncertainties import ufloat
import matplotlib.pyplot as plt


def fit_gauss_global(
        bin_centers : list[NDArray],
        smoothed_cacfs : list[NDArray],
        initial_numax : float,
        max_acf_fit_iterations : int,
        n_sigma_numax_acf : float
):
    """Global fitting"""

    res_list = []
    fit_vals = []

    for i, (bcs, scacf) in enumerate(zip(bin_centers, smoothed_cacfs)):

        numax, popt = fit_gauss_to_collapsed_acf(
            smoothed_acf = np.asarray(scacf),
            freq_centers = np.asarray(bcs),
            initial_numax = initial_numax,
            max_acf_fit_iterations = max_acf_fit_iterations,
            n_sigma_numax_acf = n_sigma_numax_acf
        ) 
        res_list.append(numax)
        fit_vals.append({
            "amp": popt[0],
            "sigma": popt[1],
            "numax": popt[2],
            "y": popt[3],
            "numax_err" : numax.s
        })

    return res_list, fit_vals 



def fit_gauss_to_collapsed_acf(smoothed_acf : NDArray, freq_centers : NDArray, initial_numax : float,
                               max_acf_fit_iterations : float, n_sigma_numax_acf : float):
    """
    Fit Gaussian to collapsed ACF:

    Input:
        collapsed_acf_numax :: collapsed 1D acf
        freq_centers :: medians of freq_windows for plotting and fitting

    Output:
        numax :: numax estimate (central value of Gauss) in muHz
    """

    if initial_numax:
        idx = np.argmin(np.abs(freq_centers-initial_numax))
        numax0 = freq_centers[idx]
    else:
        mask = (freq_centers >= 1)
        idx_max = np.argmax(smoothed_acf[mask])
        numax0 = freq_centers[idx_max]

    # Data
    x = freq_centers
    y = smoothed_acf
    
    # Safety check
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    # Iteratively maximize integral under identified envelope
    # Can help if program misidentifies numax in first iterations
    # Implemented by Enrico Corsaro, 2026, INAF - Catania
    x_res = x
    y_res = y

    numax_array = np.zeros(max_acf_fit_iterations)
    numax_sig_array = np.zeros(max_acf_fit_iterations)
    numax_err_array = np.zeros(max_acf_fit_iterations)
    integral_acf_array = np.zeros(max_acf_fit_iterations)
    popt_array = np.zeros((max_acf_fit_iterations, 4))

    acf_fit_iteration = 0

    while acf_fit_iteration < max_acf_fit_iterations:
    # We do a "try" here in case fits fails or is underresolved
        try:
        # Initial guesses
            amp0 = 0.8 * np.max(y_res)
            w0 = (2/3) * numax0 ** (22/25)
            p0s = [amp0, w0, numax0, np.median(y_res)]
            # Bounds
            lower_bounds = [0, 0, np.min(x_res), -.5]
            upper_bounds = [1.5 * amp0, 2*w0, np.max(x_res), .5]
            popt, pcov = curve_fit(
                gaussian,
                x_res,
                y_res,
                p0=p0s,
                bounds=(
                    lower_bounds,
                    upper_bounds
                )
            )
            numax_sig = popt[1]
            numax = popt[2]
            numax_err = numax_sig #np.sqrt(pcov[2,2])

            # Evaluate a proper interval around numax to compute the integral of the ACF curve
            fit_tmp = np.where((x_res >= numax-n_sigma_numax_acf*numax_sig) & (x_res <= numax+n_sigma_numax_acf*numax_sig))[0]
            if len(fit_tmp) > 0:
                x_fit = x_res[fit_tmp]
                y_fit = y_res[fit_tmp]
            else:
                x_fit = x_res
                y_fit = y_res

            y_int = integrate.trapezoid(y_fit, x_fit)

            numax_array[acf_fit_iteration] = numax
            numax_sig_array[acf_fit_iteration] = numax_sig
            numax_err_array[acf_fit_iteration] = numax_err
            integral_acf_array[acf_fit_iteration] = y_int
            for i, val in enumerate(popt):
                popt_array[acf_fit_iteration, i] = popt[i]

            res_tmp = np.where((x_res < numax-n_sigma_numax_acf*numax_sig) | (x_res > numax+n_sigma_numax_acf*numax_sig))[0]
            x_res = x_res[res_tmp]
            y_res = y_res[res_tmp]

            # Update the value of numax for the next iteration, in case it is present
            numax0_index = np.argmax(y_res)
            numax0 = x_res[numax0_index]

        except (RuntimeError, OptimizeWarning, ValueError) as e:
            print(f'Iteration {acf_fit_iteration+1} failed due to {type(e).__name__}')
            break

        acf_fit_iteration += 1
    
    # Find the most optimal peak, i.e. the one that maximizes the integral
    if len(integral_acf_array) > 0:
        numax_index = np.argmax(integral_acf_array)
        numax_final = numax_array[numax_index]
        numax_final_err = numax_err_array[numax_index]
        numax_sig_final = numax_sig_array[numax_index]
        popt_final = popt_array[numax_index, :]

        numax_final = ufloat(
            nominal_value = numax_final, 
            std_dev = np.abs(numax_final_err)
        )
        return numax_final, popt_final
    else:
        return ufloat(np.nan, np.nan), [np.nan, np.nan, np.nan, np.nan]

def gaussian(x : NDArray, A : float, sigma : float, mu : float, y : float):
    """Gaussian function"""
    return y + A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))