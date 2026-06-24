import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
from scipy import integrate
import warnings
from numpy.typing import NDArray
from uncertainties import ufloat

def collapsed_acf(acf : NDArray, freq_windows : NDArray, sliding_window_style : str = 'log_numax'):
    """
    Collapse 2D ACF into 1D ACF

    Input:
        acf :: 2D ACF
        freq_windows :: binned frequency list

    Output:
        collapsed_acf_numax :: collapsed 1D acf
        freq_centers :: medians of freq_windows for plotting and fitting
    """
    # Collapse acf with mean of each segment
    collapsed_acf = np.array([collapse_segment(seg) for seg in acf])

    # Regularize and take absolute value
    collapsed_acf = collapsed_acf - np.median(collapsed_acf)
    acfs = np.abs(collapsed_acf)
    acfs /= np.max(acfs)

    # Grab frequency as median of each segment
    frequency = np.array([np.median(seg) for seg in freq_windows])
    
    # Smooth acf (Viani+ 2019)
    unsmoothed_acf = acfs
    if sliding_window_style == 'linear':
        smoothed_acf = unsmoothed_acf
    else:
        smoothed_acf = [
            smoothing_func(center, frequency, unsmoothed_acf) for center in frequency
        ]
    
    # Regularize smoothed ACF
    smoothed_acf = normalize_0_1(smoothed_acf)
    
    return smoothed_acf, unsmoothed_acf, frequency

def collapse_segment(seg : NDArray) -> float:
    """Collapse segment of total ACF and return mean."""
    # Ignore first index which always has ACF = 1
    seg = seg[1:]

    # Check length of segment
    if len(seg) < 1:
        return np.nan
    
    # Normalize by standard deviation in the case of logarithmically spaced bins
    std = np.nanstd(seg, ddof=1)
    if std == 0:
        return np.nan
    
    # Return collapsed acf
    mean = np.nanmean(seg)
    return mean
    

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
    popt_array = np.zeros((max_acf_fit_iterations, 3))

    acf_fit_iteration = 0

    while acf_fit_iteration < max_acf_fit_iterations:
    # We do a "try" here in case fits fails or is underresolved
        try:
        # Initial guesses
            amp0 = 0.8 * np.max(y_res)
            w0 = (2/3) * numax0 ** (22/25)
            p0s = [amp0, w0, numax0]
            # Bounds
            lower_bounds = [0, 0, np.min(x_res)]
            upper_bounds = [1.5 * amp0, 2*w0, np.max(x_res)]
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
            numax_err = np.sqrt(pcov[2,2])

            # Evaluate a proper interval around numax to compute the integral of the ACF curve
            fit_tmp = np.where((x_res >= numax-n_sigma_numax_acf*numax_sig) & (x_res <= numax+n_sigma_numax_acf*numax_sig))[0]
            if len(fit_tmp) > 0:
                x_fit = x_res[fit_tmp]
                y_fit = y_res[fit_tmp]
            else:
                x_fit = x_res
                y_fit = y_res

            y_int = integrate.trapezoid(y_fit, x_fit)
            # print('integral:', y_int, 'numax estiamte:', numax)

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
        return ufloat(np.nan, np.nan), [np.nan, np.nan, np.nan]


def smoothing_func(center : float, bin_centers : NDArray, ys : NDArray) -> float:
    """Smoothing function: calculate the mean in each bin."""
    width = 0.66 * center**0.88 # Smooth with FWHM of oscillation envelope
    # width = 0.267 * center**0.764
    lower = center - width / 5
    upper = center + width / 5
    indices = np.where((bin_centers >= lower) & (bin_centers <= upper))[0]
    if len(indices) > 0 and np.sum(~np.isnan(ys[indices])) > 0:
        smoothed_val = np.nanmean(ys[indices])
    else:
        smoothed_val = np.nan
    return smoothed_val

def normalize_0_1(x : NDArray):
    """Normalize between 0 and 1."""
    x = np.asarray(x)
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    return (x - xmin) / (xmax - xmin)

def gaussian(x : NDArray, A : float, sigma : float, mu : float):
    """Gaussian function"""
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

# def fit_background(x,y):
#     try:
#         with warnings.catch_warnings():
#             warnings.simplefilter("error", OptimizeWarning)
#             # Initial guesses
#             a0 = .5
#             b0 = .5
#             p0s = [a0, b0]
#             popt, pcov = curve_fit(
#                 exponential,
#                 x,
#                 y,
#                 p0=p0s,
#                 bounds=(
#                     [0, 0],
#                     [np.inf, np.inf]
#                 )
#             )
#         return popt

#     except (RuntimeError, OptimizeWarning, ValueError):
#         return np.nan, np.nan

# def exponential(x, a ,b):
#         return a * np.exp(-b*x)

# def gauss_plus_exponential(x, A, sigma, mu, a, b):
#     return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + a * np.exp(-b*x)

# def double_check_fit(full_pg, fit_vals):
#     """
#         Check if found oscillation width has regular spaced frequencies.
#         If not, retry fit.
#     """

#     frequency = full_pg.frequency.value
#     power = full_pg.power.value

#     w = 2 * np.sqrt(2*np.log(2)) * fit_vals[1]
#     numax = fit_vals[2]
#     mask = ( (frequency >= numax - w) & (frequency <= numax + w) )

#     f = frequency[mask]
#     p = power[mask]

#     lags, corr = autocorrelation(f, p, numax)

#     _ = psps(f,p)


#     return None

# def autocorrelation(frequency, power, numax):
#     """
#         Take power spectrum of power spectrum
#     """
#     import matplotlib.pyplot as plt
#     plt.figure()
#     plt.plot(frequency, power)

#     # Perform ACF
#     p = power - np.mean(power)  
#     p /= np.std(p)
#     corr = np.correlate(p, p, mode="full")
#     corr = corr[corr.size // 2 :]  # grab only the positive lags

#     df = np.mean(np.diff(frequency))
#     lags = np.arange(len(corr))*df
    

#     deltanu = 0.267 * numax ** 0.764
#     mask = (lags >= deltanu / 2)
#     corr = np.abs(corr[mask] / np.max(corr[mask]))

#     plt.figure()
#     plt.plot(lags[mask], corr)
#     for factor in [1.0, 2.0, 3.0, 4.0]:
#         plt.axvline(factor * deltanu, c='r')
#         plt.axhline(np.median(corr), c='blue')
#         plt.axhline(5*np.median(corr), c='blue', ls='--')

#     return lags, corr

# def psps(frequency, power):
#     df = np.mean(np.diff(frequency))
#     power -= np.mean(power)
#     p = np.fft.rfft(power)
#     p = np.abs(p)**2
#     f = np.fft.rfftfreq(power.size, d=df)

#     import matplotlib.pyplot as plt
#     plt.figure()
#     plt.plot(f,p)
#     return None

# # def identify_deltanu(lags, corr, numax)