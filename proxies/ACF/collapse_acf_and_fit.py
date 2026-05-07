import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
import warnings

def collapsed_acf(acf=None, freq_windows=None, sliding_window_flag='log_numax', *args, **kwargs):
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
    if sliding_window_flag == 'linear':
        smoothed_acf = unsmoothed_acf
    else:
        smoothed_acf = [
            smoothing_func(center, frequency, unsmoothed_acf) for center in frequency
        ]
    
    # Regularize smoothed ACF
    smoothed_acf = normalize_0_1(smoothed_acf)
    
    return smoothed_acf, unsmoothed_acf, frequency

def collapse_segment(seg):
    """Collapse segment of total ACF"""
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
    

def fit_gauss_to_collapsed_acf(smoothed_acf=None, freq_centers=None, initial_numax=None, full_pg=None):
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

    # We do a "try" here in case fits fails or is underresolved
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            # Initial guesses
            amp0 = 0.8 * np.max(y)
            w0 = (2/3) * numax0 ** (22/25)
            p0s = [amp0, w0, numax0]
            # Bounds
            lower_bounds = [0, 0, np.min(x)]
            upper_bounds = [1, 2*w0, np.max(x)]
            popt, pcov = curve_fit(
                gaussian,
                x,
                y,
                p0=p0s,
                bounds=(
                    lower_bounds,
                    upper_bounds
                )
            )

            numax = popt[2]
            return numax, popt

    except (RuntimeError, OptimizeWarning, ValueError):
        return np.nan, np.nan

def smoothing_func(center, bin_centers, ys):
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

def normalize_0_1(x):
    x = np.asarray(x)
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    return (x - xmin) / (xmax - xmin)

def gaussian(x, A, sigma, mu):
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