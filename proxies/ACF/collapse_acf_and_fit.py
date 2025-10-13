import numpy as np
from scipy.optimize import curve_fit

def collapsed_acf(acf=None, freq_windows=None, *args, **kwargs):
    
    '''
        Collapse 2D ACF into 1D ACF

        Input:
            acf :: 2D ACF
            freq_windows :: binned frequency list

        Output:
            collapsed_acf_numax :: collapsed 1D acf
            freq_centers :: medians of freq_windows for plotting and fitting
    '''

    collapsed_acf_numax = np.mean(acf, axis=1)
    collapsed_acf_numax = (collapsed_acf_numax - np.median(collapsed_acf_numax))
    collapsed_acf_numax /= np.max(collapsed_acf_numax)
    freq_centers = np.median(freq_windows, axis=1)
    # if plot:
    #     plt.figure()
    #     plt.plot(freq_centers, collapsed_acf_numax)
    #     plt.xlabel("Frequency [μHz]")
    #     plt.ylabel("Collapsed ACF amplitude")
    #     plt.title("Collapsed ACF illustrating νmax")
    return collapsed_acf_numax, freq_centers 

def fit_gauss_to_collapsed_acf(collapsed_acf_numax=None, freq_centers=None):

    '''
        Fit Gaussian to collapsed ACF: 
            First estimates:
                - width = (2/3) * numax ** (22/25) (Mosser+ 2012)
                - Amplitude = 1 (because regularized)
                - mean = numax = "freq at maximum" (first numax guess)

        Input:
            collapsed_acf_numax :: collapsed 1D acf
            freq_centers :: medians of freq_windows for plotting and fitting
        
        Output:
            numax :: numax estimate (central value of Gauss) in muHz
    '''

    idx_max = np.argmax(collapsed_acf_numax[1:])
    numax = freq_centers[idx_max]

    collapsed_acf_numax = (collapsed_acf_numax - np.median(collapsed_acf_numax))
    collapsed_acf_numax /= np.max(collapsed_acf_numax) # Regularizing

    window = 4 * (2/3) * numax ** (22/25) # Mosser
    mask = (freq_centers > numax - window ) & (freq_centers < numax + window)
    # plt.plot(freq_centers[mask], collapsed_acf_numax[mask], marker='.')
    # plt.show()
    
    def gaussian(x, A, sigma, mu):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

    popt, pcov = curve_fit(gaussian, freq_centers[mask], collapsed_acf_numax[mask], p0=[1, (2/3)*numax**(22/25), numax])
    # x = np.linspace(numax - 1500, numax + 2500, 1000)
    # plt.plot(freq_centers[mask], collapsed_acf_numax[mask], marker='.')
    # plt.plot(x, gaussian(x, *popt), c='k', ls='-', label=f'{popt[2]:.2f} ± {np.sqrt(pcov[2,2]):.2f} μHz')
    # plt.legend()
    # plt.xlabel("Frequency [μHz]")
    # plt.ylabel("Collapsed ACF amplitude")
    # plt.title("Collapsed ACF illustrating νmax")
    # plt.show()
    numax = popt[2]
    return numax, popt