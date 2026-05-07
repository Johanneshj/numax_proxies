import numpy as np
from matplotlib.colors import LogNorm
import numpy as np


def plot_spec(frequency, power, smoothed_power, ax, id):
    ax.loglog(frequency, power, c="gray", label="mean spectrum")
    ax.loglog(frequency, smoothed_power, c="k", label="median filter")
    ax.set_xlabel("frequency [μHz]")
    ax.set_ylabel("power spectral density")
    ax.set_xlim(min(frequency), max(frequency))
    ax.text(0.02, 0.02, f"{id}", ha="left", va="bottom", transform=ax.transAxes)
    ax.legend()

def plot_collapsed_acf_with_gaussian_fit(collapsed_2D_acf, unsmoothed_acf, freq_centers, fit_vals, initial_numax, ax):
    def gaussian(x, A, sigma, mu):
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    
    def gauss_plus_exponential(x, A, sigma, mu, a, b):
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + a * np.exp(-b*x)

    # freq_centers = np.log10(freq_centers)
    x = np.linspace(np.min(freq_centers), np.max(freq_centers), len(freq_centers)*10)
    ax.set_xscale('log')
    ax.plot(freq_centers, unsmoothed_acf, c='gray', marker='^', label='unsmoothed ACF')
    ax.plot(freq_centers, collapsed_2D_acf, c="k", marker=".", label="collapsed 2D ACF")
    ax.plot(x, gaussian(x, *fit_vals), c="r", label="Gaussian fit")
    if initial_numax:
        ax.axvline(initial_numax, c='green', ls='-.', label='initial guess')
    ax.axvline(
        fit_vals[2], c="b", ls="--", label=f"numax = {np.round(fit_vals[2],2)}"
    )
    ax.set_xlabel("frequency [μHz]")
    ax.set_ylabel("A.U.")
    
    ax.legend()
