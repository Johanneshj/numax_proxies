# Plotting routines in case we use linear sliding window

import numpy as np
from matplotlib.colors import LogNorm
from numpy.typing import NDArray

def plot_spec_linear(frequency : NDArray, power : NDArray, smoothed_power : NDArray, ax : NDArray, id : str):
    """Plot spectrum if we used linear sliding window."""
    ax.loglog(frequency, power, c="gray", label="mean spectrum")
    ax.loglog(frequency, smoothed_power, c="k", label="median filter")
    ax.set_xlabel("frequency [μHz]")
    ax.set_ylabel("power spectral density")
    ax.set_xlim(min(frequency), max(frequency))
    ax.text(0.02, 0.02, f"{id}", ha="left", va="bottom", transform=ax.transAxes)
    ax.legend()

def plot_2D_ACF_linear(ACF : NDArray, frequency : NDArray, ax : NDArray):
    """Plot heatmap of 2D ACF."""
    if np.max(frequency) > 300:
        window_size_muHz = 249
        step = 10
    else:
        step = 2
        window_size_muHz = 49
    ax.imshow(
        ACF[::step, ::step].T,
        cmap="Greys_r",
        aspect="auto",
        norm=LogNorm(vmin=0.001, vmax=1),
        origin="lower",
        extent=(frequency[0], frequency[-1], 0, window_size_muHz),
    )
    ax.set_xlabel("frequency [μHz]")
    ax.set_ylabel("spacing [μHz]")

def plot_collapsed_acf_with_gaussian_fit_linear(collapsed_2D_acf : NDArray, freq_centers : NDArray, fit_vals : NDArray, ax : NDArray):
    """Plot ACF with Gaussian fit in linear sliding window."""
    def gaussian(x, A, sigma, mu):
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    x = np.linspace(min(freq_centers), max(freq_centers), 1000)
    ax.plot(freq_centers, collapsed_2D_acf, c="k", marker=".", label="collapsed 2D ACF")
    ax.plot(x, gaussian(x, *fit_vals), c="r", label="Gaussian fit")
    ax.axvline(
        fit_vals[2], c="gray", ls="--", label=f"numax = {np.round(fit_vals[2],2)}"
    )
    ax.set_xlabel("frequency [μHz]")
    ax.set_ylabel("A.U.")
    ax.legend()