import numpy as np
from matplotlib.colors import LogNorm
import numpy as np
from numpy.typing import NDArray
from ...data_preparation.dataclasses import ACFConfig


def plot_spec(frequency : NDArray, power : NDArray, smoothed_power : NDArray, ax : NDArray, id : str):
    """Plot spectrum."""
    ax.loglog(frequency, power, c="gray", label="mean spectrum")
    ax.loglog(frequency, smoothed_power, c="k", label="median filter")
    ax.set_xlabel("frequency [μHz]")
    ax.set_ylabel("power spectral density")
    ax.set_xlim(min(frequency), max(frequency))
    ax.text(0.02, 0.02, f"{id}", ha="left", va="bottom", transform=ax.transAxes)
    ax.legend()

def plot_collapsed_acf_with_gaussian_fit(
        collapsed_2D_acf : list[NDArray], 
        unsmoothed_acf : list[NDArray], 
        freq_centers : list[NDArray], 
        global_fit_vals : list[list], 
        initial_numax : float, 
        ax : NDArray,
        acf_config : ACFConfig
):
    """Plot collapsed ACF from log-sliding window with Gaussian fit"""
    def gaussian(x, A, sigma, mu, y):
        return y + A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    for bcs, u, s, fit_vals in zip(freq_centers, unsmoothed_acf, collapsed_2D_acf, global_fit_vals):
        ax.plot(bcs, u, c='gray', alpha=0.4, zorder=-3)
        ax.plot(bcs, s, c='k', alpha=0.4, zorder=-2)
        x = np.linspace(np.min(bcs), np.max(bcs), len(bcs)*10)
        ax.plot(x, gaussian(x, fit_vals["amp"], fit_vals["sigma"], fit_vals["numax"], fit_vals["y"]), c="r", alpha=0.4, zorder=-1)
        ax.axvline(fit_vals["numax"], c="b", ls="--", alpha=0.3, zorder=0)

    if acf_config.plot_log_scale:
        ax.set_xscale('log')

    if initial_numax:
        ax.axvline(initial_numax, c='green', ls='-.', label='initial guess')
    
    ax.set_xlabel("frequency [μHz]")
    ax.set_ylabel("norm. CACF strength")
    
    ax.legend()
