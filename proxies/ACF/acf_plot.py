import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
from numpy.typing import NDArray
from ...data_preparation.dataclasses import ACFConfig
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
from typing import Dict, Tuple

def plot_spec(frequency : NDArray, power : NDArray, smoothed_power : NDArray, ax : NDArray, id : str, acf_config : ACFConfig):
    """Plot spectrum."""
    ax.loglog(frequency, power, c="gray", label="mean spectrum")
    ax.loglog(frequency, smoothed_power, c="k", label="median filter")
    ax.set_xlabel(r"frequency [$\mu\text{Hz}$]")
    ax.set_ylabel(r"PSD [$\text{ppm}^2/\mu\text{Hz}$]")
    ax.set_xlim(min(frequency), max(frequency))
    ax.text(0.02, 0.98, f"{id}", ha="left", va="top", transform=ax.transAxes)
    ax.legend(loc='lower left')
    if acf_config.max_freq is not None:
        ax.set_xlim(acf_config.min_freq, acf_config.max_freq)
    else:
        ax.set_xlim(acf_config.min_freq, np.max(frequency))

def plot_collapsed_acf_with_gaussian_fit(
    results: Dict[Tuple[float, ...], dict],
    initial_numax: float,
    ax,
    acf_config: ACFConfig,
):
    """
    Plot collapsed ACF curves, FAP thresholds, and Gaussian fits from cacf_results dictionary.
    """
    def gaussian(x, A, sigma, mu, y):
        return y + A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    cacf_key = "cacf_smoothed"

    # Single pass loop across all parameter combinations
    for data in results.values():
        bcs = np.asarray(data["bin_centers"])
        u = np.asarray(data["cacf_raw"])
        s = np.asarray(data[cacf_key])

        # 1. Plot raw and smoothed CACFs
        ax.plot(bcs, u, c='gray', alpha=0.4, zorder=-3)
        ax.plot(bcs, s, c='k', alpha=0.4, zorder=-2)

        # 2. Plot FAP threshold curve if present
        if "fap" in data and data["fap"] is not None:
            ax.plot(bcs, data["fap"], c='green', alpha=0.1, zorder=0)

        # 3. Plot Gaussian fit and numax vertical line if present
        if "fit_params" in data and data["fit_params"] is not None:
            fit_vals = data["fit_params"]
            
            # Extract nominal float value if numax is a ufloat object
            numax_val = float(getattr(fit_vals["numax"], "n", fit_vals["numax"]))
            
            x = np.linspace(np.min(bcs), np.max(bcs), len(bcs) * 10)
            y_fit = gaussian(x, fit_vals["amp"], fit_vals["sigma"], numax_val, fit_vals["y"])

            ax.plot(x, y_fit, c="r", alpha=0.4, zorder=-1)
            ax.axvline(numax_val, c="b", ls="--", alpha=0.3, zorder=1)

    # Define custom legend handles matching the plot aesthetics
    legend_elements = [
        Line2D([0], [0], color='gray', alpha=1.0, label='unsmoothed CACF'),
        Line2D([0], [0], color='k', alpha=1.0, label='smoothed CACF'),
        Line2D([0], [0], color='r', alpha=1.0, label='Gaussian fit'),
        Line2D([0], [0], color='green', alpha=1.0, label='FAP'),
        Line2D([0], [0], color='b', ls='--', alpha=1.0, label=r'$\nu_\text{max}$ estimate'),
    ]

    if getattr(acf_config, "plot_log_scale", False):
        ax.set_xscale('log')

    if initial_numax:
        ax.axvline(initial_numax, c='green', ls='-.')
        legend_elements.append(
            Line2D([0], [0], color='green', ls='-.', label='initial guess')
        )

    ax.legend(handles=legend_elements)

    ax.set_xlabel(r"frequency [$\mu\text{Hz}$]")
    ax.set_ylabel("CACF strength")

    # Determine x-axis limits from acf_config or maximum frequency across all bins
    all_max_freqs = [np.max(data["bin_centers"]) for data in results.values()]
    fallback_max_freq = max(all_max_freqs) if all_max_freqs else 10000.0

    min_limit = getattr(acf_config, "min_freq", 100.0)
    max_limit = (
        acf_config.max_freq 
        if getattr(acf_config, "max_freq", None) is not None 
        else fallback_max_freq
    )

    ax.set_xlim(min_limit, max_limit)
