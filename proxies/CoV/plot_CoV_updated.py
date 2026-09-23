from ...data_preparation.dataclasses import COVConfig
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
from typing import Dict, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from matplotlib.lines import Line2D

def gaussian_with_offset(x: NDArray, A: float, sigma: float, mu: float) -> NDArray:
    """Gaussian function with an offset of 1."""
    return 1.0 + A * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))

def plot_CoV(
    cov_results: Dict[Tuple[float, ...], dict],
    initial_numax: Optional[float], 
    ax,
    target: str,
    cov_config: COVConfig
):
    """
    Plot CoVs, smoothed CoVs, FAPs, and Gaussian fits from the cov_results dictionary.
    """
    for data in cov_results.values():
        bcs = np.asarray(data["bin_centers"])
        covs = np.asarray(data["cov"])
        scov = np.asarray(data["cov_smoothed"])

        # 1. Plot raw unsmoothed and smoothed CoV curves
        ax.plot(bcs, covs, c='gray', alpha=0.3, zorder=-5)
        ax.plot(bcs, scov, c='k', alpha=0.3, zorder=-4)

        # 2. Plot FAP threshold if available
        if "fap" in data and data["fap"] is not None:
            ax.plot(bcs, np.asarray(data["fap"]), c='green', alpha=0.3, zorder=-3)

        # 3. Plot Gaussian fit if fitting parameters exist and are non-NaN
        fit_vals = data.get("fit_params")
        if fit_vals is not None:
            amp = fit_vals.get("amp")
            sigma = fit_vals.get("sigma")
            numax = fit_vals.get("numax")

            # Extract float values safely if parameters are ufloat objects
            amp_val = float(getattr(amp, "n", amp)) if amp is not None else np.nan
            sigma_val = float(getattr(sigma, "n", sigma)) if sigma is not None else np.nan
            numax_val = float(getattr(numax, "n", numax)) if numax is not None else np.nan

            if not (np.isnan(amp_val) or np.isnan(sigma_val) or np.isnan(numax_val)):
                x = np.linspace(np.min(bcs), np.max(bcs), len(bcs) * 10)
                fit_curve = gaussian_with_offset(x, amp_val, sigma_val, numax_val)

                ax.plot(x, fit_curve, c="r", alpha=0.3, zorder=-2)
                ax.axvline(numax_val, c="b", ls="--", alpha=0.3, zorder=-1)

    # Define custom legend handles
    legend_elements = [
        Line2D([0], [0], color='gray', alpha=1.0, label='unsmoothed CoV'),
        Line2D([0], [0], color='k', alpha=1.0, label='smoothed CoV'),
        Line2D([0], [0], color='green', alpha=1.0, label='FAP'),
        Line2D([0], [0], color='r', alpha=1.0, label='Gaussian fit'),
        Line2D([0], [0], color='b', ls='--', alpha=1.0, label=r'$\nu_\text{max}$ estimate'),
    ]

    if initial_numax and not np.isnan(initial_numax):
        ax.axvline(initial_numax, c='green', ls='-.')
        legend_elements.append(
            Line2D([0], [0], color='green', ls='-.', label='initial guess')
        )

    ax.legend(handles=legend_elements)

    if target:
        ax.text(0.98, 0.02, f"{target}", ha="right", va="bottom", transform=ax.transAxes)

    ax.set_xlabel(r"frequency [$\mu\text{Hz}$]")
    ax.set_ylabel("coefficient of variation")

    ax.set_xlim(min(bcs), max(bcs))

    if getattr(cov_config, "plot_log_scale", False):
        ax.set_xscale('log')