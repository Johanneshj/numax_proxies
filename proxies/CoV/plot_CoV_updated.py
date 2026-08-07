import numpy as np
from numpy.typing import NDArray
from ...data_preparation.dataclasses import COVConfig

def plot_CoV(
        bin_centers     :   list[NDArray],
        CoVs            :   list[NDArray],
        smoothed_CoVs   :   list[NDArray],
        FAPs            :   list[NDArray],
        global_fit_vals :   list[list],
        initial_numax   :   float, 
        ax              :   NDArray,
        target          :   str,
        cov_config      :   COVConfig
):
    # Plot CoVs, smoothed CoVs, and FAPs against bin_centers
    i = 0
    for bcs, covs, scovs, faps in zip(bin_centers, CoVs, smoothed_CoVs, FAPs):
        ax.plot(bcs, covs, c='gray', alpha=0.3, zorder=-5)
        ax.plot(bcs, faps, c='green', alpha=0.3, zorder=-3)
        x = np.linspace(np.min(bcs), np.max(bcs), len(bcs)*10)
        for scov in scovs:
            fit_vals = global_fit_vals[i]
            if fit_vals is None or any(np.isnan(val) for val in fit_vals.values() if val is not None):
                continue
            ax.plot(bcs, scov, c='k', alpha=0.3, zorder=-4)
            ax.plot(x, gaussian_with_offset(x, fit_vals["amp"], fit_vals["sigma"], fit_vals["numax"]), c="r", alpha=0.3, zorder=-2)
            ax.axvline(fit_vals["numax"], c="b", ls="--", alpha=0.3, zorder=-1)
            i += 1

    ax.text(0.98, 0.02, f"{target}", ha="right", va="bottom", transform=ax.transAxes)
    ax.set_xlabel("frequency [μHz]")
    ax.set_ylabel("coefficient of variation")

    if cov_config.plot_log_scale:
        ax.set_xscale('log')

    if initial_numax:
        ax.axvline(initial_numax, c='green', ls='-.', label='initial guess')

def gaussian_with_offset(x, A, sigma, mu):
    """Gaussian function"""
    return 1 + A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))