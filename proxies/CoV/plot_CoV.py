import numpy as np

def plot_CoV_vs_bin_centers(bin_centers, 
                            CoVs, 
                            smoothed_CoVs, 
                            numax, 
                            ax, 
                            id, 
                            initial_numax=None, 
                            fit_vals=None,
                            succesful_fit=False):
    """
        Plot CoV results.

        Inputs:
            bin_centers     : frequency values for which CoV values have been calculated
            CoVs            : unsmoothed CoV values
            smoothed_CoVs   : smoothed CoV values
            numax           : numax estimate from CoV method
            ax              : plotting axis
            id              : target identifier
            initial_numax   : initial numax guess
            fit_vals        : fitting values from gaussian fit
            succesful_fit   : flag indicating if fit was succesful (if False we don't plot)
        
        Outputs:
            A diagram
    """

    numax_err = numax.std_dev
    numax_val = numax.nominal_value
    # bin_centers = np.log10(bin_centers)
    ax.plot(bin_centers, CoVs, c="gray", marker="d", alpha=0.5, label="CoV values")
    ax.plot(bin_centers, smoothed_CoVs, c="black", marker="+", label="filtered CoV values")
    ax.axvline(
        numax_val,
        c="b",
        ls="--",
        label=f"νmax = {numax_val:.2f} ± {numax_err:.2f} μHz",
    )
    if initial_numax:
        ax.axvline(
        initial_numax,
        c="green",
        ls="-.",
        label="νmax guess",
    )
    if succesful_fit:
        def gaussian(x, A, sigma, mu):
            return 1 + A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
        x = np.linspace(np.min(bin_centers), np.max(bin_centers), 10000)
        ax.plot(x, gaussian(x, *fit_vals), c="r", label="Gaussian fit")
    
    ax.text(0.98, 0.02, f"{id}", ha="right", va="bottom", transform=ax.transAxes)
    ax.set_xscale("log")
    ax.set_xlabel("frequency [μHz]")
    ax.set_ylabel("coefficient of variation")
    ax.set_ylim(np.min(smoothed_CoVs), 3)
    ax.legend(loc="upper left")


def plot_supNyq_spec(freq, power, numax, ax, id):
    """Super nyquist spectrum ---> EXPERIMENTAL!"""
    numax_err = numax.std_dev
    numax_val = numax.nominal_value
    ax.plot(freq, power, c="k", label="super Nyquist spectrum")
    ax.axvline(
        numax_val,
        c="gray",
        ls="--",
        label=f"νmax = {numax_val:.2f} ± {numax_err:.2f} μHz",
    )
    ax.legend(loc="upper left")
    ax.set_xlabel("frequency [μHz]")
    ax.set_ylabel(r"PSD [$\text{ppm}^2$/\text{μHz}]")
    ax.legend(loc="upper left")

def plot_CoV_Bell(
        bin_centers, 
        CoVs, 
        smoothed_CoVs, 
        faps_CoV,
        numax, 
        ax, 
        id, 
        initial_numax=None, 
        fit_vals=None,
        succesful_fit=False
):
    numax_err = numax.std_dev
    numax_val = numax.nominal_value
    ax.plot(bin_centers, CoVs, c="gray", marker="d", alpha=0.5, label="CoV values")
    ax.plot(bin_centers, smoothed_CoVs, c="black", marker="+", label="filtered CoV values")
    ax.axvline(
        numax_val,
        c="b",
        ls="--",
        label=f"νmax = {numax_val:.2f} ± {numax_err:.2f} μHz",
    )
    if initial_numax:
        ax.axvline(
        initial_numax,
        c="green",
        ls="-.",
        label="νmax guess",
    )
    if succesful_fit:
        def gaussian(x, A, sigma, mu):
            return 1 + A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
        x = np.linspace(np.min(bin_centers), np.max(bin_centers), 10000)
        ax.plot(x, gaussian(x, *fit_vals), c="r", label="Gaussian fit")

    ax.plot(bin_centers, faps_CoV, c='green', ls=':', label='FAP')
    
    ax.text(0.98, 0.02, f"{id}", ha="right", va="bottom", transform=ax.transAxes)
    ax.set_xscale("log")
    ax.set_xlabel("frequency [μHz]")
    ax.set_ylabel("coefficient of variation")
    ax.set_ylim(np.min(smoothed_CoVs), 3)
    ax.legend(loc="upper left")  
