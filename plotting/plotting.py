import numpy as np
import matplotlib.pyplot as plt
import os
from ..data_preparation.dataclasses import PSDData, StarInfo, ProcessingConfig
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

def plot_spectrum_with_all_numax_estimates(
        psd : PSDData, 
        star : StarInfo, 
        numax_estimates : dict,
        config : ProcessingConfig
):
    fig, ax = plt.subplots(figsize=(4,3))
    ax.loglog(psd.frequency, psd.psd, c="gray")
    for label, numax in numax_estimates.items():
        try:
            numax_val = numax.n
            numax_err = numax.s
            val = rf"{numax_val:.1f} ± {numax_err:.1f}"
        except:
            numax_val = numax
            numax_err = None
            val = f"{numax_val:.1f}"

        if "CoV" in label:
            ls = (0, (3, 1, 1, 1))
            c = "mediumorchid"
            line_label = r"$\nu_\text{max}^\text{CoV}$: " + val
        elif "2DACF" in label:
            ls = (0, (5, 1))
            c = "forestgreen"
            line_label = r"$\nu_\text{max}^\text{2DACF}$: " + val
        elif "SR" in label:
            ls = (0, (1, 5))
            c = "dodgerblue"
            if "logg_teff" in label:
                line_label = r"$\nu_\text{max}^{gT_\text{eff}^{-1/2}}$: " + val
            else:
                line_label = r"$\nu_\text{max}^\text{SR}$: " + val
            # More to be added here
        elif "FliPer" in label:
            ls = (5, (10, 3))
            c = "darkorange"
            line_label = r"$\nu_\text{max}^\text{FliPer}$: " + val

        ax.axvline(numax_val, linestyle=ls, c=c, label=line_label)
        if numax_err is not None:
            ax.axvspan(numax_val - numax_err, numax_val + numax_err, alpha=0.2, color=c)

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power")

    ax.set_xlim(np.min(psd.frequency), np.max(psd.frequency))

    ax.legend(loc="lower left", title=f"{star.target} " + r"$\nu_\text{max} \ [\mu\text{Hz}]$")
    
    # savepath = os.path.join("numax_proxies", "results", f'{star.target}', "figures")
    savepath = os.path.join(config.results_directory, star.target, "figures")
    os.makedirs(savepath, exist_ok=True)
    fig.savefig(
        f"{savepath}/full_spectrum_with_all_estimates.png", dpi=300, bbox_inches="tight"
    )
