from .ACF import (
    calculate_relative_power,
    calculate_two_dim_ACF,
    collapsed_acf,
    fit_gauss_global,
    plot_spec,
    plot_collapsed_acf_with_gaussian_fit,
)
import json
import os
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Literal
from ..data_preparation.dataclasses import AvgPSDData, ACFConfig, ProcessingConfig
import gzip
import pickle
import pandas as pd
from uncertainties import ufloat

class NumaxFromACF:
    def __init__(
        self,
        avg_psd : AvgPSDData,
        acf_config : ACFConfig,
        config : ProcessingConfig,
        id : Optional[str] = "unknown",
        initial_numax : Optional[float] = None
    ):
            
        """Initialization"""
        # Frequency and power
        self.frequency = avg_psd.frequency
        self.avg_psd = avg_psd.psd
        
        # ACF configuration parameters and global config (only needed for noise_std)
        self.acf_config = acf_config
        self.config = config

        # identifier and initial numax
        self.id = id
        self.initial_numax = initial_numax

    def compute(self):
        """Perform 2D ACF computations"""
        # Normalize spectrum
        self.normalized_power, self.med_filter = calculate_relative_power(
            self.frequency, self.avg_psd
        )
        # Calculate 2D ACF
        self.bin_centers, self.freq_windows, self.acfs = calculate_two_dim_ACF(
            frequency   = self.frequency, 
            power       = self.normalized_power,
            acf_config  = self.acf_config
        )
        # Collapse 2D ACF and smooth and generate flattened bin_centers
        self.bin_centers, self.unsmoothed_cacfs, self.smoothed_cacfs = collapsed_acf(
            bin_centers     = self.bin_centers,  
            acfs            = self.acfs,
            acf_config      = self.acf_config
        )
        # Fit gauss to estimate numax
        self.results, self.fit_vals = fit_gauss_global(
            bin_centers             = self.bin_centers, 
            smoothed_cacfs          = self.smoothed_cacfs, 
            initial_numax           = self.initial_numax,
            max_acf_fit_iterations  = self.acf_config.max_acf_fit_iterations,
            n_sigma_numax_acf       = self.acf_config.n_sigma_numax_acf,
        )

        return self

    def plot(self):
        """Plot 2D ACF computations if specified"""
        import matplotlib.pyplot as plt
    
        fig, axs = plt.subplots(2, 1, figsize=(6, 8))
        plot_spec(
            self.frequency,
            self.avg_psd,
            self.med_filter,
            ax=axs[0],
            id=self.id,
        )
        plot_collapsed_acf_with_gaussian_fit(
            self.smoothed_cacfs, self.unsmoothed_cacfs, 
            self.bin_centers, self.fit_vals, self.initial_numax,
            ax=axs[1], acf_config=self.acf_config
        )
        savepath = os.path.join(self.config.results_directory, self.id, "figures")
        os.makedirs(savepath, exist_ok=True)
        if self.config.noise_std > 0:
            fig.savefig(
                f"{savepath}/ACF_noise-{self.config.noise_std}ppm.png", dpi=300, bbox_inches="tight"
            )
        fig.savefig(f"{savepath}/ACF.png", dpi=300, bbox_inches="tight")

    def save_all_data(self):
        """Save ACF calculations to txt file"""
        # Save path location
        savepath = Path(self.config.results_directory) / str(self.id) / "ACF_info"
        savepath.mkdir(parents=True, exist_ok=True)


        # Flattened lists
        bcs = self.bin_centers
        usacfs = self.unsmoothed_cacfs
        sacfs = self.smoothed_cacfs

        nested_dict = {
            'star'  : self.id,
            'data'  : []
        }

        # Iterate over all bin centers, unsmoothed cacfs, and smoothed cacfs
        # Also append fit values
        i = 0
        z = 0
        for ov_scale in self.acf_config.overlap_scale:
            for w_fac in self.acf_config.width_factor:

                data_entry = {
                    'overlap_scale'     : ov_scale,
                    'width_factor'      : w_fac,
                    'bin_centers'       : bcs[i],
                    'unsmoothed_cacf'   : usacfs[i],
                    'smoothed_cacfs'    : []
                }


                for smoothing_fac in self.acf_config.smoothing_factor:
                    data_entry['smoothed_cacfs'].append({
                        'smoothing_factor': smoothing_fac,
                        'smoothed_cacf'   : sacfs[z],
                        'fit_vals'        : self.fit_vals[z]
                    })
                    z += 1
                i += 1

            nested_dict['data'].append(data_entry)

        with gzip.open(f"{savepath}/all_data.pkl.gz", "wb") as f:
            pickle.dump(nested_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_numax_estimates(self):
        """Save only numax estimates"""
        savepath = Path(self.config.results_directory) / str(self.id) / "ACF_info"
        savepath.mkdir(parents=True, exist_ok=True)

        values = np.column_stack([
            [x.n for x in self.results],
            [x.s for x in self.results],
        ])

        np.savetxt(
            fname = f"{savepath}/numax_estimates.txt",
            X = values,
            header = 'numax,numax_err',
            delimiter=',',
            fmt="%.4f"
        )

    @property   
    def numax_estimate(self) -> float:
        """Return mean numax estimate"""
        numaxes = [x.n for x in self.results]
        errs = [x.s for x in self.results]
        return ufloat(np.mean(numaxes), np.mean(errs))
    

