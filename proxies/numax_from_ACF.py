from .ACF import (
    calculate_relative_power,
    calculate_two_dim_ACF,
    collapsed_acf,
    fit_gauss_global,
    plot_spec,
    plot_collapsed_acf_with_gaussian_fit,
    evaluate_faps
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
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

class NumaxFromACF:
    def __init__(
        self,
        avg_psd : AvgPSDData,
        acf_config : ACFConfig,
        config : ProcessingConfig,
        length_timeseries : Optional[float] = None,
        cadence_timeseries : Optional[float] = None,
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

        self.length_timeseries = length_timeseries
        self.cadence_timeseries = cadence_timeseries

    def compute(self):
        """Perform 2D ACF computations"""

        # Normalize spectrum
        self.normalized_power, self.med_filter = calculate_relative_power(
            self.frequency, self.avg_psd
        )
        
        # Calculate 2D ACF
        self.two_dim_acf_results = calculate_two_dim_ACF(
            frequency   = self.frequency, 
            power       = self.normalized_power,
            acf_config  = self.acf_config
        )

        # Collapse 2D ACF and smooth and generate flattened bin_centers
        self.cacf_results = collapsed_acf(
            two_dim_acf_results = self.two_dim_acf_results,  
            acf_config          = self.acf_config
        )

        # Evaluate FAP
        self.cacf_results = evaluate_faps(
            cacf_results    = self.cacf_results,
            L               = np.max([np.min([self.length_timeseries, 2500]), 20]),
        )
        # print(self.FAPs)
        # print(len(self.bin_centers), len(self.unsmoothed_cacfs), len(self.smoothed_cacfs), len(self.FAPs))
        # Fit gauss to estimate numax
        self.results = fit_gauss_global(
            cacf_results            = self.cacf_results, 
            initial_numax           = self.initial_numax,
            max_acf_fit_iterations  = self.acf_config.max_acf_fit_iterations,
            n_sigma_numax_acf       = self.acf_config.n_sigma_numax_acf,
        )
        return self

    def plot(self):
        """Plot 2D ACF computations if specified"""
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 1, figsize=(5, 7))
        plot_spec(
            self.frequency,
            self.avg_psd,
            self.med_filter,
            ax=axs[0],
            id=self.id,
            acf_config=self.acf_config
        )
        plot_collapsed_acf_with_gaussian_fit(
            results         = self.results, 
            initial_numax   = self.initial_numax,
            ax              = axs[1], 
            acf_config      = self.acf_config
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

        with gzip.open(f"{savepath}/all_data.pkl.gz", "wb") as f:
            pickle.dump(self.results, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_numax_estimates(self):
        """Save only numax estimates"""
        savepath = Path(self.config.results_directory) / str(self.id) / "CoV_info"
        savepath.mkdir(parents=True, exist_ok=True)

        numax_objs = [
            data["fit_numax"] 
            for data in self.results.values() 
            if "fit_numax" in data
        ]

        values = np.column_stack([
            [x.n for x in numax_objs],
            [x.s for x in numax_objs],
        ])

        np.savetxt(
            fname = f"{savepath}/numax_estimates.txt",
            X = values,
            header = 'numax,numax_err',
            delimiter=',',
            fmt="%.4f"
        )

    @property   
    def numax_estimate(self) -> ufloat:
        """Return median numax estimate and error across all parameter combinations."""
        # Extract fit_numax objects from each parameter sub-dictionary
        numax_objs = [
            data["fit_numax"] 
            for data in self.results.values() 
            if "fit_numax" in data
        ]

        if not numax_objs:
            return ufloat(np.nan, np.nan)

        # Extract nominal value (.n) and standard error (.s) safely
        numaxes = np.array([
            x.n for x in numax_objs
        ])
        errs = np.array([
            x.s for x in numax_objs
        ])

        # Filter out NaNs across both nominal values and errors
        valid_mask = ~np.isnan(numaxes) & ~np.isnan(errs)

        if not np.any(valid_mask):
            return ufloat(np.nan, np.nan)

        median_numax = np.median(numaxes[valid_mask])
        median_err = np.median(errs[valid_mask])

        return ufloat(median_numax, median_err)
    

