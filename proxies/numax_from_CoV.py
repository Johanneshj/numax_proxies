from .CoV import (
    log_numax_binning,
    calculate_CoVs,
    smooth_CoVs,
    evaluate_faps,
    global_fitting,
    plot_CoV
    # numax_estimate_CoV,
    # plot_CoV_Bell
)
from pathlib import Path
# from .CoV import Keaton_bell_alternative as Bell
import os
import numpy as np
import gzip
import pickle
from uncertainties import ufloat
import matplotlib.pyplot as plt
from typing import Optional, Literal
from ..data_preparation.dataclasses import PSDData, ProcessingConfig, COVConfig


class NumaxFromCoefficientsOfVariation:
    def __init__(
            self, 
            psd : PSDData, 
            config : ProcessingConfig,
            cov_config : COVConfig,
            length_timeseries : Optional[float] = None,
            cadence_timeseries : Optional[float] = None,
            id : Optional[str] = "unknown", 
            initial_numax : Optional[float] = None
        ):
        """
            Initialize class: lightcurve, periodogram, identifier, initial numax
            Allocate properties we'll need later.
        """
        self.id = id or "unknown"
        self.frequency = psd.frequency
        self.power = psd.psd
        self.config = config
        self.cov_config = cov_config
        self.length_timeseries = length_timeseries
        self.cadence_timeseries = cadence_timeseries
        self.initial_numax = initial_numax

    def compute(self):
        """
            Compute numax from CoV;
            1. Spectrum is binned and CoVs calculated.
            2. CoV values are smoothed with moving average 
                in window sizes corresponding to width of potential oscillation envelope.
            3. Numax is estimated.
            4. Numax is defined as ufloat object.
        """
        # Bin spectrum (grey diamonds in plot)
        self.binning_results = log_numax_binning(
            frequency=self.frequency, 
            power=self.power,
            min_freq=self.cov_config.min_freq,
            max_freq=self.cov_config.max_freq,
            overlap_scales=self.cov_config.overlap_scale,
            width_factors=self.cov_config.width_factor
        )

        # Generate CoV values
        self.CoVs = calculate_CoVs(
            binning_results = self.binning_results
        )

        # Smooth CoV values
        self.CoVs = smooth_CoVs(
            cov_results = self.CoVs,
            cov_config = self.cov_config
        )        

        # Evaluate FAPs
        self.CoVs = evaluate_faps(
            CoV_results = self.CoVs,
            L           = np.max([np.min([self.length_timeseries, 2500]), 20]),
        )

        # Results
        self.results = global_fitting(
            CoV_results     = self.CoVs,
            initial_numax   = self.initial_numax
        )           

        return self
    
    
    # @property
    # def numax_estimate(self):
    #     """Get numax and uncertainty"""
    #     return self.numax
    
    def plot(self):
        """Plot if specified"""

        fig, ax = plt.subplots(figsize=(5,4))
        plot_CoV(
            cov_results = self.results,
            initial_numax = self.initial_numax,
            ax = ax,
            target = self.id,
            cov_config = self.cov_config
        )
        savepath = os.path.join(self.config.results_directory, self.id, "figures")
        os.makedirs(savepath, exist_ok=True)
        fig.savefig(f"{savepath}/CoVs.png", dpi=300, bbox_inches="tight")

    def save_all_data(self):
            """Save ACF calculations to txt file"""
            # Save path location
            savepath = Path(self.config.results_directory) / str(self.id) / "CoV_info"
            savepath.mkdir(parents=True, exist_ok=True)
    
            with gzip.open(f"{savepath}/all_data.pkl.gz", "wb") as f:
                pickle.dump(self.results, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def save_numax_estimates(self):
        """Save only numax estimates"""
        savepath = Path(self.config.results_directory) / str(self.id) / "ACF_info"
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