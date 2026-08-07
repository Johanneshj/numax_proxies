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
        self.bin_centers, self.freq_windows, self.power_windows = log_numax_binning(
            frequency=self.frequency, 
            power=self.power,
            min_freq=self.cov_config.min_freq,
            max_freq=self.cov_config.max_freq,
            overlap_scales=self.cov_config.overlap_scale,
            width_factors=self.cov_config.width_factor
        )

        # Generate CoV values
        self.CoVs = calculate_CoVs(
            bin_centers = self.bin_centers,
            power_windows = self.power_windows
        )

        # Smooth CoV values
        self.smoothed_CoVs = smooth_CoVs(
            bin_centers = self.bin_centers,
            CoVs = self.CoVs,
            cov_config = self.cov_config
        )

        # Evaluate FAPs
        self.FAPs = evaluate_faps(
            bin_centers = self.bin_centers,
            length = np.max([np.min([self.length_timeseries, 1400]), 27]),
            cadence = np.max([np.min([self.cadence_timeseries, 1800]), 20]),
            overlap_scales = self.cov_config.overlap_scale,
            width_factors = self.cov_config.width_factor,
            FAP_threshold=self.cov_config.FAP_threshold
        )

        self.results, self.fit_vals = global_fitting(
            bin_centers     = self.bin_centers,
            CoVs            = self.CoVs,
            smoothed_CoVs   = self.smoothed_CoVs,
            FAPs            = self.FAPs,
            initial_numax   = self.initial_numax
        )           

        return self
    
    
    @property
    def numax_estimate(self):
        """Get numax and uncertainty"""
        return self.numax
    
    def plot(self):
        """Plot if specified"""

        fig, ax = plt.subplots()
        plot_CoV(
            bin_centers = self.bin_centers,
            CoVs = self.CoVs,
            smoothed_CoVs = self.smoothed_CoVs,
            FAPs = self.FAPs,
            global_fit_vals = self.fit_vals,
            initial_numax = self.initial_numax,
            ax = ax,
            target = self.id,
            cov_config = self.cov_config
        )
        savepath = os.path.join(self.config.results_directory, self.id, "figures")
        os.makedirs(savepath, exist_ok=True)
        fig.savefig(f"{savepath}/CoVs.png", dpi=300, bbox_inches="tight")

    def save_all_data(self):
        """Save CoV calculations to txt file"""
        # Save path location
        savepath = Path(self.config.results_directory) / str(self.id) / "CoV_info"
        savepath.mkdir(parents=True, exist_ok=True)


        # Flattened lists
        bcs = self.bin_centers
        usacfs = self.CoVs
        sacfs = self.smoothed_CoVs

        nested_dict = {
            'star'  : self.id,
            'data'  : []
        }

        # Iterate over all bin centers, unsmoothed CoVs, and smoothed CoVs
        # Also append fit values
        i = 0
        z = 0
        for ov_scale in self.cov_config.overlap_scale:
            for w_fac in self.cov_config.width_factor:

                data_entry = {
                    'overlap_scale'     : ov_scale,
                    'width_factor'      : w_fac,
                    'bin_centers'       : bcs[i],
                    'unsmoothed_CoV'    : usacfs[i],
                    'smoothed_CoVs'     : []
                }

                z = 0
                for smoothing_fac in self.cov_config.smoothing_factor:
                    data_entry['smoothed_CoVs'].append({
                        'smoothing_factor'  : smoothing_fac,
                        'smoothed_CoV'      : sacfs[i][z],
                        'fit_vals'          : self.fit_vals[z]
                    })
                    z += 1
                i += 1

            nested_dict['data'].append(data_entry)

        with gzip.open(f"{savepath}/all_data.pkl.gz", "wb") as f:
            pickle.dump(nested_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def save_numax_estimates(self):
        """Save only numax estimates"""
        savepath = Path(self.config.results_directory) / str(self.id) / "CoV_info"
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