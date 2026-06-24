from .CoV import (
    bin_spectrum,
    plot_CoV_vs_bin_centers,
    smooth_CoV_values,
    numax_estimate_CoV,
    plot_CoV_Bell
)
from .CoV import Keaton_bell_alternative as Bell
import os
import numpy as np
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
        self.bin_centers, self.CoVs = bin_spectrum(
            frequency=self.frequency, 
            power=self.power,
            min_freq=self.cov_config.min_freq,
            overlap_factor=self.cov_config.overlap_factor,
            use_linear_bins=self.cov_config.use_linear_bins
        )
        # Smooth CoV values (black crosses)
        self.smoothed_CoVs = smooth_CoV_values(
            self.bin_centers, self.CoVs, self.cov_config.smoothing_width_factor
        )
        # Estimate numax
        numax, numax_err, self.fit_vals, self.succesful_fit = numax_estimate_CoV(
            self.bin_centers, self.smoothed_CoVs, self.initial_numax
        )

        self.numax = ufloat(numax, numax_err)
        return self
    
    def compute_Bell(self):
        """
            Compute numax using method of Bell et al. (2019).
            Note! should not be used together with compute() function.
        """
        self.bin_centers, self.CoVs, self.faps_CoV = Bell.bin_spectrum(
            frequency=self.frequency,
            power=self.power,
            overlap_factor=self.cov_config.overlap_factor,
            min_freq=self.cov_config.min_freq
        )
        self.smoothed_CoVs = Bell.smooth_CoV_values(
            self.bin_centers, self.CoVs
        )
        numax, numax_err, self.fit_vals, self.succesful_fit = Bell.numax_estimate_CoV(
            self.bin_centers,
            self.smoothed_CoVs,
            self.CoVs,
            self.faps_CoV,
            self.initial_numax
        )
        self.numax = ufloat(numax, np.abs(numax_err))
        return self

    @property
    def numax_estimate(self):
        """Get numax and uncertainty"""
        return self.numax
    
    def plot(self):
        """Plot if specified"""

        fig, ax = plt.subplots()
        plot_CoV_vs_bin_centers(
            self.bin_centers,
            self.CoVs,
            self.smoothed_CoVs,
            self.numax,
            ax=ax,
            id=self.id,
            initial_numax=self.initial_numax,
            fit_vals=self.fit_vals,
            succesful_fit=self.succesful_fit
        )
        savepath = os.path.join("numax_proxies", "results", self.id, "figures")
        os.makedirs(savepath, exist_ok=True)
        fig.savefig(f"{savepath}/CoVs.png", dpi=300, bbox_inches="tight")
    
    def plot_Bell(self):
        fig, ax = plt.subplots()
        plot_CoV_Bell(
            self.bin_centers,
            self.CoVs,
            self.smoothed_CoVs,
            self.faps_CoV,
            self.numax,
            ax=ax,
            id=self.id,
            initial_numax=self.initial_numax,
            fit_vals=self.fit_vals,
            succesful_fit=self.succesful_fit
        )
        savepath = os.path.join("numax_proxies", "results", self.id, "figures")
        os.makedirs(savepath, exist_ok=True)
        fig.savefig(f"{savepath}/CoVs_Bell.png", dpi=300, bbox_inches="tight")

    def save_to_txt(self):
        """Save CoV calculations to txt file"""
        results_dir = os.path.join("numax_proxies", "results", self.id)
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)

        # Save path location
        savepath = os.path.join("numax_proxies", "results", self.id, 'CoV_info')
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        # Save CoV info
        np.savetxt(
            fname = f'{savepath}/{self.id}_CoV.txt',
            X = np.column_stack((self.bin_centers, self.CoVs, self.smoothed_CoVs, self.faps_CoV)),
            header = 'bin_centers,CoVs,smoothed_CoVs,FAPs',
            delimiter = ','
        )

        # Save fitting parameters
        if self.succesful_fit:
            np.savetxt(
                fname = f'{savepath}/{self.id}_CoV_fit_params.txt',
                X = np.column_stack(self.fit_vals),
                header = 'amp,sigma,numax',
                delimiter = ','
            )