from .ACF import (
    calculate_relative_power,
    calculate_two_dim_ACF,
    collapsed_acf,
    fit_gauss_to_collapsed_acf,
    plot_spec,
    plot_collapsed_acf_with_gaussian_fit,
    plot_spec_linear,
    plot_collapsed_acf_with_gaussian_fit_linear,
    plot_2D_ACF_linear
)
import os
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Literal
from ..data_preparation.dataclasses import AvgPSDData, ACFConfig, ProcessingConfig


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
        self.twodim_ACF, self.freq_windows = calculate_two_dim_ACF(
            frequency   = self.frequency, 
            power       = self.normalized_power,
            acf_config  = self.acf_config
        )
        # Collapse 2D ACF and smooth
        self.smoothed_acf, self.unsmoothed_acf, self.freq_centers = collapsed_acf(
            acf                     = self.twodim_ACF, 
            freq_windows            = self.freq_windows, 
            sliding_window_style    = self.acf_config.sliding_window_style
        )
        # Fit gauss to estimate numax
        self.numax, self.fit_vals = fit_gauss_to_collapsed_acf(
            smoothed_acf            = self.smoothed_acf, 
            freq_centers            = self.freq_centers, 
            initial_numax           = self.initial_numax,
            max_acf_fit_iterations  = self.acf_config.max_acf_fit_iterations,
            n_sigma_numax_acf       = self.acf_config.n_sigma_numax_acf
        )
        return self

    def plot(self):
        """Plot 2D ACF computations if specified"""
        import matplotlib.pyplot as plt

        if self.acf_config.sliding_window_style == 'linear':
            # If sliding window is linear we plot 2D ACF map
            fig, axs = plt.subplots(3, 1, figsize=(6, 12))
            plot_spec_linear(
                self.frequency,
                self.avg_psd,
                self.med_filter,
                ax=axs[0],
                id=self.id,
            )
            plot_2D_ACF_linear(self.twodim_ACF, self.frequency, ax=axs[1])
            plot_collapsed_acf_with_gaussian_fit_linear(
                self.smoothed_acf, self.freq_centers, self.fit_vals, ax=axs[2]
            )
            savepath = os.path.join("numax_proxies", "results", self.id, "figures")
            os.makedirs(savepath, exist_ok=True)
            if self.config.noise_std > 0:
                fig.savefig(
                    f"{savepath}/ACF_noise-{self.config.noise_std}ppm.png", dpi=300, bbox_inches="tight"
                )
            fig.savefig(f"{savepath}/ACF.png", dpi=300, bbox_inches="tight")
        else:
            # If sliding window is log-something, we only draw collapsed 2D ACF
            fig, axs = plt.subplots(2, 1, figsize=(6, 8))
            plot_spec(
                self.frequency,
                self.avg_psd,
                self.med_filter,
                ax=axs[0],
                id=self.id,
            )
            plot_collapsed_acf_with_gaussian_fit(
                self.smoothed_acf, self.unsmoothed_acf, 
                self.freq_centers, self.fit_vals, self.initial_numax,
                ax=axs[1]
            )
            savepath = os.path.join("numax_proxies", "results", self.id, "figures")
            os.makedirs(savepath, exist_ok=True)
            if self.config.noise_std > 0:
                fig.savefig(
                    f"{savepath}/ACF_noise-{self.config.noise_std}ppm.png", dpi=300, bbox_inches="tight"
                )
            fig.savefig(f"{savepath}/ACF.png", dpi=300, bbox_inches="tight")

    def save_to_txt(self):
        """Save ACF calculations to txt file"""
        # Save path location
        results_dir = os.path.join("numax_proxies", "results", self.id)
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)

        savepath = os.path.join("numax_proxies", "results", self.id, 'acf_info')
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        # Save acf info
        fc = np.asarray(self.freq_centers)
        usacf = np.asarray(self.unsmoothed_acf)
        sacf = np.asarray(self.smoothed_acf)
        np.savetxt(
            fname = f'{savepath}/{self.id}_2DACF.txt',
            X = np.column_stack((fc, usacf, sacf)),
            header = 'freq_centers,unsmoothed_acf,smoothed_acf',
            delimiter = ','
        )

        # Save fitting parameters
        np.savetxt(
            fname = f'{savepath}/{self.id}_2DACF_fit_params.txt',
            X = np.column_stack(self.fit_vals),
            header = 'amp,sigma,numax',
            delimiter = ','
        )

    @property   
    def numax_estimate(self) -> float:
        return self.numax
    

