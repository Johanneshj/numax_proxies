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


class NumaxFromACF:
    def __init__(self, pg=None, full_pg=None, id=None, initial_numax=None, 
                 sliding_window_flag='log_numax', acf_params={}, *args, **kwargs):
        """Initialization"""
        self._id = id or "unknown"
        self._pg = pg
        self._full_pg = pg if full_pg is None else full_pg
        self._initial_numax = initial_numax
        self._normalized_pg = None
        self._filter = None

        self._2D_ACF = None
        self._freq_windows = None
        self._smoothed_acf = None
        self._unsmoothed_acf = None
        self._freq_centers = None
        self._fit_vals = None
        self._numax = None

        # Sliding window flag ('linear', 'log', or 'log_numax').
        # 'log_numax' seems to perform the best.
        self._sliding_window_flag = sliding_window_flag
        # Load acf parameters dictionary, if not specified defaults to sensible standard values
        self._acf_params = acf_params

    def compute(self, *args, **kwargs):
        """Perform 2D ACF computations"""
        # Normalize spectrum
        self._normalized_pg, self._filter = calculate_relative_power(self._pg)
        # Calculate 2D ACF
        self._2D_ACF, self._freq_windows = calculate_two_dim_ACF(
            self._normalized_pg.frequency.value, 
            self._normalized_pg.power.value,
            sliding_window_flag=self._sliding_window_flag, 
            acf_params=self._acf_params,
        )
        # Collapse 2D ACF and smooth
        self._smoothed_acf, self._unsmoothed_acf, self._freq_centers = collapsed_acf(
            self._2D_ACF, 
            self._freq_windows, 
            sliding_window_flag=self._sliding_window_flag
        )
        # Fit gauss to estimate numax
        self._numax, self._fit_vals = fit_gauss_to_collapsed_acf(
            self._smoothed_acf, 
            self._freq_centers, 
            self._initial_numax, 
            self._full_pg
        )
        self._numax = self._numax
        return self._numax

    def plot(self, noise_std, *args, **kwargs):
        """Plot 2D ACF computations if specified"""
        import matplotlib.pyplot as plt

        if self._sliding_window_flag == 'linear':
            # If sliding window is linear we plot 2D ACF map
            fig, axs = plt.subplots(3, 1, figsize=(6, 12))
            plot_spec_linear(
                self._pg.frequency.value,
                self._pg.power.value,
                self._filter,
                ax=axs[0],
                id=self._id,
            )
            plot_2D_ACF_linear(self._2D_ACF, self._pg.frequency.value, ax=axs[1])
            plot_collapsed_acf_with_gaussian_fit_linear(
                self._smoothed_acf, self._freq_centers, self._fit_vals, ax=axs[2]
            )
            savepath = os.path.join("numax_proxies", "results", self._id, "figures")
            os.makedirs(savepath, exist_ok=True)
            if noise_std > 0:
                fig.savefig(
                    f"{savepath}/ACF_noise-{noise_std}ppm.png", dpi=300, bbox_inches="tight"
                )
            fig.savefig(f"{savepath}/ACF.png", dpi=300, bbox_inches="tight")
        else:
            # If sliding window is log-something, we only draw collapsed 2D ACF
            fig, axs = plt.subplots(2, 1, figsize=(6, 8))
            plot_spec(
                self._pg.frequency.value,
                self._pg.power.value,
                self._filter,
                ax=axs[0],
                id=self._id,
            )
            plot_collapsed_acf_with_gaussian_fit(
                self._smoothed_acf, self._unsmoothed_acf, 
                self._freq_centers, self._fit_vals, self._initial_numax,
                ax=axs[1]
            )
            savepath = os.path.join("numax_proxies", "results", self._id, "figures")
            os.makedirs(savepath, exist_ok=True)
            if noise_std > 0:
                fig.savefig(
                    f"{savepath}/ACF_noise-{noise_std}ppm.png", dpi=300, bbox_inches="tight"
                )
            fig.savefig(f"{savepath}/ACF.png", dpi=300, bbox_inches="tight")
