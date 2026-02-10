from . import (
    calculate_relative_power,
    calculate_two_dim_ACF,
    collapsed_acf,
    fit_gauss_to_collapsed_acf,
    plot_spec,
    plot_collapsed_acf_with_gaussian_fit,
    plot_2D_ACF
)
import os


class NumaxFromACF:
    def __init__(self, lc=None, pg=None, id=None, *args, **kwargs):
        self._id = id or "unknown"
        self._lc = lc
        self._pg = pg
        self._normalized_pg = None
        self._filter = None

        self._2D_ACF = None
        self._freq_windows = None
        self._collapsed_2D_ACF = None
        self._freq_centers = None
        self._fit_vals = None
        self._numax = None
    
    def compute(self, *args, **kwargs):
        '''Perform 2D ACF computations'''
        # Normalize spectrum
        self._normalized_pg, self._filter = calculate_relative_power(self._pg)
        # Calculate 2D ACF
        self._2D_ACF, self._freq_windows = calculate_two_dim_ACF(
            self._normalized_pg.frequency.value, self._normalized_pg.power.value
        )
        # Collapse 2D ACF
        self._collapsed_2D_ACF, self._freq_centers = collapsed_acf(
            self._2D_ACF, self._freq_windows
        )
        # Fit gauss to estimate numax
        self._numax, self._fit_vals = fit_gauss_to_collapsed_acf(
            self._collapsed_2D_ACF, self._freq_centers
        )
        return self._numax
    
    def plot(self, noise_std, *args, **kwargs):
        '''Plot 2D ACF computations if specified'''
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 1, figsize=(6,12))
        plot_spec(self._pg.frequency.value, 
                  self._pg.power.value,
                  self._filter,
                  ax=axs[0],
                  id=self._id)
        plot_2D_ACF(self._2D_ACF,
                    self._pg.frequency.value,
                    ax=axs[1])
        plot_collapsed_acf_with_gaussian_fit(self._collapsed_2D_ACF,
                                             self._freq_centers,
                                             self._fit_vals,
                                             ax=axs[2])
        savepath = os.path.join('numax_proxies', 'results', self._id, 'figures')
        os.makedirs(savepath, exist_ok=True)
        if noise_std > 0: fig.savefig(f'{savepath}/ACF_noise-{noise_std}ppm.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{savepath}/ACF.png', dpi=300, bbox_inches='tight')
