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
        if lc is not None and pg is None:
            from numax_proxies.data_preparation import calculate_psd
            pg = calculate_psd(lc)
        self._lc = lc
        self._pg = pg

        self._filter = None
        self._2D_ACF = None
        self._freq_windows = None
        self._collapsed_2D_ACF = None
        self._freq_centers = None
        self._fit_vals = None
        self._numax = None
    
    def compute(self, *args, **kwargs):
        relative_power, self._filter = calculate_relative_power(self._pg)
        self._2D_ACF, self._freq_windows = calculate_two_dim_ACF(
            self._pg.frequency.value, relative_power
        )
        self._collapsed_2D_ACF, self._freq_centers = collapsed_acf(
            self._2D_ACF, self._freq_windows
        )
        self._numax, self._fit_vals = fit_gauss_to_collapsed_acf(
            self._collapsed_2D_ACF, self._freq_centers
        )
        return self._numax
    
    def plot(self, *args, **kwargs):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 1, figsize=(6,12))
        plot_spec(self._pg.frequency.value, 
                  self._pg.power.value,
                  self._filter,
                  ax=axs[0])
        plot_2D_ACF(self._2D_ACF,
                    self._pg.frequency.value,
                    ax=axs[1])
        plot_collapsed_acf_with_gaussian_fit(self._collapsed_2D_ACF,
                                             self._freq_centers,
                                             self._fit_vals,
                                             ax=axs[2])
        savepath = os.path.join(self._id, 'figures', 'numax_proxies')
        os.makedirs(savepath, exist_ok=True)
        fig.savefig(f'{savepath}/ACF.png', dpi=300, bbox_inches='tight')
