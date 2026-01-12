from . import (
    calculate_CoV,
    bin_spectrum,
    plot_CoV_vs_bin_centers,
    smooth_CoV_values,
    numax_estimate_CoV
)
import os
from uncertainties import ufloat

class CoefficientsOfVariation:
    def __init__(self, lc=None, pg=None, id=None, *args, **kwargs):
        self._id = id or "unknown"
        if lc is not None and pg is None:
            from numax_proxies.data_preparation import calculate_psd
            pg = calculate_psd(lc)
        self._lc = lc
        self._pg = pg

        self._bin_centers = None
        self._CoVs = None
        self._smoothed_CoVs = None 
        self._numax = None
        self._numax_error = None

    def compute(self, *args, **kwargs):
        self._bin_centers, self._CoVs = bin_spectrum(
            frequency=self._pg.frequency.value,
            power=self._pg.power.value
        )
        self._smoothed_CoVs = smooth_CoV_values(
            self._bin_centers,
            self._CoVs
        )
        self._numax, self._numax_error = numax_estimate_CoV(
            self._bin_centers,
            self._smoothed_CoVs
        )
        self._numax = ufloat(self._numax, self._numax_error)
        return self._numax
    
    def plot(self, *args, **kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plot_CoV_vs_bin_centers(self._bin_centers,
                                self._CoVs,
                                self._smoothed_CoVs,
                                self._numax,
                                ax=ax,
                                id=self._id)
        savepath = os.path.join('numax_proxies', 'results', self._id, 'figures')
        os.makedirs(savepath, exist_ok=True)
        fig.savefig(f'{savepath}/CoVs.png', dpi=300, bbox_inches='tight')

        