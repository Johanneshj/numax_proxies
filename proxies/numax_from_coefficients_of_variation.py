from . import (
    calculate_CoV,
    bin_spectrum,
    plot_CoV_vs_bin_centers,
    smooth_CoV_values,
    numax_estimate_CoV
)
import os
from uncertainties import ufloat
import matplotlib.pyplot as plt

class CoefficientsOfVariation:
    def __init__(self, lc=None, pg=None, id=None, *args, **kwargs):
        self._id = id or "unknown"
        self._lc = lc
        self._pg = pg

        self._bin_centers = None
        self._CoVs = None
        self._smoothed_CoVs = None 
        self._numax = None
        self._numax_error = None

    def compute(self, *args, **kwargs):
        '''Compute numax from CoV'''
        # Bin spectrum (black diamonds in plot)
        self._bin_centers, self._CoVs = bin_spectrum(
            frequency=self._pg.frequency.value,
            power=self._pg.power.value
        )
        # Smooth CoV values (red crosses)
        self._smoothed_CoVs = smooth_CoV_values(
            self._bin_centers,
            self._CoVs
        )
        # Estimate numax
        self._numax, self._numax_error = numax_estimate_CoV(
            self._bin_centers,
            self._smoothed_CoVs
        )
        self._numax = ufloat(self._numax, self._numax_error)
        return self._numax
    
    def plot(self, *args, **kwargs):
        '''Plot if specified'''
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

        