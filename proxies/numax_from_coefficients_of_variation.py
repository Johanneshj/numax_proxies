from .CoV import (
    bin_spectrum,
    plot_CoV_vs_bin_centers,
    smooth_CoV_values,
    numax_estimate_CoV,
)
import os
from uncertainties import ufloat
import matplotlib.pyplot as plt


class NumaxFromCoefficientsOfVariation:
    def __init__(self, pg=None, id=None, initial_numax=None, *args, **kwargs):
        """
            Initialize class: lightcurve, periodogram, identifier, initial numax
            Allocate properties we'll need later.
        """
        self._id = id or "unknown"
        self._pg = pg
        self._initial_numax = initial_numax

        self._bin_centers = None
        self._CoVs = None
        self._smoothed_CoVs = None
        self._numax = None
        self._numax_error = None

    def compute(self, *args, **kwargs):
        """
            Compute numax from CoV;
            1. Spectrum is binned and CoVs calculated.
            2. CoV values are smoothed with moving average 
                in window sizes corresponding to width of potential oscillation envelope.
            3. Numax is estimated.
            4. Numax is defined as ufloat object.
        """
        # Bin spectrum (black diamonds in plot)
        self._bin_centers, self._CoVs = bin_spectrum(
            frequency=self._pg.frequency.value, 
            power=self._pg.power.value
        )
        # Smooth CoV values (red crosses)
        self._smoothed_CoVs = smooth_CoV_values(self._bin_centers, self._CoVs)
        # Estimate numax
        self._numax, self._numax_error, self._fit_vals, self._succesful_fit = numax_estimate_CoV(
            self._bin_centers, self._smoothed_CoVs, self._initial_numax
        )
        self._numax = ufloat(self._numax, self._numax_error)
        return self._numax

    @property
    def numax_estimate(self):
        """Get numax and uncertainty"""
        if self._numax is not None:
            if hasattr(self._numax, "nominal_value"):
                # Return value and error if numax is ufloat object (it should be!)
                return self._numax.nominal_value, self._numax.std_dev
            else:
                return self._numax

    def plot(self, *args, **kwargs):
        """Plot if specified"""

        fig, ax = plt.subplots()
        plot_CoV_vs_bin_centers(
            self._bin_centers,
            self._CoVs,
            self._smoothed_CoVs,
            self._numax,
            ax=ax,
            id=self._id,
            initial_numax=self._initial_numax,
            fit_vals=self._fit_vals,
            succesful_fit=self._succesful_fit
        )
        savepath = os.path.join("numax_proxies", "results", self._id, "figures")
        os.makedirs(savepath, exist_ok=True)
        fig.savefig(f"{savepath}/CoVs.png", dpi=300, bbox_inches="tight")

        # fig, ax = plt.subplots()
        # plot_supNyq_spec(
        #     self._pg.frequency.value,
        #     self._pg.power.value,
        #     self._numax,
        #     ax=ax,
        #     id=self._id
        # )
        # savepath = os.path.join('numax_proxies', 'results', self._id, 'figures')
        # os.makedirs(savepath, exist_ok=True)
        # fig.savefig(f'{savepath}/super_Nyq_spec.png', dpi=300, bbox_inches='tight')
