# Numax estimate from EACF method (Mosser & Appourchaux 2009, I. W. Roxburg 2009)
from ..data_preparation.dataclasses import PSDData, ProcessingConfig, EACFConfig, StarInfo
from typing import Optional, Literal
from .EACF.eacf_plot import plot
from .EACF.calculate_envelope import calculate_envelope
import matplotlib.pyplot as plt
import os

class NumaxFromEACF:
    def __init__(
            self, 
            star : StarInfo,
            psd : PSDData,
            config : ProcessingConfig,
            eacf_config : EACFConfig
    ):
        """ d
            Initialize class
        """
        self.star = star
        self.frequency = psd.frequency
        self.power = psd.psd
        self.config = config
        self.eacf_config = eacf_config
    
    def compute(self):
        """
            Compute numax from EACF method.
        """
        self.envelope = calculate_envelope(self.frequency, self.power)
        return self

    @property
    def numax_estimate(self):
        """Get numax and uncertainty"""
        return self.numax
    
    def plot(self):
        """Plot EACF results"""
        fig, ax = plt.subplots()
        plot(
            frequency=self.frequency,
            power=self.power,
            envelope=self.envelope,
            ax=ax,
            analytical_signal=None
        )
        savepath = os.path.join("numax_proxies", "results", self.star.target, "figures")
        os.makedirs(savepath, exist_ok=True)
        fig.savefig(f"{savepath}/CoVs_Bell.png", dpi=300, bbox_inches="tight")
