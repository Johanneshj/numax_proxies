from . import (
    estimate_noise,
    highpass_filter,
    plot_spectrum,
    calculate_FliPer_values,
    ML
)
from uncertainties import ufloat
import numpy as np

class NumaxFromFliPer:
    def __init__(self, id, gmag, lc=None, pg=None, *args, **kwargs):
        self._id = id or "unknown"
        self._gmag = gmag
        self._mission = 0 if 'KIC' in self._id else 1 # 1 = TESS
        if lc is not None and pg is None:
            from numax_proxies.data_preparation import calculate_psd
            pg = calculate_psd(lc)
        self._lc = lc
        self._pg = pg

        self._noise = estimate_noise(self._pg) # estimate noise as median power of last 100 freq bins
        self._filter_pg_20d, self._filter_20d = highpass_filter(self._pg, 20) # 20 days high pass filter
        self._filter_pg_80d, self._filter_80d = highpass_filter(self._pg, 80) # 80 days high pass filter 
        self._PATH_TO_TRAINING_FILE_NUMAX = 'numax_proxies/proxies/FliPer/FliPer_model.pkl' 
        
    def compute(self, *args, **kwargs):
        '''Estimate numax from FliPer values'''
        # Calculate FliPer values, and cadence
        self._Fp02, self._Fp07, self._Fp7, self._Fp20, self._Fp50, self._cadence = calculate_FliPer_values(
            self._lc,
            self._filter_pg_80d,
            self._filter_pg_20d,
            self._noise
        )
        val =(ML().PREDICTION(
            self._Fp02, self._Fp07, self._Fp7,
            self._Fp20, self._Fp50, 
            self._noise,
            self._cadence,
            self._mission,
            self._PATH_TO_TRAINING_FILE_NUMAX,
        )) 
        if val is not None and np.isfinite(val) and val > 0:
            numax = 10**val
        else:
            numax = np.nan
        return numax

    def plot(self, *args, **kwargs):
        import matplotlib.pyplot as plt
        plot_spectrum(
            id=self._id,
            pg=self._pg,
            filter_20d = self._filter_20d,
            filter_80d = self._filter_80d,
            noise=self._noise
        )