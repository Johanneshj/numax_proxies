from .data_preparation import get_lightcurve, prepare_lightcurve, calculate_psd, mean_psd
from .proxies.numax_from_ACF import NumaxFromACF

class NumaxProxies:
    #This happens when you create (initialize) the class i.e. class = FreqProxy(args,kwargs) somewhere
    def __init__(self, *args, **kwargs):
        # Fetch light curve (from ID or arrays)
        self._lc, self._id, self._cadence = get_lightcurve(*args, **kwargs)

        # Preprocess and compute periodogram
        self._savgol_iters = kwargs.pop("savgol_iters", 0)
        self._lc = prepare_lightcurve(self._lc, self._id, 
                                      plot=True, savgol=True,
                                      savgol_iters=self._savgol_iters)
        self._pg = mean_psd(lc=self._lc, cadence=self._cadence)

        # Frequency proxies container
        self._freqs = {}
    

    def compute_acf(self, plot=True):
        """
        Compute νmax using the 2D autocorrelation proxy.
        """
        acf_proxy = NumaxFromACF(lc=self._lc, pg=self._pg, id=self._id)
        numax = acf_proxy.compute()

        self._freqs["numax"] = numax

        if plot:
            acf_proxy.plot()

        return numax

    @property
    def lc(self):
        return self._lc

    @property
    def pg(self):
        return self._pg

    @property
    def freqs(self):
        return self._freqs