from .data_preparation import get_lightcurve, prepare_lightcurve, calculate_psd, mean_psd, read_logg_and_teff
from .proxies.numax_from_ACF import NumaxFromACF
from .proxies.numax_from_scaling_relations import NumaxFromScalingRelations
from .proxies.numax_from_coefficients_of_variation import CoefficientsOfVariation

class NumaxProxies:
    #This happens when you create (initialize) the class i.e. class = FreqProxy(args,kwargs) somewhere
    def __init__(self, *args, **kwargs):
        # Fetch light curve and other info (from JSON and/or light curve file)
        self._logg, self._teff = read_logg_and_teff(*args, **kwargs) # logg and teff
        self._lc, self._id, self._cadence = get_lightcurve(*args, **kwargs)
        self._noise_std = kwargs.get('noise_std', 0) if kwargs.get('add_noise', False) else 0
        
        # Preprocess and compute periodogram
        self._savgol_iters = kwargs.pop("savgol_iters", 0)
        self._lc, self._dt = prepare_lightcurve(self._lc, self._id, 
                                                plot=True, savgol=True,
                                                savgol_iters=self._savgol_iters,
                                                *args, **kwargs)
        self._full_pg = calculate_psd(lc=self._lc)
        avg_psd = kwargs.get('avg_psd', False)
        if avg_psd:
            if max(self._lc.time.value) > 2*365 and self._dt < 120:
                self._pg = mean_psd(lc=self._lc, cadence=self._cadence)
            else:
                self._pg = self._full_pg
        else:
            self._pg = self._full_pg

        # Frequency proxies container
        self._freqs = {}

    def compute_numax_from_acf(self, plot=True):
        """
        Compute νmax using the 2D autocorrelation proxy.
        """
        acf_proxy = NumaxFromACF(lc=self._lc, pg=self._pg, id=self._id)
        numax = acf_proxy.compute()

        self._freqs["numax_ACF"] = numax

        if plot:
            acf_proxy.plot(noise_std=self._noise_std)


        return numax

    def compute_numax_from_scaling_relations(self):
        """
        Compute νmax using the scaling relations.
        """
        scaling_relations_proxy = NumaxFromScalingRelations(id=self._id)#, self._logg, self._teff)
        numaxes = scaling_relations_proxy.compute()
        
        # print(numaxes)
        return numaxes
    
    def compute_numax_from_CoV(self, plot=True):
        """
        Compute νmax using coefficients of variation (Vianni et al. 2018)
        
        :param self: Description
        :param plot: Description
        """
        CoV_proxy = CoefficientsOfVariation(lc=self._lc, pg=self._pg, id=self._id)
        CoV_proxy.compute()
        
        if plot:
            CoV_proxy.plot()

    def plotting(self):
        '''
            Here we are going to plot the full spectrum with all numax estimates
        '''
        return


    @property
    def lc(self):
        return self._lc

    @property
    def pg(self):
        return self._pg

    @property
    def freqs(self):
        return self._freqs