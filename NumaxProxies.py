from .data_preparation import GetLightcurve, DataProcessing, mean_psd, read_json_file
from .plotting import plot_spectrum_with_all_numax_estimates
from .proxies.numax_from_ACF import NumaxFromACF
from .proxies.numax_from_scaling_relations import NumaxFromScalingRelations, query_gaia
from .proxies.numax_from_coefficients_of_variation import CoefficientsOfVariation
from .proxies.numax_from_FliPer import NumaxFromFliPer
import numpy as np
import lightkurve as lk
from lightkurve.periodogram import Periodogram
import matplotlib.pyplot as plt
from astropy import units as u
import time as t

class NumaxProxies:
    #This happens when you create (initialize) the class i.e. class = FreqProxy(args,kwargs) somewhere
    def __init__(self, *args, **kwargs):
        # Fetch from json file
        star = read_json_file(*args)
        self._id = star['target']
        self._cadence = star['cadence']
        self._author = star['author']
        self._quarter = star['quarter']
        self._sector = star['sector']
        self._mission = star['mission']
        self._logg = star['logg']
        self._teff = star['teff']
        self._mag = star['mag']

        # Get the lightcurve: either from fits files, .csv file, or identifier.
        lc_file = kwargs.get('lc_file', None)
        fits_files_folder = kwargs.get('fits_files_folder', None)
        start = t.time()
        gl = GetLightcurve(
            target=self._id,
            cadence=self._cadence,
            sector=self._sector,
            quarter=self._quarter,
            mission=self._mission,
            author=self._author,
            fits_files_folder=fits_files_folder,
            lc_file=lc_file,
        )
        self._time, self._flux, self._flux_err = gl.final_lc
        end = t.time()
        # print(f"computation time for getting lc: {end - start:.2f} seconds")
        
        # Do we want to experiment with noise?
        self._noise_std = kwargs.get('noise_std', 0) if kwargs.get('add_noise', False) else 0
        
        # Query GAIA for info
        start = t.time()
        self._gaia_query_dict = query_gaia(id=self._id)
        end = t.time()
        # print(f"computation time for gaia query: {end - start:.2f} seconds")
        # print(self._gaia_query_dict)

        # Process the light curve: we normalize, sort by time, close gaps, and finally compute periodogram
        start = t.time()
        dp = DataProcessing(time=self._time, 
                            flux=self._flux, 
                            flux_err=self._flux_err, 
                            id=self._id)
        normalize = kwargs.get('normalize', True) # Sometimes we dont want to normalize, but default is true
        if normalize:
            dp.normalize_flux() # Normalize to ppm
        dp.sort_and_close_gaps() # Sort by time and close gaps larger than 3 days
        dp.microHz_periodogram() # Compute PSD with frequencies in microHz

        # Define values
        self._time, self._flux, self._flux_err = dp.final_lc # light curve
        self._frequency, self._power = dp.final_psd # psd
        if kwargs.get('plot_lc', False):
            dp.plot_lc_and_pg()
        end = t.time()
        # print(f"computation time for data processing: {end - start:.2f} seconds")
    

        # Also define LightCurve object and Periodogram object
        self._lc = lk.LightCurve(
            time=self._time,
            flux=self._flux,
            flux_err=self._flux_err
        )
        self._pg = Periodogram(
            frequency=self._frequency * u.uHz,
            power=self._power * (1 / u.uHz)
        )
        
        # Compute averaged psd (Sylvain Breton)
        # We only use this in 2D ACF method, but maybe it's also useful in the future
        start = t.time()
        avg_psd = kwargs.get('avg_psd', False)
        if avg_psd:
            self._dt = np.mean(np.diff(self._lc.time.value)) * 86400
            if max(self._lc.time.value) > 365 and self._dt < 120: # Ignore if long cadence
                dp.averaged_psd()
                self._avgpsd_freq, self._avgpsd_power = dp.avg_psd
                self._avg_pg = Periodogram(
                    frequency=self._avgpsd_freq * u.uHz,
                    power=self._avgpsd_power * (1/u.Hz)
                )
            else:
                self._avg_pg = None
        else:
            self._avg_pg = None
        end = t.time()
        # print(f"computation time for avg. psd: {end - start:.2f} seconds")

        # Frequency proxies container
        self._numax_estimates = {}

    def compute_numax_from_acf(self, plot=True):
        """
        Compute νmax using the 2D autocorrelation proxy.
        """
        start = t.time()
        # Check if we should use averaged psd or not
        if self._avg_pg is not None:
            acf_proxy = NumaxFromACF(
                lc=self._lc, pg=self._avg_pg, id=self._id
            )
        else:
            acf_proxy = NumaxFromACF(
                lc=self._lc, pg=self._pg, id=self._id
            )
        numax = acf_proxy.compute()
        end = t.time()
        # print(f"computation time for ACF: {end - start:.2f} seconds")

        if plot:
            acf_proxy.plot(noise_std=self._noise_std)

        self._numax_estimates["numax_2DACF"] = numax

    def compute_numax_from_scaling_relations(self):
        """
        Compute νmax using the scaling relations.
        """
        start = t.time()
        scaling_relations_proxy = NumaxFromScalingRelations(id=self._id, 
                                                            gaia_query_dict=self._gaia_query_dict) 
                                                            #, self._logg, self._teff)
        numaxes = scaling_relations_proxy.compute()
        self._numax_estimates.update(numaxes)
        end = t.time()
        # print(f"computation time for scaling relations: {end - start:.2f} seconds")
    
    def compute_numax_from_CoV(self, plot=True):
        """
        Compute νmax using coefficients of variation (Vianni et al. 2018)
        """
        start = t.time()
        CoV_proxy = CoefficientsOfVariation(
            lc=self._lc, pg=self._pg, id=self._id
        )
        numax = CoV_proxy.compute()
        end = t.time()
        # print(f"computation time for CoV: {end - start:.2f} seconds")
        
        if plot:
            CoV_proxy.plot()
        
        self._numax_estimates['numax_CoV'] = numax

    def compute_numax_from_FliPer(self, plot=True):
        """
        Compute numax with method from Bugnet et al. (2018).
        
        Input:
            Noise estimate (usually done with magnitude, but we have to be a bit smarter)
            Teff
        """
        start = t.time()
        gmag = self._mag
        FliPer_proxy = NumaxFromFliPer(
            lc=self._lc, pg=self._pg, id=self._id, gmag=gmag
        )

        numax = FliPer_proxy.compute()
        end = t.time()
        # print(f"computation time for FliPer: {end - start:.2f} seconds")

        if plot:
            FliPer_proxy.plot()

        self._numax_estimates['numax_FliPer'] = numax

    def plotting(self):
        '''
            Here we are going to plot the full spectrum with all numax estimates
        '''
        plot_spectrum_with_all_numax_estimates(
            self._pg.frequency,
            self._pg.power,
            self._numax_estimates,
            self._id
        )

    @property
    def lc(self):
        return self._lc

    @property
    def pg(self):
        return self._pg

    @property
    def numax_estimates(self):
        def is_valid(v):
            try:
                if hasattr(v, "nominal_value"):
                    return np.isfinite(v.nominal_value)
                return np.isfinite(v)
            except Exception:
                return False

        return {k: v for k, v in self._numax_estimates.items() if is_valid(v)}
