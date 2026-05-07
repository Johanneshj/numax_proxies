# Python package imports
import numpy as np
from uncertainties import unumpy as unp
import lightkurve as lk
from lightkurve.periodogram import Periodogram
from astropy import units as u
from dataclasses import dataclass

# Internal imports
from .data_preparation import GetLightcurve, DataProcessing, read_json_file
from .plotting import plot_spectrum_with_all_numax_estimates
from .proxies.ScalingRelations import query_gaia
from .proxies import (
    NumaxFromACF,
    NumaxFromScalingRelations,
    NumaxFromCoefficientsOfVariation,
    NumaxFromFliPer
)

class NumaxProxies:

    def __init__(self, *args, **kwargs):
        # Frequency proxies container
        self._numax_estimates = {}

        # Fetch from json file
        # star = StarInfo(read_json_file(*args))
        star            = read_json_file(*args)
        self._id        = star["target"]
        self._cadence   = star["cadence"]
        self._author    = star["author"]
        self._quarter   = star["quarter"]
        self._sector    = star["sector"]
        self._mission   = star["mission"]
        self._logg      = star["logg"]
        self._teff      = star["teff"]
        self._mag       = star["mag"]

        # Get the lightcurve: either from fits files, .csv file, arrays, or identifier.
        lc_file             = kwargs.get("lc_file", None)
        fits_files_folder   = kwargs.get("fits_files_folder", None)
        time                = kwargs.get("time", None)    
        flux                = kwargs.get("flux", None)  
        flux_err            = kwargs.get("flux_er", None)       

        gl = GetLightcurve(
            target=self._id,
            cadence=self._cadence,
            sector=self._sector,
            quarter=self._quarter,
            mission=self._mission,
            author=self._author,
            fits_files_folder=fits_files_folder,
            lc_file=lc_file,
            time=time,
            flux=flux,
            flux_err=flux_err
        )
        self._time, self._flux, self._flux_err = gl.final_lc

        # Do we want to experiment with noise?
        self._noise_std = (
            kwargs.get("noise_std", 0) if kwargs.get("add_noise", False) else 0
        )

        # Query GAIA for info
        query_flag = kwargs.get("query_gaia", False)
        if query_flag:
            self._gaia_query_dict = query_gaia(id=self._id)
            self._initial_numax_flag = kwargs.get("initial_numax", False)
            self._initial_numax = None
            if self._initial_numax_flag:
                self.compute_numax_from_scaling_relations()
        else:
            self._gaia_query_dict = {}
            self._initial_numax_flag = False
            self._initial_numax = None

        # Process the light curve: we normalize, sort by time, close gaps, and finally compute periodogram
        dp = DataProcessing(
            time=self._time, flux=self._flux, flux_err=self._flux_err, id=self._id
        )
        dp.sort_data_by_time()
        savgol = kwargs.get(
            "savgol", True
        )
        if savgol:
            ws = kwargs.get("savgol_window_size_in_days", 90)
            dp.savgol_smooth(ws=ws)
        normalize = kwargs.get(
            "normalize", True
        )  # Sometimes we dont want to normalize, but default is true
        if normalize:
            dp.normalize_flux()  # Normalize to ppm
        dp.sort_and_close_gaps()  # Sort by time and close gaps larger than 3 days
        dp.microHz_periodogram()  # Compute PSD with frequencies in microHz
        dp.super_Nyquist_spectrum()  # Compute super Nyquist spectrum

        # Define values
        self._time, self._flux, self._flux_err = dp.final_lc  # light curve
        self._frequency, self._power = dp.final_psd  # psd
        self._supNyq_freq, self._supNyq_power = dp.supNyq_psd
        if kwargs.get("plot_lc", False):
            dp.plot_lc_and_pg()

        # Also define LightCurve object and Periodogram object
        self._lc = lk.LightCurve(
            time=self._time, flux=self._flux, flux_err=self._flux_err
        )
        self._pg = Periodogram(
            frequency=self._frequency * u.uHz, power=self._power * (1 / u.uHz)
        )

        # Compute averaged psd (Sylvain Breton)
        # We only use this in 2D ACF method, but maybe it's also useful in the future
        avg_psd = kwargs.get("avg_psd", False)
        if avg_psd:
            # Define chunk length, defaults to 90 days
            chunk_length = kwargs.get("avg_psd_chunk_len_days", 90)
            # Compute averaged PSD
            dp.averaged_psd(chunk_len=chunk_length)
            # Create periodogram object
            self._avgpsd_freq, self._avgpsd_power = dp.avg_psd
            self._avg_pg = Periodogram(
                frequency=self._avgpsd_freq * u.uHz,
                power=self._avgpsd_power * (1 / u.Hz),
            )
        else:
            self._avg_pg = self._pg

        # ACF misc.
        ## Sliding window flag
        self._sliding_window_flag = kwargs.get("sliding_window_flag", 'log_numax')
        ## Parameters for log_numax binning - standard values have be tested with trial-and-error
        self._acf_params = {
            "ACF_min_freq": kwargs.get("ACF_min_freq", None),
            "min_num_points": kwargs.get('ACF_min_num_points', None),
            "ACF_overlap_scale": kwargs.get("ACF_overlap_scale", None),
            "ACF_width_factor": kwargs.get("ACF_width_factor", None),
        }

    def compute_numax_from_acf(self, plot=True):
        """
        Compute νmax using the 2D autocorrelation proxy.
        """
        # Check if we should use averaged psd or not
        if self._avg_pg is not None:
            acf_proxy = NumaxFromACF(
                lc=self._lc, 
                pg=self._avg_pg, 
                full_pg=self._pg,
                id=self._id,
                initial_numax=self._initial_numax,
                sliding_window_flag=self._sliding_window_flag,
                acf_params=self._acf_params
            )
        else:
            acf_proxy = NumaxFromACF(
                lc=self._lc, 
                pg=self._pg, 
                id=self._id,
                initial_numax=self._initial_numax,
                sliding_window_flag=self._sliding_window_flag,
                acf_params=self._acf_params
            )
        numax = acf_proxy.compute()

        if plot:
            acf_proxy.plot(noise_std=self._noise_std)

        self._numax_estimates["numax_2DACF"] = numax

    def compute_numax_from_scaling_relations(self):
        """
        Compute νmax using the scaling relations.
        """
        
        if self._initial_numax_flag:
            # If we want initial guess for other methods
            scaling_relations_proxy = NumaxFromScalingRelations(
                id=self._id, 
                gaia_query_dict=self._gaia_query_dict
            )
            # , self._logg, self._teff)
            numaxes = scaling_relations_proxy.compute()
            if len(numaxes) > 0:
                values = list(numaxes.values())
                self._initial_numax = np.mean(values)
                if hasattr(self._initial_numax, "nominal_value"):
                    self._initial_numax = self._initial_numax.nominal_value
            else:
                self._initial_numax = None
            self._numax_estimates.update(numaxes)
        else:
            # If other methods should speak for themselves
            scaling_relations_proxy = NumaxFromScalingRelations(
                id=self._id, 
                gaia_query_dict=self._gaia_query_dict
            )
            # , self._logg, self._teff)
            numaxes = scaling_relations_proxy.compute()
            self._numax_estimates.update(numaxes)

    def compute_numax_from_CoV(self, plot=True):
        """
        Compute νmax using coefficients of variation (Vianni et al. 2018)
        """
        # self._supNyq_pg = Periodogram(
        #     frequency=self._supNyq_freq * u.uHz,
        #     power=self._supNyq_power * (1 / u.uHz)
        # )
        CoV_proxy = NumaxFromCoefficientsOfVariation(
            lc=self._lc, 
            pg=self._avg_pg, 
            id=self._id,
            initial_numax=self._initial_numax
        )
        numax = CoV_proxy.compute()

        if plot:
            CoV_proxy.plot()

        self._numax_estimates["numax_CoV"] = numax

    def compute_numax_from_FliPer(self, plot=True):
        """
        Compute numax with method from Bugnet et al. (2018).

        Input:
            Noise estimate (usually done with magnitude, but we have to be a bit smarter)
            Teff
        """
        gmag = self._mag
        FliPer_proxy = NumaxFromFliPer(lc=self._lc, pg=self._pg, id=self._id, gmag=gmag)

        numax = FliPer_proxy.compute()

        if plot:
            FliPer_proxy.plot()

        self._numax_estimates["numax_FliPer"] = numax

    def plotting(self):
        """
        Here we are going to plot the full spectrum with all numax estimates
        """
        plot_spectrum_with_all_numax_estimates(
            self._pg.frequency, self._pg.power, self._numax_estimates, self._id
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
    
# @dataclass
# class StarInfo:
#     target      :   str
#     cadence     :   str = 'long'
#     author      :   str
#     quarter     :   list = np.range(0,100)
#     sector      :   list = np.range(0,100)
#     mission     :   str
#     logg        :   float
#     teff        :   float
#     mag         :   float


