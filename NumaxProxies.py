# Python package imports
import numpy as np
from uncertainties import unumpy as unp
import lightkurve as lk
from lightkurve.periodogram import Periodogram
from astropy import units as u
from dataclasses import dataclass, field
from typing import Optional, Literal
from numpy.typing import NDArray
import pandas as pd
import yaml
import pyarrow.feather as feather
import time as t

# Internal imports
from .data_preparation import GetLightcurve, DataProcessing, read_json_file
from .data_preparation.dataclasses import *
from .plotting import plot_spectrum_with_all_numax_estimates
from .proxies.ScalingRelations import query_gaia
from .proxies import (
    NumaxFromACF,
    NumaxFromScalingRelations,
    NumaxFromCoefficientsOfVariation,
    NumaxFromFliPer,
    NumaxFromEACF
)

class NumaxProxies:

    def __init__(self, global_config : GlobalConfig):
        """Initialize NumaxProxies class: Read all information from yaml file"""
        # Frequency proxies container
        self.numax_estimates = {}
        # Read inputs
        self.star: StarInfo = global_config.star
        self.config: ProcessingConfig = global_config.config
        self.lc_input: LightCurveInput = global_config.lightcurve
        self.psd_input: PSDInput = global_config.psd
        self.acf_config: ACFConfig = global_config.acf_config
        self.cov_config: COVConfig = global_config.cov_config
        self.eacf_config: EACFConfig = global_config.eacf_config

    def compute_numax_from_acf(self) -> float:
        """
        Compute νmax using the 2D autocorrelation proxy.
        """
        # import matplotlib.pyplot as plt
        acf_proxy = NumaxFromACF(
            avg_psd = self.avg_psd if self.config.do_avg_psd else self.psd, # sometimes we don't want to use averaged psd
            acf_config = self.acf_config,
            config = self.config,
            id = self.star.target,
        )
        numax = acf_proxy.compute().numax_estimate

        if self.acf_config.plot:
            acf_proxy.plot()

        if self.acf_config.save_info:
            acf_proxy.save_to_txt()

        self.numax_estimates["numax_2DACF"] = numax

    def compute_numax_from_scaling_relations(self):
        """
        Compute νmax using the scaling relations.
        """
        scaling_relations_proxy = NumaxFromScalingRelations(
            star = self.star,
            config = self.config,
            gaia_data = self.gaia_data if self.gaia_data else None
        )
        numaxes = scaling_relations_proxy.compute().numax_estimates
        self.numax_estimates.update(numaxes)

    def compute_numax_from_CoV(self):
        """
        Compute νmax using coefficients of variation (Vianni et al. 2018)
        """
        CoV_proxy = NumaxFromCoefficientsOfVariation( 
            psd=self.welch_psd if self.cov_config.use_welch else self.psd, 
            config=self.config,
            cov_config = self.cov_config,
            id=self.star.target,
            initial_numax=self.config.initial_numax
        )
        # use formalism of Bell+ (2019)?
        if self.cov_config.use_Bell:
            numax = CoV_proxy.compute_Bell().numax_estimate
        else:
            numax = CoV_proxy.compute().numax_estimate

        if self.cov_config.plot:
            if self.cov_config.use_Bell:
                CoV_proxy.plot_Bell()
            else:
                CoV_proxy.plot()
        
        if self.cov_config.save_info:
            CoV_proxy.save_to_txt()

        self.numax_estimates["numax_CoV"] = numax

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

    def compute_numax_from_EACF(self):
        """Compute numax with method from Mosser & Appourchaux (2009) and I.W. Roxburgh (2009)"""
        EACF_proxy = NumaxFromEACF(
            star = self.star,
            psd = self.psd,
            config = self.config,
            eacf_config = self.eacf_config
        )
        EACF_proxy.compute()

        if self.eacf_config.plot:
            EACF_proxy.plot()

    def plotting(self):
        """
        Here we are going to plot the full spectrum with all numax estimates
        """
        plot_spectrum_with_all_numax_estimates(
            psd = self.psd,
            star = self.star,
            numax_estimates = self.numax_estimates
        )

    @property
    def results(self):
        rows = []
        for label, numax in self.numax_estimates.items():
            try:
                numax_val = numax.n
                numax_err = numax.s
            except:
                numax_val = numax
                numax_err = None
            rows.append({
                "label": label,
                "numax": numax_val,
                "numax_err": numax_err,
            })
        df = pd.DataFrame(rows)
        
        # Save results?
        if self.config.save_results:
            df.to_csv(f'numax_proxies/results/{self.star.target}/{self.star.target}_results.txt', index=False)
        
        return pd.DataFrame(rows)

    def _load_lightcurve(self):
        """Load light curve"""
        gl = GetLightcurve(
            target              =   self.star.target,
            cadence             =   self.star.cadence,
            sector              =   self.star.sector,
            quarter             =   self.star.quarter,
            mission             =   self.star.mission,
            author              =   self.star.author,

            fits_files_folder   =   self.lc_input.fits_file_folder,
            lc_file             =   self.lc_input.lc_file,
        )
        # (potentially change GetLightcurve to output dataclasses rather than tuples)
        time, flux, flux_err = gl.final_lc
        self.unprocessed_lc = UnprocessedLightCurveData(
            time = time,
            flux = flux,
            flux_err = flux_err
        )

    def _process_lightcurve(self):
        """
        Process lightcurve: 
            1) sort by time
            2) normalize
            3) close gaps
            4) compute periodogram
        """

        dp = DataProcessing(
            lc=self.unprocessed_lc, 
            config=self.config,
            cov_config=self.cov_config,
            id=self.star.target
        )
        # sort
        if self.config.sort:
            dp.sort_data_by_time()

        # normalize to ppm
        if self.config.normalize:
            dp.normalize_flux()

        # Sort by time and close gaps larger than "gap_size_days"
        if self.config.close_gaps:
            dp.close_gaps() 

        # Save light curve as feather file
        if self.config.save_lc:
            dp.save_lc()

        # Inject noise in ppm?
        if self.config.add_noise:
            dp.inject_noise()

        # Savgol
        if self.config.savgol:
            dp.savgol_smooth()

        # Compute PSD with frequencies in microHz 
        dp.microHz_periodogram()  

        # Light curve (potentially change DataProcessing to output dataclasses rather than tuples)
        time, flux, flux_err = dp.final_lc
        self.lc = LightCurveData(
            time = time,
            flux = flux,
            flux_err = flux_err
        )

        # PSD (potentially change DataProcessing to output dataclasses rather than tuples)
        frequency, psd = dp.final_psd
        self.psd = PSDData(
            frequency = frequency,
            psd = psd
        )

        # Averaged PSD
        if self.config.do_avg_psd:
            chunk_length = self.config.avg_psd_chunk
            dp.averaged_psd(chunk_len=chunk_length)
            avg_psd_freq, avg_psd_power = dp.avg_psd
            self.avg_psd = AvgPSDData(
                frequency = avg_psd_freq,
                psd = avg_psd_power
            )
        
        # Welch PSD
        if self.cov_config.use_welch:
            welch_freq, welch_psd = dp.calculate_welch_spectrum().welch_psd
            self.welch_psd = AvgPSDData(
                frequency = welch_freq,
                psd = welch_psd
            )            

        # Plot lc and pg
        if self.config.plot_lc:
            dp.plot_lc_and_pg()

        if self.config.save_psd:
            dp.save_periodogram()

        if self.config.save_avgpsd:
            dp.save_avg_periodogram()

    def _query_gaia(self):
        """Query gaia if specified"""
        if self.config.query_gaia:
            gaia_dictionary = query_gaia(id=self.star.target)
            if gaia_dictionary:
                self.gaia_data = GaiaData(**gaia_dictionary)
            else:
                self.gaia_data = GaiaData()
        else:
            self.gaia_data = GaiaData()
            
    def _load_psd(self, filename : str):
        """Load PSD file from filename if specified in YAML input file"""
        if filename.endswith('.csv'):
            data = np.genfromtxt(
                filename, delimiter=",", names=["frequency", "power"]
            )
        elif filename.endswith('feather') or filename.endswith('.ftr'):
            data = feather.read_feather(filename)
        
        mask = (
                ~np.isnan(data["frequency"])
                & ~np.isnan(data["power"])
            )
        data = data[mask]
        return data['frequency'], data['power']

    @classmethod
    def read_yaml(cls, yaml_path: str):
        """Grab config parameters from yaml file"""
        with open(yaml_path, 'r') as f:
            yaml_file = yaml.safe_load(f)
        # Store information in GlobalConfig structure
        settings = GlobalConfig(
            star=StarInfo(**yaml_file["STAR"]),
            lightcurve=LightCurveInput(**yaml_file["LIGHTCURVE"]),
            psd=PSDInput(**yaml_file['PSD']),
            config=ProcessingConfig(**yaml_file["CONFIG"]),
            acf_config=ACFConfig(**yaml_file["ACF_CONFIG"]),
            cov_config=COVConfig(**yaml_file["COV_CONFIG"])
        )
        return cls(global_config=settings)
    
    def run(self) -> "NumaxProxies":
        """Run pipeline"""
        # Query Gaia DR3/2 for data
        self._query_gaia()

        if self.psd_input.psd_file:
            self.psd = PSDData(
                self._load_psd(self.psd_input.psd_file)
            )

        if self.psd_input.avg_psd_file:
            self.avg_psd = AvgPSDData(
                self._load_psd(self.psd_input.avg_psd_file)
            )
        else:
            self._load_lightcurve()
            self._process_lightcurve() 
        
        return self
        
        


        








    

    





