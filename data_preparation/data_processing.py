import numpy as np
from astropy.timeseries import LombScargle
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import pandas as pd
import time as t
import os
from scipy.signal import savgol_filter, welch
from scipy.signal.windows import hann
from .dataclasses import LightCurveData, ProcessingConfig, COVConfig
from typing import Optional, Literal
from numpy.lib.stride_tricks import sliding_window_view as slw
import nifty_ls
import matplotlib.pyplot as plt

class DataProcessing:
    def __init__(
        self, 
        lc : LightCurveData,
        config : ProcessingConfig,
        cov_config : COVConfig,
        id : Optional[str] = 'unknown'
    ):
        # Load LC
        self.id = id or "unknown"
        self.time = lc.time.copy()
        self.flux = lc.flux.copy()
        self.flux_err = lc.flux_err.copy()

        # Load config settings
        self.cfg = config
        self.cov_config = cov_config

    # ----------------------------
    # Light curve
    # ----------------------------
    def normalize_flux(self):
        """Normalize flux to ppm"""
        flux = self.flux
        med = np.nanmedian(flux)
        self.flux = (flux / med - 1) * 1e6
        self.flux_err = self.flux_err / med * 1e6
        return self

    def sort_data_by_time(self):
        """Sort data by time values"""
        time, flux, flux_err = self.time, self.flux, self.flux_err
        _, unique_idx = np.unique(time, return_index=True)
        time = time[unique_idx]
        flux = flux[unique_idx]
        flux_err = flux_err[unique_idx]
        sort_idx = np.argsort(time)
        self.time = time[sort_idx] - time[sort_idx][0]
        self.flux = flux[sort_idx]
        self.flux_err = flux_err[sort_idx]
        return self

    def close_gaps(self):
        """close gaps larger than gap_size_days"""
        gap_size = self.cfg.gap_size_days
        gap_indices = np.concatenate(
            (np.where(np.diff(self.time) > gap_size)[0], [len(self.time) - 1])
        )
        for i in range(1, len(gap_indices)):
            start = gap_indices[i - 1] + 1
            end = gap_indices[i]
            gap = (self.time[start] - self.time[start - 1]) - 3
            self.time[start : end + 1] -= gap
        return self
    
    def savgol_smooth(self):
        """
            Perfom savgol-golay filter smoothing.
            Important! We should update this so we can do iterative savgol filtering.
        """
        self.wl_days = self.cfg.savgol_window
        dt = np.median(np.diff(self.time))
        wl = int(self.wl_days / dt) 
        if (wl % 2) == 0:
            wl += 1
        self.sg_filter = savgol_filter(self.flux, window_length=wl, polyorder=3)
        self.old_flux = self.flux.copy()
        self.flux -= self.sg_filter
        return self
    
    def inject_noise(self):
        """Inject artifical noise (ppm units) in light curve"""
        noise = np.random.normal(0, self.cfg.noise_std, len(self.flux))
        self.flux += noise
        return self

    # ----------------------------
    # Periodogram (routines adopted from Martin Nielsen at Porto summer school 2024)
    # ----------------------------
    def compute_lombscargle(self):
        """Initialize LombScargle object"""
        self.ls = LombScargle(
            t=self.time,
            y=self.flux,
            dy=self.flux_err,
            fit_mean=False,
            center_data=True,
        )
        return self

    def microHz_periodogram(self):
        """Compute frequencies in muHz and power in spectral density"""
        # Generate LombScargle object
        self.compute_lombscargle()
        # Calculate nyquist frequency
        nyq = self.nyquist(time=self.time)
        # Calculate frequency spacing (accounting for spectral window)
        df = self.freq_spacing(time=self.time, nyq=nyq)
        # Define frequency range including oversampling
        self.frequency = np.arange(df/self.cfg.oversampling, nyq, df/self.cfg.oversampling)
        # Calculate power
        self.power = self.ls.power(
            self.frequency, normalization="psd", method="fast", assume_regular_frequency=True
        )
        self.frequency *= 1e6 / 86400
        self.nyq = np.max(self.frequency)
        return self

    # ----------------------------
    # Averaged PSD (Sylvain Breton)
    # ----------------------------
    # def averaged_psd(self, chunk_len=90):
    #     """
    #     Author: Sylvain Breton
    #     email: sylvain.breton@inaf.it
    #     Created: 22 Nov 2024
    #     INAF-OACT

    #     Compute mean PSD of a light curve, by subdividing it
    #     into chunks of equal length. The light curve sampling
    #     is assumed to be regular.

    #     Parameters
    #     ----------
    #     param time: ndarray
    #     Input time vector in days

    #     param flux: ndarray
    #     Input flux of the light curve in ppm

    #     param len_chunk: float
    #     Length of the chunks in days.
    #     Optional, default 90

    #     Returns
    #     -------
    #     tuple of arrays
    #     A tuple with frequency and power spectral density
    #     vectors.
    #     """
    #     time = self.time
    #     flux = self.flux
    #     flux_err = self.flux_err

    #     dt = np.median(np.diff(time))
    #     dt_sec = dt * 86400.0
    #     df = 1 / (np.max(time)*86400.0)
    #     len_chunk = chunk_len
    #     if len_chunk >= np.max(time) / 2:
    #         raise ValueError('chunk length for averaged PSD too long. Should be atleast 1/2 length of time series.')
    #     size_chunk = int(len_chunk / dt)
    #     n_chunk = len(time) // size_chunk

    #     time = (
    #         time[: size_chunk * n_chunk].reshape((n_chunk, size_chunk)) * 86400
    #     )  # convert back to seconds
    #     flux = flux[: size_chunk * n_chunk].reshape((n_chunk, size_chunk))
    #     flux_err = flux_err[: size_chunk * n_chunk].reshape((n_chunk, size_chunk))

    #     # Common frequency grid in Hz
    #     freq_grid = np.arange(df / self.cfg.oversampling, 1/(2*dt_sec), df / self.cfg.oversampling)
    #     # print(self.cfg.oversampling * df, 1/(2*dt_sec))
    #     # freq, psd = self.calculate_psd_for_avg_psd(
    #     #     time[0], flux[0], flux_err[0], freq_grid=None
    #     # )

    #     # PSD for each segment on common frequency grid
    #     psd = [
    #         self.calculate_psd_for_avg_psd(t_chunk, f_chunk, ferr_chunk, freq_grid=freq_grid)
    #         for t_chunk, f_chunk, ferr_chunk in zip(time, flux, flux_err)
    #     ]
    #     # psd = sum(psd)

    #     # psd /= n_chunk
    #     psd_matrix = np.array(psd)  
    #     psd = np.median(psd_matrix, axis=0, ddof=1)

    #     self.avgpsd_freq = freq_grid * 1e6  # convert to microHz
    #     self.avgpsd_power = psd
    #     return self

    def averaged_psd(self, chunk_len=90, overlap=0.0):
            """
            Author: Sylvain Breton
            email: sylvain.breton@inaf.it
            Created: 22 Nov 2024
            INAF-OACT
    
            Compute mean PSD of a light curve, by subdividing it
            into chunks of equal length. The light curve sampling
            is assumed to be regular.
    
            Parameters
            ----------
            param time: ndarray
            Input time vector in days
    
            param flux: ndarray
            Input flux of the light curve in ppm
    
            param len_chunk: float
            Length of the chunks in days.
            Optional, default 90
    
            Returns
            -------
            tuple of arrays
            A tuple with frequency and power spectral density
            vectors.
            """
            
            time = self.time.astype(np.float64)
            flux = self.flux.astype(np.float64)
            flux_err = self.flux_err.astype(np.float64)

            dt = np.median(np.diff(time))
            len_chunk = chunk_len
            while len_chunk >= np.max(time) / 2:
                len_chunk -= 1
            # if len_chunk >= np.max(time) / 2:
            #     raise ValueError('chunk length for averaged PSD too long. Should be atleast 1/2 length of time series.')
            size_chunk = int(len_chunk / dt)          

            # Generate overlapping chunks using sliding window view
            chunk_overlap = int(overlap * size_chunk)
            step = size_chunk - chunk_overlap

            time = slw(time * 86400, window_shape=size_chunk, writeable=True)[::step]
            flux = slw(flux, window_shape=size_chunk, writeable=True)[::step]
            hann_window = hann(size_chunk, sym=True)
            flux_err = slw(flux_err, window_shape=size_chunk, writeable=True)[::step]
    
            # Define common frequency grid in Hz based on chunk length and oversampling         
            dt = np.median(np.diff(time))
            df = 1 / np.max(time[0, :])
            freq_grid = np.arange(df / self.cfg.oversampling, 1/(2*dt), df / self.cfg.oversampling, dtype=np.float64)       

            # Compute lombscargle for each chunk using nifty_ls
            start = t.time()
            n_chunks = time.shape[0]
            n_freqs = len(freq_grid)
            psd_matrix = nifty_ls.lombscargle_heterobatch(
                t_list=list(time),
                y_list=list(flux * hann_window), # APPLY HANNING WINDOW
                dy_list=list(flux_err),
                normalization='psd',
                fmin_list=np.full(n_chunks, freq_grid[0], dtype=np.float64),
                fmax_list=np.full(n_chunks, freq_grid[-1], dtype=np.float64),
                Nf_list=np.full(n_chunks, n_freqs, dtype=int),
                nthreads=8
            )
            end = t.time()
            print(f'nifty windows time: {np.round(end-start, 2)} seconds')

            # Compute median PSD; median is more robust than mean for outliers
            psd_matrix = psd_matrix.power_list
            psd = np.median(psd_matrix, axis=0)

            # Return to muHz
            self.avgpsd_freq = freq_grid * 1e6
            self.avgpsd_power = psd
            return self

    def calculate_psd_for_avg_psd(self, time, flux, flux_err, freq_grid=None):
        """Calculate freq and power to later sum up for averaged psd"""
        flux *= hann(len(flux), sym=True)
        ls = LombScargle(t=time, y=flux, dy=flux_err, fit_mean=False, center_data=True)

        if freq_grid is None:
            nyq = self.nyquist(time)
            freq_Hz, power = ls.autopower(
                normalization="psd", method="fast", maximum_frequency=nyq
            )
        else:
            power = ls.power(
                freq_grid,
                normalization="psd",
                method="fast",
                assume_regular_frequency=True,
            )

        if freq_grid is None:
            return freq_Hz, power
        else:
            return power
        
    def calculate_welch_spectrum(self):
        """Calculate Welch spectrum"""
        x = np.asarray(self.flux)

        dt_days = np.mean(np.diff(self.time))
        dt_seconds = dt_days * 24 * 60 * 60

        fs_uHz = 1e6 / dt_seconds
        
        seg_size = int(self.cov_config.welch_seg_size / dt_days)
        start = t.time()
        self.welch_f, self.welch_p = welch(
            x=x, 
            fs=fs_uHz, 
            scaling='density', 
            window='hann',
            nperseg=seg_size,
            average='median',
            noverlap=int(0.9 * seg_size)
        )
        end = t.time()
        print(f'Time to produce Welch spectrum: {np.round(end-start, 3)} seconds')
        # self.welch_f *= 1e6 / 86400.0
        return self


    # ----------------------------
    # Misc. routines
    # ----------------------------   
    def nyquist(self, time):
        return 1 / (2 * np.mean(np.diff(time)))    
    
    def freq_spacing(self, time, nyq):
        """Calculate frequency spacing accounting for spectral window"""
        df = 1 / (np.nanmax(time) - np.nanmin(time))
        f, w = self.windowfunction(df=df, nyq=nyq)#, width=None, oversampling=self.cfg.oversampling)
        df = simpson(w, x=f)
        return df
    
    def windowfunction(self, df, nyq):
        """Calculate spectral window function"""
        if self.cfg.width_for_wf is None:
            width = 100 * df
        time = self.time.copy()
        freq_cen = 0.5 * nyq
        Nfreq = int(self.cfg.oversampling * width / df)
        freq = freq_cen + (df / self.cfg.oversampling) * np.arange(-Nfreq, Nfreq, 1)
        x = 0.5 * np.sin(2 * np.pi * freq_cen * time) + 0.5 * np.cos(
            2 * np.pi * freq_cen * time
        )
        ls = LombScargle(time, x, center_data=True, fit_mean=False)
        power = ls.power(
            freq, method="fast", normalization="psd", assume_regular_frequency=True
        )
        power /= power[int(len(power) / 2)]  # Normalize to have maximum of one
        freq -= freq_cen
        return freq, power

    def attenuation(self):
        eta = np.sinc(0.5 * self.frequency / self.nyq)
        return eta

    # ----------------------------
    # Super Nyquist spectrum (EXPERIMENTAL!)
    # ----------------------------
    @staticmethod
    def sup_nyquist(time):
        # Return frequency slightly above Nyquist frequency
        nu_Nyq = 1 / (2 * np.mean(np.diff(time)))
        return 2.0 * nu_Nyq

    def super_Nyquist_spectrum(self):
        """Compute super Nyquist spectrum in muHz and PSD"""
        self.compute_lombscargle()
        self.supNyq_lim = self.sup_nyquist(self.time)
        self.df = self.freq_spacing(self.time)
        freq_Hz = np.arange(self.df, self.supNyq_lim, self.df)

        self.supNyq_freq = freq_Hz * 1e6
        self.supNyq_power = self.power(
            freq_Hz, normalization="psd", method="fast", assume_regular_frequency=True
        )
        return self

    # ----------------------------
    # Access properties
    # ----------------------------
    @property
    def final_lc(self):
        # Final light curve
        return self.time, self.flux, self.flux_err

    @property
    def final_psd(self):
        # Final psd
        return self.frequency, self.power

    @property
    def avg_psd(self):
        # Averaged psd
        return self.avgpsd_freq, self.avgpsd_power

    @property
    def welch_psd(self):
        # Welch psd
        return self.welch_f, self.welch_p

    @property
    def supNyq_psd(self):
        # Super nyquist PSD
        return self.supNyq_freq, self.supNyq_power

    # ----------------------------
    # Plotting
    # ----------------------------
    def plot_lc_and_pg(self):
        """Plot final lc and pg"""
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        time, flux, _ = self.final_lc
        freq, power = self.final_psd

        # axs[0].plot(time, flux, c="k", lw=0, ms=3, marker=".")
        axs[0].set_xlabel("time [days]")
        axs[0].set_ylabel("rel. amp. [ppm]")
        axs[0].set_xlim(np.min(time), np.max(time))
        if self.cfg.savgol:
            axs[0].plot(time, self.old_flux, c="k", lw=2, ls="-")
            axs[0].plot(time, self.sg_filter , c="r", lw=2, ls="-")
            axs[0].text(0.02, 0.98, 
                        f"window size for savgol: {self.wl_days} days", 
                        ha="left", va="top", transform=axs[0].transAxes,
                        color='red')
        else:
            axs[0].plot(time, flux, c="k", lw=0, ms=3, marker=".")

        axs[1].loglog(freq, power, c="k")
        axs[1].set_xlabel("frequency [muHz]")
        axs[1].set_ylabel("PSD [ppm^2/muHz]")
        axs[1].set_xlim(np.min(freq), np.max(freq))

        savepath = os.path.join("numax_proxies", "results", self.id, "figures")
        os.makedirs(savepath, exist_ok=True)
        fig.savefig(f"{savepath}/lc_and_pg.png", dpi=300, bbox_inches="tight")

    # ----------------------------
    # Saving data
    # ----------------------------
    def save_periodogram(self, folder=None, id=None):
        """Save frequency and power to .csv file"""
        freq, power = self.final_psd
        if id is None:
            id = self.id
        if folder is None:
            folder = 'numax_proxies/results'
        df = pd.DataFrame(
            np.column_stack((freq, power)), 
            columns=['frequency', 'power']
        )
        df.to_feather(f"{folder}/PSD_{id}.csv")

    def save_avg_periodogram(self, folder=None, id=None):
        """Save frequency and power to .csv file"""
        freq, power = self.avg_psd
        if id is None:
            id = self.id
        if folder is None:
            folder = 'numax_proxies/results'
        df = pd.DataFrame(
            np.column_stack((freq, power)), 
            columns=['frequency', 'power']
        )
        df.to_feather(f"{folder}/AvgPSD_{id}.csv")

    def save_lc(self):
        df = pd.DataFrame(
            np.column_stack((self.time, self.flux, self.flux_err)),
            columns=['time', 'flux', 'flux_err']
        )
        savepath_1 = os.path.join('numax_proxies', 'results', f'{self.id}')
        if not os.path.exists(savepath_1):
            os.mkdir(savepath_1)
        
        savepath_2 = os.path.join(savepath_1, 'data')
        if not os.path.exists(savepath_2):
            os.mkdir(savepath_2)

        full_savepath = os.path.join(savepath_2, f"LC_{self.id}.ftr")

        df.to_feather(full_savepath)