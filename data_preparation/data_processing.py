import numpy as np
from astropy.timeseries import LombScargle
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import os


class DataProcessing:
    def __init__(
        self, lc=None, time=None, flux=None, flux_err=None, id=None, *args, **kwargs
    ):
        self._id = id or "unknown"
        if lc is not None:
            self._time = lc.time.value
            self._flux = lc.flux.value
            self._flux_err = lc.flux_err.value
        else:
            self._time = np.array(time, dtype=float)
            self._flux = np.array(flux, dtype=float)
            self._flux_err = np.array(flux_err, dtype=float)
        self._time *= 86400.0  # Convert days to seconds for periodogram calculations

        # Normalize to ppm if specified - use if light curve is in flux or normalized flux
        if kwargs.get("normalize", False):
            self.normalize_flux()

        # Periodogram attributes
        self._ls = None
        self._nyq = None
        self._df = None

        # Frequency and power
        self._frequency = None
        self._power = None

        # Averaged psd
        self._avgpsd_freq = None
        self._avgpsd_power = None

        # Super-Nyquist PSD
        self._supNyq_lim = None
        self._supNyq_freq = None
        self._supNyq_power = None

    # ----------------------------
    # Light curve
    # ----------------------------
    def normalize_flux(self):
        """Normalize flux to ppm"""
        flux = self._flux
        med = np.nanmedian(flux)
        self._flux = (flux / med - 1) * 1e6
        self._flux_err = self._flux_err / med * 1e6
        return self

    def sort_data_by_time(self):
        """Sort data by time values"""
        time, flux, flux_err = self._time, self._flux, self._flux_err
        _, unique_idx = np.unique(time, return_index=True)
        time = time[unique_idx]
        flux = flux[unique_idx]
        flux_err = flux_err[unique_idx]
        sort_idx = np.argsort(time)
        self._time = time[sort_idx] - time[sort_idx][0]
        self._flux = flux[sort_idx]
        self._flux_err = flux_err[sort_idx]
        return self

    def close_gaps(self):
        """close gaps larger than three days"""
        time = self._time
        gap_size_secs = 3 * 86400.0
        gap_indices = np.concatenate(
            (np.where(np.diff(time) > gap_size_secs)[0], [len(time) - 1])
        )
        for i in range(1, len(gap_indices)):
            start = gap_indices[i - 1] + 1
            end = gap_indices[i]
            gap = (time[start] - time[start - 1]) - 3
            time[start : end + 1] -= gap
        self._time = time
        return self

    def sort_and_close_gaps(self):
        """do sort and close gaps in one call"""
        self.sort_data_by_time()
        self.close_gaps()
        return self

    # ----------------------------
    # Periodogram (routines adopted from Martin Nielsen at Porto summer school 2024)
    # ----------------------------
    def compute_lombscargle(self):
        """Initialize LombScargle object"""
        self._ls = LombScargle(
            t=self._time,
            y=self._flux,
            dy=self._flux_err,
            fit_mean=False,
            center_data=True,
        )
        return self

    def microHz_periodogram(self):
        """Compute frequencies in muHz and power in spectral density"""
        self.compute_lombscargle()
        self._nyq = self.nyquist(self._time)
        self._df = self.freq_spacing(self._time)
        freq_Hz = np.arange(self._df, self._nyq, self._df)

        self._frequency = freq_Hz * 1e6
        self._power = self._ls.power(
            freq_Hz, normalization="psd", method="fast", assume_regular_frequency=True
        )
        return self

    # ----------------------------
    # Averaged PSD (Sylvain Breton)
    # ----------------------------
    def averaged_psd(self):
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
        time = self._time / 86400.0  # convert to days
        flux = self._flux
        flux_err = self._flux_err

        dt = np.median(np.diff(time))
        len_chunk = 90
        size_chunk = int(len_chunk / dt)
        n_chunk = len(time) // size_chunk

        time = (
            time[: size_chunk * n_chunk].reshape((n_chunk, size_chunk)) * 86400
        )  # convert back to seconds
        flux = flux[: size_chunk * n_chunk].reshape((n_chunk, size_chunk))
        flux_err = flux_err[: size_chunk * n_chunk].reshape((n_chunk, size_chunk))

        freq, psd = self.calculate_psd_for_avg_psd(
            time[0], flux[0], flux_err[0], freq_grid=None
        )

        psd = [
            self.calculate_psd_for_avg_psd(t_chunk, f_chunk, ferr_chunk, freq_grid=freq)
            for t_chunk, f_chunk, ferr_chunk in zip(time, flux, flux_err)
        ]
        psd = sum(psd)

        psd /= n_chunk

        self._avgpsd_freq = freq * 1e6  # convert to microHz
        self._avgpsd_power = psd
        return self

    def calculate_psd_for_avg_psd(self, time, flux, flux_err, freq_grid=None):
        """Calculate freq and power to later sum up for averaged psd"""
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

    # ----------------------------
    # Misc. routines
    # ----------------------------
    def freqHz(self):
        return np.arange(self._df, self._nyq, self._df, dtype="float64")

    # def super_Nyq_freqHz(self):
    #     return np.arange(self._df, self._sup_nyq, self._df, dtype='float64')

    @staticmethod
    def nyquist(time):
        return 1 / (2 * np.mean(np.diff(time)))

    def freq_spacing(self, time, oversampling=5):
        """Calculate frequency spacing accounting for spectral window"""
        df0 = 1 / (np.nanmax(time) - np.nanmin(time))
        f, w = self.windowfunction(df=df0, width=None, oversampling=oversampling)
        df = simpson(w, x=f)
        return df * 1e-6

    def windowfunction(self, df, width=None, oversampling=10):
        """Calculate spectral window function"""
        if width is None:
            width = 100 * df
        freq_cen = 0.5 * self._nyq
        Nfreq = int(oversampling * width / df)
        freq = freq_cen + (df / oversampling) * np.arange(-Nfreq, Nfreq, 1)
        x = 0.5 * np.sin(2 * np.pi * freq_cen * self._ls.t) + 0.5 * np.cos(
            2 * np.pi * freq_cen * self._ls.t
        )
        ls = LombScargle(self._ls.t, x, center_data=True, fit_mean=False)
        power = ls.power(
            freq, method="fast", normalization="psd", assume_regular_frequency=True
        )
        power /= power[int(len(power) / 2)]  # Normalize to have maximum of one
        freq -= freq_cen
        freq *= 1e6
        return freq, power

    def attenuation(self):
        eta = np.sinc(0.5 * self._frequency / self._nyq)
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
        self._supNyq_lim = self.sup_nyquist(self._time)
        self._df = self.freq_spacing(self._time)
        freq_Hz = np.arange(self._df, self._supNyq_lim, self._df)

        self._supNyq_freq = freq_Hz * 1e6
        self._supNyq_power = self._ls.power(
            freq_Hz, normalization="psd", method="fast", assume_regular_frequency=True
        )
        return self

    # ----------------------------
    # Access properties
    # ----------------------------
    @property
    def final_lc(self):
        # Convert time back to days
        return self._time / 86400.0, self._flux, self._flux_err

    @property
    def final_psd(self):
        # Final psd
        return self._frequency, self._power

    @property
    def avg_psd(self):
        # Averaged psd
        return self._avgpsd_freq, self._avgpsd_power

    @property
    def supNyq_psd(self):
        # Super nyquist PSD
        return self._supNyq_freq, self._supNyq_power

    # ----------------------------
    # Plotting
    # ----------------------------
    def plot_lc_and_pg(self):
        """Plot final lc and pg"""
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        time, flux, _ = self.final_lc
        freq, power = self.final_psd

        axs[0].plot(time, flux, c="k", lw=0, ms=3, marker=".")
        axs[0].set_xlabel("time [days]")
        axs[0].set_ylabel("rel. amp. [ppm]")
        axs[0].set_xlim(np.min(time), np.max(time))

        axs[1].loglog(freq, power, c="k")
        axs[1].set_xlabel("frequency [muHz]")
        axs[1].set_ylabel("PSD [1/muHz^2]")
        axs[1].set_xlim(np.min(freq), np.max(freq))

        savepath = os.path.join("numax_proxies", "results", self._id, "figures")
        os.makedirs(savepath, exist_ok=True)
        fig.savefig(f"{savepath}/lc_and_pg.png", dpi=300, bbox_inches="tight")

    # ----------------------------
    # Saving data
    # ----------------------------
    def save_periodogram(self, folder="./", id=None):
        """Save frequency and power to .csv file"""
        freq, power = self.final_psd
        if id is None:
            id = self._id
        np.savetxt(f"{folder}/{id}.csv", np.column_stack((freq, power)), delimiter=",")
