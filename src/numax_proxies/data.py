"""Containers for light curves, spectra and Gaia parameters."""
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class LightCurveData:
    """Light curve: time [days], flux and flux_err (ppm after normalisation)."""
    time: NDArray[np.float64]
    flux: NDArray[np.float64]
    flux_err: NDArray[np.float64]

    @property
    def length_days(self) -> float:
        """Time span [days]."""
        return float(self.time[-1] - self.time[0])

    @property
    def cadence_secs(self) -> float:
        """Median cadence [s]."""
        return float(86400 * np.median(np.diff(self.time)))


@dataclass
class PSDData:
    """Power spectral density: frequency [muHz], psd."""
    frequency: NDArray[np.float64]
    psd: NDArray[np.float64]


# Kept as a separate name so that it is clear which spectrum is used
AvgPSDData = PSDData


@dataclass
class GaiaData:
    """Stellar parameters from Gaia DR3 (uncertainties.ufloat or None)."""
    teff_gspspec: Optional[object] = None
    teff_gspphot: Optional[object] = None
    logg_gspspec: Optional[object] = None
    logg_gspphot: Optional[object] = None
    mass_flame: Optional[object] = None
    lum_flame: Optional[object] = None
    rad_flame: Optional[object] = None
    rad_gspphot: Optional[object] = None

    def has_data(self) -> bool:
        """True if any parameter is set."""
        return any(getattr(self, f.name) is not None for f in fields(self))
