"""Reading and writing spectra, light curves and results."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .data import LightCurveData, PSDData

logger = logging.getLogger(__name__)

PSD_FORMATS = ".csv (frequency,power), .txt (2 columns), .pow (2 columns), .ftr/.feather, .parquet"
LC_FORMATS = ".csv (time,flux[,flux_err]), .ftr/.feather, .parquet"


# --------------------------------------------------------------------------------------
# Spectra
# --------------------------------------------------------------------------------------
def load_psd(filename: str) -> PSDData:
    """Load a PSD file (rows with NaNs are dropped).

    Args:
        filename: .csv, .txt, .pow, .ftr/.feather or .parquet with frequency [muHz], power.

    Returns:
        PSDData.
    """
    filename = str(filename)
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        data = np.genfromtxt(filename, delimiter=",", names=["frequency", "power"])
        f, p = data["frequency"], data["power"]
    elif suffix == ".txt":
        data = np.genfromtxt(filename, dtype=float)
        f, p = data[:, 0], data[:, 1]
    elif suffix == ".pow":
        df = pd.read_csv(filename, sep=r"\s+", names=["frequency", "power"], comment="#", header=None)
        f, p = df["frequency"].to_numpy(), df["power"].to_numpy()
    elif suffix in (".ftr", ".feather"):
        df = pd.read_feather(filename)
        f, p = df["frequency"].to_numpy(), df["power"].to_numpy()
    elif suffix == ".parquet":
        df = pd.read_parquet(filename, engine="fastparquet")
        f, p = df["frequency"].to_numpy(), df["power"].to_numpy()
    else:
        raise ValueError(f"Unsupported PSD file '{filename}'. Supported: {PSD_FORMATS}")

    f, p = np.asarray(f, dtype=float), np.asarray(p, dtype=float)
    mask = ~np.isnan(f) & ~np.isnan(p)
    return PSDData(frequency=f[mask], psd=p[mask])


def save_psd(psd: PSDData, path: Path) -> None:
    """Save a PSD as parquet (columns frequency, power).

    Args:
        psd: PSDData.
        path: output file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frequency": psd.frequency, "power": psd.psd}).to_parquet(path, index=False)


# --------------------------------------------------------------------------------------
# Light curves
# --------------------------------------------------------------------------------------
def load_lightcurve_file(filename: str) -> LightCurveData:
    """Load a light curve file.

    Args:
        filename: .csv (time, flux[, flux_err]), .ftr/.feather or .parquet; time in days.

    Returns:
        LightCurveData. Missing flux_err: 50 (csv) or the flux standard deviation.
    """
    filename = str(filename)
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        data = np.genfromtxt(filename, delimiter=",")
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"{filename}: expected 2 or 3 comma-separated columns")
        time, flux = data[:, 0], data[:, 1]
        if data.shape[1] >= 3:
            flux_err = data[:, 2]
        else:
            logger.warning("%s has no flux_err column; using flux_err = 50 for all points", filename)
            flux_err = np.full_like(flux, 50.0)
        mask = ~np.isnan(time) & ~np.isnan(flux) & ~np.isnan(flux_err)
        return LightCurveData(time[mask], flux[mask], flux_err[mask])
    if suffix in (".ftr", ".feather"):
        df = pd.read_feather(filename)
    elif suffix == ".parquet":
        df = pd.read_parquet(filename, engine="fastparquet")
    else:
        raise ValueError(f"Unsupported light-curve file '{filename}'. Supported: {LC_FORMATS}")
    time, flux = df["time"].to_numpy(float), df["flux"].to_numpy(float)
    if "flux_err" in df:
        flux_err = df["flux_err"].to_numpy(float)
    else:
        logger.warning("%s has no flux_err column; using the flux standard deviation", filename)
        flux_err = np.full_like(flux, np.nanstd(flux))
    return LightCurveData(time, flux, flux_err)


def save_lightcurve(lc: LightCurveData, path: Path) -> None:
    """Save a light curve as parquet (columns time, flux, flux_err).

    Args:
        lc: LightCurveData.
        path: output file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time": lc.time, "flux": lc.flux, "flux_err": lc.flux_err}).to_parquet(path, index=False)
