"""Stellar parameters from Gaia DR3 via SIMBAD (only imported when query_gaia is used)."""
import logging
import time

import numpy as np
from uncertainties import ufloat

from .data import GaiaData

logger = logging.getLogger(__name__)

COLUMNS = ["teff_gspspec", "teff_gspphot", "logg_gspspec", "logg_gspphot",
           "mass_flame", "lum_flame", "radius_flame", "radius_gspphot"]
# GaiaData field name for each DR3 column
FIELD = {"radius_flame": "rad_flame", "radius_gspphot": "rad_gspphot"}


def query_simbad_gaia_id(name: str, retries: int = 3, delay: float = 1.5):
    """Latest Gaia identifier of an object in SIMBAD, or None."""
    from astroquery.simbad import Simbad

    for attempt in range(retries):
        try:
            res = Simbad.query_objectids(object_name=name)
            if res is None or len(res) == 0:
                return None
            ids = [str(row[0]) for row in res if "gaia" in str(row[0]).lower()]
            return sorted(ids)[-1] if ids else None
        except (OSError, ConnectionError) as exc:
            logger.warning("SIMBAD query failed (attempt %d): %s", attempt + 1, exc)
            time.sleep(delay)
    return None


def _query(gaia_id: str) -> str:
    """ADQL query for the DR3 astrophysical parameters of a Gaia id (DR2 ids are cross-matched)."""
    _, release, number = gaia_id.split(" ")[:3]
    cols = ",\n".join(f"dr3.{c}, dr3.{c}_lower, dr3.{c}_upper" for c in COLUMNS)
    if release == "DR2":
        where = (f"JOIN (SELECT * FROM gaiadr3.dr2_neighbourhood WHERE dr2_source_id = {int(number)}) "
                 "AS xmatch ON xmatch.dr3_source_id = dr3.source_id")
    else:
        where = f"WHERE dr3.source_id = {int(number)}"
    return f"SELECT dr3.source_id, {cols} FROM gaiadr3.astrophysical_parameters AS dr3 {where}"


def _error(val, lower, upper):
    """Symmetric error from the lower / upper confidence values."""
    if np.isfinite(lower) and np.isfinite(upper):
        return (upper - lower) / 2
    if np.isfinite(lower):
        return val - lower
    if np.isfinite(upper):
        return upper - val
    return np.nan


def query_gaia(target: str) -> GaiaData:
    """Gaia DR3 Teff, logg, mass, luminosity and radius of a target (empty GaiaData if it fails)."""
    from astroquery.gaia import Gaia

    gaia_id = query_simbad_gaia_id(target)
    if gaia_id is None:
        logger.warning("No Gaia identifier found for %s", target)
        return GaiaData()
    try:
        res = Gaia.launch_job_async(_query(gaia_id)).get_results()
    except Exception as exc:  # network/archive errors of many kinds
        logger.warning("Gaia query failed for %s: %s", target, exc)
        return GaiaData()
    data = GaiaData()
    if len(res) == 0:
        return data
    row = res[0]
    for col in COLUMNS:
        val = float(np.ma.filled(row[col], np.nan))
        err = _error(val, float(np.ma.filled(row[f"{col}_lower"], np.nan)),
                     float(np.ma.filled(row[f"{col}_upper"], np.nan)))
        if np.isfinite(val) and np.isfinite(err):
            setattr(data, FIELD.get(col, col), ufloat(val, abs(err)))
    return data
