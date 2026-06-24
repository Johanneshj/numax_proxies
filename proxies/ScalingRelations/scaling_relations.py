import numpy as np
from uncertainties import unumpy as unp
from uncertainties import ufloat
from .query import query_gaia
from ...data_preparation.dataclasses import GaiaData, StarInfo
from typing import Optional
from itertools import product


def compute_numaxes(star : StarInfo, gaia_data : Optional[GaiaData]) -> dict:
    """Compute numax estimates from the scaling relations"""

    # Did we feed the function gaia_data?
    if gaia_data is not None:
        print("Gaia query already completed")
    else:
        gaia_data = query_gaia(id=star.target)

    # Collect gaia data in lists
    teffs = collect(gaia_data.teff_gspspec, gaia_data.teff_gspphot)
    loggs = collect(gaia_data.logg_gspspec, gaia_data.logg_gspphot)
    radii = collect(gaia_data.rad_flame, gaia_data.rad_gspphot)
    masses = collect(gaia_data.mass_flame)
    lums = collect(gaia_data.lum_flame)

    # Check if user specified parmaeters
    if star.teff is not None:
        teffs.extend(unp.uarray(star.teff, star.teff_err))

    if star.logg is not None:
        loggs.extend(unp.uarray(star.logg, star.logg_err))
    
    if star.radius is not None:
        radii.extend(unp.uarray(star.radius, star.radius_err))
    
    if star.mass is not None:
        masses.extend(unp.uarray(star.mass, star.mass_err))

    if star.luminosity is not None:
        lums.extend(unp.uarray(star.luminosity, star.luminosity_err))

    numaxes = numax_scaling_relations(
        logg=loggs, teff=teffs, lum=lums, mass=masses, radius=radii
    )
    return numaxes

def numax_scaling_relations(
            logg=None, 
            teff=None, 
            lum=None, 
            mass=None, 
            radius=None
    ) -> dict:
    """
    Compute three νmax (numax) scaling relations for stars.

    Parameters
    ----------
    logg : float or array-like
        Surface gravity (dex)
    teff : float or array-like
        Effective temperature (K)
    lum : float or array-like
        Luminosity (L_sun)
    mass : float or array-like
        Mass (M_sun)
    radius : float or array-like
        Radius (R_sun)

    Returns
    -------
    list of float or np.ndarray
        [numax_1, numax_2, numax_3] scaling relation values in microHz
    """
    numaxes = {}

    # Solar reference values
    logg_Sun = 4.44  # dex
    teff_Sun = 5777.0  # K
    numax_Sun = 3090.0  # microHz

    # Values
    teffs = make_uarray(teff)
    loggs = make_uarray(logg)
    masses = make_uarray(mass)
    radii = make_uarray(radius)
    lums = make_uarray(lum)

    # Define lists
    numax_1 = []
    numax_2 = []
    numax_3 = []

    # logg–Teff relation
    for logg, teff in product(loggs, teffs):
        numax_1.append((numax_Sun * (10 ** (logg - logg_Sun)) / unp.sqrt(teff / teff_Sun))[0])

    # Mass–Radius–Teff relation
    for radius, mass, teff in product(radii, masses, teffs):
        numax_2.append((numax_Sun * radius ** (-2) * mass / unp.sqrt(teff / teff_Sun))[0])
    
    # Mass–Luminosity–Teff relation
    for lum, mass, teff in product(lums, masses, teffs):
        mass_div_lum = mass / lum
        numax_3.append((numax_Sun * mass_div_lum * (teff / teff_Sun) ** 3.5)[0])

    # Put estimates into dictionary
    for i, val in enumerate(numax_1):
        numaxes[f"numax_SR_logg_teff_{i}"] = val
    
    for i, val in enumerate(numax_2):
        numaxes[f"numax_SR_mass_radius_teff_{i}"] = val

    for i, val in enumerate(numax_3):
        numaxes[f"numax_SR_mass_luminosity_teff_{i}"] = val

    return numaxes

def make_uarray(vals):
    """Make lists broadcastable arrays"""
    return unp.uarray([[v.nominal_value] for v in vals], [[v.std_dev] for v in vals])

def collect(*values):
    return [v for v in values if v is not None]
