import numpy as np
from uncertainties import unumpy as unp
from uncertainties import ufloat
from .query import query_gaia


def compute_numaxes(*args, **kwargs) -> dict:
    if kwargs.get("gaia_query_dict"):
        numax_dict = kwargs.get("gaia_query_dict")
    else:
        numax_dict = query_gaia(id=kwargs.get("id"))

    # We do a "try" here to catch if the entire query failed
    try:
        if not isinstance(numax_dict, dict):
            raise ValueError("numax_dict is not a dictionary")

        params = ["logg", "teff", "luminosity", "radius", "mass"]
        for param in params:
            if kwargs.get(f"{param}"):
                numax_dict.setdefault(
                    f"{param}",
                    ufloat(
                        kwargs.get(f"{param}", np.nan), kwargs.get(f"{param}_err", 0)
                    ),
                )
            else:
                continue

        loggs = make_broadcastable_uarray("logg", numax_dict)
        teffs = make_broadcastable_uarray("teff", numax_dict)
        lums = make_broadcastable_uarray("lum", numax_dict)
        radii = make_broadcastable_uarray("radius", numax_dict)
        masses = make_broadcastable_uarray("mass", numax_dict)

        numaxes = numax_scaling_relations(
            logg=loggs, teff=teffs, lum=lums, mass=masses, radius=radii
        )
        return numaxes
    except Exception:
        # print('Scaling relations failed:', e)
        return {}


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

    # logg–Teff relation
    numax_1 = numax_Sun * (10 ** (logg - logg_Sun)) / unp.sqrt(teff / teff_Sun)
    for i, val in enumerate(numax_1):
        if len(val) > 0:
            numaxes[f"numax_SR_logg_teff_{i}"] = val[0]

    # Mass–Radius–Teff relation
    numax_2 = numax_Sun * radius ** (-2) * mass.T / unp.sqrt(teff / teff_Sun)
    for i, val in enumerate(numax_2):
        if len(val) > 0:
            numaxes[f"numax_SR_mass_radius_teff_{i}"] = val[0]

    # Mass–Luminosity–Teff relation
    mass_div_lum = mass / lum.T
    numax_3 = numax_Sun * mass_div_lum.ravel() * (teff / teff_Sun) ** 3.5
    for i, val in enumerate(numax_3):
        if len(val) > 0:
            numaxes[f"numax_SR_mass_luminosity_teff_{i}"] = val[0]

    return numaxes


def make_broadcastable_uarray(param, dictionary):
    numax_dict = dictionary
    vals = [
        entry[1]
        for entry in numax_dict.items()
        if f"{param}" in entry[0] and entry[1] is not np.nan
    ]
    arr = unp.uarray(
        [[val.nominal_value] for val in vals], [[val.std_dev] for val in vals]
    )
    return arr
