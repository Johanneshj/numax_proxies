"""numax from scaling relations, using Gaia and/or user-given stellar parameters."""
from itertools import product

from uncertainties import ufloat

from ..seismic import (numax_from_logg_teff, numax_from_mass_luminosity_teff,
                       numax_from_mass_radius_teff)


def _values(gaia_values, star_values, star_errors):
    """Gaia values plus ufloats of the user values."""
    out = [v for v in gaia_values if v is not None]
    if star_values is not None:
        out += [ufloat(v, e) for v, e in zip(star_values, star_errors)]
    return out


def numax_from_scaling_relations(star, gaia_data) -> dict:
    """numax from every combination of the available stellar parameters.

    Args:
        star: StarInfo (user values and errors).
        gaia_data: GaiaData.

    Returns:
        {"numax_SR_<relation>_<i>": numax (ufloat)}.
    """
    g = gaia_data
    teffs = _values([g.teff_gspspec, g.teff_gspphot], star.teff, star.teff_err)
    loggs = _values([g.logg_gspspec, g.logg_gspphot], star.logg, star.logg_err)
    radii = _values([g.rad_flame, g.rad_gspphot], star.radius, star.radius_err)
    masses = _values([g.mass_flame], star.mass, star.mass_err)
    lums = _values([g.lum_flame], star.luminosity, star.luminosity_err)

    estimates = {}
    for i, (logg, teff) in enumerate(product(loggs, teffs)):
        estimates[f"numax_SR_logg_teff_{i}"] = numax_from_logg_teff(logg, teff)
    for i, (radius, mass, teff) in enumerate(product(radii, masses, teffs)):
        estimates[f"numax_SR_mass_radius_teff_{i}"] = numax_from_mass_radius_teff(mass, radius, teff)
    for i, (lum, mass, teff) in enumerate(product(lums, masses, teffs)):
        estimates[f"numax_SR_mass_luminosity_teff_{i}"] = numax_from_mass_luminosity_teff(mass, lum, teff)
    return estimates
