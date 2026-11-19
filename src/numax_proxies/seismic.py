"""Asteroseismic scaling relations used throughout the package (single source of truth).

All frequencies in muHz.
"""
import numpy as np

# Delta nu as a function of numax: Delta nu = 0.267 * numax**0.764 (Stello et al. 2009)
DNU_COEFF = 0.267
DNU_EXP = 0.764

# FWHM of the oscillation envelope: 0.66 * numax**0.88 (Mosser et al. 2012)
ENV_COEFF = 0.66
ENV_EXP = 0.88

# Upper limit of the CoV expected from solar-like oscillations: 2.69 * nu**0.154
COV_LIMIT_COEFF = 2.69
COV_LIMIT_EXP = 0.154

# Solar reference values for the numax scaling relations
NUMAX_SUN = 3090.0   # muHz
TEFF_SUN = 5777.0    # K
LOGG_SUN = 4.44      # dex


def delta_nu(nu, factor=1.0):
    """Large frequency separation expected at nu: factor x 0.267 nu^0.764.

    Args:
        nu: frequency [muHz].
        factor: multiplier.

    Returns:
        factor x Delta-nu(nu) [muHz].
    """
    # written as factor * coeff * nu**exp so results are bit-identical to the original code
    return factor * DNU_COEFF * np.asarray(nu) ** DNU_EXP


def envelope_fwhm(nu, factor=1.0):
    """FWHM of the oscillation envelope at nu: factor x 0.66 nu^0.88.

    Args:
        nu: frequency [muHz].
        factor: multiplier.

    Returns:
        factor x FWHM(nu) [muHz].
    """
    return factor * ENV_COEFF * np.asarray(nu) ** ENV_EXP


def cov_upper_limit(nu):
    """Upper limit of the CoV from solar-like oscillations: 2.69 nu^0.154.

    Args:
        nu: frequency [muHz].

    Returns:
        CoV limit.
    """
    return COV_LIMIT_COEFF * np.asarray(nu) ** COV_LIMIT_EXP


# --------------------------------------------------------------------------------------
# numax scaling relations (work with floats or uncertainties objects)
# --------------------------------------------------------------------------------------
def numax_from_logg_teff(logg, teff):
    """numax = numax_sun (g / g_sun) (Teff / Teff_sun)^-0.5.

    Args:
        logg: log g [dex] (float or ufloat).
        teff: Teff [K] (float or ufloat).

    Returns:
        numax [muHz].
    """
    return NUMAX_SUN * 10 ** (logg - LOGG_SUN) / (teff / TEFF_SUN) ** 0.5


def numax_from_mass_radius_teff(mass, radius, teff):
    """numax = numax_sun M R^-2 (Teff / Teff_sun)^-0.5.

    Args:
        mass: [M_sun]. radius: [R_sun]. teff: [K] (floats or ufloats).

    Returns:
        numax [muHz].
    """
    return NUMAX_SUN * radius ** (-2) * mass / (teff / TEFF_SUN) ** 0.5


def numax_from_mass_luminosity_teff(mass, luminosity, teff):
    """numax = numax_sun M L^-1 (Teff / Teff_sun)^3.5.

    Args:
        mass: [M_sun]. luminosity: [L_sun]. teff: [K] (floats or ufloats).

    Returns:
        numax [muHz].
    """
    return NUMAX_SUN * mass / luminosity * (teff / TEFF_SUN) ** 3.5
