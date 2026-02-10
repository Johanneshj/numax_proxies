from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
import numpy as np
from uncertainties import ufloat
from pyvo.dal.exceptions import DALFormatError
from requests.exceptions import ConnectionError
import time as t

def query_gaia(id=None, ra=None, dec=None):

    gaia_id = query_simbad(object_name=id)#Simbad.query_objectids(object_name=id)
    if gaia_id is not None:
        QUERY = get_query(gaia_id)
        job = Gaia.launch_job(QUERY)
        results = return_dict(job)
        return results
    else:
        return {}

def get_query(id):

    data_release = id.split(' ')[1]
    gaia_id = id.split(' ')[2]

    dr2_string = f"""
                JOIN(
                    SELECT dr2todr3.*
                    FROM gaiadr3.dr2_neighbourhood as dr2todr3
                        WHERE dr2todr3.dr2_source_id = {gaia_id}
                    ) AS xmatch
                    ON xmatch.dr3_source_id = dr3.source_id
                """
    
    QUERY = f"""
            SELECT
                dr3.source_id,
                dr3.radius_gspphot,
                dr3.radius_gspphot_lower,
                dr3.radius_gspphot_upper,
                dr3.teff_gspspec,
                dr3.teff_gspspec_lower,
                dr3.teff_gspspec_upper,
                dr3.teff_gspphot,
                dr3.teff_gspphot_lower,
                dr3.teff_gspphot_upper,
                dr3.logg_gspspec,
                dr3.logg_gspspec_lower,
                dr3.logg_gspspec_upper,
                dr3.logg_gspphot,
                dr3.logg_gspphot_lower,
                dr3.logg_gspphot_upper,
                dr3.lum_flame,
                dr3.lum_flame_lower,
                dr3.lum_flame_upper,
                dr3.mass_flame,
                dr3.mass_flame_lower,
                dr3.mass_flame_upper,
                dr3.radius_flame,
                dr3.radius_flame_lower,
                dr3.radius_flame_upper
            FROM gaiadr3.astrophysical_parameters AS dr3
            {dr2_string if data_release == 'DR2' else f'WHERE dr3.source_id = {gaia_id}'}
            """

    return QUERY

def return_dict(job=None):
    from uncertainties import ufloat

    res = job.get_results()
    dictionary = {}

    if len(res) < 1:
        # No results, return empty dictionary
        return dictionary
    if len(res) > 1:
        # Multiple results, just take the first one
        # Can happen if fx a Gaia DR2 id has multiple matches in DR3
        return res[0]

    for col in res.colnames:
        if col.endswith('_lower') or col.endswith('_upper') or col.endswith('source_id') or col.endswith('_uncertainty'):
            continue
        lower_col = f"{col}_lower"
        upper_col = f"{col}_upper"
        uncertainty_col = f"{col}_uncertainty"

        val = res[col] #if col is res.colnames else np.nan
        val_lower = res[lower_col] if lower_col in res.colnames else np.nan
        val_upper = res[upper_col] if upper_col in res.colnames else np.nan
        val_uncertainty = res[uncertainty_col] if uncertainty_col in res.colnames else np.nan

        if np.isnan(val_uncertainty) and np.isfinite(val_lower) and np.isnan(val_upper):
            err = val - val_lower
        elif np.isnan(val_uncertainty) and np.isfinite(val_upper) and np.isnan(val_lower):
            err = val_upper - val
        elif np.isfinite(val_uncertainty):
            err = val_uncertainty
        elif np.isnan(val_uncertainty) and np.isfinite(val_lower) and np.isfinite(val_upper):
            err = (val_upper - val_lower) / 2
        else:
            err = np.nan

        if np.isfinite(val) and np.isfinite(err):
            dictionary.setdefault(col, ufloat(val, abs(err)))
        else:
            continue

    return dictionary

def query_simbad(object_name, retries=3, delay=1.5):
    '''Query SIMBAD with retries to account for time-outs'''
    for attempt in range(retries):
        try:
            res = Simbad.query_objectids(object_name=object_name)
            if res is None or len(res) == 0:
                return None # No results
            Gaia_ids = [
                res[i][0]
                for i in range(len(res))
                if 'gaia' in res[i][0].lower()
            ]
            if len(Gaia_ids) == 0: 
                return None # No Gaia IDs
            gaia_id = sorted(Gaia_ids)[-1]
            return gaia_id
        except (DALFormatError, ConnectionError, OSError) as e:
            if attempt < retries - 1:
                t.sleep(delay)
            else:
                return None # SIMBAD query time-out even after retries

# def get_gaia_id(object_name):
#     '''Query object and get Gaia id'''
#     kic_number = object_name.replace('KIC', '').strip()
#     query = f"""
#             SELECT source_id
#             FROM gaiadr3.kepler_neighbourhood
#             WHERE kic_id = {kic_number}
#             """
#     job = Gaia.launch_job(query)
#     results = job.get_results()
    
#     if len(results) == 0:
#         return None
    
#     return results["source_id"][0]

# def ensure_val(param):
#     '''Basically a fail safe'''
#     if param is not float:
#         return param
#     else:
#         return np.nan

## Different queries
# QUERY = f"""
#         SELECT
#             dr3.source_id,
#                 dr3.radius_gspphot,
#                 dr3.radius_gspphot_lower,
#                 dr3.radius_gspphot_upper,
#                 dr3.teff_gspspec,
#                 dr3.teff_gspspec_lower,
#                 dr3.teff_gspspec_upper,
#                 dr3.teff_gspphot,
#                 dr3.teff_gspphot_lower,
#                 dr3.teff_gspphot_upper,
#                 dr3.logg_gspspec,
#                 dr3.logg_gspspec_lower,
#                 dr3.logg_gspspec_upper,
#                 dr3.logg_gspphot,
#                 dr3.logg_gspphot_lower,
#                 dr3.logg_gspphot_upper,
#                 dr3.lum_flame,
#                 dr3.lum_flame_lower,
#                 dr3.lum_flame_upper,
#                 dr3.mass_flame,
#                 dr3.mass_flame_lower,
#                 dr3.mass_flame_upper,
#                 dr3.radius_flame,
#                 dr3.radius_flame_lower,
#                 dr3.radius_flame_upper,
#                 dr3.teff_esphs,
#                 dr3.teff_esphs_uncertainty,
#                 dr3.logg_esphs,
#                 dr3.logg_esphs_uncertainty,
#                 dr3.teff_espucd,
#                 dr3.teff_espucd_uncertainty,
#                 dr3.logg_msc1,
#                 dr3.logg_msc1_lower,
#                 dr3.logg_msc1_upper,
#                 dr3.logg_msc2,
#                 dr3.logg_msc2_lower,
#                 dr3.logg_msc2_upper
#         FROM gaiadr3.astrophysical_parameters AS dr3
#         {dr2_string if data_release == 'DR2' else f'WHERE dr3.source_id = {gaia_id}'}
#         """

# QUERY = f"""
#         SELECT
#             dr3.source_id,
#                 dr3.radius_gspphot,
#                 dr3.radius_gspphot_lower,
#                 dr3.radius_gspphot_upper,
#                 dr3.teff_gspspec,
#                 dr3.teff_gspspec_lower,
#                 dr3.teff_gspspec_upper,
#                 dr3.teff_gspphot,
#                 dr3.teff_gspphot_lower,
#                 dr3.teff_gspphot_upper,
#                 dr3.logg_gspspec,
#                 dr3.logg_gspspec_lower,
#                 dr3.logg_gspspec_upper,
#                 dr3.logg_gspphot,
#                 dr3.logg_gspphot_lower,
#                 dr3.logg_gspphot_upper,
#                 dr3.lum_flame,
#                 dr3.lum_flame_lower,
#                 dr3.lum_flame_upper,
#                 dr3.mass_flame,
#                 dr3.mass_flame_lower,
#                 dr3.mass_flame_upper,
#                 dr3.radius_flame,
#                 dr3.radius_flame_lower,
#                 dr3.radius_flame_upper
#         FROM gaiadr3.astrophysical_parameters AS dr3
#         {dr2_string if data_release == 'DR2' else f'WHERE dr3.source_id = {gaia_id}'}
#         """

# QUERY = f"""
#         SELECT
#             dr3.source_id,
#                 dr3.teff_gspspec,
#                 dr3.teff_gspspec_lower,
#                 dr3.teff_gspspec_upper,
#                 dr3.teff_gspphot,
#                 dr3.teff_gspphot_lower,
#                 dr3.teff_gspphot_upper,
#                 dr3.logg_gspspec,
#                 dr3.logg_gspspec_lower,
#                 dr3.logg_gspspec_upper,
#                 dr3.logg_gspphot,
#                 dr3.logg_gspphot_lower,
#                 dr3.logg_gspphot_upper
#         FROM gaiadr3.astrophysical_parameters AS dr3
#         {dr2_string if data_release == 'DR2' else f'WHERE dr3.source_id = {gaia_id}'}
#         """


# Old query
# dr2_string = f"""
#                 JOIN(
#                     SELECT dr2todr3.*
#                     FROM gaiadr3.dr2_neighbourhood as dr2todr3
#                         WHERE dr2todr3.dr2_source_id = {gaia_id}
#                     ) AS xmatch
#                     ON xmatch.dr3_source_id = dr3.source_id
#                 """
    
#     QUERY = f"""
#             SELECT
#                 dr3.source_id,
#                 dr3.radius_gspphot,
#                 dr3.radius_gspphot_lower,
#                 dr3.radius_gspphot_upper,
#                 dr3.teff_gspspec,
#                 dr3.teff_gspspec_lower,
#                 dr3.teff_gspspec_upper,
#                 dr3.teff_gspphot,
#                 dr3.teff_gspphot_lower,
#                 dr3.teff_gspphot_upper,
#                 dr3.logg_gspspec,
#                 dr3.logg_gspspec_lower,
#                 dr3.logg_gspspec_upper,
#                 dr3.logg_gspphot,
#                 dr3.logg_gspphot_lower,
#                 dr3.logg_gspphot_upper,
#                 dr3.lum_flame,
#                 dr3.lum_flame_lower,
#                 dr3.lum_flame_upper,
#                 dr3.mass_flame,
#                 dr3.mass_flame_lower,
#                 dr3.mass_flame_upper,
#                 dr3.radius_flame,
#                 dr3.radius_flame_lower,
#                 dr3.radius_flame_upper
#             FROM gaiadr3.astrophysical_parameters AS dr3
#             {dr2_string if data_release == 'DR2' else f'WHERE dr3.source_id = {gaia_id}'}
#             """