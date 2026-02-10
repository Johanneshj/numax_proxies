import numpy as np
import lightkurve as lk
from collections import Counter
import json
import time as t
from .add_noise import calculate_noise

def get_lightcurve(*args, **kwargs):
    """
    Fetch a light curve either from:
    1) A target ID (e.g., 'KIC12345678' or 'TIC12345678'), with optional cadence.
    2) Time, flux, and optional flux_err arrays.
    3) No arguments — fallback default target.

    Returns:
        lightkurve.LightCurve
    """

    # -------------------------------
    # Case 1: No arguments -> fallback
    if len(args) == 0:
        search_results = lk.search_lightcurve('KIC12008916',
                                              mission='Kepler',
                                              cadence='long',
                                              author='Kepler',
                                              quarter=np.arange(0, 60))
        id = 'KIC12008916'
        return search_results.download_all().stitch(), id
    
    # -------------------------------    
    id, cadence, author, quarter, sector, mission = read_json_file(*args, **kwargs)
    if author: kwargs['author'] = author 
    if quarter: kwargs['quarter'] = quarter
    if sector: kwargs['sector'] = sector
    
    # Case 2: JSON file + lightcurve file
    lc_file = kwargs.get('lc_file', None)
    if lc_file is not None:
        # Assumes csv data file with columns time (days), flux (ppm) and flux_err (ppm)
        data        = np.genfromtxt(lc_file, delimiter=',', names=['time', 'flux', 'flux_err'])
        mask        = ~np.isnan(data['time']) & ~np.isnan(data['flux']) & ~np.isnan(data['flux_err'])
        data        = data[mask]
        time        = np.array(data['time'])
        flux        = np.array(data['flux'])
        flux_err    = np.array(data['flux_err'])
        lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)
        return lc, id, cadence
    
    # Case 3: JSON file only
    elif lc_file is None:        
        kwargs = dict(target=id, mission=mission, cadence=cadence)

        if not author:
            kwargs['author'] = find_author(search_results)
        else:
            kwargs['author'] = author

        if sector is not None:
            kwargs['sector'] = sector
        
        if quarter is not None:
            kwargs['quarter'] = quarter

        search_results = lk.search_lightcurve(**kwargs)
        if len(search_results) == 0:
            raise ValueError(f"No light curves found for {id} ({mission}).")

        lc_collection = search_results.download_all()
        lc_list = []
        for lc in lc_collection:
            flux = lc.flux.value
            flux /= np.median(flux)
            flux = (flux - 1)*1e6
            lc = lk.LightCurve(time=lc.time.value, 
                                flux=flux, 
                                flux_err=lc.flux_err.value)
            lc_list.append(lc)
        lc = lc.append(lc_list)
        return lc, id, cadence
    
    raise ValueError("Invalid arguments. Provide either an ID string or time+flux arrays.")

def prepare_lightcurve(lc=None, id=None, *args, **kwargs):
    '''
        Prepare lightcurve: 1. sort data, 
                            2. close gaps, 
                            3. smooth with savgol filter (if stated)
                            4. plot lightcurve (if stated)

        Inputs:
            lc      : lightkurve LightCurve object
            id      : target id
            args    : only relevant if lc is not provided
            kwargs  : can contain savgol=True, plot=True, savgol_iters=n
        
        Outputs:
            lc      : processed LightCurve object
            fig     : plot of lightcurve
    '''
    if lc is None:
        lc = get_lightcurve(*args, **kwargs)

    if hasattr(lc, "stitch"):
        lc = lc.stitch().remove_nans().remove_outliers(5)
    else:
        lc = lc.remove_nans().remove_outliers(5)
    # print(lc.time.value)
    time, original_flux, flux_err = sort_data_by_time(lc)
    # print(time)
    time = close_gaps(time)
    # print(time)
    original_flux += calculate_noise(original_flux, kwargs.get('noise_std', None)) if kwargs.get('add_noise', None) else 0 # Add noise for testing

    lc = lk.LightCurve(time=time, flux=original_flux, flux_err=flux_err)

    savgol = kwargs.get("savgol", False)
    plot = kwargs.get("plot", False)
    savgol_iters = kwargs.get("savgol_iters", 0)
    dt = np.mean(np.diff(time)) * 24 * 60 * 60 # seconds
    if savgol:
        lc = lc
        filters = np.zeros((savgol_iters, len(lc.flux.value)))
        smoothed_fluxes = np.zeros((savgol_iters, len(lc.flux.value)))

        for i in range(savgol_iters):
            filter = get_savgol_filter(lc)
            filters[i,:] = filter
            
            lc = lk.LightCurve(time=lc.time.value, 
                               flux=lc.flux.value-filter, 
                               flux_err=lc.flux_err.value)
            
            smoothed_fluxes[i,:] = lc.flux.value

        if plot:
            plot_lc(time=time, original_flux=original_flux, 
                    filters=filters, smoothed_fluxes=smoothed_fluxes, 
                    id=id)
        return lc, dt 
            
    else:
        if plot:
            plot_lc(time=lc.time.value, flux=lc.flux.value, 
                    id=id)
        return lc, dt

def calculate_psd(lc=None, *args, **kwargs):
    '''
        Calculate PSD using Lightkurve's .to_periodogram()

        Inputs:
            lc      : lightkurve LightCurve object
            args    : only applies if lc is not provided
            kwargs  : only applies if lc is not provided
        
        Outputs:
            pg      : lightkurve Periodogram object
    '''
    if lc is None:
        lc = get_lightcurve(*args, **kwargs)

    pg = lc.to_periodogram(freq_unit='uHz', normalization='psd')
    return pg

def find_author(search_results):
    '''
        Find most suitable author if no author is given.
        Author priority given by priority list.

        Inputs:
            search_results  : output from Lightkurve search_lightcurve function
        
        Outputs:
            author          : best author
    '''
    priority = ['KASOC', 'Kepler', 'TASOC', 'SPOC', 'TGLC', 'QLP']
    author_counts = Counter(search_results.author)
    most_common = author_counts.most_common()
    author = sorted(
        most_common,
        key=lambda x: (priority.index(x[0]) if x[0] in priority else len(priority), -x[1])
    )[0][0]
    return author
    
def sort_data_by_time(lc):
    '''
        Sort lightcurve by time values.

        Input:
            lc          : Lightkurve LightCurve object

        Output:
            time        : sorted time array
            flux        : sorted flux array
            flux_err    : sorted flux_err array
    '''
    time = lc.time.value
    flux = lc.flux.value
    flux_err = lc.flux_err.value

    _, unique_idx = np.unique(time, return_index=True)
    time = time[unique_idx]
    flux = flux[unique_idx]
    flux_err = flux_err[unique_idx]

    sort_idx = np.argsort(time)
    time = time[sort_idx]
    flux = flux[sort_idx]
    flux_err = flux_err[sort_idx]

    time -= time[0]
    flux_err /= np.median(flux_err)*1e6

    return time, flux, flux_err

def close_gaps(time):
    '''
        Function that closes gaps, but forces minimum gap size of three days.

        Input:
            time    : time array of light curve.
        
        Output:
            time    : new time array with closed gaps.
    '''
    gap_indices = np.concatenate((np.where(np.diff(time) > 3)[0], [len(time)-1]))
    gap = 0
    for i in range(1, len(gap_indices)):
        start = gap_indices[i-1]+1
        end = gap_indices[i]
        gap = (time[start]-time[start-1]) - 3
        time[start:end+1] -= gap
    return time

def get_savgol_filter(lc):
    '''
        Function that applies savgol filter.
        Polyorder always 2.
        Window length equal to half amount of data points contained in period of maximum power

        Inputs:
            LC  : input Lightkurve LightCurve object
        
        Outputs:
            filter  : savgol filter
    '''
    from scipy.signal import savgol_filter
    start = t.time() 
    pg = lc.to_periodogram()
    pmax = pg.period_at_max_power.value
    end = t.time()
    print(f"computation time for pmax: {end - start:.4f} seconds")
    wl = int(pmax/np.mean(np.diff(lc.time.value)) / 2)
    wl = wl if wl % 2 != 0 else wl + 1
    
    chunk_len = max(lc.time.value) / 20
    def chunk_up_array(array, time, chunk_len):
        chunk_size = int(chunk_len / np.mean(np.diff(time)))
        n_chunk = len(lc.time.value) // chunk_size
        return array[:chunk_size*n_chunk].reshape((n_chunk, chunk_size))
    
    start = t.time() 
    # filter = np.array([savgol_filter(flux, window_length=wl, polyorder=2) for flux in chunk_up_array(lc.flux.value, lc.time.value, chunk_len)]).flatten()
    filter = savgol_filter(lc.flux.value, window_length=wl, polyorder=2)
    end = t.time()
    print(f"computation time for savgol: {end - start:.4f} seconds")

    # print(filter, len(filter), len(lc.time.value))
    return filter

def plot_lc(time=None, original_flux=None, filters=None, smoothed_fluxes=None, id=None):
    '''
        Plots processed lightcurve including showcasing smoothings with savgol filter

        Inputs:
            time            : time array
            original_flux   : original supplied flux array
            filters         : filters from savgol iterations
            smoothed_fluxes : fluxes with savgol filter subtracted
            id              : target id
        
        Outputs:
            Plots LCs and saves 1 figure in target folder
    '''
    import matplotlib.pyplot as plt
    import os

    fig, axs = plt.subplots(2,1,figsize=(10,6))

    axs[0].scatter(time, original_flux, c='k', s=2, zorder=-1, label='original flux')
    axs[0].set_ylabel('flux [ppm]')

    mid = max(time)/2
    mask = np.where((time > mid) & (time <= mid+10))[0]
    axs[1].scatter(time[mask], original_flux[mask], c='k', s=2, zorder=-1) 
    axs[1].set_xlabel('time [days]')
    axs[1].set_ylabel('flux [ppm]')

    if filters is not None:
        n_filters = filters.shape[0]
        for i in range(n_filters):
            filt = filters[i,:]
            smoothed_flux = smoothed_fluxes[i,:]
            axs[0].plot(time, filt, zorder=i, label=f'filter iteration {int(i)}')
            axs[0].scatter(time, smoothed_flux, s=2, zorder=i+1, label=f'smoothing iteration {int(i)}') 
            axs[1].plot(time[mask], filt[mask], zorder=i)
            axs[1].scatter(time[mask], smoothed_flux[mask], s=2, zorder=i+1)
    axs[0].legend(loc='upper left')       

    savepath = os.path.join('numax_proxies', 'results', id, 'figures')
    os.makedirs(savepath, exist_ok=True)

    fig.savefig(f'{savepath}/lightcurve.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

# def read_json_file(*args, **kwargs):
#     '''
#         Reads JSON file if provided.
#     '''
#     json_args = [a for a in args if isinstance(a, str) and a.endswith('.json')]
#     if len(json_args) > 0:
#         with open(json_args[0], 'r') as f:
#             cfg = json.load(f)
#         id = cfg.get('target')
#         cadence = cfg.get('cadence', 'long')
#         author = cfg.get('author', None)
#         quarter = cfg.get('quarter', None)
#         sector = cfg.get('sector', None)
#         mission = 'Kepler' if 'KIC' in id.upper() else 'TESS'
        
#     return id, cadence, author, quarter, sector, mission

def read_json_file(*args):
    '''
        Reads JSON file if provided.
    '''
    json_args = [a for a in args if isinstance(a, str) and a.endswith('.json')]

    data = dict(
        target=None,
        cadence="long",
        author=None,
        quarter=None,
        sector=None,
        mission=None,
        logg=None,
        teff=None,
        mag=None
    )

    if json_args:
        with open(json_args[0]) as f:
            cfg = json.load(f)

        data["target"] = cfg.get("target", "KIC12008916")
        data["cadence"] = cfg.get("cadence", "long")
        data["author"] = cfg.get("author", "Kepler")
        data["quarter"] = cfg.get("quarter", np.arange(0, 100))
        data["sector"] = cfg.get("sector", None)
        data["mission"] = (
            "Kepler" if data["target"] and "KIC" in data["target"].upper() else "TESS"
        )
        data["logg"] = cfg.get("logg", None)
        data["teff"] = cfg.get("teff", None)
        data["mag"] = cfg.get("mag", None)
    
    return data
        

def read_logg_and_teff_and_mag(*args, **kwargs):
    '''
        Reads JSON file if provided.
    '''
    json_args = [a for a in args if isinstance(a, str) and a.endswith('.json')]
    if len(json_args) > 0:
        with open(json_args[0], 'r') as f:
            cfg = json.load(f)
        logg = cfg.get('logg', None)
        teff = cfg.get('teff', None)
        mag = cfg.get('mag', None)
    return logg, teff, mag
