import numpy as np
import matplotlib.pyplot as plt
import os

def plot_spectrum(id, pg, filter_20d, filter_80d, noise):
    fig, ax = plt.subplots()
    ax.loglog(
        pg.frequency.value,
        pg.power.value,
        c='gray',
        label='original'
    )
    ax.loglog(
        pg.frequency.value,
        filter_80d,
        c='k',
        label='80 day filtered'
    )
    ax.loglog(
        pg.frequency.value,
        filter_20d,
        c='r',
        label='20 day filtered'
    )
    ax.axhline(noise, c='b', ls='--', label='noise est.')
    ax.set_xlim(
        np.min(pg.frequency.value),
        np.max(pg.frequency.value)
    )
    ax.legend(loc='center left')
    ax.text(0.02, 0.02, f'{id}', ha='left', va='bottom', transform=ax.transAxes)
    ax.text(0.02, 0.98, f'FliPer', ha='left', va='top', transform=ax.transAxes)
    savepath = os.path.join('numax_proxies', 'results', id, 'figures')
    os.makedirs(savepath, exist_ok=True)
    fig.savefig(f'{savepath}/FliPer.png', dpi=300, bbox_inches='tight')
