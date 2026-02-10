import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat, UFloat
from uncertainties import unumpy as unp
import os

def plot_spectrum_with_all_numax_estimates(frequency, power, numax_dict, id):
    fig, ax = plt.subplots()
    ax.loglog(frequency, power, 
            c='gray')
    for label, numax in numax_dict.items():
        try:
            numax_val = numax.n
            numax_err = numax.s
            line_label = rf'{label}: {numax_val:.1f} ± {numax_err:.1f}'
        except:
            numax_val = numax
            numax_err = None
            line_label = f'{label}: {numax_val:.1f}'

        if 'CoV' in label:
            ls = (0, (3, 1, 1, 1))
            c = 'mediumorchid'
        elif '2DACF' in label:
            ls = (0, (5, 1))
            c = 'forestgreen'
        elif 'SR' in label:
            ls = (0, (1, 5))
            c = 'dodgerblue'
        elif 'FliPer' in label:
            ls = (5, (10, 3))
            c = 'darkorange'

        ax.axvline(numax_val, linestyle=ls, c=c, label=line_label)
        if numax_err is not None:
            ax.axvspan(numax_val - numax_err,
                       numax_val + numax_err,
                       alpha=0.2,
                       color=c)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power')
    ax.set_xlim(np.min(frequency.value), np.max(frequency.value))
    ax.text(0.02, 0.02, f'{id}', ha='left', va='bottom', transform=ax.transAxes)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    savepath = os.path.join('numax_proxies', 'results', id, 'figures')
    os.makedirs(savepath, exist_ok=True)
    fig.savefig(f'{savepath}/full_spectrum_with_all_estimates.png', dpi=300, bbox_inches='tight')
