import matplotlib.pyplot as plt

def plot_CoV_vs_bin_centers(bin_centers, CoVs, smoothed_CoVs, numax, numax_err, ax, id):
    ax.plot(bin_centers, CoVs, c='k', marker='d', label='CoV values')
    ax.plot(bin_centers, smoothed_CoVs, c='r', marker='+', label='filtered CoV values')
    ax.axvline(numax, c='gray', ls='--', label=f'νmax = {numax:.2f} ± {numax_err:.2f} μHz')
    ax.set_xscale('log')
    ax.set_xlabel('frequency [μHz]')
    ax.set_ylabel('coefficient of variation')
    ax.legend(loc='upper left')
