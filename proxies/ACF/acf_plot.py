import numpy as np
from matplotlib.colors import LogNorm


def plot_spec(frequency, power, smoothed_power, ax, id):
    ax.loglog(frequency, power, c="gray", label="mean spectrum")
    ax.loglog(frequency, smoothed_power, c="k", label="median filter")
    ax.set_xlabel("frequency [μHz]")
    ax.set_ylabel("power spectral density")
    ax.set_xlim(min(frequency), max(frequency))
    ax.text(0.02, 0.02, f"{id}", ha="left", va="bottom", transform=ax.transAxes)
    ax.legend()


def plot_2D_ACF(ACF, frequency, ax):
    if np.max(frequency) > 300:
        window_size_muHz = 249
        step = 10
    else:
        step = 2
        window_size_muHz = 49
    # start = t.time()
    ax.imshow(
        ACF[::step, ::step].T,
        cmap="Greys_r",
        aspect="auto",
        norm=LogNorm(vmin=0.001, vmax=1),
        origin="lower",
        extent=(frequency[0], frequency[-1], 0, window_size_muHz),
    )
    ax.set_xlabel("frequency [μHz]")
    ax.set_ylabel("spacing [μHz]")
    # end = t.time()
    # print(f"computation time for plotting ACF: {end - start:.4f} seconds")
    # plt.colorbar(label='|ACF| (log scale)', cax=ax)


def plot_collapsed_acf_with_gaussian_fit(collapsed_2D_acf, freq_centers, fit_vals, ax):
    def gaussian(x, A, sigma, mu):
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    x = np.linspace(min(freq_centers), max(freq_centers), 1000)
    ax.plot(freq_centers, collapsed_2D_acf, c="k", marker=".", label="collapsed 2D ACF")
    ax.plot(x, gaussian(x, *fit_vals), c="r", label="Gaussian fit")
    ax.axvline(
        fit_vals[2], c="gray", ls="--", label=f"numax = {np.round(fit_vals[2],2)}"
    )
    ax.set_xlabel("frequency [μHz]")
    ax.set_ylabel("A.U.")
    ax.legend()
