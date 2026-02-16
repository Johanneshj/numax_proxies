

def plot_CoV_vs_bin_centers(bin_centers, CoVs, smoothed_CoVs, numax, ax, id):
    numax_err = numax.std_dev
    numax_val = numax.nominal_value
    ax.plot(bin_centers, CoVs, c="k", marker="d", label="CoV values")
    ax.plot(bin_centers, smoothed_CoVs, c="r", marker="+", label="filtered CoV values")
    ax.axvline(
        numax_val,
        c="gray",
        ls="--",
        label=f"νmax = {numax_val:.2f} ± {numax_err:.2f} μHz",
    )
    ax.set_xscale("log")
    ax.set_xlabel("frequency [μHz]")
    ax.set_ylabel("coefficient of variation")
    ax.legend(loc="upper left")


def plot_supNyq_spec(freq, power, numax, ax, id):
    numax_err = numax.std_dev
    numax_val = numax.nominal_value
    ax.plot(freq, power, c="k", label="super Nyquist spectrum")
    ax.axvline(
        numax_val,
        c="gray",
        ls="--",
        label=f"νmax = {numax_val:.2f} ± {numax_err:.2f} μHz",
    )
    ax.legend(loc="upper left")
    ax.set_xlabel("frequency [μHz]")
    ax.set_ylabel(r"PSD [$\text{ppm}^2$/\text{μHz}]")
    ax.legend(loc="upper left")
