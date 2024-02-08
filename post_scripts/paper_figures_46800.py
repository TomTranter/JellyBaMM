# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 08:14:54 2020

@author: Tom
"""

import jellybamm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from sklearn.preprocessing import StandardScaler
import scipy
import pandas as pd

# Turn off code warnings (this is not recommended for routine use)
import warnings
from string import ascii_lowercase as abc
import seaborn as sns


plt.style.use("seaborn-dark")
cmap = sns.color_palette("rocket")
warnings.filterwarnings("ignore")

root = "D:\\pybamm_pnm_results\\46800"
save_im_path = "D:\\pybamm_pnm_results\\46800\\figures"
plt.close("all")

savefigs = True

tab_1 = [
    0,
    1,
    2,
    3,
]
tab_tesla = [
    4,
    5,
    6,
    7,
]

amps = jellybamm.get_amp_cases()
d = jellybamm.load_all_data()
cases = jellybamm.get_cases()
soc_list = [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]]
mini_soc_list = [[0.8, 0.6], [0.4, 0.2]]
grp = "neg"
data = d[0][17.5][0]["data"]


def find_best_fit(y, report_results=False):
    # Set up list of candidate distributions to use
    # See https://docs.scipy.org/doc/scipy/reference/stats.html for more
    # y = data_spm.copy()
    size = len(y)
    dist_names = ["norm", "gumbel_l", "gumbel_r"]
    # Set up empty lists to stroe results
    chi_square = []
    p_values = []
    params = []

    sc = StandardScaler()
    yy = y.reshape(-1, 1)
    sc.fit(yy)
    y_std = sc.transform(yy)
    y_std = y_std.flatten()

    # Set up 50 bins for chi-square test
    # Observed data will be approximately evenly distrubuted aross all bins
    percentile_bins = np.linspace(0, 100, 51)
    percentile_cutoffs = np.percentile(y_std, percentile_bins)
    observed_frequency, bins = np.histogram(y_std, bins=percentile_cutoffs)
    cum_observed_frequency = np.cumsum(observed_frequency)

    # Loop through candidate distributions

    for distribution in dist_names:
        # Set up distribution and get fitted distribution parameters
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y_std)
        params.append(param)
        # Obtain the KS test P statistic, round it to 5 decimal places
        p = scipy.stats.kstest(y_std, distribution, args=param)[1]
        p = np.around(p, 5)
        p_values.append(p)

        # Get expected counts in percentile bins
        # This is based on a 'cumulative distrubution function' (cdf)
        cdf_fitted = dist.cdf(
            percentile_cutoffs, *param[:-2], loc=param[-2], scale=param[-1]
        )
        expected_frequency = []
        for bin in range(len(percentile_bins) - 1):
            expected_cdf_area = cdf_fitted[bin + 1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)

        # calculate chi-squared
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = sum(
            ((cum_expected_frequency - cum_observed_frequency) ** 2)
            / cum_observed_frequency
        )
        chi_square.append(ss)

    # Collate results and sort by goodness of fit (best at top)

    results = pd.DataFrame()
    results["Distribution"] = dist_names
    results["chi_square"] = chi_square
    results["p_value"] = p_values
    results.sort_values(["chi_square"], inplace=True)

    # Report results
    if report_results:
        print("\nDistributions sorted by goodness of fit:")
        print("----------------------------------------")
        print(results)
    best_dist_name = results.values[0][0]
    best_chi_square = results.values[0][1]
    dist = getattr(scipy.stats, best_dist_name)
    args = dist.fit(y)
    return (
        best_dist_name,
        best_chi_square,
        dist,
        args,
        dist.mean(*args),
        dist.std(*args),
    )


input_dir = jellybamm.INPUT_DIR


def jellyroll_one_plot(data, title, dp=3):
    fig, ax = plt.subplots(figsize=(12, 12))
    spm_map = np.load(os.path.join(input_dir, "im_spm_map.npz"))["arr_0"]
    spm_map_copy = spm_map.copy()
    spm_map_copy[np.isnan(spm_map_copy)] = -1
    spm_map_copy = spm_map_copy.astype(int)
    mask = np.isnan(spm_map)
    arr = np.ones_like(spm_map).astype(float)
    arr[~mask] = data[spm_map_copy][~mask]
    arr[mask] = np.nan
    im = ax.imshow(arr, cmap=cm.inferno)
    ax.set_axis_off()
    plt.colorbar(im, ax=ax, format="%." + str(dp) + "f")
    ax.set_title(title)
    return fig


# Heat Data by Layer comparison
net = jellybamm.get_net()
abs_xcoords = np.abs(net["pore.coords"][:, 0])
r_max = abs_xcoords.max()
r_min = abs_xcoords.min()
spm_vol = net["throat.volume"][net["throat.spm_resistor"]]

V = np.pi * (r_max**2 - r_min**2) * 0.08
fig, axes = plt.subplots(2, 1, sharex=False, sharey=True, figsize=(6, 8))
variables = [17, 18, 16, 19]
subi = 0
for ax, case in enumerate([0, 4]):
    Q_tot = d[case][17.5][7]["data"]
    Q_ohm = d[case][17.5][16]["data"]
    Q_irr = d[case][17.5][17]["data"]
    Q_rev = d[case][17.5][18]["data"]
    Q_ohm_cc = d[case][17.5][19]["data"]
    nt, nspm = Q_ohm.shape
    spm_vol_t = np.tile(spm_vol[:, np.newaxis], nt).T
    tot_heat = np.zeros(nt)
    sum_Q_tot = np.sum(Q_tot * spm_vol_t, axis=1)
    sum_Q_ohm = np.sum(Q_ohm * spm_vol_t, axis=1)
    sum_Q_irr = np.sum(Q_irr * spm_vol_t, axis=1)
    sum_Q_rev = np.sum(Q_rev * spm_vol_t, axis=1)
    sum_Q_ohm_cc = np.sum(Q_ohm_cc * spm_vol_t, axis=1)

    base = np.zeros(len(sum_Q_ohm))
    tot_heat = sum_Q_ohm + sum_Q_irr + sum_Q_rev + sum_Q_ohm_cc
    cols = cmap(np.linspace(0.1, 0.9, 4))
    labels = []
    for i in [18, 17, 16, 19]:
        text = jellybamm.format_label(i).strip("X-averaged").strip("[W.m-3]")
        labels.append(text.lstrip().rstrip().capitalize())
    for si, source in enumerate([sum_Q_rev, sum_Q_irr, sum_Q_ohm, sum_Q_ohm_cc]):
        axes[ax].fill_between(
            d[case][17.5][10]["mean"],
            base,
            base + source,
            color=cols[si],
            label=labels[si],
        )
        base += source

    axes[ax].set_xlabel("Time [h]")
    axes[ax].set_ylabel("Total Heat Produced [W]")
    axes[ax].grid()
    t = axes[ax].text(
        -0.1, 1.15, abc[subi], transform=axes[ax].transAxes, fontsize=14, va="top"
    )
    t.set_bbox(dict(facecolor="white", alpha=1.0, edgecolor="black"))
    plt.legend()
    subi += 1
plt.tight_layout()


fig, axes = plt.subplots(2, 1, sharex=False, sharey=True, figsize=(6, 8))
for subi, case in enumerate([0, 4]):
    jellybamm.stacked_variables(net, d, case, 17.5, [18, 17, 16, 19], axes[subi], subi)
plt.tight_layout()
plt.savefig(os.path.join(save_im_path, "fig7.png"), dpi=600)
spm_res = net["throat.spm_resistor"]
spm_res_arc_index = net["pore.arc_index"][net["throat.conns"][spm_res]][:, 0]
spm_res_layer = np.ones_like(spm_res_arc_index) * -1
l = 0
nspm = len(spm_res_layer)
for ires in range(nspm):
    spm_res_layer[ires] = l
    if ires == nspm - 1:
        pass
    elif spm_res_arc_index[ires + 1] < spm_res_arc_index[ires]:
        l += 1
    if l > 39:
        l = 0
dp = 2
fig, axes = plt.subplots(2, 1)
arrs = []
vmin = 1e9
vmax = -1e9
for ax, case in enumerate([1, 5]):
    heat_data = d[case][17.5][19]["data"]
    heat_data = heat_data[:1000, :]
    arrs.append(heat_data)
    if heat_data.min() < vmin:
        vmin = heat_data.min()
    if heat_data.max() > vmax:
        vmax = heat_data.max()
heat_layer_maps = []
for ax, case in enumerate([1, 5]):
    heat_data = arrs[ax]
    nt, nspm = heat_data.shape
    nbin = int(np.floor(nt / 10))
    heat_layer_map = np.zeros([spm_res_layer.max(), nbin])

    for t in range(nbin):
        mean_heat = np.mean(heat_data[t * 10 : (t + 1) * 10, :], axis=0)
        for nl in range(spm_res_layer.max()):
            heat_layer_map[nl, t] = np.mean(mean_heat[spm_res_layer == nl])

    im = axes[ax].imshow(heat_layer_map, vmin=vmin, vmax=vmax, cmap=cm.seismic)
    plt.colorbar(im, ax=axes[ax], format="%." + str(dp) + "f")
    heat_layer_maps.append(heat_layer_map)

plt.figure()
plt.imshow(heat_layer_maps[0] - heat_layer_maps[1], cmap=cm.seismic)


jellybamm.super_subplot(net, d, tab_1, tab_tesla, 17.5)
if savefigs:
    plt.savefig(os.path.join(save_im_path, "fig2.png"), dpi=600)
# 2nd Case 5.25 Amps - HTC 100 - 2 Tab
fig5 = jellybamm.jellyroll_subplot(
    d, 0, amps[-1], var=0, soc_list=soc_list, global_range=True, dp=1
)
if savefigs:
    plt.savefig(os.path.join(save_im_path, "fig5.png"), dpi=600)
fig6 = jellybamm.jellyroll_subplot(
    d, 4, amps[-1], var=0, soc_list=soc_list, global_range=True, dp=1
)
if savefigs:
    plt.savefig(os.path.join(save_im_path, "fig6.png"), dpi=600)
