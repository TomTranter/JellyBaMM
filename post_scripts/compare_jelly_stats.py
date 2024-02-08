# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 10:53:59 2020

@author: Tom
"""

import jellybamm
import os
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

root = "D:\\pybamm_pnm_results\\Chen2020_v3"
cases = jellybamm.get_cases()
amps = jellybamm.get_amp_cases()
data_suff = ["mean", "std", "chi"]
d = {}
for case in cases.keys():
    d[case] = {}
    case_path = os.path.join(root, cases[case]["file"])
    for amp in amps:
        d[case][amp] = {}
        amp_path = os.path.join(case_path, str(amp) + "A")
        file_prefix = "current_density_case_" + str(case) + "_amp_" + str(amp) + "_"
        for suff in data_suff:
            fp = os.path.join(amp_path, file_prefix + suff)
            d[case][amp][suff] = io.loadmat(fp)["data"].flatten()

input_dir = "C:\\Code\\pybamm_pnm_couple\\input"


def jellyroll_multiplot(
    data,
    cases=[0, 1, 2],
    amps=[1.75, 3.5, 5.25],
    var="std",
    title="Current Density Distribution Log(STD)",
    dp=3,
    do_log=True,
    global_scale=True,
):
    nrows = len(cases)
    ncols = len(amps)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12), sharex=True, sharey=True)
    spm_map = np.load(os.path.join(input_dir, "im_spm_map.npz"))["arr_0"]
    spm_map_copy = spm_map.copy()
    spm_map_copy[np.isnan(spm_map_copy)] = -1
    spm_map_copy = spm_map_copy.astype(int)
    mask = np.isnan(spm_map)
    all_data = []
    for ir, case in enumerate(cases):
        for ic, amp in enumerate(amps):
            case_data = data[case][amp][var]
            if do_log:
                case_data = np.log(case_data)
            all_data.append(case_data)
    all_data_arr = np.asarray(all_data)
    vmin = all_data_arr.min()
    vmax = all_data_arr.max()
    for ir, case in enumerate(cases):
        for ic, amp in enumerate(amps):
            ax = axes[ir][ic]
            case_data = all_data.pop(0)
            arr = np.ones_like(spm_map).astype(float)
            arr[~mask] = case_data[spm_map_copy][~mask]
            arr[mask] = np.nan
            if global_scale:
                im = ax.imshow(arr, cmap=cm.inferno, vmin=vmin, vmax=vmax)
            else:
                im = ax.imshow(arr, cmap=cm.inferno)
            ax.set_axis_off()
            plt.colorbar(im, ax=ax, format="%." + str(dp) + "f")
            ax.set_title(jellybamm.format_case(case, amp, expanded=False))

    return fig


jellyroll_multiplot(
    d,
    cases=[0, 5, 10],
    var="std",
    title="Current density distribution log(STD)",
    do_log=True,
    global_scale=True,
)
plt.savefig("figZglobal.png", dpi=1200)
