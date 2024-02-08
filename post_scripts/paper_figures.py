# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 08:14:54 2020

@author: Tom
"""

import jellybamm
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
import scipy
import pandas as pd
from matplotlib import cm
import configparser
# Turn off code warnings (this is not recommended for routine use)
import warnings

warnings.filterwarnings("ignore")


root = 'D:\\pybamm_pnm_results\\Chen2020_v3'
save_im_path = 'D:\\pybamm_pnm_results\\figures'

exp_root = 'D:\\pybamm_pnm_results\\experimental'
exp_files = ['MJ1_0.5C.csv',
             'MJ1_1.0C.csv',
             'MJ1_1.5C.csv']
base = 'pybamm_pnm_case'
plt.close('all')

savefigs = False

tab_1 = [0, 1, 2, 3, 4]
tab_2 = [5, 6, 7, 8, 9]
tab_5 = [10, 11, 12, 13, 14]
tab_2_third = [15, 16, 17, 18, 19]
tab_1_2 = [20, 21, 22, 23, 24]


amps = jellybamm.get_amp_cases()
d = jellybamm.load_all_data()
cases = jellybamm.get_cases()
#soc_list=[[0.9, 0.8, 0.7],[0.6, 0.5, 0.4],[0.3, 0.2, 0.1]]
#mini_soc_list=[[0.99, 0.98, 0.97],[0.96, 0.95, 0.94],[0.93, 0.92, 0.91]]
soc_list = [[0.9, 0.5, 0.4],
            [0.3, 0.2, 0.1]]
mini_soc_list = [[0.09, 0.08],
                 [0.07, 0.06]]
grp = 'neg'

data = d[0][5.25][0]['data']


def load_experimental():
    data_list = []
    for ef in exp_files:
        fp = os.path.join(exp_root, ef)
        data_list.append(pd.read_csv(fp))
    return data_list


def get_cases():
    cases = [
        '1_Chen2020',
        '2_Chen2020',
        '5_Chen2020',
        '3_Chen2020',
        '4_Chen2020',
        '1_Chen2020c',
        '2_Chen2020c',
        '5_Chen2020c',
        '3_Chen2020c',
        '4_Chen2020c',
        '1_Chen2020b',
        '2_Chen2020b',
        '5_Chen2020b',
        '3_Chen2020b',
        '4_Chen2020b',
        '1_Chen2020_third',
        '2_Chen2020_third',
        '5_Chen2020_third',
        '3_Chen2020_third',
        '4_Chen2020_third',
    ]
    full = [base + case for case in cases]
    cases = {
        0: {'file': full[0], 'htc': 5, 'tabs': 1},
        1: {'file': full[1], 'htc': 10, 'tabs': 1},
        2: {'file': full[2], 'htc': 28, 'tabs': 1},
        3: {'file': full[3], 'htc': 50, 'tabs': 1},
        4: {'file': full[4], 'htc': 100, 'tabs': 1},
        5: {'file': full[5], 'htc': 5, 'tabs': 2},
        6: {'file': full[6], 'htc': 10, 'tabs': 2},
        7: {'file': full[7], 'htc': 28, 'tabs': 2},
        8: {'file': full[8], 'htc': 50, 'tabs': 2},
        9: {'file': full[9], 'htc': 100, 'tabs': 2},
        10: {'file': full[10], 'htc': 5, 'tabs': 5},
        11: {'file': full[11], 'htc': 10, 'tabs': 5},
        12: {'file': full[12], 'htc': 28, 'tabs': 5},
        13: {'file': full[13], 'htc': 50, 'tabs': 5},
        14: {'file': full[14], 'htc': 100, 'tabs': 5},
        15: {'file': full[15], 'htc': 5, 'tabs': 1},
        16: {'file': full[16], 'htc': 10, 'tabs': 1},
        17: {'file': full[17], 'htc': 28, 'tabs': 1},
        18: {'file': full[18], 'htc': 50, 'tabs': 1},
        19: {'file': full[19], 'htc': 100, 'tabs': 1},
    }

    return cases


def get_case_details(key):
    cases = get_cases()
    return cases[key]['htc'], cases[key]['tabs']


def abc(x):
    alphabet = np.array(['a', 'b', 'c', 'd',
                         'e', 'f', 'g', 'h',
                         'i', 'j', 'k', 'l',
                         'm', 'n', 'o', 'p',
                         'q', 'r', 's', 't',
                         'u', 'v', 'w', 'x',
                         'y', 'z'])
    return alphabet[x].upper()


def format_case(x, a, expanded=False, print_amps=True):
    htc, tabs = get_case_details(x)
    if expanded:
        text = ('Case ' + abc(x) + ': h=' + str(htc) + ' [W.m-2.K-1] #tabs='
                + str(tabs).capitalize() + ': I=' + str(a) + ' [A]')
    else:
        if print_amps:
            text = 'Case ' + abc(x) + ': I=' + str(a) + ' [A]'
        else:
            text = 'Case ' + abc(x)
    return text


def load_all_data():
    config = configparser.ConfigParser()
    net = jellybamm.get_net()
    weights = jellybamm.get_weights(net)
    cases = get_cases()
    amps = jellybamm.get_amp_cases()
    variables = jellybamm.get_saved_var_names()
    data = {}
    for ci in range(len(cases.keys())):
        case_folder = os.path.join(root, cases[ci]['file'])
        data[ci] = {}
        config.read(os.path.join(case_folder, 'config.txt'))
        data[ci]['config'] = jellybamm.config2dict(config)
        for amp in amps:
            amp_folder = os.path.join(case_folder, str(amp) + 'A')
            data[ci][amp] = {}
            for vi, v in enumerate(variables):
                data[ci][amp][vi] = {}
                temp = jellybamm.load_and_amalgamate(amp_folder, v)
                if temp is not None:
                    if vi == 0:
                        check_nans = np.any(np.isnan(temp), axis=1)
                        if np.any(check_nans):
                            print('Nans removed from', amp_folder)
                    if np.any(check_nans):
                        temp = temp[~check_nans, :]
                    data[ci][amp][vi]['data'] = temp
                    means = np.zeros(temp.shape[0])
                    for t in range(temp.shape[0]):
                        (mean, std_dev) = jellybamm.weighted_avg_and_std(temp[t, :], weights)
                        means[t] = mean
                    data[ci][amp][vi]['mean'] = means
                    data[ci][amp][vi]['min'] = np.min(temp, axis=1)
                    data[ci][amp][vi]['max'] = np.max(temp, axis=1)
            if temp is not None:
                t_hrs = data[ci][amp][10]['data'][:, 0]
                cap = t_hrs * amp
                data[ci][amp]['capacity'] = cap
    return data


def jellyroll_one_plot(data, title, dp=3):
    input_dir = jellybamm.INPUT_DIR
    fig, ax = plt.subplots(figsize=(12, 12))
    spm_map = np.load(os.path.join(input_dir, 'im_spm_map.npz'))['arr_0']
    spm_map_copy = spm_map.copy()
    spm_map_copy[np.isnan(spm_map_copy)] = -1
    spm_map_copy = spm_map_copy.astype(int)
    mask = np.isnan(spm_map)
    arr = np.ones_like(spm_map).astype(float)
    arr[~mask] = data[spm_map_copy][~mask]
    arr[mask] = np.nan
    im = ax.imshow(arr, cmap=cm.inferno)
    ax.set_axis_off()
    plt.colorbar(im, ax=ax, format='%.' + str(dp) + 'f')
    ax.set_title(title)
    return fig


def find_best_fit(y, report_results=False):
    # Set up list of candidate distributions to use
    # See https://docs.scipy.org/doc/scipy/reference/stats.html for more
    #y = data_spm.copy()
    size = len(y)
    dist_names = ['norm',
                  'gumbel_l',
                  'gumbel_r']

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
    observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
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
        cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2],
                              scale=param[-1])
        expected_frequency = []
        for bin in range(len(percentile_bins) - 1):
            expected_cdf_area = cdf_fitted[bin + 1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)

        # calculate chi-squared
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = sum(((cum_expected_frequency -
                   cum_observed_frequency) ** 2) / cum_observed_frequency)
        chi_square.append(ss)

    # Collate results and sort by goodness of fit (best at top)

    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['chi_square'] = chi_square
    results['p_value'] = p_values
    results.sort_values(['chi_square'], inplace=True)

    # Report results
    if report_results:
        print('\nDistributions sorted by goodness of fit:')
        print('----------------------------------------')
        print(results)
    best_dist_name = results.values[0][0]
    best_chi_square = results.values[0][1]
    dist = getattr(scipy.stats, best_dist_name)
    args = dist.fit(y)
    return (best_dist_name, best_chi_square, dist,
            args, dist.mean(*args), dist.std(*args))


# Base Case 5.25 Amps - HTC 28 - 1 Tab
fig1 = jellybamm.jellyroll_subplot(d, 2, amps[-1], var=0,
                             soc_list=soc_list, global_range=False, dp=1)
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig1.png'), dpi=600)
# Base Case all Amps - HTC 28 - 2 Tabs
fig2 = jellybamm.multi_var_subplot(d, [0], amps, [2, 0], landscape=False)
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig2.png'), dpi=600)
# All HTC cases - 1 tabs, 10 A
fig3 = jellybamm.multi_var_subplot(d, tab_1, [amps[-1]], [0, 1])
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig3.png'), dpi=600)
# 2nd Case 5.25 Amps - HTC 100 - 2 Tab
fig4 = jellybamm.jellyroll_subplot(d, 7, amps[-1], var=0,
                             soc_list=soc_list, global_range=False, dp=1)
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig4.png'), dpi=600)
# 3rd Case 5.25 Amps - HTC 100 - 5 Tab
fig5 = jellybamm.jellyroll_subplot(d, 12, amps[-1],
                             var=0, soc_list=soc_list, global_range=False, dp=1)
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig5.png'), dpi=600)
# All Tabs, all currents HTC 5
fig6 = jellybamm.spacetime(d, [0, 5, 10], amps, var=0, group=grp, normed=True)
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig6.png'), dpi=600)
# All Tabs, highest currents HTC 5
fig7 = jellybamm.multi_var_subplot(d, [0, 5, 10], [amps[-1]], [0, 1])
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig7.png'), dpi=600)
# All Tabs, highest currents HTC 100
fig8 = jellybamm.multi_var_subplot(d, [4, 9, 14], [amps[-1]], [0, 1])
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig8.png'), dpi=600)
# All Tabs, all currents HTC 5
fig9a = jellybamm.spacetime(d, [0, 5, 10], amps, var=0, group=grp, normed=True)
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig9a.png'), dpi=600)
fig9b = jellybamm.chargeogram(d, [0, 5, 10], amps, group=grp)
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig9b.png'), dpi=600)
# All Tabs, all currents HTC 100
fig10a = jellybamm.spacetime(d, [4, 9, 14], amps, var=0, group=grp, normed=True)
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig10a.png'), dpi=600)
fig10b = jellybamm.chargeogram(d, [4, 9, 14], amps, group=grp)
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig10b.png'), dpi=600)
fig11a = jellybamm.spacetime(d, [9, 17, 19], amps, var=0, group=grp, normed=True)
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig11a.png'), dpi=600)
fig11b = jellybamm.chargeogram(d, [9, 17, 19], amps, group=grp)
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig11b.png'), dpi=600)
# Third Heating
fig12 = jellybamm.jellyroll_subplot(d, 19, 5.25, var=0, soc_list=soc_list, global_range=False)
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig12.png'), dpi=600)
fig13 = jellyroll_one_plot(d[19][5.25][1]['data'][-1, :],
                           'Temperature [K] with uneven cooling\n' +
                           jellybamm.format_case(19, 5.25, True))
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig13.png'), dpi=600)
fig14 = jellybamm.multi_var_subplot(d, [2, 4, 17, 19], [5.25], [0, 1])
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig14.png'), dpi=600)
exp_data = jellybamm.load_experimental()
#sim_data = [d[2][1.75], d[2][3.5], d[2][5.25]]
fig, ax = plt.subplots()
for i in range(3):
    ed = exp_data[i]
    sd = d[2][amps[i]]
    ax.scatter(ed['Q discharge [mA.h]'].values / 1000, ed['Temperature [K]'].values)
    ax.plot(sd['capacity'], sd[1]['mean'], label='I=' + str(amps[i]) + ' [A]')
ax.set_xlabel('Capacity [Ah]')
ax.set_ylabel('Temperature [K]')
plt.legend()

if savefigs:
    plt.savefig(os.path.join(save_im_path, 'figX.png'), dpi=600)

figY = jellybamm.jellyroll_subplot(d, 19, 5.25, var=1, soc_list=[[0.9, 0.7], [0.5, 0.3]],
                             global_range=False, dp=1)
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'figY.png'), dpi=600)
