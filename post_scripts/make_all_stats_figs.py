# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 13:43:52 2020

@author: Tom
"""

import jellybamm
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import kstest
from sklearn.preprocessing import StandardScaler
import scipy
import pandas as pd
from matplotlib import cm
import warnings
from scipy import io

warnings.filterwarnings("ignore")

input_dir = 'C:\\Code\\pybamm_pnm_couple\\input'
root = 'D:\\pybamm_pnm_results\\Chen2020_v3'
#save_im_path = 'D:\\pybamm_pnm_results\\figures'
plt.close('all')

savefigs = True

tab_1 = [0, 1, 2, 3, 4]
tab_2 = [5, 6, 7, 8, 9]
tab_5 = [10, 11, 12, 13, 14]
tab_2_third = [15, 16, 17, 18, 19]
tab_1_2 = [20, 21, 22, 23, 24]

amps = jellybamm.get_amp_cases()
d = jellybamm.load_all_data()
cases = jellybamm.get_cases()
soc_list = [[0.9, 0.8, 0.7],
            [0.6, 0.5, 0.4],
            [0.3, 0.2, 0.1]]
mini_soc_list = [[0.09, 0.08],
                 [0.07, 0.06]]
grp = 'neg'


def plot_one_spm_dist(data_spm, dist_name, args=None):
    dist = getattr(scipy.stats, dist_name)
    if args is None:
        args = dist.fit(data_spm)
    x = np.linspace(data_spm.min(), data_spm.max(), 101)
    fig, ax = plt.subplots()
    ax.hist(data_spm, density=True)
    ax.plot(x, dist.pdf(x, *args))
    ks_res = kstest(data_spm, dist_name, args=args)
    ax.set_title(dist_name + ': ' + str(np.around(ks_res.pvalue, 6)))


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
    return (best_dist_name, best_chi_square, dist, args, dist.mean(*args),
            dist.std(*args))


def jellyroll_one_plot(data, title, dp=3):
    fig, ax = plt.subplots()
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


cases = jellybamm.get_cases()
amps = jellybamm.get_amp_cases()
d = jellybamm.load_all_data()
for key in cases.keys():
    case_path = os.path.join(root, cases[key]['file'])
    for amp in amps:
        amp_path = os.path.join(case_path, str(amp) + 'A')
        print(amp_path)
        save_file = os.path.join(amp_path, 'current_density_case_' +
                                 str(key) + '_amp_' + str(amp))
        data = d[key][amp][0]['data']
        dist_names = []
        chi_squares = []
        args = []
        means = []
        stds = []
        for i in range(data.shape[1]):
            (dist_name, chi_square, dist, arg,
             dist_mean, dist_std) = find_best_fit(data[:, i])
            dist_names.append(dist_name)
            chi_squares.append(chi_square)
            args.append(arg)
            means.append(dist_mean)
            stds.append(dist_std)
        means = np.asarray(means)
        stds = np.asarray(stds)
        chi_squares = np.asarray(chi_squares)
        jellyroll_one_plot(np.log(stds), 'Current Density Distribution Log(STD)')
        io.savemat(file_name=save_file + '_std',
                   mdict={'data': stds},
                   long_field_names=True)
        if savefigs:
            plt.savefig(os.path.join(save_file + '_log_std.png'), dpi=600)
        jellyroll_one_plot(means, 'Current Density Distribution Means')
        io.savemat(file_name=save_file + '_mean',
                   mdict={'data': means},
                   long_field_names=True)
        if savefigs:
            plt.savefig(os.path.join(save_file + '_mean.png'), dpi=600)
        jellyroll_one_plot(chi_squares, 'Current Density Distribution Chi-Square')
        io.savemat(file_name=save_file + '_chi',
                   mdict={'data': chi_squares},
                   long_field_names=True)
        if savefigs:
            plt.savefig(os.path.join(save_file + '_chi_sq.png'), dpi=600)
