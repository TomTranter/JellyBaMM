# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 08:14:54 2020

@author: Tom
"""

import ecm
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import lognorm, gumbel_l, gumbel_r, norm, cauchy, kstest
from sklearn.preprocessing import StandardScaler
import scipy
import pandas as pd
# Turn off code warnings (this is not recommended for routine use)
import warnings
warnings.filterwarnings("ignore")


root = 'C:\\Users\\Tom\\Documents\\Chen2020_v3'
save_im_path = 'C:\\Users\\Tom\Documents\\Chen2020_v3\\figures'
plt.close('all')

savefigs = True

tab_1 = [0, 1, 2, 3, 4]
tab_2 = [5, 6, 7, 8, 9]
tab_5 = [10, 11, 12, 13, 14]
tab_2_third = [15, 16, 17, 18, 19]
tab_1_2 = [20, 21, 22, 23, 24]


amps = ecm.get_amp_cases()
d = ecm.load_all_data()
cases = ecm.get_cases()
#soc_list=[[0.9, 0.8, 0.7],[0.6, 0.5, 0.4],[0.3, 0.2, 0.1]]
#mini_soc_list=[[0.99, 0.98, 0.97],[0.96, 0.95, 0.94],[0.93, 0.92, 0.91]]
soc_list = [[0.9, 0.5, 0.4],[0.3, 0.2, 0.1]]
mini_soc_list=[[0.09, 0.08],[0.07, 0.06]]
grp = 'neg'


data = d[0][5.25][0]['data']


input_dir = 'C:\\Code\\pybamm_pnm\\input'
from matplotlib import cm
def jellyroll_one_plot(data, title, dp=3):

    fig, ax = plt.subplots(figsize=(12, 12))
    spm_map = np.load(os.path.join(input_dir, 'im_spm_map.npz'))['arr_0']
    spm_map_copy = spm_map.copy()
    spm_map_copy[np.isnan(spm_map_copy)] = -1
    spm_map_copy = spm_map_copy.astype(int)
    mask = np.isnan(spm_map)
    arr = np.ones_like(spm_map).astype(float)
    arr[~mask] = data[spm_map_copy][~mask]
    arr[mask] = np.nan
    im = ax.imshow(arr,  cmap=cm.inferno)
    ax.set_axis_off()
    plt.colorbar(im, ax=ax, format='%.'+str(dp)+'f')
    ax.set_title(title)
    return fig
vlist = [[2, 7],[0, 1]]
big_soc_list = np.linspace(0, 1, 36)[::-1]
big_soc_list = big_soc_list.reshape([6, 6])
bespoke_soc_list = [[0.95, 0.28, 0.26],[0.15, 0.09, 0.06]]
bespoke_soc_list = [[0.9, 0.5, 0.4],[0.3, 0.2, 0.1]]
if 1 == 1:
    # Base Case 5.25 Amps - HTC 28 - 1 Tab
    fig1 = ecm.jellyroll_subplot(d, 2, amps[-1], var=0, soc_list=bespoke_soc_list, global_range=False, dp=1)
    if savefigs:
        plt.savefig(os.path.join(save_im_path, 'fig1.png'), dpi=600)
    # Base Case all Amps - HTC 28 - 2 Tabs
    fig2 = ecm.multi_var_subplot(d, [0], amps, vlist)
    if savefigs:
        plt.savefig(os.path.join(save_im_path, 'fig2.png'), dpi=600)
    ## All HTC cases - 1 tabs, 10 A
    fig3 = ecm.multi_var_subplot(d, tab_1, [amps[-1]], vlist)
    if savefigs:
        plt.savefig(os.path.join(save_im_path, 'fig3.png'), dpi=600)
    # 2nd Case 5.25 Amps - HTC 100 - 2 Tab
    fig4 = ecm.jellyroll_subplot(d, 7, amps[-1], var=0, soc_list=soc_list, global_range=False, dp=1)
    if savefigs:
        plt.savefig(os.path.join(save_im_path, 'fig4.png'), dpi=600)
    # 3rd Case 5.25 Amps - HTC 100 - 5 Tab
    fig5 = ecm.jellyroll_subplot(d, 12, amps[-1], var=0, soc_list=soc_list, global_range=False, dp=1)
    if savefigs:
        plt.savefig(os.path.join(save_im_path, 'fig5.png'), dpi=600)
    # All Tabs, all currents HTC 5
    fig6 = ecm.spacetime(d, [0, 5, 10], [amps[-1]], var=0, group=grp, normed=True)
    if savefigs:
        plt.savefig(os.path.join(save_im_path, 'fig6.png'), dpi=600)
    # All Tabs, highest currents HTC 5
    fig7 = ecm.multi_var_subplot(d, [0, 5, 10], [amps[-1]], vlist)
    if savefigs:
        plt.savefig(os.path.join(save_im_path, 'fig7.png'), dpi=600)
    # All Tabs, highest currents HTC 100
    fig8 = ecm.multi_var_subplot(d, [4, 9, 14], [amps[-1]], vlist)
    if savefigs:
        plt.savefig(os.path.join(save_im_path, 'fig8.png'), dpi=600)
    # All Tabs, all currents HTC 5
    fig9a = ecm.spacetime(d, [0, 5, 10], amps, var=0, group=grp, normed=True)
    if savefigs:
        plt.savefig(os.path.join(save_im_path, 'fig9a.png'), dpi=600)
#    fig9b = ecm.chargeogram(d, [0, 5, 10], amps, group=grp)
#    if savefigs:
#        plt.savefig(os.path.join(save_im_path, 'fig9b.png'), dpi=600)
    # All Tabs, all currents HTC 100
    fig10a = ecm.spacetime(d, [4, 9, 14], amps, var=0, group=grp, normed=True)
    if savefigs:
        plt.savefig(os.path.join(save_im_path, 'fig10a.png'), dpi=600)
#    fig10b = ecm.chargeogram(d, [4, 9, 14], amps, group=grp)
#    if savefigs:
#        plt.savefig(os.path.join(save_im_path, 'fig10b.png'), dpi=600)
    fig11a = ecm.spacetime(d, [9, 17, 19], amps, var=0, group=grp, normed=True)
    if savefigs:
        plt.savefig(os.path.join(save_im_path, 'fig11a.png'), dpi=600)
#    fig11b = ecm.chargeogram(d, [9, 17, 19], amps, group=grp)
#    if savefigs:
#        plt.savefig(os.path.join(save_im_path, 'fig11b.png'), dpi=600)
    fig11c = ecm.multi_var_subplot(d, [2, 4, 17, 19], [5.25], vlist)
    if savefigs:
        plt.savefig(os.path.join(save_im_path, 'fig11c.png'), dpi=600)
    # Third Heating
    fig12 = ecm.jellyroll_subplot(d, 19, 5.25, var=0, soc_list=soc_list, global_range=False)
    if savefigs:
        plt.savefig(os.path.join(save_im_path, 'fig12.png'), dpi=600)
    fig13 = jellyroll_one_plot(d[19][5.25][1]['data'][-1, :],
                               'Temperature [K] with uneven cooling\n' +
                               ecm.format_case(19, 5.25, True))
    if savefigs:
        plt.savefig(os.path.join(save_im_path, 'fig13.png'), dpi=600)
    fig14 =ecm.multi_var_subplot(d, [2, 4, 17, 19], [5.25], vlist)
    if savefigs:
        plt.savefig(os.path.join(save_im_path, 'fig14.png'), dpi=600)
#exp_data = ecm.load_experimental()
##sim_data = [d[2][1.75], d[2][3.5], d[2][5.25]]
#fig, ax = plt.subplots()
#for i in range(3):
#    ed = exp_data[i]
#    sd = d[2][amps[i]]
#    ax.scatter(ed['Q discharge [mA.h]'].values/1000, ed['Temperature [K]'].values)
#    ax.plot(sd['capacity'], sd[1]['mean'], label='I='+str(amps[i])+' [A]')
#ax.set_xlabel('Capacity [Ah]')
#ax.set_ylabel('Temperature [K]')
#plt.legend()
#
#if savefigs:
#    plt.savefig(os.path.join(save_im_path, 'figX.png'), dpi=600)
