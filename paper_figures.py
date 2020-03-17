# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 08:14:54 2020

@author: Tom
"""

import ecm
import numpy as np
import matplotlib.pyplot as plt
import os
root = 'D:\\pybamm_pnm_results\\Chen2020_Q_cc'
save_im_path = 'D:\\pybamm_pnm_results\\figures'
plt.close('all')

savefigs = True

tab_1 = [0, 1, 2, 3, 4]
tab_2 = [5, 6, 7, 8, 9]
tab_5 = [10, 11, 12, 13, 14]
tab_2_third = [15, 16, 17, 18, 19]

amps = ecm.get_amp_cases()
d = ecm.load_all_data()
cases = ecm.get_cases()
soc_list=[[0.9, 0.8, 0.7],[0.6, 0.5, 0.4],[0.3, 0.2, 0.1]]
#mini_soc_list=[[0.99, 0.98, 0.97],[0.96, 0.95, 0.94],[0.93, 0.92, 0.91]]
mini_soc_list=[[0.09, 0.08],[0.07, 0.06]]
grp = 'neg'

# Base Case 5.25 Amps - HTC 28 - 1 Tab
fig1 = ecm.jellyroll_subplot(d, 2, amps[-1], var=0, soc_list=soc_list, global_range=False, dp=1)
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig1.png'), dpi=600)
# Base Case all Amps - HTC 28 - 2 Tabs
fig2 = ecm.multi_var_subplot(d, [0], amps, [0, 1])
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig2.png'), dpi=600)
## All HTC cases - 1 tabs, 10 A
fig3 = ecm.multi_var_subplot(d, tab_1, [amps[-1]], [0, 1])
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
fig6 = ecm.spacetime(d, [0, 5, 10], amps, var=0, group=grp, normed=True)
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig6.png'), dpi=600)
# All Tabs, highest currents HTC 5
fig7 = ecm.multi_var_subplot(d, [0, 5, 10], [amps[-1]], [0, 1])
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig7.png'), dpi=600)
# All Tabs, highest currents HTC 100
fig8 = ecm.multi_var_subplot(d, [4, 9, 14], [amps[-1]], [0, 1])
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig8.png'), dpi=600)
# All Tabs, all currents HTC 5
fig9a = ecm.spacetime(d, [0, 5, 10], amps, var=0, group=grp, normed=True)
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig9a.png'), dpi=600)
fig9b = ecm.chargeogram(d, [0, 5, 10], amps, group=grp)
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig9b.png'), dpi=600)
# All Tabs, all currents HTC 100
fig10a = ecm.spacetime(d, [4, 9, 14], amps, var=0, group=grp, normed=True)
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig10a.png'), dpi=600)
fig10b = ecm.chargeogram(d, [4, 9, 14], amps, group=grp)
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig10b.png'), dpi=600)
fig11a = ecm.spacetime(d, [9, 17, 19], amps, var=0, group=grp, normed=True)
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig11a.png'), dpi=600)
fig11b = ecm.chargeogram(d, [9, 17, 19], amps, group=grp)
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig11b.png'), dpi=600)
# Third Heating
fig12 = ecm.jellyroll_subplot(d, 19, 5.25, var=0, soc_list=soc_list, global_range=False)
if savefigs:
    plt.savefig(os.path.join(save_im_path, 'fig12.png'), dpi=600)
#ecm.animate_data4(d, 0, 10, [0, 1], 'test_new_ani')
exp_data = ecm.load_experimental()
sim_data = [d[2][1.75], d[2][3.5], d[2][5.25]]
fig, ax = plt.subplots()
for i in range(3):
    ed = exp_data[i]
    sd = sim_data[i]
    plt.scatter(ed['Q discharge [mA.h]'].values, ed['Temperature [K]'].values)
    plt.plot(sd['capacity']*1000, sd[1]['mean'])