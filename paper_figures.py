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

tab_1 = [0, 1, 2, 3]
tab_2 = [4, 5, 6, 7]
tab_5 = [8, 9, 10, 11]
tab_2_third = [12, 13, 14, 15]
even_third = [6, 7, 14, 15]
amps = ecm.get_amp_cases()
d = ecm.load_all_data()
cases = ecm.get_cases()
soc_list=[[0.9, 0.8, 0.7],[0.6, 0.5, 0.4],[0.3, 0.2, 0.1]]
# Base Case 5.25 Amps - HTC 100 - 1 Tab
fig1 = ecm.jellyroll_subplot(d, 3, 5.25, var=0, soc_list=soc_list, global_range=False, dp=1)
plt.savefig(os.path.join(save_im_path, 'fig1.png'), dpi=600)
# Base Case all Amps - HTC 5 - 2 Tabs
fig2 = ecm.multi_var_subplot(d, [0], amps, [0, 1])
plt.savefig(os.path.join(save_im_path, 'fig2.png'), dpi=600)
## All HTC cases - 1 tabs, 10 A
fig3 = ecm.multi_var_subplot(d, [0, 1, 2, 3], [amps[-1]], [0, 1])
plt.savefig(os.path.join(save_im_path, 'fig3.png'), dpi=600)
# 2nd Case 5.25 Amps - HTC 100 - 2 Tab
fig4 = ecm.jellyroll_subplot(d, 7, 5.25, var=0, soc_list=soc_list, global_range=False, dp=1)
plt.savefig(os.path.join(save_im_path, 'fig4.png'), dpi=600)
# 3rd Case 5.25 Amps - HTC 100 - 5 Tab
fig5 = ecm.jellyroll_subplot(d, 11, 5.25, var=0, soc_list=soc_list, global_range=False, dp=1)
plt.savefig(os.path.join(save_im_path, 'fig5.png'), dpi=600)
# All Tabs, all currents HTC 5
fig6 = ecm.spacetime(d, [0, 4, 8], amps, var=0, group='pos', normed=True)
plt.savefig(os.path.join(save_im_path, 'fig6.png'), dpi=600)
# All Tabs, all currents HTC 100
fig7 = ecm.spacetime(d, [3, 7, 11], amps, var=0, group='pos', normed=True)
plt.savefig(os.path.join(save_im_path, 'fig7.png'), dpi=600)
# All Tabs, all currents HTC 5
fig8 = ecm.chargeogram(d, [0, 4, 8], amps, group='pos')
plt.savefig(os.path.join(save_im_path, 'fig8.png'), dpi=600)
### All HTC cases - 2 tabs, 10 A
#fig4 = ecm.multi_var_subplot(d, [4, 5, 6, 7], [amps[-1]], [0, 1])
### All HTC cases - 5 tabs, 10 A
#fig5 = ecm.multi_var_subplot(d, [8, 9, 10, 11], [amps[-1]], [0, 1])
### 100 HTC cases - 2 tabs, 4, 10 A - neg_cc_econd
##fig6 = ecm.multi_var_subplot(d, [3, 12], [6, 10], [0, 1])
### 100 HTC cases - 2 tabs, 4, 10 A - variable k
##fig7 = ecm.multi_var_subplot(d, [3, 13], [4, 10], [0, 1])
### Base Case all Amps - Average Heating
#fig8 = ecm.multi_var_subplot(d, [0], amps, [7, 1])
#
#filename = os.path.join(root, cases[0])
#filename = os.path.join(filename, 'current_density_and_temperature')
#
fig9 = ecm.spacetime(d, [3, 7, 11 ], amps, var=0, group='pos', normed=True)
#fig9 = ecm.spacetime(d, [4, 5, 6, 7], amps, var=0, group='pos', normed=False)
#fig9 = ecm.spacetime(d, [8, 9, 10, 11], amps, var=1, group='pos', normed=False)

#ecm.animate_data4(d, 0, 10, [0, 1], 'test_new_ani')
