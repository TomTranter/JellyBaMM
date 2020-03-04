# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 08:46:59 2020

@author: Tom
"""

import pybamm
import openpnm as op
import matplotlib.pyplot as plt
import os
from scipy import io
import numpy as np
from scipy.stats import lognorm as dist, kstest
import ecm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#plt.style.use('default')
plt.style.use('default')

plt.close("all")

pybamm.set_logging_level("INFO")
wrk = op.Workspace()
wrk.clear()


save_parent = 'C:\\Code\\pybamm_pnm_case'
cases = ['1_Chen2020', '2_Chen2020', '3_Chen2020', '4_Chen2020']
cases = ['4_Chen2020', '4_Chen2020b', '4_Chen2020c', '4_Chen2020econd']
amps = np.array([4, 6, 8, 10])
subs = [str(a)+'A' for a in amps]
cwd = os.getcwd()
input_dir = os.path.join(cwd, 'input')
wrk.load_project(os.path.join(input_dir, 'MJ141-mid-top_m_cc_new.pnm'))
sim_name = list(wrk.keys())[-1]
project = wrk[sim_name]
net = project.network
Nspm = net.num_throats('spm_resistor')
weights = net['throat.arc_length'][net.throats('spm_resistor')]
fig, axes = plt.subplots(len(amps), len(cases), figsize=(12, 12), sharex=True, sharey=True)
colors = ['r', 'g', 'b', 'y']
for ci, case in enumerate(cases):
    for si, sub in enumerate(subs):
        ax = axes[si][ci]
        save_root = save_parent + case + '\\' + sub
        file_lower = os.path.join(save_root, 'var_Current_collector_current_density_lower')
        file_upper = os.path.join(save_root, 'var_Current_collector_current_density_upper')
#        file_lower = os.path.join(save_root, 'var_Temperature_lower')
#        file_upper = os.path.join(save_root, 'var_Temperature_upper')        
        data_lower = io.loadmat(file_lower)['data']
        data_upper = io.loadmat(file_upper)['data']
        file_lower = os.path.join(save_root, 'var_Temperature_lower')
        file_upper = os.path.join(save_root, 'var_Temperature_upper')
        temp_lower = io.loadmat(file_lower)['data']
        temp_upper = io.loadmat(file_upper)['data']
        cwd = os.getcwd()
        file_lower = os.path.join(save_root, 'var_Time_lower')
        file_upper = os.path.join(save_root, 'var_Time_upper')
        time_lower = io.loadmat(file_lower)['data']
        time_upper = io.loadmat(file_upper)['data']

        data_amalg = np.hstack((data_lower, data_upper))
        temp_amalg = np.hstack((temp_lower, temp_upper))
        time_amalg = np.hstack((time_lower, time_upper))
        check_nans = np.any(np.isnan(data_amalg), axis=1)
        if np.any(check_nans):
            data_amalg = data_amalg[~check_nans, :]
            temp_amalg = temp_amalg[~check_nans, :]
            time_amalg = time_amalg[~check_nans, :]
        cap = time_amalg[:, 0] * amps[si]
        mean_t = []
        for t in range(data_amalg.shape[0]):
            data_t = data_amalg[t, :]
            mean, _ = ecm.weighted_avg_and_std(data_t, weights)
            mean_t.append(mean)
        min_t = np.min(data_amalg, axis=1)
        max_t = np.max(data_amalg, axis=1)
    
    
        roll_pos = np.cumsum(net['throat.arc_length'][net.throats('spm_neg_inner')])
        norm_roll_pos = roll_pos/roll_pos[-1]
        norm_roll_pos *= 100
        data_amalg /= mean
        data_amalg *= 100
        spm_ids = np.argwhere(net['pore.arc_index'][net['throat.conns'][net.throats('spm_neg_inner')]][:, 0] < 37 )
        filtered_data = data_amalg[:, spm_ids]
        fmin = np.int(np.floor(filtered_data.min()))
        fmax = np.int(np.ceil(filtered_data.max()))
        nbins = fmax-fmin
        data_2d = np.zeros([len(spm_ids), nbins], dtype=float)
        for i in range(len(spm_ids)):
            data, bins = np.histogram(filtered_data[:, i], bins=nbins, range=(fmin, fmax), density=True)
#            print(np.sum(data))
            data_2d[i, :] = data*100

        centers = (bins[1:] + bins[:-1])/2
        x_data, y_data = np.meshgrid( centers,
                                      norm_roll_pos[spm_ids] )
        heatmap = data_2d.astype(float)
        heatmap[heatmap == 0.0] = np.nan
        im = ax.pcolormesh(x_data-100, y_data, heatmap, cmap=cm.coolwarm, vmin=0.0, vmax=100)
        ax.set_title(sub + ' ' + case)
#plt.colorbar(im)
