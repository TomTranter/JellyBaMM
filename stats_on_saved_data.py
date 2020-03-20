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
from scipy.stats import lognorm, gumbel_l, gumbel_r, kstest
import ecm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#plt.style.use('default')
plt.style.use('default')

plt.close("all")

pybamm.set_logging_level("INFO")
wrk = op.Workspace()
wrk.clear()


save_parent = 'C:\\Code\\pybamm_pnm_case4_Chen2020c'
amps = np.array([10])
subs = [str(a)+'A' for a in amps]
cwd = os.getcwd()
input_dir = os.path.join(cwd, 'input')
wrk.load_project(os.path.join(input_dir, 'MJ141-mid-top_m_cc_new.pnm'))
sim_name = list(wrk.keys())[-1]
project = wrk[sim_name]
net = project.network
Nspm = net.num_throats('spm_resistor')
weights = net['throat.arc_length'][net.throats('spm_resistor')]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
colors = ['r', 'g', 'b', 'y']
for si, sub in enumerate(subs):
    save_root = save_parent + '\\' + sub
    file_lower = os.path.join(save_root, 'var_Current_collector_current_density_lower')
    file_upper = os.path.join(save_root, 'var_Current_collector_current_density_upper')
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
    
    if np.any(np.isnan(data_amalg[-1, :])):
        data_amalg = data_amalg[:-1, :]
        temp_amalg = temp_amalg[:-1, :]
        time_amalg = time_amalg[:-1, :]
    cap = time_amalg[:, 0] * amps[si]
    abs_min = data_amalg.min()
    abs_max = data_amalg.max()
    
    #int_weights = np.around(weights/weights.min()*100, 0).astype(int)
    #plt.figure()
    #all_args = []
    #all_ks = []
    #for t in range(data_amalg.shape[0]-1):
    #    print(t)
    #    data_t = data_amalg[t, :]
    #    full_data_t = np.repeat(data_t, int_weights)
    #    sample = np.random.choice(full_data_t, 5000)
    #    args = dist.fit(sample)
    #    all_args.append(args)
    #    all_ks.append(kstest(sample, 'lognorm', args=args))
    #    start = data_t.min()
    #    end = data_t.max()
    #    x = np.linspace(start, end, 1000)
    #    plt.plot(x, dist.pdf(x, *args))
    neg_inner_weights = net['throat.arc_length'][net.throats('spm_neg_inner')]
    neg_inner_distance = np.cumsum(neg_inner_weights)
    pos_inner_weights = net['throat.arc_length'][net.throats('spm_pos_inner')]
    pos_inner_distance = np.cumsum(pos_inner_weights)
    mean_t = []
    m_minus_t = []
    m_plus_t = []
    Iy_t = []
    for t in range(data_amalg.shape[0]):
        data_t = data_amalg[t, :]
        mean, _ = ecm.weighted_avg_and_std(data_t, weights)
        diff = (data_t - mean)*weights
        Iy = np.sum(weights*((data_t-mean)**3)/3)
        m_p = np.mean(diff[diff > 0])/np.mean(weights[diff > 0])
        m_m = np.mean(diff[diff <=0])/np.mean(weights[diff <=0])
        mean_t.append(mean)
        m_minus_t.append(m_m)
        m_plus_t.append(m_p)
        Iy_t.append(Iy)
    min_t = np.min(data_amalg, axis=1)
    max_t = np.max(data_amalg, axis=1)
    ax1.plot(cap, Iy_t/mean, c=colors[si], label=sub)
    ax2.plot(cap, max_t/mean, c=colors[si], label=sub)
    ax2.plot(cap, min_t/mean,c=colors[si])
ax1.legend()
ax2.legend()
roll_pos = np.cumsum(net['throat.arc_length'][net.throats('spm_neg_inner')])
norm_roll_pos = roll_pos/roll_pos[-1]
norm_roll_pos *= 100
data_amalg /= mean
data_amalg *= 100
spm_ids = np.argwhere(net['pore.arc_index'][net['throat.conns'][net.throats('spm_neg_inner')]][:, 0] < 37 )

#for _t in range(data_amalg)

filtered_data = data_amalg[:-2, spm_ids]
fmin = np.int(np.floor(filtered_data.min()))
fmax = np.int(np.ceil(filtered_data.max()))
nbins = fmax-fmin
data_2d = np.zeros([len(spm_ids), nbins], dtype=float)
for i in range(len(spm_ids)):
    data, bins = np.histogram(filtered_data[:, i], bins=nbins, range=(fmin, fmax), density=True)
    print(np.sum(data))
    data_2d[i, :] = data*100



data_array = np.array(data_2d)
#data_array[data_array==0.0] = np.nan
#
# Create a figure for plotting the data as a 3D histogram.
#

#
# Create an X-Y mesh of the same dimension as the 2D data. You can
# think of this as the floor of the plot.
#
centers = (bins[1:] + bins[:-1])/2
bin_widths = bins[:-1] - bins[1:]
#x_data, y_data = np.meshgrid( centers,
#                              np.arange(data_array.shape[0]) )
x_data, y_data = np.meshgrid( centers,
                              norm_roll_pos[spm_ids] )
x_len , y_len = np.meshgrid( bin_widths,
                             net['throat.arc_length'][net.throats('spm_neg_inner')][spm_ids] )
#
# Flatten out the arrays so that they may be passed to "ax.bar3d".
# Basically, ax.bar3d expects three one-dimensional arrays:
# x_data, y_data, z_data. The following call boils down to picking
# one entry from each array and plotting a bar to from
# (x_data[i], y_data[i], 0) to (x_data[i], y_data[i], z_data[i]).
#
fig = plt.figure()
#ax = fig.gca(projection='3d')
ax = fig.gca()
#ax.plot_surface(x_data, y_data, data_array,
#                cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
heatmap = data_array.astype(float)
heatmap[heatmap == 0.0] = np.nan
im = ax.pcolormesh(x_data-100, y_data, heatmap, cmap=cm.coolwarm)
plt.colorbar(im)

# Plot 2
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#x_data = x_data.flatten()
#y_data = y_data.flatten()
#y_len = y_len.flatten()
#x_len = x_len.flatten()
#z_data = data_array.flatten()
#z_data_log = np.log(z_data)
#z_data[z_data == 0] = 0
##z_data[z_data == 0.0] = np.nan
#x_data = x_data[z_data > 0]
#y_data = y_data[z_data > 0]
#x_len = x_len[z_data > 0]
#y_len = y_len[z_data > 0]
#z_data = z_data[z_data > 0]
#ax.bar3d( x_data,
#          y_data,
#          np.zeros(len(z_data)),
#          x_len, y_len, z_data)
#plt.show()

## Plot 3
#im_data = np.log(data_2d)
#im_data[data_2d == 0] = np.nan
#plt.figure()
#plt.imshow(im_data, aspect=data_2d.shape[1]/data_2d.shape[0])