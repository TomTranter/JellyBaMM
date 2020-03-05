# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:32:23 2020

@author: Tom
"""
import os
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import openpnm as op
import math
from matplotlib import cm
import configparser
import matplotlib.animation as animation
from scipy.interpolate import griddata
from scipy.interpolate import NearestNDInterpolator
from matplotlib import gridspec
import matplotlib.ticker as mtick


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


input_dir = 'C:\\Code\\pybamm_pnm_couple\\input'
root = 'D:\\pybamm_pnm_results\\Chen2020'
base = 'pybamm_pnm_case'
wrk = op.Workspace()

def get_saved_var_names():
    saved_vars = ['var_Current_collector_current_density',
                  'var_Temperature',
                  'var_Terminal_voltage',
                  'var_Negative_electrode_average_extent_of_lithiation',
                  'var_Positive_electrode_average_extent_of_lithiation',
                  'var_X-averaged_negative_particle_surface_concentration',
                  'var_X-averaged_positive_particle_surface_concentration',
                  'var_X-averaged_total_heating',
                  'var_ECM_I_Local',
                  'var_ECM_R_local',
                  'var_Time',
                  'eta_Change_in_measured_open_circuit_voltage',
                  'eta_X-averaged_battery_reaction_overpotential',
                  'eta_X-averaged_battery_concentration_overpotential',
                  'eta_X-averaged_battery_electrolyte_ohmic_losses',
                  'eta_X-averaged_battery_solid_phase_ohmic_losses',
                  ]
    return saved_vars

def get_saved_var_units():
    units = ['A.m-2',
             'K',
             'V',
             '-',
             '-',
             'mol.m-3',
             'mol.m-3',
             'W.m-3',
             'A',
             'Ohm',
             'h',
             'V',
             'V',
             'V',
             'V',
             'V',
             ]
    return units

def get_cases():
    
    cases = [
            '1_Chen2020',
            '2_Chen2020',
            '3_Chen2020',
            '4_Chen2020',
            '1_Chen2020c_third',
            '2_Chen2020c_third',
            '3_Chen2020c_third',
            '4_Chen2020c_third',
            '1_Chen2020b',
            '2_Chen2020b',
            '3_Chen2020b',
            '4_Chen2020b',
            '4_Chen2020_econd',
            '4_Chen2020_lowk',
            '4_Chen2020_third',
            '4_Chen2020_lowecond_third',
            ]
    full = [base + case for case in cases]
    return full

def get_amp_cases():
    return [4, 6, 8, 10]

def load_and_amalgamate(save_root, var_name):
    file_lower = os.path.join(save_root, var_name+'_lower')
    file_upper = os.path.join(save_root, var_name+'_upper')
    data_lower = io.loadmat(file_lower)['data']
    data_upper = io.loadmat(file_upper)['data']
    data_amalg = np.hstack((data_lower, data_upper))
    return data_amalg

def format_label(i):
    saved_vars = get_saved_var_names()
    units = get_saved_var_units()
    label = saved_vars[i]
    var_axis_name = label.replace('var_', '')
    var_axis_name = var_axis_name.replace('eta_', '')
    var_axis_name = var_axis_name.replace('_', ' ')
    var_axis_name = var_axis_name + ' [' + units[i]+']'
    return var_axis_name

def config2dict(config):
    temp = {}
    for sec in config.sections():
        temp[sec] = {}
        for key in config[sec]:
            temp[sec][key] = config.get(sec, key)
    return temp

def compare_config(config_a, config_b):
    for sec in config_a.keys():
        for option in config_a[sec].keys():
            if 'i_app' not in option:
                opt_a = config_a[sec][option]
                opt_b = config_b[sec][option]
                if opt_a != opt_b:
                    print(sec, option, opt_a, opt_b)

def load_all_data():
    config = configparser.ConfigParser()
    net = get_net()
    weights = get_weights(net)
    cases = get_cases()
    amps = get_amp_cases()
    variables = get_saved_var_names()
    data = {}
    for ci, case in enumerate(cases):
        case_folder = os.path.join(root, case)
        data[ci] = {}
        config.read(os.path.join(case_folder, 'config.txt'))
        data[ci]['config'] = config2dict(config)
        for amp in amps:
            amp_folder = os.path.join(case_folder, str(amp)+'A')
            data[ci][amp] = {}
            for vi, v in enumerate(variables):                    
                data[ci][amp][vi] = {} 
                temp = load_and_amalgamate(amp_folder, v)
                if vi == 0:
                    check_nans = np.any(np.isnan(temp), axis=1)
                    if np.any(check_nans):
                        print('Nans removed from', amp_folder)
                if np.any(check_nans):
                    temp = temp[~check_nans, :]
                data[ci][amp][vi]['data'] = temp
                means = np.zeros(temp.shape[0])
                for t in range(temp.shape[0]):
                    (mean, std_dev) = weighted_avg_and_std(temp[t, :], weights)
                    means[t] = mean
                data[ci][amp][vi]['mean'] = means
                data[ci][amp][vi]['min'] = np.min(temp, axis=1)
                data[ci][amp][vi]['max'] = np.max(temp, axis=1)
            t_hrs = data[ci][amp][10]['data'][:, 0]
            cap = t_hrs * amp
            data[ci][amp]['capacity'] = cap
    return data

def get_net():
    wrk.load_project(os.path.join(input_dir, 'MJ141-mid-top_m_cc_new.pnm'))
    sim_name = list(wrk.keys())[-1]
    project = wrk[sim_name]
    net = project.network
    return net

def get_weights(net):
    weights = net['throat.arc_length'][net.throats('spm_resistor')]
    return weights

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))

def min_mean_max_subplot(data, case=0, amp=4, var=0, normed=False, c='k', ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    cap = data[case][amp]['capacity']
    dmin = data[case][amp][var]['min']
    dmean = data[case][amp][var]['mean']
    dmax = data[case][amp][var]['max']
    if normed:
        ax.plot(cap, dmin/dmean, c=c)
        ax.plot(cap, dmean/dmean, c=c, label='Case '+str(case)+': Amps='+str(amp))
        ax.plot(cap, dmax/dmean, c=c)
    else:
        ax.plot(cap, dmin, c=c)
        ax.plot(cap, dmean, c=c, label='Case '+str(case)+': Amps='+str(amp))
        ax.plot(cap, dmax, c=c)
    ax.set
    return ax

def chargeogram(data, case_list, amp_list, group='neg'):
    wrk.clear()
    net = get_net()
    nrows = len(case_list)
    ncols = len(amp_list)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(int(5*ncols), int(5*nrows)),
                             sharex=True,
                             sharey=True)
    var = 0  # Current density
    Ts = net.throats('spm_'+group+'_inner')
    roll_pos = np.cumsum(net['throat.arc_length'][Ts])
    norm_roll_pos = roll_pos/roll_pos[-1]
    norm_roll_pos *= 100
    Nspm = net.num_throats('spm_resistor')
    if group == 'neg':
        spm_ids = np.arange(Nspm)[:len(Ts)]
    else:
        spm_ids = np.arange(Nspm)[len(Ts):]
    for ci, case in enumerate(case_list):
        for ai, amp in enumerate(amp_list):
            ax = axes[ci][ai]
            data_amalg = data[case][amp][var]['data'].copy()
            mean = data[case][amp][var]['mean'][0]
            data_amalg /= mean
            data_amalg *= 100
#            spm_ids = np.argwhere(net['pore.arc_index'][net['throat.conns'][Ts]][:, 0] < 37 )
            filtered_data = data_amalg[:, spm_ids]
            fmin = np.int(np.floor(filtered_data.min()))
            fmax = np.int(np.ceil(filtered_data.max()))
            nbins = fmax-fmin
            data_2d = np.zeros([len(spm_ids), nbins], dtype=float)
            for i in range(len(spm_ids)):
                hdata, bins = np.histogram(filtered_data[:, i], bins=nbins, range=(fmin, fmax), density=True)
                data_2d[i, :] = hdata*100
        
            centers = (bins[1:] + bins[:-1])/2
            x_data, y_data = np.meshgrid( centers,
                                          norm_roll_pos[spm_ids] )
            heatmap = data_2d.astype(float)
            heatmap[heatmap == 0.0] = np.nan
            im = ax.pcolormesh(x_data-100, y_data, heatmap, cmap=cm.coolwarm, vmin=0.0, vmax=100)
            ax.set_title('Case '+str(case)+': Amps='+str(amp))

def combined_subplot(data, case_list, amp_list, var=0, normed=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    col_array = np.asarray(colors)
    for case in case_list:
        for amp in amp_list:
            c = col_array[0]
            ax = min_mean_max_subplot(data, case, amp, var, normed, c=c, ax=ax)
            col_array = np.roll(col_array, -1)
    ax.set_xlabel('Capacity [Ah]')
    ax.set_ylabel(format_label(var))
    plt.legend()

def multi_var_subplot(data, case_list, amp_list, var_list, normed=False):
    nrows = 1
    ncols = len(var_list)
    fig, axes = plt.subplots(nrows, ncols, figsize=(int(5*ncols), int(5*nrows)))
    for vi in range(ncols):
        ax = axes[vi]
        combined_subplot(data, case_list, amp_list, var=var_list[vi], normed=normed, ax=ax)
    return fig, axes

def animate_init():
    pass

def animate_data4(data, case, amp, variables=None, filename=None):
    net = get_net()
    weights = get_weights(net)
    project = net.project
    im_spm_map = np.load(os.path.join(input_dir, 'im_spm_map.npz'))['arr_0']
    title = filename.split("\\")
    if len(title) == 1:
        title = title[0]
    else:
        title = title[-1]
    plot_left = format_label(variables[0])
    plot_right = format_label(variables[1])
    fig = setup_animation_subplots(plot_left, plot_right)
    mask = np.isnan(im_spm_map)
    spm_map_copy = im_spm_map.copy()
    spm_map_copy[np.isnan(spm_map_copy)] = -1
    spm_map_copy = spm_map_copy.astype(int)
    time_var = 'Time [h]'
    time = data[case][amp][10]['mean']
    vars2plot = {}
    vars2plot[plot_left] = data[case][amp][variables[0]]['data']
    vars2plot[plot_right] = data[case][amp][variables[1]]['data']
    func_ani = animation.FuncAnimation(fig=fig,
                                       func=update_multi_animation_subplots,
#                                       frames=time.shape[0],
                                       frames=5,
                                       init_func=animate_init,
                                       fargs=(fig, project,
                                              vars2plot,
                                              [plot_left, plot_right],
                                              spm_map_copy, mask,
                                              time_var, time, weights))
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=2, metadata=dict(artist='Tom Tranter'), bitrate=-1)

#    im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000,
#                                       blit=True)
    if '.mp4' not in filename:
        filename = filename + '.mp4'
    func_ani.save(filename, writer=writer, dpi=300)

def setup_animation_subplots(plot_left, plot_right):
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(3, 2, height_ratios=[12, 1, 4], width_ratios=[1, 1])
    plt.subplot(gs[0, 0])
    plt.subplot(gs[1, 0])
    plt.subplot(gs[2, 0])
    plt.subplot(gs[0, 1])
    plt.subplot(gs[1, 1])
    plt.subplot(gs[2, 1])
    return fig


def update_multi_animation_subplots(t, fig, project, variables, plot_vars, spm_map, mask, time_var, time, weights):
    for i, side in enumerate(['left', 'right']):
        data = variables[plot_vars[i]]
        if i == 0:
            global_range = False
        else:
            global_range = False
        fig  = update_animation_subplot(t, fig, data, plot_vars[i], spm_map, mask, time_var, time, weights, side=side, global_range=global_range)


def update_animation_subplot(t, fig, data, data_name,
                             spm_map, mask, time_var, time, weights,
                             side='left', global_range=True):
    print('Updating animation ' + side + ' frame', t)
    if side == 'left':
        ax1 = fig.axes[0]
        ax1c = fig.axes[1]
        ax2 = fig.axes[2]
    else:
        ax1 = fig.axes[3]
        ax1c = fig.axes[4]
        ax2 = fig.axes[5]
    ax1.clear()
    ax1c.clear()
    ax2.clear()
    ax1.set(title=data_name)
    ax2.set(xlabel=time_var)
#    ax3.set(ylabel=plot_right)
    
    arr = np.ones_like(spm_map).astype(float)
    t_data = data[t, :]
    arr[~mask] = t_data[spm_map][~mask]
    arr[mask] = np.nan
    gmin = np.min(data[~np.isnan(data)])
    gmax = np.max(data[~np.isnan(data)])
    if global_range:
        vmin = np.min(data)
        vmax = np.max(data)
    else:
        vmin = np.min(data[t, :])
        vmax = np.max(data[t, :])
    im = ax1.imshow(arr, vmax=vmax, vmin=vmin, cmap=cm.inferno)
#    ax1.set_axis_off()
    cbar = plt.colorbar(im, cax=ax1c, orientation="horizontal", format='%.2f')
    cbar.ax.locator_params(nbins=6)
    ax2.plot(time, np.max(data, axis=1), 'k--')
    ax2.plot(time, np.min(data, axis=1), 'k--')
    means = np.zeros(data.shape[0])
    std_devs = np.zeros(data.shape[0])
    if weights is None:
        weights = np.ones_like(data[0, :])
    for _t in range(data.shape[0]):
        (mean, std_dev) = weighted_avg_and_std(data[_t, :], weights)
        means[_t] = mean
        std_devs[_t] = std_dev
    ax2.plot(time, means, 'b--')
    ax2.fill_between(time,
                     means-std_devs,
                     means+std_devs)
    ax2.plot([time[t], time[t]], [vmin, vmax], 'r')
    grange = gmax-gmin
#    if global_range:
    ax2.set_ylim(gmin-grange*0.05,
                gmax+grange*0.05)
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax2.yaxis.tick_right()
    if t == 0:
        plt.tight_layout()
    return fig