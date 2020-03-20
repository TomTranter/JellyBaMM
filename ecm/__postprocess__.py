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
import pandas as pd


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


input_dir = 'C:\\Code\\pybamm_pnm_couple\\input'
root = 'D:\\pybamm_pnm_results\\Chen2020_Q_cc'
base = 'pybamm_pnm_case'
exp_root = 'D:\\pybamm_pnm_results\\experimental'
exp_files = ['MJ1_0.5C.csv',
             'MJ1_1.0C.csv',
             'MJ1_1.5C.csv']
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
            '1_Chen2020d',
            '2_Chen2020d',
            '5_Chen2020d',
            '3_Chen2020d',
            '4_Chen2020d',
#            '4_Chen2020_econd',
#            '4_Chen2020_lowk',
#            '4_Chen2020_third',
#            '4_Chen2020_lowecond_third',
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
            20: {'file': full[20], 'htc': 5, 'tabs': 2},
            21: {'file': full[21], 'htc': 10, 'tabs': 2},
            22: {'file': full[22], 'htc': 28, 'tabs': 2},
            23: {'file': full[23], 'htc': 30, 'tabs': 2},
            24: {'file': full[24], 'htc': 100, 'tabs': 2},}
    
    return cases

def get_case_details(key):
    cases = get_cases()    
    return cases[key]['htc'], cases[key]['tabs']

def format_case(x, a, expanded=True):
    htc, tabs = get_case_details(x)
    if expanded:
        text = 'Case ' +abc(x)+': h='+str(htc)+' [W.m-2.K-1] #tabs='+str(tabs) +': I='+str(a)+ ' [A]'
    else:
        text = 'Case ' +abc(x)+': I='+str(a)+ ' [A]'
    return text

def abc(x):
    alphabet = np.array(['a', 'b', 'c', 'd',
                         'e', 'f', 'g', 'h',
                         'i', 'j', 'k', 'l',
                         'm', 'n', 'o', 'p',
                         'q', 'r', 's', 't',
                         'u', 'v', 'w', 'x',
                         'y', 'z'])
    return alphabet[x].upper()
    

def get_amp_cases():
    return [1.75, 3.5, 5.25]

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
    for ci in range(len(cases.keys())):
        case_folder = os.path.join(root, cases[ci]['file'])
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
        ax.plot(cap, dmean/dmean, c=c, label='Case '+abc(case)+': I = '+str(amp)+ '[A]')
        ax.plot(cap, dmax/dmean, c=c)
    else:
        ax.plot(cap, dmin, c=c)
        ax.plot(cap, dmean, c=c, label=format_case(case, amp, expanded=False))
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
#    norm_roll_pos *= 100
    Nspm = net.num_throats('spm_resistor')
    if group == 'neg':
        spm_ids = np.arange(Nspm)[:len(Ts)]
    else:
        spm_ids = np.arange(Nspm)[-len(Ts):]
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
            x_data, y_data = np.meshgrid( norm_roll_pos,
                                          centers
                                           )
            heatmap = data_2d.astype(float)
            heatmap[heatmap == 0.0] = np.nan
            im = ax.pcolormesh(x_data, y_data-100, heatmap.T, cmap=cm.coolwarm, vmin=0.0, vmax=100)
            ax.set_title(format_case(case, amp, expanded=False))
#            if ai == 0 and ci == 1:
#                ax.set_ylabel()
            if ci == len(case_list) - 1:
                ax.set_xlabel('Normalized roll position')
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.locator_params(nbins=6)
    fig.suptitle('Current Density Distribution \n' +
                 'Percentage deviation from mean: '+format_label(0))

    return fig

def spacetime(data, case_list, amp_list, var=0, group='neg', normed=False):
    wrk.clear()
    net = get_net()
    nrows = len(case_list)
    ncols = len(amp_list)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(int(5*ncols), int(5*nrows)),
                             sharex=True,
                             sharey=True)
#    var = 0  # Current density
    Ts = net.throats('spm_'+group+'_inner')
    roll_pos = np.cumsum(net['throat.arc_length'][Ts])
    norm_roll_pos = roll_pos/roll_pos[-1]
#    norm_roll_pos *= 100
    Nspm = net.num_throats('spm_resistor')
    if group == 'neg':
        spm_ids = np.arange(Nspm)[:len(Ts)]
    else:
        spm_ids = np.arange(Nspm)[-(len(Ts)):]
    ax_list = []
    x_list = []
    y_list = []
    data_list = []
    for ci, case in enumerate(case_list):
        for ai, amp in enumerate(amp_list):
            ax = axes[ci][ai]
            data_amalg = data[case][amp][var]['data'].copy()
            ax_list.append(ax)
            cap = data[case][amp]['capacity']
            if normed:
                mean = data[case][amp][var]['mean'][0]
                data_amalg /= mean
                data_amalg *= 100
                data_amalg -= 100
#            spm_ids = np.argwhere(net['pore.arc_index'][net['throat.conns'][Ts]][:, 0] < 37 )
            filtered_data = data_amalg[:, spm_ids]
#            fmin = np.int(np.floor(filtered_data.min()))
#            fmax = np.int(np.ceil(filtered_data.max()))
            x_data, y_data = np.meshgrid(norm_roll_pos,
                                         cap)
            heatmap = filtered_data.astype(float)
            heatmap[heatmap == 0.0] = np.nan
            x_list.append(x_data)
            y_list.append(y_data)
            data_list.append(heatmap)
#    dmin = 9e99
#    dmax = -9e99
#    for tmp in data_list:
#        tmp_min = tmp.min()
#        tmp_max = tmp.max()
#        if tmp_min < dmin:
#            dmin = tmp_min
#        if tmp_max > dmax:
#            dmax = tmp_max
#    for i in range(len(ax_list)):
#        ax = ax_list[i]
#        x_data = x_list[i]
#        y_data = y_list[i]
#        heatmap = data_list[i]
            im = ax.pcolormesh(x_data, y_data, heatmap, cmap=cm.inferno)
            ax.set_title(format_case(case, amp, expanded=False))
            if ai == 0:
                ax.set_ylabel('Capacity [Ah]')
            if ci == len(case_list) - 1:
                ax.set_xlabel('Normalized Roll Position')
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.locator_params(nbins=6)
    if normed:
        fig.suptitle('Percentage deviation from mean: \n'+format_label(var))
    else:
        fig.suptitle(format_label(var))
    return fig


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
                                       frames=time.shape[0],
#                                       frames=5,
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

def jellyroll_subplot(data, case, amp, var=0, soc_list=[[0.9, 0.7], [0.5, 0.3]], global_range=False, dp=3):
    soc_arr = np.asarray(soc_list)
    (nrows, ncols) = soc_arr.shape
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12), sharex=True, sharey=True)
    spm_map = np.load(os.path.join(input_dir, 'im_spm_map.npz'))['arr_0']
    spm_map_copy = spm_map.copy()
    spm_map_copy[np.isnan(spm_map_copy)] = -1
    spm_map_copy = spm_map_copy.astype(int)
    var_data = data[case][amp][var]['data']
    mask = np.isnan(spm_map)
    arr = np.ones_like(spm_map).astype(float)
    soc = get_SOC_vs_cap(data, case, amp)[:, 1]
    gmax = -1e12
    gmin = 1e12
    arrs = []
    for ir in range(nrows):
        for ic in range(ncols):
            ax = axes[ir][ic]
            soc_target = soc_arr[ir][ic]
            t = np.argmin((soc-soc_target)**2)
            t_data = var_data[t, :]
            arr[~mask] = t_data[spm_map_copy][~mask]
            arr[mask] = np.nan
#            if global_range:
#                vmin = np.min(var_data)
#                vmax = np.max(var_data)
#            else:
            vmin = np.min(var_data[t, :])
            vmax = np.max(var_data[t, :])
            if vmin < gmin:
                gmin = vmin
            if vmax > gmax:
                gmax = vmax
            arrs.append(arr.copy())
    for ir in range(nrows):
        for ic in range(ncols):
            ax = axes[ir][ic]
            soc_target = soc_arr[ir][ic]
            t = np.argmin((soc-soc_target)**2)
            arr = arrs.pop(0)
            if global_range:
                im = ax.imshow(arr, vmax=gmax, vmin=gmin, cmap=cm.inferno)
            else:
                im = ax.imshow(arr,  cmap=cm.inferno)
            ax.set_axis_off()
            plt.colorbar(im, ax=ax, format='%.'+str(dp)+'f')
            ax.set_title('SOC: '+str(np.around(soc[t], 2)))
            fig.suptitle(format_case(case, amp) + '\n' + format_label(var))
    return fig

def get_SOC_vs_cap(data, case, amp):
    var_names = get_saved_var_names()
    i = [i for i, vn in enumerate(var_names) if 'Negative' in vn and 'lithiation' in vn][0]
    lith = data[case][amp][i]['mean']
    cap = data[case][amp]['capacity']
    soc = lith - lith[-1]
    soc = soc/soc.max()
    return np.vstack((cap, soc)).T

def load_experimental():
    data_list = []
    for ef in exp_files:
        fp = os.path.join(exp_root, ef)
        data_list.append(pd.read_csv(fp))
    return data_list
