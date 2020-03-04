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


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


input_dir = 'C:\\Code\\pybamm_pnm_couple\\input'
root = 'C:\\Code'
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
            '1_Chen2020b',
            '2_Chen2020b',
            '3_Chen2020b',
            '4_Chen2020b',
            '4_Chen2020_econd',
            '4_Chen2020_lowk',
            '4_Chen2020_third'
#            '1_Chen2020c',
#            '2_Chen2020c',
#            '3_Chen2020c',
#            '4_Chen2020c',
#            '4_Chen2020_lowecond',
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

def load_all_data():
    net = get_net()
    weights = get_weights(net)
    cases = get_cases()
    amps = get_amp_cases()
    variables = get_saved_var_names()
    data = {}
    for ci, case in enumerate(cases):
        case_folder = os.path.join(root, case)
        data[ci] = {}
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