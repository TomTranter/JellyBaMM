# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:44:23 2020

@author: Tom
"""

import pybamm
import openpnm as op
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import ecm
import os
from scipy import io
import numpy as np
import sys
import math
import matplotlib.ticker as mtick

plt.close("all")

pybamm.set_logging_level("INFO")
wrk = op.Workspace()
wrk.clear()


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
              'var_Local_ECM_resistance',
              'var_Time',
              'eta_Change_in_measured_open_circuit_voltage',
              'eta_X-averaged_concentration_overpotential',
              'eta_X-averaged_electrolyte_ohmic_losses',
              'eta_X-averaged_reaction_overpotential',
              'eta_X-averaged_solid_phase_ohmic_losses',
              ]

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
         'Ohm',
         'h',
         'V',
         'V',
         'V',
         'V',
         'V',
         ]


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))


def format_label(i):
    if i is None:
        return 'Total overpotential [V]'
    else:
        label = saved_vars[i]
        var_axis_name = label.replace('var_', '')
        var_axis_name = var_axis_name.replace('eta_', '')
        var_axis_name = var_axis_name.replace('_', ' ')
        var_axis_name = var_axis_name + ' [' + units[i]+']'
        return var_axis_name

var2plot = 3


same_axis = True

if __name__ == "__main__":
    save_parent = 'C:\\Code\\pybamm_pnm_case1'
#    save_parent = sys.argv[-1]
    if same_axis:
        fig, axes = plt.subplots(1, 2, figsize=(10, 7.5), sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(3, 2, figsize=(10, 7.5), sharex=True)
    for ti, prefix in enumerate(['']):
        for ax_int, sub in enumerate(['1A']):
            amps = float(sub[0])
            save_root = save_parent + prefix + '\\' + sub
            cwd = os.getcwd()
            input_dir = os.path.join(cwd, 'input')
            wrk.load_project(os.path.join(input_dir, 'MJ141-mid-top_m_cc_new.pnm'))
            sim_name = list(wrk.keys())[-1]
            project = wrk[sim_name]
            net = project.network
            Nspm = net.num_throats('spm_resistor')

            variables = {}
            variables['Total overpotential [V]'] = None
            for i, sv in enumerate(saved_vars):
                var_axis_name = format_label(i)
                variables[var_axis_name] = ecm.load_and_amalgamate(save_root, sv)
                if 'eta' in sv:
                    if variables['Total overpotential [V]'] is None:
                        variables['Total overpotential [V]'] = ecm.load_and_amalgamate(save_root, sv)
                    else:
                        variables['Total overpotential [V]'] += ecm.load_and_amalgamate(save_root, sv)
            if var2plot is None:
                data_amalg = variables['Total overpotential [V]']
            else:
                data_amalg = variables[format_label(var2plot)]
            temp_amalg = variables['Temperature [K]']
            time_amalg = variables['Time [h]']
            weights = net['throat.arc_length'][net.throats('spm_resistor')]

            means = np.zeros(data_amalg.shape[0])
            std_devs = np.zeros(data_amalg.shape[0])
            for t in range(data_amalg.shape[0]):
                (mean, std_dev) = weighted_avg_and_std(data_amalg[t, :], weights)
                means[t] = mean
                std_devs[t] = std_dev
            min_data = np.min(data_amalg, axis=1)
            max_data = np.max(data_amalg, axis=1)
            t_hrs = time_amalg[:, 0]
            cap = t_hrs * amps

            if same_axis:
                ax = axes[ti]
                ax.set(xlabel='Capacity [Ah]')
                ax.set(ylabel=format_label(var2plot))
            else:
                ax = axes[ax_int, ti]
                if ax_int == 2:
                    ax.set(xlabel='Capacity [Ah]')
                if ax_int == 1:
                    ax.set(ylabel=format_label(var2plot))
            ax.plot(cap, means, 'b--')
            ax.plot(cap, min_data, 'k--')
            ax.plot(cap, max_data, 'k--')
            label = sub[0] + ' [' + sub[1]+']'
            if ti == 0:
                label = label + ' 1 tab'
            else:
                label = label + ' 5 tabs'
            ax.fill_between(cap,
                            means-std_devs,
                            means+std_devs, label=label)

            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
            if not same_axis:
                ax.set_title(label=label, loc='left')
        if same_axis:
            ax.legend()

        spm_num = 0
        fig2, axes2 = plt.subplots(1, 1, figsize=(15, 10))
        terminal_voltage = variables['Terminal voltage [V]'][:, spm_num]
        V0 = np.zeros(len(t_hrs))
        for ei in np.arange(len(saved_vars))[-5:]:
            eta = variables[format_label(ei)][:, spm_num]
            axes2.fill_between(t_hrs, V0, V0-eta, label=format_label(ei))
            V0 -= eta
        plt.legend()
