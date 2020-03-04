# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:44:23 2020

@author: Tom
"""

import pybamm
import openpnm as op
import matplotlib.pyplot as plt
#plt.style.use('seaborn')
import ecm
import os
from scipy import io
import numpy as np
import sys
import math
import matplotlib.ticker as mtick
import warnings

plt.close("all")

pybamm.set_logging_level("INFO")
wrk = op.Workspace()
wrk.clear()

saved_vars = ecm.get_saved_var_names()
units = ecm.get_saved_var_units()

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

var2plot = 0


same_axis = True
amp_cases = ['4A', '6A', '8A', '10A']
#amp_cases = ['1A']
if __name__ == "__main__":
    save_parent = 'C:\\Code\\pybamm_pnm_case'
#    save_parent = sys.argv[-1]
    if same_axis:
        fig, axes = plt.subplots(1, 2, figsize=(10, 7.5), sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(3, 2, figsize=(10, 7.5), sharex=True)
    for ti, prefix in enumerate(['1_Chen2020', '2_Chen2020']):
        for ax_int, sub in enumerate(amp_cases):
            amps = float(sub[:-1])
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
            data_current = variables[format_label(0)]
            temp_amalg = variables['Temperature [K]']
            time_amalg = variables['Time [h]']
            weights = net['throat.arc_length'][net.throats('spm_resistor')]
            min_current = np.min(data_current, axis=1)
            max_current = np.max(data_current, axis=1)
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
#            ax.plot(cap, data_amalg[:, 0], 'r')
#            ax.plot(cap, data_amalg[:, 300], 'r')
            ax.plot(cap, min_data, 'k--')
            ax.plot(cap, max_data, 'k--')
            label = sub[:-1] + ' [' + sub[-1]+']'
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
            
            cap = cap[~np.isnan(means)]
            min_current = min_current[~np.isnan(means)]
            max_current = max_current[~np.isnan(means)]
            means = means[~np.isnan(means)]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', np.RankWarning)
                z30 = np.polyfit(cap, means, 30)
                z30_1deriv = np.polyder(z30, 1)
                z30_2deriv = np.polyder(z30, 2)
                p30 = np.poly1d(z30)
                p30_1deriv = np.poly1d(z30_1deriv)
                p30_2deriv = np.poly1d(z30_2deriv)
#            z = np.polyfit(cap, means, 6)
#            p = np.poly1d(z)
            pts = np.linspace(0, cap.max(), 100)
            print(p30(pts))
            ax.plot(pts, p30(pts), 'y--')
            fig3, axes3 = plt.subplots(4, 1, figsize=(15, 10))
            axes3[0].plot(cap, min_current, 'k--')
            axes3[0].plot(cap, max_current, 'k--')
            axes3[1].plot(pts, p30(pts))
            axes3[2].plot(pts, p30_1deriv(pts))
            axes3[3].plot(pts, p30_2deriv(pts))
        if same_axis:
            ax.legend()
        spm_num = 0
        fig2, axes2 = plt.subplots(1, 1, figsize=(15, 10))
        terminal_voltage = variables['Terminal voltage [V]'][:, spm_num]
        V0 = np.zeros(len(t_hrs))
        for ei in np.arange(len(saved_vars))[-5:]:
            eta = variables[format_label(ei)][:, spm_num]
            print(eta[0])
            axes2.fill_between(t_hrs, V0, V0-eta, label=format_label(ei))
            V0 -= eta
        plt.legend()
        

