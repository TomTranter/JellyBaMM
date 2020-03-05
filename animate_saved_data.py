# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 08:46:59 2020

@author: Tom
"""

import pybamm
import openpnm as op
import matplotlib.pyplot as plt
import ecm
import os
from scipy import io
import numpy as np
import sys
from scipy.stats import lognorm as dist, kstest

plt.style.use('default')

plt.close("all")

pybamm.set_logging_level("INFO")
wrk = op.Workspace()
wrk.clear()


if __name__ == "__main__":
    save_parent = 'C:\\Code\\pybamm_pnm_couple'
#    save_parent = sys.argv[-1]
    for prefix in ['']:#, 'b']:
        for sub in ['4A_Q_cc']:#, '2A', '3A', '4A', '5A']:
            save_root = save_parent + prefix + '\\' + sub
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
            cwd = os.getcwd()
            input_dir = os.path.join(cwd, 'input')
            wrk.load_project(os.path.join(input_dir, 'MJ141-mid-top_m_cc_new.pnm'))
            sim_name = list(wrk.keys())[-1]
            project = wrk[sim_name]
            net = project.network
            Nspm = net.num_throats('spm_resistor')
            data_amalg = np.hstack((data_lower, data_upper))
#            data_amalg = data_amalg[::-1, :]
            temp_amalg = np.hstack((temp_lower, temp_upper))
            time_amalg = np.hstack((time_lower, time_upper))
            weights = net['throat.arc_length'][net.throats('spm_resistor')]
            variables = {}
            plot_left='Current Collector Current Density [A.m-2]'
            plot_right='Temperature [K]'
            plot_time ='Time [h]'
#            overpotentials = [
#                              'eta_Change_in_measured_open_circuit_voltage',
#                              'eta_X-averaged_concentration_overpotential',
#                              'eta_X-averaged_electrolyte_ohmic_losses',
#                              'eta_X-averaged_reaction_overpotential',
#                              'eta_X-averaged_solid_phase_ohmic_losses',
#                              ]
#            variables['Total overpotential [V]'] = None
#            for sv in overpotentials:
#                if variables['Total overpotential [V]'] is None:
#                    variables['Total overpotential [V]'] = ecm.load_and_amalgamate(save_root, sv)
#                else:
#                    variables['Total overpotential [V]'] += ecm.load_and_amalgamate(save_root, sv)
            variables[plot_left] = data_amalg
            variables[plot_right] = temp_amalg
            variables[plot_time] = time_amalg

            save_path = os.path.join(save_root, 'Current collector current density')
            ecm.animate_data3(project, variables, plot_left, plot_right, weights,
                              filename=save_path)
#            int_weights = np.around(weights/weights.min()*100, 0).astype(int)
#            plt.figure()
#            all_args = []
#            all_ks = []
#            for t in range(data_amalg.shape[0]-1):
#                print(t)
#                data_t = data_amalg[t, :]
#                full_data_t = np.repeat(data_t, int_weights)
#                sample = np.random.choice(full_data_t, 5000)
#                args = dist.fit(sample)
#                all_args.append(args)
#                all_ks.append(kstest(sample, 'lognorm', args=args))
#                start = data_t.min()
#                end = data_t.max()
#                x = np.linspace(start, end, 1000)
#                plt.plot(x, dist.pdf(x, *args))