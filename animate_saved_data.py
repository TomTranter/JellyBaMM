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

plt.close("all")

pybamm.set_logging_level("INFO")
wrk = op.Workspace()
wrk.clear()


if __name__ == "__main__":
    save_parent = 'C:\\Code\\pybamm_pnm_case1'
#    save_parent = sys.argv[-1]
    for prefix in ['']:#, 'b']:
        for sub in ['5A']:#, '2A', '3A', '4A', '5A']:
            save_root = save_parent + prefix + '\\' + sub
            file_lower = os.path.join(save_root, 'var_ECM_sigma_local_lower')
            file_upper = os.path.join(save_root, 'var_ECM_sigma_local_upper')
            data_lower = io.loadmat(file_lower)['data']
            data_upper = io.loadmat(file_upper)['data']
            file_lower = os.path.join(save_root, 'var_Negative_electrode_average_extent_of_lithiation_lower')
            file_upper = os.path.join(save_root, 'var_Negative_electrode_average_extent_of_lithiation_upper')
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
            temp_amalg = np.hstack((temp_lower, temp_upper))
            time_amalg = np.hstack((time_lower, time_upper))
            weights = net['throat.arc_length'][net.throats('spm_resistor')]
            weights = None
            variables = {}
            plot_left='Total overpotential [V]'
            plot_right='Negative electrode average extent of lithiation [-]'
            plot_time ='Time [h]'
            overpotentials = [
                              'eta_Change_in_measured_open_circuit_voltage',
                              'eta_X-averaged_concentration_overpotential',
                              'eta_X-averaged_electrolyte_ohmic_losses',
                              'eta_X-averaged_reaction_overpotential',
                              'eta_X-averaged_solid_phase_ohmic_losses',
                              ]
            variables['Total overpotential [V]'] = None
            for sv in overpotentials:
                if variables['Total overpotential [V]'] is None:
                    variables['Total overpotential [V]'] = ecm.load_and_amalgamate(save_root, sv)
                else:
                    variables['Total overpotential [V]'] += ecm.load_and_amalgamate(save_root, sv)

            variables[plot_right] = temp_amalg
            variables[plot_time] = time_amalg

            save_path = os.path.join(save_root, 'total overpotential and lithiation')
            ecm.animate_data3(project, variables, plot_left, plot_right, weights,
                              filename=save_path)