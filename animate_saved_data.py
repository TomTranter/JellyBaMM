# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 08:46:59 2020

@author: Tom
"""

import pybamm
import openpnm as op
import matplotlib.pyplot as plt
import ecm
import configparser
import os
from scipy import io
import numpy as np

plt.close("all")

pybamm.set_logging_level("INFO")
wrk = op.Workspace()
wrk.clear()


if __name__ == "__main__":
    save_root = 'C:\\Code\\pybamm_pnm_case1\\5A'
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
    temp_amalg = np.hstack((temp_lower, temp_upper))
    time_amalg = np.hstack((time_lower, time_upper))
    weights = net['throat.arc_length'][net.throats('spm_resistor')]
    variables = {}
    plot_left='Current collector current density [A.m-2]'
    plot_right='Temperature [K]'
    plot_time ='Time [h]'
    time_cut = 3
    if time_cut is None:
        variables[plot_left] = data_amalg
        variables[plot_right] = temp_amalg
        variables[plot_time] = time_amalg
    else:
        variables[plot_left] = data_amalg[:time_cut, :]
        variables[plot_right] = temp_amalg[:time_cut, :]
        variables[plot_time] = time_amalg[:time_cut, :]
    save_path = os.path.join(save_root, 'test')
    ecm.animate_data2(project, variables, plot_left, plot_right, weights,
                      filename=save_path)