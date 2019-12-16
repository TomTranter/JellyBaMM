#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:14:46 2019

@author: tom
"""

import pybamm
import numpy as np
import matplotlib.pyplot as plt
import ecm
import os
import time
import copy


plt.close("all")
# set logging level
pybamm.set_logging_level("INFO")


if __name__ == '__main__':
    Nunit = 10
    max_workers = int(os.cpu_count()/2)
    I_app = 1.0
    spm_sim = ecm.make_spm(Nunit)
    height = spm_sim.parameter_values["Electrode height [m]"]
    width = spm_sim.parameter_values["Electrode width [m]"]
    A_cc = height*width
    variables = ['Local ECM resistance [Ohm.m2]',
                 'Local ECM voltage [V]',
                 'Local voltage [V]',
#                 'X-averaged positive particle surface concentration [mol.m-3]',
#                 'X-averaged negative particle surface concentration [mol.m-3]'
                ]
    pool_vars = [variables for i in range(Nunit)]
    spm_sim, results = ecm.step_spm((spm_sim, I_app/Nunit, 1e-6, variables, False))
    R = results[0]/A_cc
    V_ecm = results[1]
    print(R)
    R_max = R*10
    net, alg, phase = ecm.make_net(spm_sim, Nunit, R, spacing=height)
    (V_local_pnm, I_local_pnm, R_local_pnm) = ecm.run_ecm(net, alg, V_ecm)
    print('*'*30)
    print('V local pnm', V_local_pnm, '[V]')
    print('I local pnm', I_local_pnm, '[A]')
    print('R local pnm', R_local_pnm, '[Ohm]')
    spm_models = [copy.deepcopy(spm_sim) for i in range(len(net.throats('spm_resistor')))]
    dt = 5e-2
    Nsteps = 120
    res_Ts = net.throats('spm_resistor')
    terminal_voltages = np.zeros(Nsteps)
    V_test = V_ecm
    tol = 1e-5
    local_R = np.zeros([Nunit, Nsteps])
    stop_R = np.zeros(Nunit, dtype=bool)
    st = time.time()
    all_time_results = np.zeros([Nsteps, Nunit, len(variables)])
    all_time_I_local = np.zeros([Nsteps, Nunit])
    param = spm_sim.parameter_values
    tau = param.process_symbol(pybamm.standard_parameters_lithium_ion.tau_discharge)
    t_end = 3600 / tau.evaluate(0)
    t_eval = np.linspace(0, t_end, Nsteps)
    dt = t_end/(Nsteps-1)
    dead = np.zeros(Nunit, dtype=bool)
    pool = ecm.setup_pool(max_workers)
    outer_step = 0
    while np.any(~dead) and outer_step < Nsteps:
#    for outer_step in range(Nsteps):
        print('*'*30)
        print('Outer', outer_step)
        # Find terminal voltage that satisfy ecm total currents for R
        current_match = False
        max_inner_steps = 1000
        inner_step = 0
        damping = Nunit/10
        while (inner_step < max_inner_steps) and (not current_match):
            (V_local_pnm, I_local_pnm, R_local_pnm) = ecm.run_ecm(net,
                                                                  alg,
                                                                  V_test)
            tot_I_local_pnm = np.sum(I_local_pnm)
            diff = (I_app - tot_I_local_pnm)/I_app
            if np.absolute(diff) < tol:
                current_match = True
            else:
                V_test *= (1+(diff/damping))
            inner_step += 1
            print('Inner', inner_step, diff, V_test)
        print('N inner', inner_step)
        all_time_I_local[outer_step, :] = I_local_pnm
        terminal_voltages[outer_step] = V_test
        # I_local_pnm should now match the total applied current
        # Run the spms for the the new I_locals
        data = ecm.pool_spm(zip(spm_models,
                                I_local_pnm,
                                np.ones(Nunit)*dt,
                                pool_vars,
                                dead),
                            pool)
        data = np.asarray(data)
        spm_models = data[:, 0].tolist()
        temp = data[:, 1]
        results = np.zeros([Nunit, len(variables)])
        for i in range(Nunit):
            results[i, :] = temp[i]
        all_time_results[outer_step, :, :] = results
        temp_R = results[:, 0]/A_cc
        sig = 1/temp_R
        if np.any(temp_R > R_max):
            dead[temp_R > R_max] = True
            sig[temp_R > R_max] = 1e-6
        if np.any(np.isnan(temp_R)):
            dead[np.isnan(temp_R)] = True
            sig[np.isnan(temp_R)] = 1e-6
        phase['throat.electrical_conductance'][res_Ts] = sig
        local_R[:, outer_step] = temp_R
        print('Resistances', temp_R)
        print('Dead', dead)
        outer_step += 1


    ecm.shutdown_pool(pool)
    fig, ax = plt.subplots()
    for i in range(Nunit):
        ax.plot(local_R[i, :]) 
    plt.title('R Local [Ohm]')
    fig, ax = plt.subplots()
    for i in range(Nunit):
        ax.plot(all_time_I_local[:, i]) 
    plt.title('I Local [A]')
    for i, var in enumerate(variables):
        temp = all_time_results[:, :, i]
        fig, ax = plt.subplots()
        for i in range(Nunit):
            ax.plot(temp[:, i])
        plt.title(var)

    print('*'*30)
    print('Sim time', time.time()-st)
    print('*'*30)
