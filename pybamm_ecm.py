#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:14:46 2019

@author: tom
"""

import pybamm
import openpnm as op
from pybamm import EvaluatorPython as ep
import numpy as np
import matplotlib.pyplot as plt
import ecm
import os
import time


plt.close("all")
# set logging level
pybamm.set_logging_level("INFO")
wrk = op.Workspace()
wrk.clear()

if __name__ == "__main__":
    parallel = True
    Nlayers = 7
    layer_spacing = 195e-6
    dtheta = 10
    Narc = np.int(360 / dtheta)  # number of nodes in a wind/layer
    Nunit = np.int(Nlayers * Narc)  # nodes in each cc
    Nsteps = 2  # number of time steps
    max_workers = int(os.cpu_count() / 2)
#    max_workers = 30
    I_app = 0.5  # A
    model_name = 'blah'
    opt = {'domain': 'model',
           'Nlayers': Nlayers,
           'cp': 1399.0,
           'rho': 2055.0,
           'K0': 1.0,
           'T0': 303,
           'heat_transfer_coefficient': 5,
           'length_3d': 0.065,
           'I_app': I_app,
           'cc_cond_neg': 3e7,
           'cc_cond_pos': 3e7,
           'dtheta': dtheta,
           'spacing': 1e-5,
           'model_name': model_name}
    ###########################################################################
    project, arc_edges = ecm.make_spiral_net(Nlayers, dtheta,
                                             spacing=layer_spacing,
                                             pos_tabs=[0], neg_tabs=[-1])
    net = project.network
    # The jellyroll layers are double sided around the cc except for the inner
    # and outer layers the number of spm models is the number of throat
    # connections between cc layers
    Nspm = net.num_throats('spm_resistor')
    lens = arc_edges[1:] - arc_edges[:-1]
    l_norm = lens/lens[-1]
    total_length = arc_edges[-1]  # m
    ###########################################################################
    I_typical = I_app / Nspm
    spm_sim = ecm.make_spm(I_typical, height=lens[-1])
    height = spm_sim.parameter_values["Electrode height [m]"]
    width = spm_sim.parameter_values["Electrode width [m]"]
    t1 = spm_sim.parameter_values['Negative electrode thickness [m]']
    t2 = spm_sim.parameter_values['Positive electrode thickness [m]']
    t3 = spm_sim.parameter_values['Negative current collector thickness [m]']
    t4 = spm_sim.parameter_values['Positive current collector thickness [m]']
    t5 = spm_sim.parameter_values['Separator thickness [m]']
    ttot = t1+t2+t3+t4+t5
    A_cc = height * width
    bat_vol = height * width * ttot * Nunit
#    print('BATTERY VOLUME', bat_vol)
    variables = [
        "Local ECM resistance [Ohm.m2]",
        "Local ECM voltage [V]",
        "Measured open circuit voltage [V]",
        "Local voltage [V]",
        "Change in measured open circuit voltage [V]",
        "X-averaged total heating [W.m-3]",
    ]
    overpotentials = [
        "X-averaged reaction overpotential [V]",
        "X-averaged concentration overpotential [V]",
        "X-averaged electrolyte ohmic losses [V]",
        "X-averaged solid phase ohmic losses [V]",
        "Change in measured open circuit voltage [V]",
    ]
#    pool_vars = [variables for i in range(Nunit)]
    spm_sol = ecm.step_spm((spm_sim, None, I_app / Nunit, 1e-6,
                            opt['T0'], False))
    # Create dictionaries of evaluator functions from the discretized model
    variables_eval = {}
    overpotentials_eval = {}
    for var in variables:
        variables_eval[var] = ep(spm_sim.built_model.variables[var])
    for var in overpotentials:
        overpotentials_eval[var] = ep(spm_sim.built_model.variables[var])

    temp = ecm.evaluate_python(variables_eval, spm_sol, current=I_app/Nunit)
    R = temp[0] / A_cc
    V_ecm = temp[1]
    print(R)
    R_max = R * 10
    # Initialize with a guess for the terminal voltage
    alg = ecm.setup_ecm_alg(project, layer_spacing, R)
    phys = project.physics()['phys_01']
    phase = project.phases()['phase_01']
    (V_local_pnm, I_local_pnm, R_local_pnm) = ecm.run_ecm(net, alg, V_ecm)
    print("*" * 30)
    print("V local pnm", V_local_pnm, "[V]")
    print("I local pnm", I_local_pnm, "[A]")
    print("R local pnm", R_local_pnm, "[Ohm]")
    spm_models = [
        spm_sim for i in range(Nspm)
    ]
    solutions = [
        spm_sol for i in range(Nspm)
    ]

    res_Ts = net.throats("spm_resistor")
    terminal_voltages = np.zeros(Nsteps)
    V_test = V_ecm
    tol = 1e-5
    local_R = np.zeros([Nspm, Nsteps])
    st = time.time()
    all_time_results = np.zeros([Nsteps, Nspm, len(variables)])
    all_time_overpotentials = np.zeros([Nsteps, Nspm, len(overpotentials)])
    all_time_I_local = np.zeros([Nsteps, Nspm])
    param = spm_sim.parameter_values
    sym_tau = pybamm.standard_parameters_lithium_ion.tau_discharge
    tau = param.process_symbol(sym_tau)
    t_end = 3600 / tau.evaluate(0)
    t_eval_ecm = np.linspace(0, t_end, Nsteps)
    dt = t_end / (Nsteps - 1)
    dim_time_step = ecm.convert_time(spm_sim.parameter_values,
                                     dt, to='seconds')
    dead = np.zeros(Nspm, dtype=bool)
    if parallel:
        pool = ecm.setup_pool(max_workers, pool_type='Process')
    outer_step = 0
    print(project)
    ecm.setup_thermal(project, opt)
    print(project)
    spm_temperature = np.ones(len(res_Ts))*opt['T0']
    while np.any(~dead) and outer_step < Nsteps:
        print("*" * 30)
        print("Outer", outer_step)
        # Find terminal voltage that satisfy ecm total currents for R
        current_match = False
        max_inner_steps = 1000
        inner_step = 0
        damping = Nspm / 100
        # Iterate the ecm until the currents match
        while (inner_step < max_inner_steps) and (not current_match):
            print(inner_step, V_test)
            (V_local_pnm, I_local_pnm, R_local_pnm) = ecm.run_ecm(net,
                                                                  alg,
                                                                  V_test)
            tot_I_local_pnm = np.sum(I_local_pnm)
            diff = (I_app - tot_I_local_pnm) / I_app
            if np.absolute(diff) < tol:
                current_match = True
            else:
                V_test *= 1 + (diff / damping)
            inner_step += 1

        print("N inner", inner_step)
        all_time_I_local[outer_step, :] = I_local_pnm
        terminal_voltages[outer_step] = V_test
        # I_local_pnm should now sum to match the total applied current
        # Run the spms for the the new I_locals for the next time interval
        if parallel:
            solutions = ecm.pool_spm_new(
                zip(spm_models, solutions, I_local_pnm,
                    np.ones(Nspm) * dt, spm_temperature, dead), pool,
                    max_workers
            )
        else:
            solutions = ecm.serial_spm(
                zip(spm_models, solutions, I_local_pnm,
                    np.ones(Nspm) * dt, spm_temperature, dead)
            )
        # Gather the results for this time step
        results = np.zeros([Nspm, len(variables)])
        results_o = np.zeros([Nspm, len(overpotentials)])
        for i, solution in enumerate(solutions):
            results[i, :] = ecm.evaluate_python(variables_eval,
                                                solution,
                                                I_local_pnm[i])
            results_o[i, :] = ecm.evaluate_python(overpotentials_eval,
                                                  solution,
                                                  current=I_local_pnm[i])
        # Collate the results
        all_time_results[outer_step, :, :] = results
        all_time_overpotentials[outer_step, :, :] = results_o
        temp_local_V = results[:, 3]
        # Apply Heat Sources
        # To Do: make this better
        Q = results[:, 5] / (opt['cp'] * opt['rho'])
        ecm.apply_heat_source(project, Q)
        # Calculate Global Temperature
        ecm.run_step_transient(project, dim_time_step, opt['T0'])
        # Interpolate the node temperatures for the SPMs
        spm_temperature = phase.interpolate_data('pore.temperature')[res_Ts]
        # Get new equivalent resistances
        temp_R = ecm.calc_R_new(results_o, I_local_pnm)
        # stop simulation if any local voltage below the minimum
        # To do: check validity of using local
        if np.any(temp_local_V < 3.5):
            dead.fill(np.nan)
        # Update ecm conductivities for the spm_resistor throats
        sig = 1 / temp_R
        if np.any(temp_R > R_max):
            dead[temp_R > R_max] = True
            sig[temp_R > R_max] = 1e-6
        if np.any(np.isnan(temp_R)):
            dead[np.isnan(temp_R)] = True
            sig[np.isnan(temp_R)] = 1e-6
        phys["throat.electrical_conductance"][res_Ts] = sig
        local_R[:, outer_step] = temp_R

        print("N Dead", np.sum(dead))
        outer_step += 1

#    ecm.run_ecm(net, alg, V_test, plot=True)
#
#    all_time_results = all_time_results[:outer_step, :, :]
#    if parallel:
#        ecm.shutdown_pool(pool)
#    fig, ax = plt.subplots()
#    for i in range(Nspm):
#        ax.plot(local_R[i, :outer_step])
#    plt.title("R Local [Ohm]")
#    fig, ax = plt.subplots()
#    for i in range(Nspm):
#        ax.plot(all_time_I_local[:outer_step, i])
#    plt.title("I Local [A]")
#    for i, var in enumerate(variables):
#        temp = all_time_results[:, :, i]
#        fig, ax = plt.subplots()
#        for i in range(Nspm):
#            ax.plot(temp[:, i])
#        plt.title(var)

    ecm.plot_phase_data(project, 'pore.temperature')
    print("*" * 30)
    print("ECM Sim time", time.time() - st)
    print("*" * 30)
