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
import scipy.interpolate as interp
import sys

plt.close("all")
# set logging level
pybamm.set_logging_level("INFO")


if __name__ == "__main__":
    parallel = False
    Nunit = 10
    Nsteps = 30
    max_workers = int(os.cpu_count() / 2)
    #    max_workers = 5
    I_app = 0.25
    total_length = 0.2
    spm_sim = ecm.make_spm(Nunit, I_app, total_length)
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
    ]
    overpotentials = [
        "X-averaged reaction overpotential [V]",
        "X-averaged concentration overpotential [V]",
        "X-averaged electrolyte ohmic losses [V]",
        "X-averaged solid phase ohmic losses [V]",
        "X-averaged battery open circuit voltage [V]",
    ]
#    pool_vars = [variables for i in range(Nunit)]
    spm_sol = ecm.step_spm((spm_sim, None, I_app / Nunit, 1e-6,  False))
    variables_eval = {}
    overpotentials_eval = {}
    for var in variables:
        variables_eval[var] = pybamm.EvaluatorPython(spm_sim.built_model.variables[var])
    for var in overpotentials:
        overpotentials_eval[var] = pybamm.EvaluatorPython(spm_sim.built_model.variables[var])
    
    temp = ecm.evaluate_python(variables_eval, spm_sol, current=I_app/Nunit)
    R = temp[0] / A_cc
    V_ecm = temp[1]
    print(R)
    R_max = R * 10
    net, alg, phase = ecm.make_net(spm_sim, Nunit, R, spacing=height)
    (V_local_pnm, I_local_pnm, R_local_pnm) = ecm.run_ecm(net, alg, V_ecm)
    print("*" * 30)
    print("V local pnm", V_local_pnm, "[V]")
    print("I local pnm", I_local_pnm, "[A]")
    print("R local pnm", R_local_pnm, "[Ohm]")
    spm_models = [
        spm_sim for i in range(Nunit)
    ]
    solutions = [
        spm_sol for i in range(Nunit)
    ]
#    spm_models = [
#        copy.deepcopy(spm_sim) for i in range(len(net.throats("spm_resistor")))
#    ]

    res_Ts = net.throats("spm_resistor")
    terminal_voltages = np.zeros(Nsteps)
    V_test = V_ecm
    tol = 1e-5
    local_R = np.zeros([Nunit, Nsteps])
    stop_R = np.zeros(Nunit, dtype=bool)
    st = time.time()
    all_time_results = np.zeros([Nsteps, Nunit, len(variables)])
    all_time_overpotentials = np.zeros([Nsteps, Nunit, len(overpotentials)])
#    all_time_R = np.zeros([Nsteps, Nunit])
    all_time_I_local = np.zeros([Nsteps, Nunit])
    param = spm_sim.parameter_values
    tau = param.process_symbol(pybamm.standard_parameters_lithium_ion.tau_discharge)
    t_end = 3600 / tau.evaluate(0)
#    t_end = 0.15776182
    t_eval_ecm = np.linspace(0, t_end, Nsteps)
    dt = t_end / (Nsteps - 1)
    dead = np.zeros(Nunit, dtype=bool)
    if parallel:
        pool = ecm.setup_pool(max_workers)
    outer_step = 0

    while np.any(~dead) and outer_step < Nsteps:
        #    for outer_step in range(Nsteps):
        print("*" * 30)
        print("Outer", outer_step)
        # Find terminal voltage that satisfy ecm total currents for R
        current_match = False
        max_inner_steps = 1000
        inner_step = 0
        damping = Nunit / 10
        while (inner_step < max_inner_steps) and (not current_match):
            (V_local_pnm, I_local_pnm, R_local_pnm) = ecm.run_ecm(net, alg, V_test)
            tot_I_local_pnm = np.sum(I_local_pnm)
            diff = (I_app - tot_I_local_pnm) / I_app
            if np.absolute(diff) < tol:
                current_match = True
            else:
                V_test *= 1 + (diff / damping)
            inner_step += 1
            print("Inner", inner_step, diff, V_test)
        print("N inner", inner_step)
        all_time_I_local[outer_step, :] = I_local_pnm
        terminal_voltages[outer_step] = V_test
        # I_local_pnm should now match the total applied current
        # Run the spms for the the new I_locals
        if parallel:
            solutions = ecm.pool_spm(
                zip(spm_models, solutions, I_local_pnm, np.ones(Nunit) * dt, dead), pool
            )
        else:
            solutions = ecm.serial_spm(
                zip(spm_models, solutions, I_local_pnm, np.ones(Nunit) * dt, dead)
            )
#        data = np.asarray(data)
#        spm_models = data[:, 0].tolist()
#        solutions = data[:, 0].tolist()
        results = np.zeros([Nunit, len(variables)])
        results_o = np.zeros([Nunit, len(overpotentials)])
        for i, solution in enumerate(solutions):
            results[i, :] = ecm.evaluate_python(variables_eval, solution, I_local_pnm[i])
            results_o[i, :] = ecm.evaluate_python(overpotentials_eval, solution, current=I_local_pnm[i])
#        temp = data[:, 1]
#        results = np.zeros([Nunit, len(variables)])
#        for i in range(Nunit):
#            results[i, :] = temp[i]
        all_time_results[outer_step, :, :] = results
        all_time_overpotentials[outer_step, :, :] = results_o
        #        temp_R = results[:, 0] / A_cc
        #        temp_local_OCV = results[:, 2]
        #        temp_local_dOCV = V_ocv_0 - temp_local_OCV
        #        temp_local_ROCV = temp_local_dOCV/I_local_pnm
        temp_local_V = results[:, 3]
        #        temp_R += temp_local_ROCV
        #        temp_R /= A_cc
#        temp_R = results[:, -1]
        temp_R = ecm.calc_R_new(results_o, I_local_pnm)
#        all_time_R[outer_step, :] = temp_R
        if np.any(temp_local_V < 3.5):
            dead.fill(np.nan)
        sig = 1 / temp_R
        if np.any(temp_R > R_max):
            dead[temp_R > R_max] = True
            sig[temp_R > R_max] = 1e-6
        if np.any(np.isnan(temp_R)):
            dead[np.isnan(temp_R)] = True
            sig[np.isnan(temp_R)] = 1e-6
        phase["throat.electrical_conductance"][res_Ts] = sig
        local_R[:, outer_step] = temp_R
        print("Resistances", temp_R)
        #        print('OCV Resistances', temp_local_ROCV)
        print("Dead", dead)
        outer_step += 1

    ecm.run_ecm(net, alg, V_test, plot=True)

    all_time_results = all_time_results[:outer_step, :, :]
    if parallel:
        ecm.shutdown_pool(pool)
    fig, ax = plt.subplots()
    for i in range(Nunit):
        ax.plot(local_R[i, :outer_step])
    plt.title("R Local [Ohm]")
    fig, ax = plt.subplots()
    for i in range(Nunit):
        ax.plot(all_time_I_local[:outer_step, i])
    plt.title("I Local [A]")
    for i, var in enumerate(variables):
        temp = all_time_results[:, :, i]
        fig, ax = plt.subplots()
        for i in range(Nunit):
            ax.plot(temp[:, i])
        plt.title(var)

    print("*" * 30)
    print("ECM Sim time", time.time() - st)
    print("*" * 30)


    model_1p1D, param_1p1D, solution_1p1D, mesh_1p1D, t_eval_1p1D, I_local_1p1D = ecm.spm_1p1D(Nunit, Nsteps, I_app, total_length)
    print('ECM t_eval', t_eval_ecm)
    print('1+1D t_eval', t_eval_1p1D)
    print('ECM hours', ecm.convert_time(param, t_eval_ecm, to='hours'))
    print('1+1D hours', ecm.convert_time(param_1p1D, t_eval_1p1D, to='hours'))

    plt.figure()
    z = mesh_1p1D["current collector"][0].nodes
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = np.asarray(prop_cycle.by_key()['color'])
    for ind, t in enumerate(t_eval_ecm[:outer_step]):
        c = np.roll(colors, -ind)[0]
        plt.plot(z, all_time_I_local[ind, :].T, "o", color=c)
        plt.plot(z, I_local_1p1D[ind, :].T, "--", color=c)
    plt.title("i_local. ECM vs pybamm")


#    J_compare = all_time_I_local / A_cc
#
#    variable_name = "Current collector current density [A.m-2]"
#
#    def myinterp(t):
#        return interp.interp1d(t_eval_ecm, J_compare.T)(t)[:, np.newaxis]
#
#    # Need to append ECM to name otherwise quickplot gets confused...
#    i_local = pybamm.Function(myinterp, pybamm.t, name=variable_name + "_ECM")
#    # Set domain to be the same as the pybamm variable
#    i_local.domain = "current collector"
#
#    # Make ECM pybamm model
#    ECM_model = pybamm.BaseModel(name="ECM model")
#    ECM_model.variables = {"Current collector current density [A.m-2]": i_local}
#    processed_i_local = pybamm.ProcessedVariable(
#        ECM_model.variables["Current collector current density [A.m-2]"],
#        solution_1p1D.t,
#        solution_1p1D.y,
#        mesh=mesh_1p1D,
#    )
#    processed_i_1p1D = pybamm.ProcessedVariable(
#        model_1p1D.variables["Current collector current density [A.m-2]"],
#        solution_1p1D.t,
#        solution_1p1D.y,
#        mesh=mesh_1p1D
#    )
#    plot = pybamm.QuickPlot(
#        [model_1p1D, ECM_model],
#        mesh_1p1D,
#        [solution_1p1D, solution_1p1D],
#        output_variables=ECM_model.variables.keys(),
#    )
#    plot.dynamic_plot()
