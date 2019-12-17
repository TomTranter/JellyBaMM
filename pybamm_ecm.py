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

plt.close("all")
# set logging level
pybamm.set_logging_level("INFO")


if __name__ == "__main__":
    parallel = False
    Nunit = 10
    max_workers = int(os.cpu_count() / 2)
    #    max_workers = 5
    I_app = 1.0
    spm_sim = ecm.make_spm(Nunit)
    height = spm_sim.parameter_values["Electrode height [m]"]
    width = spm_sim.parameter_values["Electrode width [m]"]
    A_cc = height * width
    variables = [
        "Local ECM resistance [Ohm.m2]",
        "Local ECM voltage [V]",
        "Measured open circuit voltage [V]",
        "Local voltage [V]",
#        "X-averaged positive particle surface concentration [mol.m-3]",
#        "X-averaged negative particle surface concentration [mol.m-3]",
#        "X-averaged positive electrode open circuit potential [V]",
#        "X-averaged negative electrode open circuit potential [V]",
#        "Terminal voltage [V]",
        
    ]
    pool_vars = [variables for i in range(Nunit)]
    spm_sim, results = ecm.step_spm((spm_sim, I_app / Nunit, 1e-6, variables, False))
    R = results[0] / A_cc
    V_ecm = results[1]
    print(R)
    R_max = R * 10
    net, alg, phase = ecm.make_net(spm_sim, Nunit, R, spacing=height)
    (V_local_pnm, I_local_pnm, R_local_pnm) = ecm.run_ecm(net, alg, V_ecm)
    print("*" * 30)
    print("V local pnm", V_local_pnm, "[V]")
    print("I local pnm", I_local_pnm, "[A]")
    print("R local pnm", R_local_pnm, "[Ohm]")
    spm_models = [
        copy.deepcopy(spm_sim) for i in range(len(net.throats("spm_resistor")))
    ]
    Nsteps = 40
    res_Ts = net.throats("spm_resistor")
    terminal_voltages = np.zeros(Nsteps)
    V_test = V_ecm
    tol = 1e-5
    local_R = np.zeros([Nunit, Nsteps])
    stop_R = np.zeros(Nunit, dtype=bool)
    st = time.time()
    all_time_results = np.zeros([Nsteps, Nunit, len(variables)+1])
    all_time_I_local = np.zeros([Nsteps, Nunit])
    param = spm_sim.parameter_values
    tau = param.process_symbol(pybamm.standard_parameters_lithium_ion.tau_discharge)
    t_end = 3600 / tau.evaluate(0)
    t_eval = np.linspace(0, t_end, Nsteps)
    dt = t_end / (Nsteps - 1)
    dead = np.zeros(Nunit, dtype=bool)
    if parallel:
        pool = ecm.setup_pool(max_workers)
    outer_step = 0
    V_ocv_0 = 3.845
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
            data = ecm.pool_spm(
                zip(spm_models, I_local_pnm, np.ones(Nunit) * dt, pool_vars, dead), pool
            )
        else:
            data = ecm.serial_spm(
                zip(spm_models, I_local_pnm, np.ones(Nunit) * dt, pool_vars, dead)
            )
        data = np.asarray(data)
        spm_models = data[:, 0].tolist()
        temp = data[:, 1]
        results = np.zeros([Nunit, len(variables)+1])
        for i in range(Nunit):
            results[i, :] = temp[i]
        all_time_results[outer_step, :, :] = results
#        temp_R = results[:, 0] / A_cc
#        temp_local_OCV = results[:, 2]
#        temp_local_dOCV = V_ocv_0 - temp_local_OCV 
#        temp_local_ROCV = temp_local_dOCV/I_local_pnm
        temp_local_V = results[:, 3]
#        temp_R += temp_local_ROCV
#        temp_R /= A_cc
        temp_R = results[:, -1]
        if np.any(temp_local_V < 2.5):
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
    print("Sim time", time.time() - st)
    print("*" * 30)

def compare_models(all_time_I_local, A_cc, Nunit, t_eval, Nsteps):

    # load (1+1D) SPMe model
    options = {
        "current collector": "potential pair",
        "dimensionality": 1,
#        "thermal": "x-lumped",
    }
    model = pybamm.lithium_ion.SPM(options)
    model.use_simplify = False
    # create geometry
    geometry = model.default_geometry

    # load parameter values and process model and geometry
    param = model.default_parameter_values
    C_rate = 1
    current_1C = 24 * param.process_symbol(pybamm.geometric_parameters.A_cc).evaluate()
    param.update(
        {
            "Typical current [A]": C_rate * current_1C,
            "Initial temperature [K]": 298.15,
            "Negative current collector conductivity [S.m-1]": 1e5,
            "Positive current collector conductivity [S.m-1]": 1e5,
#            "Heat transfer coefficient [W.m-2.K-1]": 1,
        }
    )
    param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    var = pybamm.standard_spatial_vars
    var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.r_n: 10, var.r_p: 10, var.z: Nunit}
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

    # solve model -- simulate one hour discharge
    tau = param.process_symbol(pybamm.standard_parameters_lithium_ion.tau_discharge)
    t_end = 3600 / tau.evaluate(0)
    # solution = model.default_solver.solve(model, t_eval)
    solution = pybamm.CasadiSolver(mode="fast").solve(model, np.linspace(0, t_end, Nsteps))
    # e.g. make model with variable for I_local
    # t_eval
    # all_time_I_local
    # Set name to be same as the pybamm variable
    variable_name = "Current collector current density [A.m-2]"

    def myinterp(t):
        return interp.interp1d(
            t_eval, all_time_I_local / A_cc, axis=0
        )(t)[:, np.newaxis]

    # Use dimensional time. Need to append ECM to name otherwise quickplot gets confused...
    i_local = pybamm.Function(myinterp, pybamm.t * tau, name=variable_name + "_ECM")
    # Set domain to be the same as the pybamm variable
    i_local.domain = "current collector"

    # Make ECM pybamm model
    ECM_model = pybamm.BaseModel(name="ECM model")
    ECM_model.variables = {"Current collector current density [A.m-2]": i_local}

#    # plot
#    plot = pybamm.QuickPlot(
#        model,
#        mesh,
#        solution, 
#        output_variables=["Current collector current density [A.m-2]"],
#    )
    plot = pybamm.QuickPlot(
        [model, ECM_model],
        mesh,
        [solution, solution],
        output_variables=ECM_model.variables.keys(),
    )
    plot.dynamic_plot()

compare_models(all_time_I_local, A_cc, Nunit, t_eval, Nsteps)


#model = spm_models[0].built_model
#all_variables = model.variables
#plot_voltage_components(all_variables, t_eval, model)