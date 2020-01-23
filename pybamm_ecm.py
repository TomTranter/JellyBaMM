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
import openpnm.topotools as tt


plt.close("all")
# set logging level
pybamm.set_logging_level("INFO")
wrk = op.Workspace()
wrk.clear()

if __name__ == "__main__":
    parallel = True
    Nlayers = 3
    layer_spacing = 195e-6
    dtheta = 20
    length_3d = 0.065
    Narc = np.int(360 / dtheta)  # number of nodes in a wind/layer
    Nsteps = 600  # number of time steps
    max_workers = int(os.cpu_count() / 2)
    I_app = 1.0  # A
    model_name = 'blah'
    opt = {'domain': 'tomo',
           'Nlayers': Nlayers,
           'cp': 1399.0,
           'rho': 2055.0,
           'K0': 1.0,
           'T0': 303,
           'heat_transfer_coefficient': 5,
           'length_3d': length_3d,
           'I_app': I_app,
           'cc_cond_neg': 1e7,
           'cc_cond_pos': 1e7,
           'dtheta': dtheta,
           'spacing': 1e-5,
           'model_name': model_name}
    ###########################################################################
    if opt['domain'] == 'model':
        project, arc_edges = ecm.make_spiral_net(Nlayers, dtheta,
                                                 spacing=layer_spacing,
                                                 length_3d=length_3d)
    else:
        project, arc_edges = ecm.make_tomo_net(dtheta,
                                               spacing=layer_spacing,
                                               length_3d=length_3d)
        
#    project, arc_edges = ecm.make_spiral_net(Nlayers, dtheta,
#                                             spacing=layer_spacing,
#                                             pos_tabs=[0], neg_tabs=[-1])
    net = project.network
    # The jellyroll layers are double sided around the cc except for the inner
    # and outer layers the number of spm models is the number of throat
    # connections between cc layers
    Nspm = net.num_throats('spm_resistor')
    res_Ts = net.throats("spm_resistor")
    electrode_heights = net['throat.electrode_height'][res_Ts]
    typical_height = np.mean(electrode_heights)
    #############################################
#    electrode_heights.fill(typical_height)
    #############################################    
    I_typical = I_app / Nspm
    temp_inputs = {"Current": I_typical,
                   'Electrode height [m]': typical_height}
    lens = arc_edges[1:] - arc_edges[:-1]
    l_norm = lens/lens[-1]
    total_length = arc_edges[-1]  # m
    print('Total cc length', total_length)
    print('Total pore volume', np.sum(net['pore.volume']))
    print('Mean throat area', np.mean(net['throat.area']))
    print('Num throats', net.num_throats())
    print('Num throats SPM', Nspm)
    print('Num throats pos_cc', net.num_throats('pos_cc'))
    print('Num throats neg_cc', net.num_throats('neg_cc'))
    print('Typical height', typical_height)
    print('Typical current', I_typical)
    ###########################################################################
    spm_sim = ecm.make_spm(I_typical)
#    height = spm_sim.parameter_values["Electrode height [m]"]
    height = electrode_heights.min()
    width = spm_sim.parameter_values["Electrode width [m]"]
    t1 = spm_sim.parameter_values['Negative electrode thickness [m]']
    t2 = spm_sim.parameter_values['Positive electrode thickness [m]']
    t3 = spm_sim.parameter_values['Negative current collector thickness [m]']
    t4 = spm_sim.parameter_values['Positive current collector thickness [m]']
    t5 = spm_sim.parameter_values['Separator thickness [m]']
    ttot = t1+t2+t3+t4+t5
    A_cc = height * width
    bat_vol = height * width * ttot * Nspm
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
    spm_sol = ecm.step_spm((spm_sim, None, I_typical, typical_height, 1e-6,
                            opt['T0'], False))
    # Create dictionaries of evaluator functions from the discretized model
    variables_eval = {}
    overpotentials_eval = {}

    for var in variables:
        variables_eval[var] = ep(spm_sim.built_model.variables[var])
    for var in overpotentials:
        overpotentials_eval[var] = ep(spm_sim.built_model.variables[var])


    temp = ecm.evaluate_python(variables_eval, spm_sol, inputs=temp_inputs)
    R = temp[0] / A_cc
    V_ecm = temp[1]
    print(R)
    R_max = R * 1e6
    # Initialize with a guess for the terminal voltage
    alg = ecm.setup_ecm_alg(project, layer_spacing, R, opt['cc_cond_neg'])
    phys = project.physics()['phys_01']
    phase = project.phases()['phase_01']
    (V_local_pnm, I_local_pnm, R_local_pnm) = ecm.run_ecm(net, alg, V_ecm)
    print("*" * 30)
    print("V local pnm", V_local_pnm, "[V]")
    print("I local pnm", I_local_pnm, "[A]")
    print("R local pnm", R_local_pnm, "[Ohm]")
    spm_models = [
        ecm.make_spm(I_typical) for i in range(Nspm)
    ]
    solutions = [
        None for i in range(Nspm)
    ]


    terminal_voltages = np.ones(Nsteps)*np.nan
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
    tau_typical = tau.evaluate(u=temp_inputs)
    t_end = 1.0*3600 / tau_typical
    t_eval_ecm = np.linspace(0, t_end, Nsteps)
    dt = t_end / (Nsteps - 1)
    tau_spm = []
    for i in range(Nspm):
        temp_tau = spm_models[i].parameter_values.process_symbol(sym_tau)
        tau_input = {'Electrode height [m]': electrode_heights[i]}
        tau_spm.append(temp_tau.evaluate(u=tau_input))
    tau_spm = np.asarray(tau_spm)
    dim_time_step = ecm.convert_time(spm_sim.parameter_values,
                                     dt, to='seconds', inputs=temp_inputs)
    dead = np.zeros(Nspm, dtype=bool)
    if parallel:
        pool = ecm.setup_pool(max_workers, pool_type='Process')
    outer_step = 0
    print(project)
    ecm.setup_thermal(project, opt)
    print(project)
    spm_temperature = np.ones(len(res_Ts))*opt['T0']
    while np.any(~dead) and outer_step < Nsteps and V_test < 0.3:
        print("*" * 30)
        print("Outer", outer_step)
        # Find terminal voltage that satisfy ecm total currents for R
        current_match = False
        max_inner_steps = 1000
        inner_step = 0
        damping = 0.66
        # Iterate the ecm until the currents match
        t_ecm_start = time.time()
        while (inner_step < max_inner_steps) and (not current_match):
#            print(inner_step, V_test)
            (V_local_pnm, I_local_pnm, R_local_pnm) = ecm.run_ecm(net,
                                                                  alg,
                                                                  V_test)
            tot_I_local_pnm = np.sum(I_local_pnm)
            diff = (I_app - tot_I_local_pnm) / I_app
            if np.absolute(diff) < tol:
                current_match = True
            else:
                V_test *= 1 + (diff * damping)
            inner_step += 1

        print("N inner", inner_step, 'time per step',
              (time.time()-t_ecm_start)/inner_step)
        all_time_I_local[outer_step, :] = I_local_pnm
        terminal_voltages[outer_step] = V_test
        # I_local_pnm should now sum to match the total applied current
        # Run the spms for the the new I_locals for the next time interval
        time_steps = np.ones(Nspm) * dt * (tau_typical/tau_spm)
        bundle_inputs = zip(spm_models, solutions, I_local_pnm, electrode_heights,
                            time_steps, spm_temperature, dead)
        if parallel:
            solutions = ecm.pool_spm(
                    bundle_inputs,
                    pool,
                    max_workers
            )
        else:
            solutions = ecm.serial_spm(
                bundle_inputs
            )
        # Gather the results for this time step
        results = np.ones([Nspm, len(variables)])*np.nan
        results_o = np.ones([Nspm, len(overpotentials)])*np.nan
        for i in range(Nspm):
            if solutions[i].termination != 'final time':
                dead[i] = True
        for i, solution in enumerate(solutions):
            if not dead[i]:
                temp_inputs = {"Current": I_local_pnm[i],
                               'Electrode height [m]': electrode_heights[i]}
                results[i, :] = ecm.evaluate_python(variables_eval,
                                                    solution,
                                                    temp_inputs)
                results_o[i, :] = ecm.evaluate_python(overpotentials_eval,
                                                      solution,
                                                      temp_inputs)
        all_time_results[outer_step, :, :] = results
        all_time_overpotentials[outer_step, :, :] = results_o

        # Collate the results for last time step
#        t, y = ecm.collect_solutions(solutions)
#        for i, func in enumerate(variables_eval.values()):
#            print(i)
#            temp = func.evaluate(t, y, u={'Current': I_local_pnm})
#            all_time_results[outer_step, :, i] = temp
#        for i, func in enumerate(overpotentials_eval.values()):
#            print(i)
#            temp = func.evaluate(t, y, u={'Current': I_local_pnm})
#            all_time_overpotentials[outer_step, :, i] = temp

        temp_local_V = all_time_results[outer_step, :, 3]
        # Apply Heat Sources
        # To Do: make this better
        Q = all_time_results[outer_step, :, 5] / (opt['cp'] * opt['rho'])
#        Q = np.ones(Nspm)*25000 / (opt['cp'] * opt['rho'])
        Q[np.isnan(Q)] = 0.0
        ecm.apply_heat_source(project, Q)
        # Calculate Global Temperature
        ecm.run_step_transient(project, dim_time_step, opt['T0'])
        # Interpolate the node temperatures for the SPMs
        spm_temperature = phase.interpolate_data('pore.temperature')[res_Ts]
        # Get new equivalent resistances
        temp_R = ecm.calc_R_new(all_time_overpotentials[outer_step, :, :], I_local_pnm)
        # stop simulation if any local voltage below the minimum
        # To do: check validity of using local
#        if np.any(temp_local_V < 3.5):
#            dead.fill(np.nan)
        # Update ecm conductivities for the spm_resistor throats
        sig = 1 / temp_R
        if np.any(temp_R > R_max):
            print('Max R found')
            print(I_local_pnm[temp_R > R_max])
            dead[temp_R > R_max] = True
            sig[temp_R > R_max] = 1/R_max
        if np.any(np.isnan(temp_R)):
            print('Nans found')
            print(I_local_pnm[np.isnan(temp_R)])
            dead[np.isnan(temp_R)] = True
            sig[np.isnan(temp_R)] = 1/R_max
        phys["throat.electrical_conductance"][res_Ts] = sig
        local_R[:, outer_step] = temp_R

        print("N Dead", np.sum(dead))
        if np.any(dead):
            fig = tt.plot_connections(net, res_Ts[dead], c='r')
            fig = tt.plot_connections(net, res_Ts[~dead], c='g', fig=fig)
            plt.title('Dead SPM: step '+str(outer_step))
        outer_step += 1

    ecm.run_ecm(net, alg, V_test, plot=True)

    all_time_results = all_time_results[:outer_step, :, :]
    if parallel:
        ecm.shutdown_pool(pool)
    fig, ax = plt.subplots()
    for i in range(Nspm):
        ax.plot(1/local_R[i, :outer_step])
    plt.title("Sigma Local [S]")
    fig, ax = plt.subplots()
    for i in range(Nspm):
        ax.plot(all_time_I_local[:outer_step, i]/electrode_heights[i])
    plt.title("I Local [A.m-1]")
    for i, var in enumerate(variables):
        temp = all_time_results[:, :, i]
        fig, ax = plt.subplots()
        for i in range(Nspm):
            ax.plot(temp[:, i])
        plt.title(var)

    ecm.plot_phase_data(project, 'pore.temperature')
    print("*" * 30)
    print("ECM Sim time", time.time() - st)
    print("*" * 30)
