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
from scipy.interpolate import griddata


plt.close("all")
# set logging level
pybamm.set_logging_level("INFO")
wrk = op.Workspace()
wrk.clear()
save = False
plot = False
animate = True

if __name__ == "__main__":
    parallel = False
    Nlayers = 19
    hours = 0.05
    layer_spacing = 195e-6
    dtheta = 10
    length_3d = 0.065
    pixel_size = 10.4e-6
    max_workers = int(os.cpu_count() / 2)
    I_app = 1.0  # A
    Nsteps = np.int(hours*60*I_app)  # number of time steps
    V_over_max = 1.5
    model_name = 'blah'
    opt = {'domain': 'tomo',
           'Nlayers': Nlayers,
           'cp': 1399.0,
           'rho': 2055.0,
           'K0': 1.0,
           'T0': 298.15,
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
    # This is dodgy - figure out later - might need to initiate each spm with different typical current!!!
    # Would be better to specify current density
    electrode_heights.fill(typical_height)
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
    spm_sim = ecm.make_spm(I_typical, thermal=True,
                           length_3d=length_3d, pixel_size=pixel_size)
#    height = electrode_heights.min()
    width = spm_sim.parameter_values["Electrode width [m]"]
    t1 = spm_sim.parameter_values['Negative electrode thickness [m]']
    t2 = spm_sim.parameter_values['Positive electrode thickness [m]']
    t3 = spm_sim.parameter_values['Negative current collector thickness [m]']
    t4 = spm_sim.parameter_values['Positive current collector thickness [m]']
    t5 = spm_sim.parameter_values['Separator thickness [m]']
    ttot = t1+t2+t3+t4+t5
    A_cc = electrode_heights * width
    bat_vol = np.sum(A_cc*ttot)
    print('BATTERY ELECTRODE VOLUME', bat_vol)
    print('18650 VOLUME', 0.065*np.pi*((8.75e-3)**2-(2.0e-3)**2 ))
    ###########################################################################
    result_template = np.ones([Nsteps, Nspm])
    result_template.fill(np.nan)
    variables = {
        "Negative electrode average extent of lithiation": result_template.copy(),
        "Positive electrode average extent of lithiation": result_template.copy(),
        "X-averaged negative particle surface concentration [mol.m-3]": result_template.copy(), 
        "X-averaged positive particle surface concentration [mol.m-3]": result_template.copy(),
#        "Local ECM resistance [Ohm.m2]": result_template.copy(),
#        "Local ECM voltage [V]": result_template.copy(),
#        "Measured open circuit voltage [V]": result_template.copy(),
        "Terminal voltage [V]": result_template.copy(),
#        "Change in measured open circuit voltage [V]": result_template.copy(),
        "X-averaged total heating [W.m-3]": result_template.copy(),
        "Time [h]": result_template.copy(),
        "Current collector current density [A.m-2]": result_template.copy()
    }
    overpotentials = {
        "X-averaged reaction overpotential [V]": result_template.copy(),
        "X-averaged concentration overpotential [V]": result_template.copy(),
        "X-averaged electrolyte ohmic losses [V]": result_template.copy(),
        "X-averaged solid phase ohmic losses [V]": result_template.copy(),
        "Change in measured open circuit voltage [V]": result_template.copy(),
    }
    param = spm_sim.parameter_values
    temp_parms = spm_sim.built_model.submodels["thermal"].param
    Delta_T = param.process_symbol(temp_parms.Delta_T).evaluate(u=temp_inputs)
    Delta_T_spm = Delta_T * (typical_height/electrode_heights)
    T_ref = param.process_symbol(temp_parms.T_ref).evaluate()
    T_non_dim = (opt['T0'] - T_ref) / Delta_T
    spm_sol = ecm.step_spm((spm_sim.built_model,
                            spm_sim.solver,
                            None, I_typical, typical_height, 1e-6,
                            T_non_dim, False))
    # Create dictionaries of evaluator functions from the discretized model
    variables_eval = {}
    overpotentials_eval = {}
    
    for var in variables.keys():
        variables_eval[var] = ep(spm_sim.built_model.variables[var])
    for var in overpotentials.keys():
        overpotentials_eval[var] = ep(spm_sim.built_model.variables[var])
    variable_keys = list(variables.keys())
    overpotential_keys = list(overpotentials.keys())
    
#    temp = ecm.evaluate_python(variables_eval, spm_sol, inputs=temp_inputs)
    temp = 0.0
    for j, key in enumerate(overpotential_keys):
        temp -= overpotentials_eval[key].evaluate(
                    spm_sol.t[-1], spm_sol.y[:, -1],
                    u=temp_inputs
                )
    R = temp / I_typical
    V_ecm = temp.flatten()
    print(R)
    R_max = R[0] * 1e6
    # Initialize with a guess for the terminal voltage
    alg = ecm.setup_ecm_alg(project, layer_spacing, R, opt['cc_cond_neg'])
    phys = project.physics()['phys_01']
    phase = project.phases()['phase_01']
    (V_local_pnm, I_local_pnm, R_local_pnm) = ecm.run_ecm(net, alg, V_ecm)
    print("*" * 30)
    print("V local pnm", V_local_pnm, "[V]")
    print("I local pnm", I_local_pnm, "[A]")
    print("R local pnm", R_local_pnm, "[Ohm]")

    spm_models = [spm_sim.built_model for i in range(Nspm)]
    spm_solvers = [spm_sim.solver for i in range(Nspm)]
    spm_params = [spm_sim.parameter_values for i in range(Nspm)]

    solutions = [
        spm_sol for i in range(Nspm)
    ]
    saved_sols = [
        None for i in range(Nspm)
    ]
    terminal_voltages = np.ones(Nsteps)*np.nan
    V_test = V_ecm
    tol = 1e-5
    local_R = np.zeros([Nspm, Nsteps])
    st = time.time()
    
    all_time_I_local = np.zeros([Nsteps, Nspm])
    all_time_temperature = np.zeros([Nsteps, Nspm])

    sym_tau = pybamm.standard_parameters_lithium_ion.tau_discharge
    tau = param.process_symbol(sym_tau)
    tau_typical = tau.evaluate(u=temp_inputs)
    t_end = hours*3600 / tau_typical
    t_eval_ecm = np.linspace(0, t_end, Nsteps)
    dt = t_end / (Nsteps - 1)
    tau_spm = []
    for i in range(Nspm):
        temp_tau = spm_params[i].process_symbol(sym_tau)
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
    T_non_dim_spm = np.ones(len(res_Ts))*T_non_dim
    max_temperatures = []
    sorted_res_Ts = net['throat.spm_resistor_order'][res_Ts].argsort()
    while np.any(~dead) and outer_step < Nsteps and V_test < V_over_max:
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
            print(I_app, inner_step, V_test, tot_I_local_pnm)
        if V_test < V_over_max:
            print("N inner", inner_step, 'time per step',
                  (time.time()-t_ecm_start)/inner_step)
            all_time_I_local[outer_step, :] = I_local_pnm
            terminal_voltages[outer_step] = V_test
            # I_local_pnm should now sum to match the total applied current
            # Run the spms for the the new I_locals for the next time interval
            time_steps = np.ones(Nspm) * dt * (tau_typical/tau_spm)
            bundle_inputs = zip(spm_models, spm_solvers,
                                solutions, I_local_pnm, electrode_heights,
                                time_steps, T_non_dim_spm, dead)
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
            for i in range(Nspm):
                if saved_sols[i] is None:
                    saved_sols[i] = solutions[i]
                else:
                    saved_sols[i].append(solutions[i])
            # Gather the results for this time step
            results_o = np.ones([Nspm, len(overpotential_keys)])*np.nan
            for si, i in enumerate(sorted_res_Ts):
                if solutions[i].termination != 'final time':
                    dead[i] = True
                else:
                    temp_inputs = {"Current": I_local_pnm[i],
                                   'Electrode height [m]': electrode_heights[i]}
                    for key in variable_keys:
                        temp = saved_sols[i][key](saved_sols[i].t[-1])
                        variables[key][outer_step, si] = temp
                    for j, key in enumerate(overpotential_keys):
                        temp = overpotentials_eval[key].evaluate(
                                solutions[i].t[-1], solutions[i].y[:, -1],
                                u=temp_inputs
                                )
                        overpotentials[key][outer_step, si] = temp
                        results_o[i, j] = temp

            # Apply Heat Sources
            # To Do: make this better
            Q = variables["X-averaged total heating [W.m-3]"][outer_step, :]
            Q = Q / (opt['cp'] * opt['rho'])
            Q[np.isnan(Q)] = 0.0
            ecm.apply_heat_source(project, Q)
            # Calculate Global Temperature
            ecm.run_step_transient(project, dim_time_step, opt['T0'])
            # Interpolate the node temperatures for the SPMs
            spm_temperature = phase.interpolate_data('pore.temperature')[res_Ts]
            all_time_temperature[outer_step, :] = spm_temperature
            max_temperatures.append(spm_temperature.max())
            T_non_dim_spm = (spm_temperature - T_ref) / Delta_T_spm
            # Get new equivalent resistances
            temp_R = ecm.calc_R_new(results_o, I_local_pnm)
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
    
    if parallel:
        ecm.shutdown_pool(pool)

    if plot:
        variables['ECM sigma local'] = 1/local_R[sorted_res_Ts, :outer_step].T
        variables['ECM I Local'] = all_time_I_local[:outer_step, sorted_res_Ts]
        variables['ECM Temperature [K]'] = all_time_temperature[:outer_step, sorted_res_Ts]
    
        if outer_step < Nsteps:
            for key in variables.keys():
                variables[key] = variables[key][:outer_step-1, :]
    
        for key in variables.keys():
            fig, ax = plt.subplots()
            ax.plot(variables[key][:, sorted_res_Ts])
            plt.title(key)
            plt.show()
        
        ecm.plot_phase_data(project, 'pore.temperature')
    
        fig, ax = plt.subplots()
        ax.plot(max_temperatures)
        ax.set_xlabel('Discharge Time [h]')
        ax.set_ylabel('Maximum Temperature [K]')
        lower_mask = net['throat.spm_neg_inner'][res_Ts[sorted_res_Ts]]

    if save:
        save_path = 'C:\Code\pybamm_pnm_save_data_10A'
        ecm.export(project, save_path, variables, 'var_', lower_mask=lower_mask, save_animation=animate)
        ecm.export(project, save_path, overpotentials, 'eta_',lower_mask=lower_mask, save_animation=animate)
        project.export_data(phases=[phase], filename='ecm')

    if animate:
        data = variables['Current collector current density [A.m-2]']
        ecm.animate_data2(project, data, 'Current collector current density test')

    
    print("*" * 30)
    print("ECM Sim time", time.time() - st)
    print("*" * 30)
