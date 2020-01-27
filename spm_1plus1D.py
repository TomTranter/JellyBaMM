import pybamm
import numpy as np
import sys
import matplotlib.pyplot as plt
import ecm
import openpnm as op
import time
import os
from pybamm import EvaluatorPython as ep
import openpnm.topotools as tt
from copy import deepcopy
    

if __name__ == '__main__':

    plt.close('all')
    # set logging level
    #pybamm.set_logging_level("INFO")
    
    # load (1+1D) SPMe model
    wrk = op.Workspace()
    Nspm = 8
    Nsteps = 180
    parallel = True
    e_height = 2.0
    max_workers = 5
    #max_workers = int(os.cpu_count() / 2)
    options = {
        "current collector": "potential pair",
        "dimensionality": 1,
    #    "thermal": "x-lumped",
    }
    model = pybamm.lithium_ion.SPM(options)
    model.use_simplify = False
    # create geometry
    geometry = model.default_geometry
    
    # load parameter values and process model and geometry
    param = model.default_parameter_values
    #C_rate = 1
    
    #current_1C = 24 * A_cc
    I_app = 1.0
    I_typical = I_app/Nspm
    e_cond_cc = 1e7
    param.update(
        {
            "Typical current [A]": I_typical,
            "Current function [A]": I_app,
            "Initial temperature [K]": 298.15,
            "Negative current collector conductivity [S.m-1]": e_cond_cc,
            "Positive current collector conductivity [S.m-1]": e_cond_cc,
            "Heat transfer coefficient [W.m-2.K-1]": 1,
            "Electrode height [m]": e_height,
            "Negative tab centre z-coordinate [m]": 0.0,
            "Positive tab centre z-coordinate [m]": e_height,
        }
    )
    #e_height = param["Electrode height [m]"]
    e_width = param["Electrode width [m]"]
    z_edges = np.linspace(0, e_height, Nspm+1)
    A_cc = param.evaluate(pybamm.geometric_parameters.A_cc)
    
    param.process_model(model)
    param.process_geometry(geometry)
    
    #e_cond_cc = param['Negative current collector conductivity [S.m-1]']
    cc_thickness = param['Negative current collector thickness [m]']
    
    sys.setrecursionlimit(10000)
    
    var = pybamm.standard_spatial_vars
    var_pts = {
        var.x_n: 5,
        var.x_s: 5,
        var.x_p: 5,
        var.r_n: 10,
        var.r_p: 10,
        var.z: Nspm,
    }
    submesh_types = model.default_submesh_types
    pts = z_edges / z_edges[-1]
    z = (pts[:-1] + pts[1:])/2
    submesh_types["current collector"] = pybamm.MeshGenerator(
        pybamm.UserSupplied1DSubMesh, submesh_params={"edges": pts}
    )
    
    solver = pybamm.CasadiSolver(atol=1e-8, rtol=1e-8, mode='fast')
    
    solver = pybamm.CasadiSolver()
    sim = pybamm.Simulation(model=model,
                            geometry=geometry,
                            parameter_values=param,
                            submesh_types=submesh_types,
                            var_pts=var_pts,
                            spatial_methods=model.default_spatial_methods,
                            solver=solver)
    
    tau_sym = pybamm.standard_parameters_lithium_ion.tau_discharge
    tau = param.process_symbol(tau_sym).evaluate(0)
    t_end = 1800 / tau
    t_eval = np.linspace(0, t_end, Nsteps)
    
    
    sim.solve(t_eval)
    solution = sim.solution
    
    V_local = solution['Local voltage [V]'](solution.t, z=z).T
    cc_fracs = sim.mesh["current collector"][0].d_edges
    I_local = solution['Current collector current density [A.m-2]'](solution.t, z=z).T
    I_local *= cc_fracs*A_cc
    #I_exch = solution['X-averaged negative electrode interfacial current density [A.m-2]'](solution.t, z=z).T
    I_total = solution['Total current density [A.m-2]'](solution.t, z=z).T
    R_local = np.zeros_like(I_local)
    variables = [
        "Local ECM resistance [Ohm.m2]",
        "Local ECM voltage [V]",
        "Measured open circuit voltage [V]",
        "Local voltage [V]",
        "Change in measured open circuit voltage [V]",
        "Time [h]",
        "Current collector current density [A.m-2]"
    ]
    overpotentials = [
        "X-averaged reaction overpotential [V]",
    #    "X-averaged concentration overpotential [V]",
    #    "X-averaged electrolyte ohmic losses [V]",
    #    "X-averaged solid phase ohmic losses [V]",
        "Change in measured open circuit voltage [V]",
    ]
    
    
    V_eta_local = np.zeros_like(I_local)
    for eta in overpotentials:
    #    plt.figure()
        eta_local = solution[eta](solution.t, z=z).T
    #    plt.plot(eta_local)
    #    plt.title(eta)
        V_eta_local -= eta_local
    
    R_local = V_eta_local / I_local
    fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6))  = plt.subplots(3, 2, figsize=(20,15))
    ax1.plot(I_local)
    ax1.set_ylabel('Local Current')
    ax2.plot(V_eta_local)
    ax2.set_ylabel('Local Overpotential')
    ax3.plot(R_local)
    ax3.set_ylabel('Local R')
    
    print('I applied', I_app)
    print('Total current [A]', I_total*A_cc)
    print('CC Current', np.sum(I_local, axis=1))
    spacing = e_height/Nspm
    R_av = np.mean(R_local[0, :])
    R_max = R_av * 1e6
    V_ecm = np.mean(V_eta_local[0, :])
    
    
    typical_height = spacing
    temperature = 303.0
    
    spm_sim = ecm.make_spm(I_typical=I_typical, thermal=False)
    spm_models = [spm_sim.built_model for i in range(Nspm)]
    spm_solvers = [spm_sim.solver for i in range(Nspm)]
    #spm_params = [spm_sim.parameter_values for i in range(Nspm)]
    spm_sol = ecm.step_spm((spm_sim.built_model,
                            spm_sim.solver,
                            None, I_typical, typical_height, 1e-6,
                            temperature, False))
    solutions = [
        spm_sol for i in range(Nspm)
    ]
    
    
    #    spm_models = [
    #        spm_sim for i in range(Nspm)
    #    ]
    #spm_models = [
    #    ecm.make_spm(I_typical=I_typical, thermal=False) for i in range(Nspm)
    #]
    
    variables_eval = {}
    overpotentials_eval = {}
    
    for var in variables:
        variables_eval[var] = ep(spm_sim.built_model.variables[var])
    for var in overpotentials:
        overpotentials_eval[var] = ep(spm_sim.built_model.variables[var])
    
    temp_inputs = {"Current": I_typical,
                   'Electrode height [m]': typical_height}
    spm_temperature = np.zeros(Nspm) # Non-dim
    
    electrode_heights = cc_fracs*e_height
    project = ecm.make_1D_net(Nunit=Nspm, R=R_local[0, :], spacing=spacing, pos_tabs=[0], neg_tabs=[-1])
    net = project.network
    res_Ts = net.throats('spm_resistor')
    alg = ecm.setup_ecm_alg(project, spacing, R_local[0, :], e_cond_cc)
    phys = project.physics()['phys_01']
    phase = project.phases()['phase_01']
    (V_local_pnm, I_local_pnm, R_local_pnm) = ecm.run_ecm(net, alg, V_ecm)
    
    
    terminal_voltages = np.ones(Nsteps)*np.nan
    V_test = V_ecm
    tol = 1e-6
    local_R = np.zeros([Nspm, Nsteps])
    st = time.time()
    all_time_results = np.zeros([Nsteps, Nspm, len(variables)])
    all_time_overpotentials = np.zeros([Nsteps, Nspm, len(overpotentials)])
    all_time_I_local = np.zeros([Nsteps, Nspm])
    all_time_V_local = np.zeros([Nsteps, Nspm])
    
    
    tau_mini = spm_sim.parameter_values.process_symbol(tau_sym).evaluate(u=temp_inputs)
    dt = np.mean(solution.t[1:]-solution.t[:-1])
    dt *= tau/tau_mini
    
    dead = np.zeros(Nspm, dtype=bool)
    outer_step = 0
    
    if parallel:
        pool = ecm.setup_pool(max_workers, pool_type='Process')
    
    A_cc_spm = electrode_heights*e_width
    
    while np.any(~dead) and outer_step < Nsteps and V_test < 1.0:
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
    
        print("N inner", inner_step, 'time per step',
              (time.time()-t_ecm_start)/inner_step)
        all_time_I_local[outer_step, :] = I_local_pnm
        all_time_V_local[outer_step, :] = V_local_pnm
        terminal_voltages[outer_step] = V_test
        # I_local_pnm should now sum to match the total applied current
        # Run the spms for the the new I_locals for the next time interval
        bundle_inputs = zip(spm_models, spm_solvers,
                            solutions, I_local_pnm, electrode_heights,
                            np.ones(Nspm) * dt, spm_temperature, dead)
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
        for i, isol in enumerate(solutions):
            if not dead[i]:
                temp_inputs = {"Current": I_local_pnm[i],
                               'Electrode height [m]': electrode_heights[i]}
                results[i, :] = ecm.evaluate_python(variables_eval,
                                                    isol,
                                                    temp_inputs)
                results_o[i, :] = ecm.evaluate_python(overpotentials_eval,
                                                      isol,
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
    
    #    temp_local_V = all_time_results[outer_step, :, 3]
        # Apply Heat Sources
        # To Do: make this better
    #    Q = all_time_results[outer_step, :, 5] / (opt['cp'] * opt['rho'])
    #        Q = np.ones(Nspm)*25000 / (opt['cp'] * opt['rho'])
    #    Q[np.isnan(Q)] = 0.0
    #    ecm.apply_heat_source(project, Q)
        # Calculate Global Temperature
    #    ecm.run_step_transient(project, dim_time_step, opt['T0'])
        # Interpolate the node temperatures for the SPMs
    #    spm_temperature = phase.interpolate_data('pore.temperature')[res_Ts]
        # Get new equivalent resistances
        temp_R = ecm.calc_R_new(all_time_overpotentials[outer_step, :, :], I_local_pnm)
    #    temp_R = results[:, 0] / A_cc_spm
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
    #    if np.any(dead):
    #        fig = tt.plot_connections(net, res_Ts[dead], c='r')
    #        fig = tt.plot_connections(net, res_Ts[~dead], c='g', fig=fig)
    #        plt.title('Dead SPM: step '+str(outer_step))
        outer_step += 1
    
    #solutions[0][variables[0]]
    
    #ecm.run_ecm(net, alg, V_test, plot=True)
    
    all_time_results = all_time_results[:outer_step, :, :]
    if parallel:
        ecm.shutdown_pool(pool)
    
    
    ax4.plot(all_time_I_local)
    ax4.set_ylabel('Local Current')
    ax5.plot(-np.sum(all_time_overpotentials, axis=2))
    #ax5.plot(all_time_V_local)
    ax5.set_ylabel('Local Overpotential')
    ax6.plot(local_R.T)
    ax6.set_ylabel('Local R')
    
    A_cc_spm = electrode_heights*e_width
    temp = all_time_results[:, :, 0]/A_cc_spm
    fig, ax = plt.subplots()
    ax.plot(temp)
    plt.title('ECM Resistance [Ohm]')
    
    for i, var in enumerate(variables):
        temp = all_time_results[:, :, i]
        fig, ax = plt.subplots()
        for i in range(Nspm):
            ax.plot(temp[:, i])
        plt.title(var)
    
    #V_test = V_ecm
    #re_I_local = np.zeros([Nsteps, Nspm])
    #for i in range(Nsteps):
    #    R_spm = R_local[i, :]
    #    sig = 1/R_spm
    #    phys["throat.electrical_conductance"][res_Ts] = sig
    #    inner_step = 0
    #    current_match = False
    #    while (inner_step < max_inner_steps) and (not current_match):
    #
    #        (V_local_pnm, I_local_pnm, R_local_pnm) = ecm.run_ecm(net,
    #                                                              alg,
    #                                                              V_test)
    #        tot_I_local_pnm = np.sum(I_local_pnm)
    #        diff = (I_app - tot_I_local_pnm) / I_app
    #        if np.absolute(diff) < tol:
    #            current_match = True
    #        else:
    #            V_test *= 1 + (diff * damping)
    #        inner_step += 1
    #    re_I_local[i, :] = I_local_pnm

#    main()