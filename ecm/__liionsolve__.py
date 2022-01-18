#
# Liionpack solve
#
import os
import numpy as np
import ecm
import pybamm
import time
import matplotlib.pyplot as plt
import openpnm as op


wrk = op.Workspace()


def run_simulation_lp(I_app, save_path, config):
    ###########################################################################
    # Simulation information                                                  #
    ###########################################################################
    max_workers = int(os.cpu_count() / 2)
    hours = config.getfloat("RUN", "hours")
    Nsteps = np.int(hours * 60 * I_app) + 1  # number of time steps
    V_over_max = 2.0
    if config.get("GEOMETRY", "domain") == "model":
        project, arc_edges = ecm.make_spiral_net(config)
    elif config.get("GEOMETRY", "domain") == "1d":
        project, arc_edges = ecm.make_1D_net(config)
    else:
        project, arc_edges = ecm.make_tomo_net(config)

    net = project.network
    if config.get("GEOMETRY", "domain") != "1d":
        ecm.plot_topology(net)
    # The jellyroll layers are double sided around the cc except for the inner
    # and outer layers the number of spm models is the number of throat
    # connections between cc layers
    Nspm = net.num_throats("spm_resistor")
    res_Ts = net.throats("spm_resistor")
    sorted_res_Ts = net["throat.spm_resistor_order"][res_Ts].argsort()
    electrode_heights = net["throat.electrode_height"][res_Ts]
    print("Total Electrode Height", np.around(np.sum(electrode_heights), 2), "m")
    typical_height = np.mean(electrode_heights)
    I_typical = I_app / Nspm
    temp_inputs = {"Current": I_typical, "Electrode height [m]": typical_height}
    total_length = arc_edges[-1]  # m
    print("Total cc length", total_length)
    print("Total pore volume", np.sum(net["pore.volume"]))
    print("Mean throat area", np.mean(net["throat.area"]))
    print("Num throats", net.num_throats())
    print("Num throats SPM", Nspm)
    print("Num throats pos_cc", net.num_throats("pos_cc"))
    print("Num throats neg_cc", net.num_throats("neg_cc"))
    print("Typical height", typical_height)
    print("Typical current", I_typical)
    ###########################################################################
    # Make the pybamm simulation - should be moved to a simfunc               #
    ###########################################################################
    # To Do - make a parameter values object instead
    spm_sim = ecm.make_spm(I_typical, config)
    width = spm_sim.parameter_values["Electrode width [m]"]
    t1 = spm_sim.parameter_values["Negative electrode thickness [m]"]
    t2 = spm_sim.parameter_values["Positive electrode thickness [m]"]
    t3 = spm_sim.parameter_values["Negative current collector thickness [m]"]
    t4 = spm_sim.parameter_values["Positive current collector thickness [m]"]
    t5 = spm_sim.parameter_values["Separator thickness [m]"]
    ttot = t1 + t2 + t3 + t4 + t5
    A_cc = electrode_heights * width
    bat_vol = np.sum(A_cc * ttot)
    print("BATTERY ELECTRODE VOLUME", bat_vol)
    print("18650 VOLUME", 0.065 * np.pi * ((8.75e-3) ** 2 - (2.0e-3) ** 2))
    ###########################################################################
    # Output variables                                                        #
    ###########################################################################
    temp = np.ones([Nsteps, Nspm])
    temp.fill(np.nan)
    lithiations = {
        "X-averaged negative electrode extent of lithiation": temp.copy(),
        "X-averaged positive electrode extent of lithiation": temp.copy(),
    }
    variables = {
        "X-averaged negative particle surface concentration [mol.m-3]": temp.copy(),
        "X-averaged positive particle surface concentration [mol.m-3]": temp.copy(),
        "Terminal voltage [V]": temp.copy(),
        "Time [h]": temp.copy(),
        "Current collector current density [A.m-2]": temp.copy(),
    }
    overpotentials = {
        "X-averaged battery reaction overpotential [V]": temp.copy(),
        "X-averaged battery concentration overpotential [V]": temp.copy(),
        "X-averaged battery electrolyte ohmic losses [V]": temp.copy(),
        "X-averaged battery solid phase ohmic losses [V]": temp.copy(),
        "Change in measured open circuit voltage [V]": temp.copy(),
    }
    variables_heating = {
        "Volume-averaged Ohmic heating [W.m-3]": temp.copy(),
        "Volume-averaged irreversible electrochemical heating [W.m-3]": temp.copy(),
        "Volume-averaged reversible heating [W.m-3]": temp.copy(),
        "Volume-averaged total heating [W.m-3]": temp.copy(),
        "Volume-averaged Ohmic heating CC [W.m-3]": temp.copy(),
    }
    variable_keys = list(variables.keys())
    overpotential_keys = list(overpotentials.keys())
    heating_keys = list(variables_heating.keys())
    heating_keys.pop(-1)
    ###########################################################################
    # Thermal parameters                                                      #
    ###########################################################################
    param = spm_sim.parameter_values
    temp_parms = spm_sim.built_model.submodels["thermal"].param
    Delta_T = param.process_symbol(temp_parms.Delta_T).evaluate(inputs=temp_inputs)
    Delta_T_spm = Delta_T * (typical_height / electrode_heights)
    T_ref = param.process_symbol(temp_parms.T_ref).evaluate()
    T0 = config.getfloat("PHYSICS", "T0")
    lumpy_therm = ecm.lump_thermal_props(config)
    cp = lumpy_therm["lump_Cp"]
    rho = lumpy_therm["lump_rho"]
    T_non_dim = (T0 - T_ref) / Delta_T
    T_non_dim_spm = np.ones(len(res_Ts)) * T_non_dim
    ###########################################################################
    # Initial guess - no longer needed                                        #
    ###########################################################################
    spm_sol = ecm.step_spm(
        (
            spm_sim.built_model,
            spm_sim.solver,
            None,
            I_typical,
            typical_height,
            1e-6,
            T_non_dim,
            False,
        )
    )

    temp = 0.0
    for j, key in enumerate(overpotential_keys):
        temp -= spm_sol[key].entries[-1]
    R = temp / I_typical
    guess_R = R * typical_height / electrode_heights
    V_ecm = temp.flatten()
    print(R)
    R_max = R * 1e6
    # Initialize with a guess for the terminal voltage
    alg = ecm.setup_ecm_alg(project, config, guess_R)
    phys = project.physics()["phys_01"]
    phase = project.phases()["phase_01"]
    (V_local_pnm, I_local_pnm, R_local_pnm) = ecm.run_ecm(net, alg, V_ecm)
    print("*" * 30)
    print("V local pnm", V_local_pnm, "[V]")
    print("I local pnm", I_local_pnm, "[A]")
    print("R local pnm", R_local_pnm, "[Ohm]")
    ###########################################################################
    # Objects to pass to processpool - no longer needed                       #
    ###########################################################################
    spm_models = [spm_sim.built_model for i in range(Nspm)]
    spm_solvers = [pybamm.CasadiSolver() for i in range(Nspm)]
    spm_params = [spm_sim.parameter_values for i in range(Nspm)]
    solutions = [None for i in range(Nspm)]
    ###########################################################################
    # Simulation variables
    ###########################################################################
    terminal_voltages = np.ones(Nsteps) * np.nan
    V_test = V_ecm
    tol = 1e-5
    local_R = np.zeros([Nspm, Nsteps])
    st = time.time()
    all_time_I_local = np.zeros([Nsteps, Nspm])
    all_time_temperature = np.zeros([Nsteps, Nspm])
    dead = np.zeros(Nspm, dtype=bool)
    max_temperatures = []
    ###########################################################################
    # Time step variables
    ###########################################################################
    sym_tau = pybamm.LithiumIonParameters().tau_discharge
    t_end = hours * 3600
    dt = t_end / (Nsteps - 1)
    tau_spm = []
    for i in range(Nspm):
        temp_tau = spm_params[i].process_symbol(sym_tau)
        tau_input = {"Electrode height [m]": electrode_heights[i]}
        tau_spm.append(temp_tau.evaluate(inputs=tau_input))
    tau_spm = np.asarray(tau_spm)
    dim_time_step = dt
    ###########################################################################
    # Run time config                                                         #
    ###########################################################################
    if config.getboolean("RUN", "parallel"):
        pool = ecm.setup_pool(max_workers, pool_type="Process")
    outer_step = 0
    if config.getboolean("PHYSICS", "do_thermal"):
        ecm.setup_thermal(project, config)
    try:
        thermal_third = config.getboolean("RUN", "third")
    except KeyError:
        thermal_third = False
    ###########################################################################
    # Main Loop                                                               #
    ###########################################################################
    while np.any(~dead) and outer_step < Nsteps and V_test < V_over_max:
        print("*" * 30)
        print("Outer", outer_step)
        print("Elapsed Simulation Time", np.around((outer_step) * dt, 2), "s")
        # Find terminal voltage that satisfy ecm total currents for R
        current_match = False
        max_inner_steps = 1000
        inner_step = 0
        damping = 0.66
        # Iterate the ecm until the currents match
        t_ecm_start = time.time()
        while (inner_step < max_inner_steps) and (not current_match):
            (V_local_pnm, I_local_pnm, R_local_pnm) = ecm.run_ecm(net, alg, V_test)
            tot_I_local_pnm = np.sum(I_local_pnm)
            diff = (I_app - tot_I_local_pnm) / I_app
            if np.absolute(diff) < tol:
                current_match = True
            else:
                V_test *= 1 + (diff * damping)
            inner_step += 1
        ecm.get_cc_heat(net, alg, V_test)
        if V_test < V_over_max:
            print(
                "N inner",
                inner_step,
                "time per step",
                (time.time() - t_ecm_start) / inner_step,
            )
            print("Over-voltage", np.around(V_test, 2), "V")
            all_time_I_local[outer_step, :] = I_local_pnm
            terminal_voltages[outer_step] = V_test
            # I_local_pnm should now sum to match the total applied current
            # Run the spms for the the new I_locals for the next time interval
            time_steps = np.ones(Nspm) * dt
            bundle_inputs = zip(
                spm_models,
                spm_solvers,
                solutions,
                I_local_pnm,
                electrode_heights,
                time_steps,
                T_non_dim_spm,
                dead,
            )
            t_spm_start = time.time()
            if config.getboolean("RUN", "parallel"):
                solutions = ecm.pool_spm(bundle_inputs, pool, max_workers)
            else:
                solutions = ecm.serial_spm(bundle_inputs)
            print(
                "Finished stepping SPMs in ",
                np.around((time.time() - t_spm_start), 2),
                "s",
            )
            print("Solution size", solutions[0].t.shape)
            # Gather the results for this time step
            results_o = np.ones([Nspm, len(overpotential_keys)]) * np.nan
            t_eval_start = time.time()
            for si, i in enumerate(sorted_res_Ts):
                if solutions[i].termination != "final time":
                    dead[i] = True
                else:
                    temp_inputs = {
                        "Current": I_local_pnm[i],
                        "Electrode height [m]": electrode_heights[i],
                    }
                    for key in lithiations.keys():
                        temp = solutions[i][key].entries[-1]
                        lithiations[key][outer_step, si] = temp
                    for key in variable_keys:
                        # temp = solutions[i][key](solutions[i].t[-1])
                        temp = solutions[i][key].entries[-1]
                        variables[key][outer_step, si] = temp
                    for j, key in enumerate(overpotential_keys):
                        temp = solutions[i][key].entries[-1]
                        overpotentials[key][outer_step, si] = temp
                        results_o[i, j] = temp
                    if config.getboolean("PHYSICS", "do_thermal"):
                        for j, key in enumerate(heating_keys):
                            temp = solutions[i][key].entries[-1]
                            variables_heating[key][outer_step, si] = temp
            print(
                "Finished evaluating SPMs in ",
                np.around((time.time() - t_eval_start), 2),
                "s",
            )
            if config.getboolean("PHYSICS", "do_thermal"):
                # Apply Heat Sources
                # To Do: make this better
                vh = variables_heating
                Q_tot = vh["Volume-averaged total heating [W.m-3]"][outer_step, :]
                Q_ohm_cc = net.interpolate_data("pore.cc_power_loss")[res_Ts]
                Q_ohm_cc /= net["throat.volume"][res_Ts]
                key = "Volume-averaged Ohmic heating CC [W.m-3]"
                vh[key][outer_step, :] = Q_ohm_cc[sorted_res_Ts]
                Q = Q_tot
                Q[np.isnan(Q)] = 0.0
                ecm.apply_heat_source(project, Q)
                # Calculate Global Temperature
                ecm.run_step_transient(project, dim_time_step, T0, cp, rho,
                                       thermal_third)
                # Interpolate the node temperatures for the SPMs
                spm_temperature = phase.interpolate_data("pore.temperature")[res_Ts]
                all_time_temperature[outer_step, :] = spm_temperature
                max_temperatures.append(spm_temperature.max())
                T_non_dim_spm = (spm_temperature - T_ref) / Delta_T_spm
            # Get new equivalent resistances
            temp_R = ecm.calc_R(results_o, I_local_pnm)
            # Update ecm conductivities for the spm_resistor throats
            sig = 1 / temp_R
            if np.any(temp_R > R_max):
                print("Max R found")
                print(I_local_pnm[temp_R > R_max])
                dead[temp_R > R_max] = True
                sig[temp_R > R_max] = 1 / R_max
            if np.any(np.isnan(temp_R)):
                print("Nans found")
                print(I_local_pnm[np.isnan(temp_R)])
                dead[np.isnan(temp_R)] = True
                sig[np.isnan(temp_R)] = 1 / R_max
            phys["throat.electrical_conductance"][res_Ts] = sig
            local_R[:, outer_step] = temp_R
            if solutions[0].t.shape[0] > 1:
                if not ecm.check_vlim(
                    solutions[0],
                    config.getfloat("RUN", "vlim_lower"),
                    config.getfloat("RUN", "vlim_upper"),
                ):
                    dead.fill(True)
                    print("VOLTAGE LIMITS EXCEEDED")
            else:
                dead.fill(True)
                print(solutions[0].termination)

            outer_step += 1

    if config.getboolean("RUN", "parallel"):
        ecm.shutdown_pool(pool)
    ###########################################################################
    # Collect output                                                          #
    ###########################################################################
    variables["ECM R local"] = local_R[sorted_res_Ts, :outer_step].T
    variables["ECM I Local"] = all_time_I_local[:outer_step, sorted_res_Ts]
    variables["Temperature [K]"] = all_time_temperature[:outer_step, sorted_res_Ts]

    variables.update(lithiations)
    if config.getboolean("PHYSICS", "do_thermal"):
        variables.update(variables_heating)
    if outer_step < Nsteps:
        for key in variables.keys():
            variables[key] = variables[key][: outer_step - 1, :]
        for key in overpotentials.keys():
            overpotentials[key] = overpotentials[key][: outer_step - 1, :]

    if config.getboolean("OUTPUT", "plot"):
        ecm.run_ecm(net, alg, V_test, plot=True)
        for key in variables.keys():
            fig, ax = plt.subplots()
            ax.plot(variables[key][:, sorted_res_Ts])
            plt.title(key)
            plt.show()

        fig, ax = plt.subplots()
        ax.plot(max_temperatures)
        ax.set_xlabel("Discharge Time [h]")
        ax.set_ylabel("Maximum Temperature [K]")

    if config.getboolean("OUTPUT", "save"):
        print("Saving to", save_path)
        lower_mask = net["throat.spm_neg_inner"][res_Ts[sorted_res_Ts]]
        ecm.export(
            project,
            save_path,
            variables,
            "var_",
            lower_mask=lower_mask,
            save_animation=False,
        )
        ecm.export(
            project,
            save_path,
            overpotentials,
            "eta_",
            lower_mask=lower_mask,
            save_animation=False,
        )
        parent_dir = os.path.dirname(save_path)
        wrk.save_project(project=project, filename=os.path.join(parent_dir, "net"))
        # project.export_data(phases=[phase], filename='ecm.vtp')

    print("*" * 30)
    print("ECM Sim time", time.time() - st)
    print("*" * 30)
    return project, variables, solutions

