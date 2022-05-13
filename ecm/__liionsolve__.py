#
# Liionpack solve
#
import os
import numpy as np
import ecm
import pybamm
import time as ticker
import openpnm as op
import liionpack as lp
from tqdm import tqdm
import matplotlib.pyplot as plt
import configparser


wrk = op.Workspace()


def get_cc_power_loss(network, netlist):
    pnm_power = np.zeros(network.Nt)
    for i in range(network.Nt):
        T_map = netlist["pnm_throat_id"] == i
        pnm_power[i] = np.sum(netlist["power_loss"][T_map])
    return pnm_power


def fT_non_dim(parameter_values, T):
    param = pybamm.LithiumIonParameters()
    Delta_T = parameter_values.evaluate(param.Delta_T)
    T_ref = parameter_values.evaluate(param.T_ref)
    return (T - T_ref) / Delta_T


def do_heating():
    pass


def run_simulation_lp(I_app, save_path, config):
    ###########################################################################
    # Simulation information                                                  #
    ###########################################################################
    st = ticker.time()
    max_workers = int(os.cpu_count() / 2)
    hours = config.getfloat("RUN", "hours")
    try:
        dt = config.getfloat("RUN", "dt")
        Nsteps = np.int(np.ceil(hours * 3600 / dt) + 1)
    except configparser.NoOptionError:
        dt = 30
        Nsteps = np.int(hours * 60 * 2) + 1  # number of time steps
    if config.get("GEOMETRY", "domain") == "model":
        project, arc_edges = ecm.make_spiral_net(config)
    elif config.get("GEOMETRY", "domain") == "1d":
        project, arc_edges = ecm.make_1D_net(config)
    else:
        project, arc_edges = ecm.make_tomo_net(config)

    net = project.network
    if config.get("GEOMETRY", "domain") != "1d":
        ecm.plot_topology(net)
    phase = project.phases()["phase_01"]
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
    parameter_values = ecm.make_parameters(I_typical, config)
    width = parameter_values["Electrode width [m]"]
    t1 = parameter_values["Negative electrode thickness [m]"]
    t2 = parameter_values["Positive electrode thickness [m]"]
    t3 = parameter_values["Negative current collector thickness [m]"]
    t4 = parameter_values["Positive current collector thickness [m]"]
    t5 = parameter_values["Separator thickness [m]"]
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
        # "X-averaged negative particle surface concentration [mol.m-3]": temp.copy(),
        # "X-averaged positive particle surface concentration [mol.m-3]": temp.copy(),
        "Terminal voltage [V]": temp.copy(),
        "Volume-averaged cell temperature [K]": temp.copy(),
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
        # "Volume-averaged Ohmic heating [W.m-3]": temp.copy(),
        # "Volume-averaged irreversible electrochemical heating [W.m-3]": temp.copy(),
        # "Volume-averaged reversible heating [W.m-3]": temp.copy(),
        "Volume-averaged total heating [W.m-3]": temp.copy(),
    }
    lithiations_keys = list(lithiations.keys())
    variable_keys = list(variables.keys())
    heating_keys = list(variables_heating.keys())
    output_variables = variable_keys + heating_keys + lithiations_keys
    ###########################################################################
    # Thermal parameters                                                      #
    ###########################################################################
    params = pybamm.LithiumIonParameters()
    Delta_T = parameter_values.process_symbol(params.Delta_T).evaluate(
        inputs=temp_inputs
    )
    # Delta_T_spm = Delta_T * (typical_height / electrode_heights)
    T_ref = parameter_values.process_symbol(params.T_ref).evaluate()
    T0 = config.getfloat("PHYSICS", "T0")
    lumpy_therm = ecm.lump_thermal_props(config)
    cp = lumpy_therm["lump_Cp"]
    rho = lumpy_therm["lump_rho"]
    T_non_dim = (T0 - T_ref) / Delta_T
    T_non_dim_spm = np.ones(len(res_Ts)) * T_non_dim
    ###########################################################################
    # Simulation variables
    ###########################################################################
    local_R = np.zeros([Nspm, Nsteps])
    all_time_I_local = np.zeros([Nsteps, Nspm])
    ###########################################################################
    # Run time config                                                         #
    ###########################################################################
    outer_step = 0
    if config.getboolean("PHYSICS", "do_thermal"):
        ecm.setup_thermal(project, config)
    try:
        thermal_third = config.getboolean("RUN", "third")
    except KeyError:
        thermal_third = False
    ###########################################################################
    # New Liionpack code                                                      #
    ###########################################################################
    dim_time_step = 10
    neg_econd, pos_econd = ecm.cc_cond(project, config)
    Rs = 1e-2  # series resistance
    Ri = 90  # initial guess for internal resistance
    V = 3.6  # initial guess for cell voltage
    # I_app = 0.5
    netlist = ecm.network_to_netlist(net, Rs, Ri, V, I_app)
    T0 = parameter_values["Initial temperature [K]"]
    T_non_dim_spm = np.ones(Nspm) * fT_non_dim(parameter_values, T0)
    e_heights = net["throat.electrode_height"][net.throats("throat.spm_resistor")]
    # e_heights.fill(np.mean(e_heights))
    inputs = {
        "Electrode height [m]": e_heights,
    }
    ###########################################################################
    # Initialisation
    external_variables = {"Volume-averaged cell temperature": T_non_dim_spm}
    experiment = pybamm.Experiment(
        [
            f"Discharge at {I_app} A for 4 seconds",
        ],
        period="1 second",
    )
    # Solve the pack
    manager = lp.casadi_manager()
    manager.solve(
        netlist=netlist,
        sim_func=lp.thermal_external,
        parameter_values=parameter_values,
        experiment=experiment,
        output_variables=output_variables,
        inputs=inputs,
        external_variables=external_variables,
        nproc=max_workers,
        initial_soc=0.5,
        setup_only=True,
    )
    Qvar = "Volume-averaged total heating [W.m-3]"
    Qid = np.argwhere(np.asarray(manager.variable_names) == Qvar).flatten()[0]
    lp.logger.notice("Starting initial step solve")
    vlims_ok = True
    tic = ticker.time()
    netlist["power_loss"] = 0.0
    plt.figure()
    with tqdm(total=manager.Nsteps, desc="Initialising simulation") as pbar:
        step = 0
        # reset = True
        while step < manager.Nsteps and vlims_ok:
            ###################################################################
            external_variables = {"Volume-averaged cell temperature": T_non_dim_spm}
            vlims_ok = manager._step(step, external_variables)
            ###################################################################
            # Apply Heat Sources
            Q_tot = manager.output[Qid, step, :]
            Q = get_cc_power_loss(net, netlist)
            # To do - Get cc heat from netlist
            # Q_ohm_cc = net.interpolate_data("pore.cc_power_loss")[res_Ts]
            # Q_ohm_cc /= net["throat.volume"][res_Ts]
            # key = "Volume-averaged Ohmic heating CC [W.m-3]"
            # vh[key][outer_step, :] = Q_ohm_cc[sorted_res_Ts]
            Q[res_Ts] += Q_tot
            ecm.apply_heat_source_lp(project, Q)
            # Calculate Global Temperature
            ecm.run_step_transient(project, dim_time_step, T0, cp, rho, thermal_third)
            # Interpolate the node temperatures for the SPMs
            spm_temperature = phase.interpolate_data("pore.temperature")[res_Ts]
            T_non_dim_spm = fT_non_dim(parameter_values, spm_temperature)
            ###################################################################
            step += 1
            pbar.update(1)
            temp_Ri = np.array(netlist.loc[manager.Ri_map].value)
            plt.scatter(e_heights, temp_Ri, label=str(step))
    plt.legend()
    manager.step = step
    toc = ticker.time()
    lp.logger.notice("Initial step solve finished")
    lp.logger.notice("Total stepping time " + str(np.around(toc - tic, 3)) + "s")
    lp.logger.notice(
        "Time per step " + str(np.around((toc - tic) / manager.Nsteps, 3)) + "s"
    )
    ###########################################################################
    # Real Solve
    ###########################################################################
    T_non_dim_spm = np.ones(Nspm) * fT_non_dim(parameter_values, T0)
    external_variables = {"Volume-averaged cell temperature": T_non_dim_spm}
    experiment = pybamm.Experiment(
        [
            f"Discharge at {I_app} A for {hours} hours",
        ],
        period=f"{dt} seconds",
    )
    # Solve the pack
    manager = lp.casadi_manager()
    manager.solve(
        netlist=netlist,
        sim_func=lp.thermal_external,
        parameter_values=parameter_values,
        experiment=experiment,
        output_variables=output_variables,
        inputs=inputs,
        external_variables=external_variables,
        nproc=max_workers,
        initial_soc=0.5,
        setup_only=True,
    )
    Qvar = "Volume-averaged total heating [W.m-3]"
    Qid = np.argwhere(np.asarray(manager.variable_names) == Qvar).flatten()[0]
    lp.logger.notice("Starting step solve")
    vlims_ok = True
    tic = ticker.time()
    netlist["power_loss"] = 0.0
    with tqdm(total=manager.Nsteps, desc="Stepping simulation") as pbar:
        step = 0
        # reset = True
        while step < manager.Nsteps and vlims_ok:
            ###################################################################
            external_variables = {"Volume-averaged cell temperature": T_non_dim_spm}
            vlims_ok = manager._step(step, external_variables)
            ###################################################################
            # Apply Heat Sources
            Q_tot = manager.output[Qid, step, :]
            Q = get_cc_power_loss(net, netlist)
            # To do - Get cc heat from netlist
            # Q_ohm_cc = net.interpolate_data("pore.cc_power_loss")[res_Ts]
            # Q_ohm_cc /= net["throat.volume"][res_Ts]
            # key = "Volume-averaged Ohmic heating CC [W.m-3]"
            # vh[key][outer_step, :] = Q_ohm_cc[sorted_res_Ts]
            Q[res_Ts] += Q_tot
            ecm.apply_heat_source_lp(project, Q)
            # Calculate Global Temperature
            ecm.run_step_transient(project, dim_time_step, T0, cp, rho, thermal_third)
            # Interpolate the node temperatures for the SPMs
            spm_temperature = phase.interpolate_data("pore.temperature")[res_Ts]
            T_non_dim_spm = fT_non_dim(parameter_values, spm_temperature)
            ###################################################################
            step += 1
            pbar.update(1)
    manager.step = step
    toc = ticker.time()
    lp.logger.notice("Step solve finished")
    lp.logger.notice("Total stepping time " + str(np.around(toc - tic, 3)) + "s")
    lp.logger.notice(
        "Time per step " + str(np.around((toc - tic) / manager.Nsteps, 3)) + "s"
    )

    ###########################################################################
    # Collect output                                                          #
    ###########################################################################
    variables["ECM R local"] = local_R[sorted_res_Ts, :outer_step].T
    variables["ECM I Local"] = all_time_I_local[:outer_step, sorted_res_Ts]

    variables.update(lithiations)
    if config.getboolean("PHYSICS", "do_thermal"):
        variables.update(variables_heating)
    if outer_step < Nsteps:
        for key in variables.keys():
            variables[key] = variables[key][: outer_step - 1, :]
        for key in overpotentials.keys():
            overpotentials[key] = overpotentials[key][: outer_step - 1, :]

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
    print("ECM Sim time", ticker.time() - st)
    print("*" * 30)
    return project, manager.step_output()
