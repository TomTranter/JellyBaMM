#
# Liionpack solve
#
import os
import numpy as np
import jellybamm
import pybamm
import time as ticker
import openpnm as op
import liionpack as lp
from tqdm import tqdm
import matplotlib.pyplot as plt

# import configparser


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

def get_skin_temperature(project):
    net = project.network
    outer_pores = net['pore.outer']
    spms = net['throat.spm_resistor']
    outers = np.argwhere(outer_pores)

    skin_temperature = project.phases()['phase_01']['pore.temperature'][outers].flatten()
    return skin_temperature
 
def do_heating():
    pass


def run_simulation_lp(parameter_values, experiment, initial_soc, project, **kwargs):
    ###########################################################################
    # Simulation information                                                  #
    ###########################################################################
    st = ticker.time()
    max_workers = kwargs.get('max_workers', int(os.cpu_count() / 2))
    kwargs.setdefault('skin_temp_cutoff_K', None)
    # hours = config.getfloat("RUN", "hours")
    # try:
    # dt = config.getfloat("RUN", "dt")
    # Nsteps = int(np.ceil(hours * 3600 / dt) + 1)
    # except configparser.NoOptionError:
    # dt = 30
    # Nsteps = int(hours * 60 * 2) + 1  # number of time steps
    net = project.network
    phase = project.phases()["phase_01"]
    # The jellyroll layers are double sided around the cc except for the inner
    # and outer layers the number of spm models is the number of throat
    # connections between cc layers
    Nspm = net.num_throats("spm_resistor")
    res_Ts = net.throats("spm_resistor")
    # sorted_res_Ts = net["throat.spm_resistor_order"][res_Ts].argsort()
    # print("Total Electrode Height", np.around(np.sum(electrode_heights), 2), "m")
    # Take I_app from first command of the experiment
    protos, terms, types = lp.generate_protocol_from_experiment(experiment)
    # if len(protos) > 1:
    #     raise ValueError("Experiment must be a single step currently")
    I_app = protos[0][0]
    if I_app == 0:
        I_app = protos[1][0]
    I_typical = I_app / Nspm

    # print("Total pore volume", np.sum(net["pore.volume"]))
    # print("Mean throat area", np.mean(net["throat.area"]))
    # print("Num throats", net.num_throats())
    # print("Num throats SPM", Nspm)
    # print("Num throats pos_cc", net.num_throats("pos_cc"))
    # print("Num throats neg_cc", net.num_throats("neg_cc"))
    # print("Typical height", typical_height)
    # print("Typical current", I_typical)
    ###########################################################################
    # Make the pybamm simulation - should be moved to a simfunc               #
    ###########################################################################
    parameter_values = jellybamm.adjust_parameters(parameter_values, I_typical)
    # width = parameter_values["Electrode width [m]"]
    # t1 = parameter_values["Negative electrode thickness [m]"]
    # t2 = parameter_values["Positive electrode thickness [m]"]
    # t3 = parameter_values["Negative current collector thickness [m]"]
    # t4 = parameter_values["Positive current collector thickness [m]"]
    # t5 = parameter_values["Separator thickness [m]"]
    # ttot = t1 + t2 + t3 + t4 + t5
    # A_cc = electrode_heights * width
    # bat_vol = np.sum(A_cc * ttot)
    # print("BATTERY ELECTRODE VOLUME", bat_vol)
    # print("18650 VOLUME", 0.065 * np.pi * ((8.75e-3) ** 2 - (2.0e-3) ** 2))
    ###########################################################################
    # Output variables                                                        #
    ###########################################################################
    output_variables = jellybamm.output_variables()
    ###########################################################################
    # Thermal parameters                                                      #
    ###########################################################################
    T0 = parameter_values["Initial temperature [K]"]
    lumpy_therm = jellybamm.lump_thermal_props(parameter_values)
    cp = lumpy_therm["lump_Cp"]
    rho = lumpy_therm["lump_rho"]
    ###########################################################################
    # Run time config                                                         #
    ###########################################################################
    # outer_step = 0
    # if config.getboolean("PHYSICS", "do_thermal"):
    # Always do thermal
    jellybamm.setup_thermal(project, parameter_values)
    # try:
    #     thermal_third = config.getboolean("RUN", "third")
    # except KeyError:
    thermal_third = False
    ###########################################################################
    # New Liionpack code                                                      #
    ###########################################################################
    dim_time_step = experiment.period
    neg_econd, pos_econd = jellybamm.cc_cond(project, parameter_values)
    Rs = 1e-2  # series resistance
    Ri = 90  # initial guess for internal resistance
    V = 3.6  # initial guess for cell voltage
    # I_app = 0.5
    netlist = jellybamm.network_to_netlist(net, Rs, Ri, V, I_app)
    # return netlist
    desc = np.array(netlist["desc"]).astype("<U1")  # just take first character
    I_map = desc == "I"
    Terminal_Node = np.array(netlist[I_map].node1)
    def tabs_voltage_term(V_node):
        if V_node[Terminal_Node] < 2.5:
            return True

    node_termination_func = tabs_voltage_term

    # return netlist
    T0 = parameter_values["Initial temperature [K]"]
    e_heights = net["throat.electrode_height"][net.throats("throat.spm_resistor")]
    spm_temperature = np.ones(Nspm) * T0
    # e_heights.fill(np.mean(e_heights))
    inputs = {
        "Electrode height [m]": e_heights,
        "Input temperature [K]": spm_temperature,
    }
    ###########################################################################
    # Initialisation
    def initialize_simulation(I_app, netlist, project, phase, spm_temperature):
        print(I_app)
        if I_app > 0:
            experiment_string = f"Discharge at {abs(I_app)} A for 4 seconds or until 0V"
        elif I_app < 0:
            experiment_string = f"Charge at {I_app} A for 4 seconds or until 4.5V"
        else:
            raise ValueError("I_app must be non-zero")
        experiment_init = pybamm.Experiment([experiment_string], period="1 second")
        print(experiment_string)

        proto_init, term_init, type_init = lp.generate_protocol_from_experiment(experiment_init)
        # Solve the pack
        tmp_manager = lp.CasadiManager()
        tmp_manager.solve(
            netlist=netlist,
            sim_func=lp.thermal_external,
            parameter_values=parameter_values,
            experiment=experiment_init,
            output_variables=output_variables,
            inputs=inputs,
            nproc=max_workers,
            initial_soc=initial_soc,
            setup_only=True,
        )
        Qvar = "Total heating [W]"
        # Qvar = "Volume-averaged total heating [W.m-3]"
        Qid = np.argwhere(np.asarray(tmp_manager.variable_names) == Qvar).flatten()[0]
        lp.logger.notice("Starting initial step solve")
        vlims_ok = True
        tic = ticker.time()
        netlist["power_loss"] = 0.0
        # plt.figure()
        skip_vcheck = True
        tmp_manager.global_step = 0
        with tqdm(total=len(proto_init[0]), desc="Initialising simulation",disable=kwargs.get('disable_tqdm',True)) as pbar:
            step = 0
            while step < len(proto_init[0]):
                ###################################################################
                updated_inputs = {"Input temperature [K]": spm_temperature}
                vlims_ok = tmp_manager._step(step, proto_init[0], term_init[0], type_init[0], updated_inputs, skip_vcheck)
                ###################################################################
                # Apply Heat Sources
                Q_tot = tmp_manager.output[Qid, step, :]
                Q = get_cc_power_loss(net, netlist)
                # To do - Get cc heat from netlist
                # Q_ohm_cc = net.interpolate_data("pore.cc_power_loss")[res_Ts]
                # Q_ohm_cc /= net["throat.volume"][res_Ts]
                # key = "Volume-averaged Ohmic heating CC [W.m-3]"
                # vh[key][outer_step, :] = Q_ohm_cc[sorted_res_Ts]
                Q[res_Ts] += Q_tot
                jellybamm.apply_heat_source_lp(project, Q)
                # Calculate Global Temperature
                jellybamm.run_step_transient(
                    project, dim_time_step, T0, cp, rho, thermal_third
                )
                # Interpolate the node temperatures for the SPMs
                spm_temperature = phase.interpolate_data("pore.temperature")[res_Ts]
                # T_non_dim_spm = fT_non_dim(parameter_values, spm_temperature)
                ###################################################################
                if vlims_ok:
                    step += 1
                    pbar.update(1)
                    temp_Ri = np.array(netlist.loc[tmp_manager.Ri_map].value)
                    # plt.scatter(np.arange(len(temp_Ri)), temp_Ri, label=str(step))
                else:
                    break
        # plt.legend()
        tmp_manager.step = step
        return netlist, project, phase, spm_temperature, tmp_manager
        toc = ticker.time()
        lp.logger.notice("Initial step solve finished")
        lp.logger.notice("Total stepping time " + str(np.around(toc - tic, 3)) + "s")
        lp.logger.notice(
            "Time per step " + str(np.around((toc - tic) / tmp_manager.Nsteps, 3)) + "s"
        )


    # netlist, project, phase, spm_temperature, tmp_manager = initialize_simulation(I_app, netlist, project, phase, spm_temperature)
    ###########################################################################
    # Real Solve
    ###########################################################################
    spm_temperature = np.ones(Nspm) * T0
    inputs.update({"Input temperature [K]": spm_temperature})
    # Solve the pack
    manager = lp.CasadiManager()
    manager.solve(
        netlist=netlist,
        sim_func=lp.thermal_external,
        parameter_values=parameter_values,
        experiment=experiment,
        output_variables=output_variables,
        inputs=inputs,
        nproc=max_workers,
        initial_soc=initial_soc,
        node_termination_func=node_termination_func,
        setup_only=True,
    )
    # manager.temp_Ri = tmp_manager.temp_Ri
    # Qvar = "Volume-averaged total heating [W.m-3]"
    Qvar = "Total heating [W]"
    Qid = np.argwhere(np.asarray(manager.variable_names) == Qvar).flatten()[0]
    netlist["power_loss"] = 0.0
    manager.global_step = 0
    step_previous_Iapp = 0
    for ps, step_protocol in enumerate(protos):
        step_termination = terms[ps]
        step_type = types[ps]
        if step_termination == []:
            step_termination = 0.0

        # if step_protocol[0] != step_previous_Iapp and step_protocol[0] != 0:
        #     manager._step(0, step_protocol, step_termination, step_type, None, True)
        #     manager._step(0, step_protocol, step_termination, step_type, None, True)
        # do reinitialization if switching directions
        if step_protocol[0] != step_previous_Iapp and step_protocol[0] != 0 and ps > 0:
            netlist, project, phase, _, tmp_manager = initialize_simulation(step_protocol[0]/1, netlist, project, phase, spm_temperature)
            manager.temp_Ri = tmp_manager.temp_Ri
            manager.netlist = netlist
        step_previous_Iapp = step_protocol[-1]

        ## Now Solve
        tic = ticker.time()
        lp.logger.notice(f"Starting step solve for step {ps}")

        vlims_ok = True
        skip_vcheck = True
        with tqdm(total=len(step_protocol), desc=f"Stepping simulation ({ps+1}/{len(protos)})",disable=kwargs.get('disable_tqdm',True)) as pbar:
            step = 0
            while step < len(step_protocol):
                ###################################################################
                # print(spm_temperature.min(), spm_temperature.max())
                updated_inputs = {"Input temperature [K]": spm_temperature}
                vlims_ok = manager._step(step, step_protocol, step_termination, step_type, updated_inputs, skip_vcheck)
                if step > 5:
                    skip_vcheck = False
                ###################################################################
                # Apply Heat Sources
                Q_tot = manager.output[Qid, manager.global_step, :]
                Q = get_cc_power_loss(net, netlist)

                # print(Q_tot)
                # To do - Get cc heat from netlist
                # Q_ohm_cc = net.interpolate_data("pore.cc_power_loss")[res_Ts]
                # Q_ohm_cc /= net["throat.volume"][res_Ts]
                # key = "Volume-averaged Ohmic heating CC [W.m-3]"
                # vh[key][outer_step, :] = Q_ohm_cc[sorted_res_Ts]
                Q[res_Ts] += Q_tot
                jellybamm.apply_heat_source_lp(project, Q)
                # Calculate Global Temperature
                jellybamm.run_step_transient(
                    project, dim_time_step, T0, cp, rho, thermal_third
                )
                # Interpolate the node temperatures for the SPMs
                spm_temperature = phase.interpolate_data("pore.temperature")[res_Ts]
                skin_temps = get_skin_temperature(project=project)
                if kwargs.get('skin_temp_cutoff_K',None) is not None:
                    if any(skin_temps>kwargs['skin_temp_cutoff_K']):
                        vlims_ok = False
               ###################################################################
                if vlims_ok:
                    manager.global_step += 1
                    step += 1
                    pbar.update(1)
                else:
                    # This break statement will only break out of the innermost while loop
                    break
        manager.step = step
        toc = ticker.time()
        lp.logger.notice("Step solve finished")
        lp.logger.notice("Total stepping time " + str(np.around(toc - tic, 3)) + "s")
        lp.logger.notice(
            "Time per step " + str(np.around((toc - tic) / manager.Nsteps, 3)) + "s"
        )

    print("*" * 30)
    print("Step Sim time", ticker.time() - st)
    print("*" * 30)
    return project, manager.step_output()
