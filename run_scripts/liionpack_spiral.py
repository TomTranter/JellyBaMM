#
# Test the spiral net
#

import openpnm as op
import matplotlib.pyplot as plt
import ecm
import configparser
import os
import liionpack as lp
# import pandas as pd
import pybamm
import numpy as np

plt.close("all")

# pybamm.set_logging_level("INFO")
wrk = op.Workspace()
wrk.clear()


if __name__ == "__main__":
    save_root = os.path.join(ecm.OUTPUT_DIR, "spiral")
    print(save_root)
    config = configparser.ConfigParser()
    config.read(os.path.join(save_root, "config.txt"))
    print(ecm.lump_thermal_props(config))
    I_apps = [config.get("RUN", key) for key in config["RUN"] if "i_app" in key]
    prj, arc_edges = ecm.make_spiral_net(config)
    net = prj.network
    ecm.plot_topology(net)

    # Now make network into liionpack netlist
    Rbn = 1e-4
    Rbp = 1e-4
    Rs = 1e-5
    Ri = 60
    V = 3.6
    I_app = -5.0
    netlist = ecm.network_to_netlist(net, Rbn, Rbp, Rs, Ri, V, I_app)
    lp.power_loss(netlist, include_Ri=False)
    R_map =netlist["desc"].str.find("R") > -1
    
    pnm_power = np.zeros(net.Nt)
    for i in range(net.Nt):
        T_map = netlist["pnm_throat_id"] == i
        pnm_power[i] = np.sum(netlist["power_loss"][T_map])

    V_node, I_batt = lp.solve_circuit(netlist)
    # Cycling experiment, using PyBaMM
    experiment = pybamm.Experiment(
        [
            "Discharge at 0.5 A for 1 hour",
        ],
        period="30 seconds",
    )

    # PyBaMM battery parameters
    chemistry = pybamm.parameter_sets.Chen2020
    param = pybamm.ParameterValues(chemistry=chemistry)

    # Make the electrode width an input parameter and use arc edges
    param.update(
        {
            # "Electrode height [m]": "[input]",
            "Electrode height [m]": 0.005,
            "Electrode width [m]": 0.065,
            # "Upper voltage cut-off [V]": 4.5,
        },
        check_already_exists=False,
    )
    e_heights = net["throat.electrode_height"][net.throats("throat.spm_resistor")]
    output_variables = ["Volume-averaged cell temperature"]
    # Solve the pack problem
    output = lp.solve(
        netlist=netlist,
        parameter_values=param,
        experiment=experiment,
        output_variables=output_variables,
        initial_soc=0.5,
        # inputs={
        #     "Electrode height [m]": e_heights,
        # },
        external_variables={
            "Volume-averaged cell temperature": np.ones_like(e_heights) * 0.0,
        },
        sim_func=lp.thermal_external,
    )
    lp.plot_output(output)
