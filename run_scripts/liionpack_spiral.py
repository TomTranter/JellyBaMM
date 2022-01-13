#
# Test the spiral net
#

import openpnm as op
import matplotlib.pyplot as plt
import ecm
import configparser
import os
import liionpack as lp
import pandas as pd
import pybamm
import numpy as np

plt.close("all")

# pybamm.set_logging_level("INFO")
wrk = op.Workspace()
wrk.clear()


def fn(n):
    if n == 0:
        return "0"
    else:
        return "N" + str(n).zfill(3)


if __name__ == "__main__":
    save_root = os.path.join(ecm.OUTPUT_DIR, "spiral")
    print(save_root)
    config = configparser.ConfigParser()
    config.read(os.path.join(save_root, "config.txt"))
    print(ecm.lump_thermal_props(config))
    I_apps = [config.get("RUN", key) for key in config["RUN"] if "i_app" in key]
    prj, arc_edges = ecm.make_spiral_net(config)
    net = prj.network
    # ecm.plot_topology(net)

    # Now make network into liionpack netlist
    Rbn = 1e-4
    Rbp = 1e-4
    Rs = 1e-5
    Ri = 60
    V = 3.6
    I_app = -5.0
    desc = []
    node1 = []
    node2 = []
    value = []
    node1_x = []
    node1_y = []
    node2_x = []
    node2_y = []
    xs = net["pore.coords"][:, 0]
    ys = net["pore.coords"][:, 1]

    # Negative current collector
    for t in net.throats("neg_cc"):
        desc.append("Rbn" + str(t))
        n1, n2 = net["throat.conns"][t]
        node1.append(n1)
        node2.append(n2)
        value.append(Rbn)
        node1_x.append(xs[n1])
        node1_y.append(ys[n1])
        node2_x.append(xs[n2])
        node2_y.append(ys[n2])

    # Positive current collector
    for t in net.throats("pos_cc"):
        desc.append("Rbp" + str(t))
        n1, n2 = net["throat.conns"][t]
        node1.append(n1)
        node2.append(n2)
        value.append(Rbp)
        node1_x.append(xs[n1])
        node1_y.append(ys[n1])
        node2_x.append(xs[n2])
        node2_y.append(ys[n2])

    # check contiguous
    node_max = max((max(node1), max(node2)))
    for i in range(node_max):
        if i not in node1:
            if i not in node2:
                print("Missing", i)
    add_res = True
    nn = node_max
    # Battery Segment
    for t in net.throats("throat.spm_resistor"):
        n1, n2 = net["throat.conns"][t]
        # swap node if n1 is negative
        n1_neg = net["pore.neg_cc"][n1]
        if n1_neg:
            n1, n2 = net["throat.conns"][t][::-1]
        vx = xs[n2] - xs[n1]
        vy = ys[n2] - ys[n1]
        vax = xs[n1] + vx / 3
        vbx = xs[n1] + vx * 2 / 3
        vay = ys[n1] + vy / 3
        vby = ys[n1] + vy * 2 / 3
        if add_res:
            # Make a new connection resistor from neg to V
            nn += 1
            desc.append("Rs" + str(t))
            node1.append(n1)
            node2.append(nn)
            value.append(Rs)
            node1_x.append(xs[n1])
            node1_y.append(ys[n1])
            node2_x.append(vax)
            node2_y.append(vay)
            # Make a battery node Va to Vb
            nn += 1
            desc.append("V" + str(t))
            node1.append(nn - 1)
            node2.append(nn)
            value.append(V)
            node1_x.append(vax)
            node1_y.append(vay)
            node2_x.append(vbx)
            node2_y.append(vby)
            # Make an intenal resistor from Vb to pos
            desc.append("Ri" + str(t))
            node1.append(nn)
            node2.append(n2)
            value.append(Ri)
            node1_x.append(vbx)
            node1_y.append(vby)
            node2_x.append(xs[n2])
            node2_y.append(ys[n2])
        else:
            desc.append("V" + str(t))
            node1.append(n1)
            node2.append(n2)
            value.append(V)
            node1_x.append(xs[n1])
            node1_y.append(ys[n1])
            node2_x.append(xs[n2])
            node2_y.append(ys[n2])

    # Terminals
    n1 = net.pores("pos_cc")[-1]
    n2 = net.pores("neg_cc")[0]
    desc.append("I0")
    node1.append(n1)
    node2.append(n2)
    value.append(I_app)
    node1_x.append(xs[n1])
    node1_y.append(ys[n1])
    node2_x.append(xs[n2])
    node2_y.append(ys[n2])

    netlist_data = {
        "desc": desc,
        "node1": node1,
        "node2": node2,
        "value": value,
        "node1_x": node1_x,
        "node1_y": node1_y,
        "node2_x": node2_x,
        "node2_y": node2_y,
    }
    # add internal resistors
    netlist = pd.DataFrame(netlist_data)
    # lp.simple_netlist_plot(netlist)
    V_node, I_batt = lp.solve_circuit(netlist)
    fname = "jellyroll.cir"
    lines = ["* " + os.path.join(os.getcwd(), fname)]
    for (i, r) in netlist.iterrows():
        line = r.desc + " " + fn(r.node1) + " " + fn(r.node2) + " " + str(r.value)
        lines.append(line)
    lines.append(".op")
    lines.append(".backanno")
    lines.append(".end")
    with open(fname, "w") as f:
        for line in lines:
            f.write(line)
            f.write("\n")

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
            "Electrode height [m]": "[input]",
            "Electrode width [m]": 0.065,
            "Upper voltage cut-off [V]": 5.0,
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
        inputs={
            "Electrode height [m]": e_heights,
        },
        external_variables={
            "Volume-averaged cell temperature": np.ones_like(e_heights) * 303,
        },
        sim_func=lp.thermal_external,
    )
    lp.plot_output(output)
