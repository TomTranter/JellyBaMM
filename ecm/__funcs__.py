#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:11:13 2019

@author: thomas
"""
import numpy as np
import openpnm as op
import openpnm.topotools as tt
import pybamm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

plt.close("all")
# set logging level
pybamm.set_logging_level("INFO")


def convert_time(param, non_dim_time, to="seconds"):
    s_parms = pybamm.standard_parameters_lithium_ion
    t_sec = param.process_symbol(s_parms.tau_discharge).evaluate()
    t = non_dim_time * t_sec
    if to == "hours":
        t *= 1 / 3600
    return t


def current_function(t):
    return pybamm.InputParameter("Current")


def make_spm(Nunit, I_app):
    model = pybamm.lithium_ion.SPM()
    geometry = model.default_geometry
    param = model.default_parameter_values
    h = param["Electrode height [m]"]
    new_h = h / Nunit
    param.update(
        {
            "Typical current [A]": I_app / Nunit,
            "Current function": current_function,
            "Current": "[input]",
            "Electrode height [m]": new_h,
        }
    )
    param.process_model(model)
    param.process_geometry(geometry)
    var = pybamm.standard_spatial_vars
    var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.r_n: 10, var.r_p: 10}
    spatial_methods = model.default_spatial_methods
    solver = pybamm.CasadiSolver()
    sim = pybamm.Simulation(
        model=model,
        geometry=geometry,
        parameter_values=param,
        var_pts=var_pts,
        spatial_methods=spatial_methods,
        solver=solver,
    )
    return sim


def calc_R(sim, current):
    overpotentials = [
        "X-averaged reaction overpotential [V]",
        "X-averaged concentration overpotential [V]",
        "X-averaged electrolyte ohmic losses [V]",
        "X-averaged solid phase ohmic losses [V]",
    ]
    initial_ocv = 3.8518206633137266
    ocv = evaluate(sim, "X-averaged battery open circuit voltage [V]", current)
    totdV = initial_ocv - ocv
    for overpotential in overpotentials:
        totdV -= evaluate(sim, overpotential, current)
    return totdV / current


def evaluate(sim, var="Current collector current density [A.m-2]", current=0.0):
    model = sim.built_model
    #    mesh = sim.mesh
    solution = sim.solution
    #    proc = pybamm.ProcessedVariable(
    #        model.variables[var], solution.t, solution.y, mesh=mesh,
    #        inputs={"Current": current}
    #    )
    value = model.variables[var].evaluate(
        solution.t[-1], solution.y[:, -1], u={"Current": current}
    )
    # should move this definition to the main script...
    python_eval = pybamm.EvaluatorPython(model.variables[var])
    python_value = python_eval.evaluate(
        solution.t[-1], solution.y[:, -1], u={"Current": current}
    )

    #    return proc(solution.t[-1])
    return value


def step_spm(zipped):
    sim, I_app, dt, variables, dead = zipped
    #    h = sim.parameter_values['Electrode height [m]']
    #    w = sim.parameter_values['Electrode width [m]']
    #    A_cc = h*w
    results = np.zeros(len(variables) + 1)
    if ~dead:
        sim.step(dt=dt, inputs={"Current": I_app}, save=False)
        for i, key in enumerate(variables):
            results[i] = evaluate(sim, key, I_app)
        results[-1] = calc_R(sim, I_app)
    #    V_ecm = evaluate(sim, 'Local ECM voltage [V]', I_app)
    else:
        #        results = np.zeros(len(variables))
        results.fill(np.nan)
    return sim, results


def make_net(spm_sim, Nunit, R, spacing):
    net = op.network.Cubic([Nunit + 2, 2, 1], spacing)
    net["pore.pos_cc"] = net["pore.right"]
    net["pore.neg_cc"] = net["pore.left"]

    T = net.find_neighbor_throats(net.pores("front"), mode="xnor")
    tt.trim(net, throats=T)
    T = net.find_neighbor_throats(net.pores("back"), mode="xnor")
    tt.trim(net, throats=T)
    pos_cc_Ts = net.find_neighbor_throats(net.pores("pos_cc"), mode="xnor")
    neg_cc_Ts = net.find_neighbor_throats(net.pores("neg_cc"), mode="xnor")

    P_pos_a = net.pores(["pos_cc", "front"], "and")
    P_neg_a = net.pores(["neg_cc", "front"], "and")
    P_pos_b = net.pores(["pos_cc", "back"], "and")
    P_neg_b = net.pores(["neg_cc", "back"], "and")

    net["pore.pos_terminal_a"] = False
    net["pore.neg_terminal_a"] = False
    net["pore.pos_terminal_a"][P_pos_a] = True
    net["pore.neg_terminal_a"][P_neg_a] = True
    net["pore.pos_terminal_b"] = False
    net["pore.neg_terminal_b"] = False
    net["pore.pos_terminal_b"][P_pos_b] = True
    net["pore.neg_terminal_b"][P_neg_b] = True
    net["throat.pos_cc"] = False
    net["throat.neg_cc"] = False
    net["throat.pos_cc"][pos_cc_Ts] = True
    net["throat.neg_cc"][neg_cc_Ts] = True
    net["throat.spm_resistor"] = True
    net["throat.spm_resistor"][pos_cc_Ts] = False
    net["throat.spm_resistor"][neg_cc_Ts] = False

    del net["pore.left"]
    del net["pore.right"]
    del net["pore.front"]
    del net["pore.back"]
    del net["pore.internal"]
    del net["pore.surface"]
    del net["throat.internal"]
    del net["throat.surface"]

    fig = tt.plot_coordinates(net, net.pores("pos_cc"), c="b")
    fig = tt.plot_coordinates(net, net.pores("pos_terminal_b"), c="y", fig=fig)
    fig = tt.plot_coordinates(net, net.pores("neg_cc"), c="r", fig=fig)
    fig = tt.plot_coordinates(net, net.pores("neg_terminal_b"), c="g", fig=fig)
    fig = tt.plot_connections(net, net.throats("pos_cc"), c="b", fig=fig)
    fig = tt.plot_connections(net, net.throats("neg_cc"), c="r", fig=fig)
    fig = tt.plot_connections(net, net.throats("spm_resistor"), c="k", fig=fig)

    phase = op.phases.GenericPhase(network=net)
    cc_cond = 1e5
    cc_unit_len = spacing
    cc_unit_area = 25e-6 * 0.207
    phase["throat.electrical_conductance"] = cc_cond * cc_unit_area / cc_unit_len
    phase["throat.electrical_conductance"][net.throats("spm_resistor")] = 1 / R
    alg = op.algorithms.OhmicConduction(network=net)
    alg.setup(
        phase=phase,
        quantity="pore.potential",
        conductance="throat.electrical_conductance",
    )
    alg.settings["rxn_tolerance"] = 1e-8
    return net, alg, phase


def run_ecm(net, alg, V_terminal, plot=False):
    potential_pairs = net["throat.conns"][net.throats("spm_resistor")]
    P1 = potential_pairs[:, 0]
    P2 = potential_pairs[:, 1]
    adj = np.random.random(1) / 1e3
    alg.set_value_BC(net.pores("pos_terminal_b"), values=V_terminal + adj)
    alg.set_value_BC(net.pores("neg_terminal_b"), values=adj)
    #    alg['pore.potential'] -= adj
    alg.run()
    V_local_pnm = alg["pore.potential"][P2] - alg["pore.potential"][P1]
    I_local_pnm = alg.rate(throats=net.throats("spm_resistor"), mode="single")
    R_local_pnm = V_local_pnm / I_local_pnm
    if plot:
        plt.figure()
        plt.plot(alg["pore.potential"][P1])
        plt.plot(alg["pore.potential"][P2])

    return (V_local_pnm, I_local_pnm, R_local_pnm)


def setup_pool(max_workers):
    pool = ThreadPoolExecutor(max_workers=max_workers)
    return pool


def pool_spm(spm_models, pool):
    data = list(pool.map(step_spm, spm_models))
    return data


def shutdown_pool(pool):
    pool.shutdown()
    del pool


def serial_spm(inputs):
    outputs = []
    for bundle in inputs:
        outputs.append(step_spm(bundle))
    return outputs
