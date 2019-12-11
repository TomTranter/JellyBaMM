#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:11:13 2019

@author: thomas
"""
import openpnm as op
import openpnm.topotools as tt
import pybamm
import matplotlib.pyplot as plt
import copy
from concurrent.futures import ProcessPoolExecutor

plt.close("all")
# set logging level
pybamm.set_logging_level("INFO")


def current_function(t):
    return pybamm.InputParameter("Current")


def make_spm():
    model = pybamm.lithium_ion.SPM()
    geometry = model.default_geometry
    param = model.default_parameter_values
    param.update({"Current function": current_function, "Current": "[input]"})
    param.process_model(model)
    param.process_geometry(geometry)
    var_pts = model.default_var_pts
    spatial_methods = model.default_spatial_methods
    solver = pybamm.CasadiSolver()
    sim = pybamm.Simulation(model=model,
                            geometry=geometry,
                            parameter_values=param,
                            var_pts=var_pts,
                            spatial_methods=spatial_methods,
                            solver=solver)
    return sim


def evaluate(sim, var="Current collector current density [A.m-2]", current=0.0):
    model = sim.built_model
    mesh = sim.mesh
    solution = sim.solution
    proc = pybamm.ProcessedVariable(
        model.variables[var], solution.t, solution.y, mesh=mesh,
        inputs={"Current": current}
    )
    return proc(solution.t[-1])


def step_spm(zipped):
    sim, I_app, dt = zipped
    h = sim.parameter_values['Electrode height [m]']
    w = sim.parameter_values['Electrode width [m]']
    A_cc = h*w
    sim.step(dt=dt, inputs={"Current": I_app}, save=False)
    R_ecm = evaluate(sim, 'Local ECM resistance [Ohm.m2]', I_app)
    V_ecm = evaluate(sim, 'Local ECM voltage [V]', I_app)
    R = R_ecm/A_cc
    return sim, R, V_ecm


def make_net(spm_sim, Nunit, R):
    net = op.network.Cubic([Nunit+1, 2, 1])
    net['pore.pos_cc'] = net['pore.right']
    net['pore.neg_cc'] = net['pore.left']

    T = net.find_neighbor_throats(net.pores('front'), mode='xnor')
    tt.trim(net, throats=T)
    pos_cc_Ts = net.find_neighbor_throats(net.pores('pos_cc'), mode='xnor')
    neg_cc_Ts = net.find_neighbor_throats(net.pores('neg_cc'), mode='xnor')

    P_pos = net.pores(['pos_cc', 'front'], 'and')
    P_neg = net.pores(['neg_cc', 'front'], 'and')

    net['pore.pos_terminal'] = False
    net['pore.neg_terminal'] = False
    net['pore.pos_terminal'][P_pos] = True
    net['pore.neg_terminal'][P_neg] = True
    net['throat.pos_cc'] = False
    net['throat.neg_cc'] = False
    net['throat.pos_cc'][pos_cc_Ts] = True
    net['throat.neg_cc'][neg_cc_Ts] = True
    net['throat.spm_resistor'] = True
    net['throat.spm_resistor'][pos_cc_Ts] = False
    net['throat.spm_resistor'][neg_cc_Ts] = False

    del net['pore.left']
    del net['pore.right']
    del net['pore.front']
    del net['pore.back']
    del net['pore.internal']
    del net['pore.surface']
    del net['throat.internal']
    del net['throat.surface']

    fig = tt.plot_coordinates(net, net.pores('pos_cc'), c='b')
    fig = tt.plot_coordinates(net, net.pores('pos_terminal'), c='y', fig=fig)
    fig = tt.plot_coordinates(net, net.pores('neg_cc'), c='r', fig=fig)
    fig = tt.plot_coordinates(net, net.pores('neg_terminal'), c='g', fig=fig)
    fig = tt.plot_connections(net, net.throats('pos_cc'), c='b', fig=fig)
    fig = tt.plot_connections(net, net.throats('neg_cc'), c='r', fig=fig)
    fig = tt.plot_connections(net, net.throats('spm_resistor'), c='k', fig=fig)

    phase = op.phases.GenericPhase(network=net)
    cc_cond = 3e7
    cc_unit_len = 5e-2
    cc_unit_area = 1e-4
    phase['throat.electrical_conductance'] = cc_cond*cc_unit_area/cc_unit_len
    phase['throat.electrical_conductance'][net.throats('spm_resistor')] = 1/R
    spm_models = [copy.deepcopy(spm_sim) for i in range(len(net.throats('spm_resistor')))]
    alg = op.algorithms.OhmicConduction(network=net)
    alg.setup(phase=phase,
              quantity="pore.potential",
              conductance="throat.electrical_conductance",
              )
    alg.set_value_BC(net.pores('neg_terminal'), values=0.0)
    alg.settings['solver_rtol'] = 1e-15
    alg.settings['solver_atol'] = 1e-15
    return net, alg, phase, spm_models


def run_ecm(net, alg, V_terminal):
    potential_pairs = net['throat.conns'][net.throats('spm_resistor')]
    P1 = potential_pairs[:, 0]
    P2 = potential_pairs[:, 1]
    alg.set_value_BC(net.pores('pos_terminal'), values=V_terminal)
    alg.run()
    V_local_pnm = alg['pore.potential'][P2] - alg['pore.potential'][P1]
    I_local_pnm = alg.rate(throats=net.throats('spm_resistor'), mode='single')
    R_local_pnm = V_local_pnm / I_local_pnm
    return (V_local_pnm, I_local_pnm, R_local_pnm)


def pool_spm(spm_models):
    pool = ProcessPoolExecutor(max_workers=10)
    data = list(pool.map(step_spm, spm_models))
    pool.shutdown()
    del pool
    return data
