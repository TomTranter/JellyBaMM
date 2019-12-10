#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:14:46 2019

@author: tom
"""

import openpnm as op
import openpnm.topotools as tt
from matplotlib import cm
import matplotlib as mpl
from matplotlib.collections import LineCollection
import pybamm
import numpy as np
import sys
import matplotlib.pyplot as plt
import copy

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


def step_spm(sim, dt, I_app):
    h = sim.parameter_values['Electrode height [m]']
    w = sim.parameter_values['Electrode width [m]']
    A_cc = h*w
    sim.step(dt=dt, inputs={"Current": I_app}, save=False)
    R_ecm = evaluate(sim, 'Local ECM resistance [Ohm.m2]', I_app)
    V_ecm = evaluate(sim, 'Local ECM voltage [V]', I_app)
    R = R_ecm/A_cc
    return R, V_ecm


solution = None
current_state = np.array([])
#key = "Current collector current density [A.m-2]"
dt = 1e-12
I_app = 1.0
spm_sim = make_spm()
R, V_ecm = step_spm(spm_sim, dt, I_app)
print(R)
#resistances = []
#y0 = None
#for i in range(3):
#    current = I_app
#    R, V_ecm = step_spm(sim, dt, current)
#    step_sol = copy.deepcopy(sim.solution)
##    step_sol.t += i * dt
#    if y0 is None:
#        y0 = sim.solver.y0
#    if solution is None:
#        solution = step_sol
#    else:
#        solution.append(step_sol)
#    proc = pybamm.ProcessedVariable(
#        sim.built_model.variables[key],
#        sim.solution.t,
#        sim.solution.y,
#        2mesh=sim.mesh,
#        inputs={"Current": current},
#    )
#    current_state = np.concatenate([current_state, proc(step_sol.t[-1:])])
#    resistances.append(R[0])
#    sim.reset()

#plt.figure()
#plt.plot(solution.t[1:], current_state, "*")
#plt.show()
#plt.figure()
#plt.plot(solution.t[1:], resistances, "*")
#plt.show()



#################################################################
Nunit = 3
I_app = 1.0
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
#spm_sim, R, V_ecm = spm_init(I_app=I_app/Nunit)
phase['throat.electrical_conductance'] = cc_cond*cc_unit_area/cc_unit_len
phase['throat.electrical_conductance'][net.throats('spm_resistor')] = 1/R
spm_models = [copy.deepcopy(spm_sim) for i in range(len(net.throats('spm_resistor')))]
potential_pairs = net['throat.conns'][net.throats('spm_resistor')]
P1 = potential_pairs[:, 0]
P2 = potential_pairs[:, 1]
alg = op.algorithms.OhmicConduction(network=net)
alg.setup(phase=phase,
          quantity="pore.potential",
          conductance="throat.electrical_conductance",
          )
alg.set_value_BC(net.pores('neg_terminal'), values=0.0)
alg.settings['solver_rtol'] = 1e-15
alg.settings['solver_atol'] = 1e-15

def run_ecm(V_terminal):
    alg.set_value_BC(net.pores('pos_terminal'), values=V_terminal)
    alg.run()
    V_local_pnm = alg['pore.potential'][P2] - alg['pore.potential'][P1]
    I_local_pnm = alg.rate(throats=net.throats('spm_resistor'), mode='single')
    R_local_pnm = V_local_pnm / I_local_pnm
    return (V_local_pnm, I_local_pnm, R_local_pnm)


def plot(spm_models):
    # Plotting
#    z = np.arange(0, len(spm_models), 1)
    pvs = {
#        "X-averaged reversible heating [W.m-3]": plt.subplots(),
#        "X-averaged irreversible electrochemical heating [W.m-3]": plt.subplots(),
#        "X-averaged Ohmic heating [W.m-3]": plt.subplots(),
#        "X-averaged total heating [W.m-3]": plt.subplots(),
        "Current collector current density [A.m-2]": plt.subplots(),
        "X-averaged positive particle " +
        "surface concentration [mol.m-3]": plt.subplots(),
        "X-averaged negative particle " +
        "surface concentration [mol.m-3]": plt.subplots(),
        "X-averaged cell temperature [K]": plt.subplots(),
        "Negative current collector potential [V]": plt.subplots(),
        "Positive current collector potential [V]": plt.subplots(),
    }
#    hrs = sim.solution.t
    for key in pvs.keys():
        for sim in spm_models:
            sol = sim.solution
            proc = pybamm.ProcessedVariable(
                sim.built_model.variables[key], sol.t, sol.y, mesh=sim.mesh
            )
            pvs[key][1].plot(proc(sol.t))
    
#    for key in pvs.keys():
#        fig, ax = plt.subplots()
#        lines = []
#        data = pvs[key](sol.t)
#        for bat_id in range(len(spm_models)):
#            lines.append(np.column_stack((hrs, data[bat_id, :])))
#        line_segments = LineCollection(lines)
#        line_segments.set_array(z)
#        ax.yaxis.set_major_formatter(
#            mpl.ticker.ScalarFormatter(useMathText=True, useOffset=False)
#        )
#        ax.add_collection(line_segments)
#        plt.xlabel("t")
#        plt.ylabel(key)
#        plt.xlim(hrs.min(), hrs.max())
#        plt.ylim(data.min(), data.max())
#        #            plt.ticklabel_format(axis='y', style='sci')
#        plt.show()


(V_local_pnm, I_local_pnm, R_local_pnm) = run_ecm(V_ecm)
#fig = tt.plot_coordinates(net, net.Ps, c=alg['pore.potential'])
print('*'*30)
print('V local pnm', V_local_pnm, '[V]')
print('I local pnm', I_local_pnm, '[A]')
print('R local pnm', R_local_pnm, '[Ohm]')

dt = 2e-3
N_steps = 100
res_Ts = net.throats('spm_resistor')
terminal_voltages = np.zeros(N_steps)
V_test = V_ecm
tol = 1e-6
for outer_step in range(N_steps):
    # Find terminal voltage that satisfy ecm total currents for R
    current_match = False
    max_inner_steps = 100
    inner_step = 0
    damping = 1
    while (inner_step < max_inner_steps) and (not current_match):
        (V_local_pnm, I_local_pnm, R_local_pnm) = run_ecm(V_test)
        tot_I_local_pnm = np.sum(I_local_pnm)
        diff = (I_app - tot_I_local_pnm)/I_app
        if np.absolute(diff) < tol:
            current_match = True
        else:
            V_test *= (1+(diff/damping))
        inner_step += 1
        print('Inner', inner_step, diff, V_test)
    terminal_voltages[outer_step] = V_test
    # I_local_pnm should now match the total applied current
    # Run the spms for the the new I_locals
    for i in range(len(spm_models)):
        sim = spm_models[i]
        temp_R, temp_V_ecm = step_spm(sim, dt, I_local_pnm[i])
        # Update conductance
        print(i, temp_R)
        phase['throat.electrical_conductance'][res_Ts[i]] = 1/temp_R
    print('Outer', outer_step)

fig, ax = plt.subplots()
ax.plot(terminal_voltages)
#plot(spm_models)