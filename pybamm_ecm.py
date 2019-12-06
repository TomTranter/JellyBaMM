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

plt.close('all')
# set logging level
pybamm.set_logging_level("INFO")


def spm_init(height=None, width=None, I_app=None):
    model = pybamm.lithium_ion.SPMe()
    var = pybamm.standard_spatial_vars
    var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.r_n: 5, var.r_p: 5}
    sim = pybamm.Simulation(model, var_pts=var_pts)
    dt = 1e-6
    sim.step(dt)
    t_final = sim.solution.t[-1]
    pv = pybamm.post_process_variables(sim.built_model.variables,
                                       sim.solution.t,
                                       sim.solution.y,
                                       sim.mesh)
    R_ecm = pv['Local ECM resistance [Ohm.m2]'](t_final)
    V_ecm = pv['Local ECM voltage [V]'](t_final)
    h = sim.parameter_values['Electrode height [m]']
    w = sim.parameter_values['Electrode width [m]']
    A = h*w
    R = R_ecm/A
    return sim, R, V_ecm


def spm_full_run(sim):
    t_eval = np.linspace(0, 1.0, 201)
    sim.solve(t_eval=t_eval)
    pv = pybamm.post_process_variables(sim.built_model.variables,
                                       sim.solution.t,
                                       sim.solution.y,
                                       sim.mesh)
    t_final = sim.solution.t
    V_ocv_spm = pv['Measured open circuit voltage [V]'](t_final)
    V_local_spm = pv['Local voltage [V]'](t_final)
    R_ecm = pv['Local ECM resistance [Ohm.m2]'](t_final)
    V_ecm = pv['Local ECM voltage [V]'](t_final)
    I_ecm = pv['Current collector current density [A.m-2]'](t_final)
    etas = ["X-averaged battery reaction overpotential [V]",
            "X-averaged battery concentration overpotential [V]",
            "X-averaged battery electrolyte ohmic losses [V]",
            "X-averaged battery solid phase ohmic losses [V]"]
    overpotential_sum = 0.0
    for eta in etas:
        overpotential_sum -= pv[eta](t_final)
    R_ecm = pv['Local ECM resistance [Ohm.m2]'](t_final)
    h = sim.parameter_values['Electrode height [m]']
    w = sim.parameter_values['Electrode width [m]']
    A = h*w
    R = R_ecm/A
    print('*'*30)
    print('V local spm', V_ecm, '[V]')
    print('I local spm', I_ecm*A, '[A]')
    print('R local spm', R, '[Ohm]')
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(t_final, V_ocv_spm, 'b--')
    ax1.plot(t_final, V_local_spm, 'r*-')
    ax1.scatter(t_final, V_ecm+V_local_spm, c='orange', s=100)
    ax2.plot(t_final, V_ecm, 'k-')


#################################################################
Nunit = 1
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
cc_cond = 3e12
cc_unit_len = 5e-1
cc_unit_area = 20e-4
spm_sim, R, V_ecm = spm_init()
phase['throat.electrical_conductance'] = cc_cond*cc_unit_area/cc_unit_len
phase['throat.electrical_conductance'][net.throats('spm_resistor')] = 1/R


alg = op.algorithms.OhmicConduction(network=net)
alg.setup(phase=phase,
          quantity="pore.potential",
          conductance="throat.electrical_conductance",
          )
alg.set_value_BC(net.pores('pos_terminal'), values=V_ecm)
alg.set_value_BC(net.pores('neg_terminal'), values=0.0)
alg.settings['solver_rtol'] = 1e-15
alg.settings['solver_atol'] = 1e-15
alg.run()
fig = tt.plot_coordinates(net, net.Ps, c=alg['pore.potential'])

potential_pairs = net['throat.conns'][net.throats('spm_resistor')]
P1 = potential_pairs[:, 0]
P2 = potential_pairs[:, 1]
V_local_pnm = alg['pore.potential'][P2] - alg['pore.potential'][P1]
I_local_pnm = alg.rate(throats=net.throats('spm_resistor'), mode='single')
R_local_pnm = V_local_pnm / I_local_pnm

print('*'*30)
print('V local pnm', V_local_pnm, '[V]')
print('I local pnm', I_local_pnm, '[A]')
print('R local pnm', R_local_pnm, '[Ohm]')
