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
Nunit = 10
net = op.network.Cubic([Nunit+1, 2, 1])
print(net.labels())
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
print(net.labels())

fig = tt.plot_coordinates(net, net.pores('pos_cc'), c='b')
fig = tt.plot_coordinates(net, net.pores('pos_terminal'), c='y', fig=fig)
fig = tt.plot_coordinates(net, net.pores('neg_cc'), c='r', fig=fig)
fig = tt.plot_coordinates(net, net.pores('neg_terminal'), c='g', fig=fig)
fig = tt.plot_connections(net, net.throats('pos_cc'), c='b', fig=fig)
fig = tt.plot_connections(net, net.throats('neg_cc'), c='r', fig=fig)
fig = tt.plot_connections(net, net.throats('spm_resistor'), c='k', fig=fig)

phase = op.phases.GenericPhase(network=net)
phase['throat.electrical_conductance'] = 1.0

I = 1.0
alg = op.algorithms.OhmicConduction(network=net)
alg.setup(phase=phase,
          quantity="pore.potential",
          conductance="throat.electrical_conductance",
          )
alg.set_rate_BC(net.pores('pos_terminal'), values=I)
alg.set_value_BC(net.pores('neg_terminal'), values=0.0)
alg.run()
fig = tt.plot_coordinates(net, net.Ps, c=alg['pore.potential'])

potential_pairs = net['throat.conns'][net.throats('spm_resistor')]
P1 = potential_pairs[:, 0]
P2 = potential_pairs[:, 1]
dV_local = alg['pore.potential'][P2] - alg['pore.potential'][P1]
I_local = alg.rate(throats=net.throats('spm_resistor'), mode='single')
R_local = dV_local / I_local
print(dV_local)
print(I_local)
print(R_local)

# set logging level
pybamm.set_logging_level("INFO")

model = pybamm.lithium_ion.SPMe()
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.r_n: 5, var.r_p: 5}

sim = pybamm.Simulation(model, var_pts=var_pts)

t_eval = np.linspace(0, 0.13, 2)
# step through the solver, setting the temperature at each timestep
for i in np.arange(1, len(t_eval) - 1):
    dt = t_eval[i + 1] - t_eval[i]
    sim.step(dt)
