#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Call OpenPNM to set up a global domain comprised of unit cells that have
electrochemistry calculated by PyBAMM.
Heat is generated in the unit cells which act as little batteries connected
in parallel.
The global heat equation is solved on the larger domain as well as the
global potentials which are used as boundary conditions for PyBAMM and a
lumped unit cell temperature is applied and updated over time
"""

import openpnm as op
import matplotlib.pyplot as plt
import os
import jellysim as js
import numpy as np
import openpnm.topotools as tt


def plot_pore_data(net, data):
    fig, ax = plt.subplots(1)
    bulk_Ps = net.pores("free_stream", mode="not")
    coords = net["pore.coords"][bulk_Ps]
    xmin = coords[:, 0].min() * 1.05
    ymin = coords[:, 1].min() * 1.05
    xmax = coords[:, 0].max() * 1.05
    ymax = coords[:, 1].max() * 1.05
    mappable = ax.scatter(coords[:, 0], coords[:, 1], c=data[bulk_Ps])
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.colorbar(mappable)

plt.close("all")
use_tomo = True
wrk = op.Workspace()
input_dir = os.path.join(os.getcwd(), 'input')
#pybamm.set_logging_level(10)
I_app = 1.0
# Simulation options
opt = {'domain': 'model',
       'Nlayers': 17,
       'cp': 1399.0,
       'rho': 2055.0,
       'K0': 1.0,
       'T0': 303,
       'heat_transfer_coefficient': 10,
       'length_3d': 0.065,
       'I_app_mag': I_app*1.0,
       'cc_cond_neg': 3e7,
       'cc_cond_pos': 3e7,
       'dtheta': 10,
       'spacing': 1e-5}

sim = js.coupledSim()
sim.setup(opt)

pnm = sim.runners['pnm']
pnm.plot_topology()


terminal_an = np.logical_and(pnm.net['pore.cell_id'] == 0,
                             pnm.net['pore.region_id'] == 2)

max_id = pnm.net['pore.cell_id'].max()

terminal_cat = np.logical_and(pnm.net['pore.cell_id'] == max_id,
                              pnm.net['pore.region_id'] == 4)

pnm.net['pore.terminal_an'] = False
pnm.net['pore.terminal_an'][terminal_an] = True
pnm.net['pore.terminal_cat'] = False
pnm.net['pore.terminal_cat'][terminal_cat] = True

phase = pnm.phase

an = pnm.net["pore.region_id"] == 1
an_cc = pnm.net["pore.region_id"] == 2
cat = pnm.net["pore.region_id"] == 3
cat_cc = pnm.net["pore.region_id"] == 4
sep = pnm.net["pore.region_id"] == 5
inner = pnm.net["pore.left"]
outer = pnm.net["pore.right"]
an_Ts = pnm.net.find_neighbor_throats(pores=an, mode='xnor')
an_cc_Ts = pnm.net.find_neighbor_throats(pores=an_cc, mode='xnor')
cat_Ts = pnm.net.find_neighbor_throats(pores=cat, mode='xnor')
cat_cc_Ts = pnm.net.find_neighbor_throats(pores=cat_cc, mode='xnor')
t_sep = pnm.net.throats("separator*")


# Set up Phase and Physics
alpha = 0.1
phase["pore.electrical_conductivity"] = alpha  # [W/(m.K)]
phase["throat.electrical_conductance"] = (
    alpha * pnm.geo["throat.area"] / pnm.geo["throat.length"]
)
phase["throat.electrical_conductance"][an_cc_Ts] *= 1e3
phase["throat.electrical_conductance"][cat_cc_Ts] *= 1e3
phase["throat.electrical_conductance"][an_Ts] *= 0.00001
phase["throat.electrical_conductance"][cat_Ts] *= 0.00001
phase["throat.electrical_conductance"][t_sep] *= 0.00001
phys = pnm.phys

wrk.copy_project(wrk['sim_01'], 'sim_02')
tt.trim(pnm.net, pnm.net["pore.region_id"] == 1)
tt.trim(pnm.net, pnm.net["pore.region_id"] == 2)
p_sep = pnm.net["pore.region_id"]==5
alg = op.algorithms.OhmicConduction(network=pnm.net)
alg.setup(
    phase=phase,
    quantity="pore.potential",
    conductance="throat.electrical_conductance",
)
#bulk_Ps = self.net.pores("free_stream", mode="not")
#alg.set_source("pore.source", bulk_Ps)
j = pnm.net['pore.volume'][p_sep]
alg.set_rate_BC(p_sep, values=-I_app*j)
alg.set_value_BC(pnm.net.pores("terminal_cat"), values=3.40)
alg.run()
plot_pore_data(pnm.net, data=alg['pore.potential'])

net_2 = wrk['sim_02'].network
phase_2 = wrk['sim_02'].phases('phase_01')
tt.trim(net_2, net_2["pore.region_id"] == 1)
tt.trim(net_2, net_2["pore.region_id"] == 2)
p_sep = net_2["pore.region_id"]==5

alg2 = op.algorithms.OhmicConduction(network=net_2)
alg2.setup(
        phase=phase_2,
        quantity="pore.potential",
        conductance="throat.electrical_conductance",
)
#bulk_Ps = self.net.pores("free_stream", mode="not")
#alg.set_source("pore.source", bulk_Ps)
j = net_2['pore.volume'][p_sep]
alg2.set_rate_BC(p_sep, values=I_app*j)
alg2.set_value_BC(net_2.pores("terminal_cat"), values=0.0)
alg2.run()

plot_pore_data(net_2, data=alg2['pore.potential'])
