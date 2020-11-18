# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:54:00 2020

@author: Tom
"""

import openpnm as op
import openpnm.topotools as tt
import matplotlib.pyplot as plt

plt.close('all')
net = op.network.Cubic([5, 4, 1], spacing=1.0)
conns = net['throat.conns']
P1 = conns[:, 0]
P2 = conns[:, 1]
v = net['pore.coords'][P2] - net['pore.coords'][P1]
net['throat.vector'] = v
net['throat.x'] = v[:, 0] == 1.0
net['throat.y'] = v[:, 1] == 1.0
fig = tt.plot_coordinates(net, pores=net.pores('left'), c='b')
fig = tt.plot_coordinates(net, pores=net.pores('right'), c='r', fig=fig)
fig = tt.plot_coordinates(net, pores=net.pores('front'), c='y', fig=fig)
fig = tt.plot_coordinates(net, pores=net.pores('back'), c='g', fig=fig)
fig = tt.plot_connections(net, throats=net.throats()[:15], c='pink', fig=fig)
fig = tt.plot_connections(net, throats=net.throats()[15:15 + 8], c='k', fig=fig)
fig = tt.plot_connections(net, throats=net.throats()[15 + 8:], c='r', fig=fig)

phase = op.phases.GenericPhase(network=net)
phase['throat.diffusive_conductance'] = 1.0
alg = op.algorithms.FickianDiffusion(network=net)
alg.setup(phase=phase)
alg.set_value_BC(pores=net.pores('front'), values=1.0)
alg.set_value_BC(pores=net.pores('back'), values=0.0)
alg.run()
deff = alg.calc_effective_diffusivity(domain_area=4.0, domain_length=4.0)
print(deff)
phase['throat.diffusive_conductance'] = 1.0
phase['throat.diffusive_conductance'][:15] = 0.0
alg.run()
deff = alg.calc_effective_diffusivity(domain_area=4.0, domain_length=4.0)
print(deff)
phase['throat.diffusive_conductance'] = 1.0
phase['throat.diffusive_conductance'][:15] = 0.0
phase['throat.diffusive_conductance'][15:15 + 8] = 2.0
fig = tt.plot_coordinates(net, pores=net.pores(), c='b')
blue_ts = phase['throat.diffusive_conductance'] == 1.0
red_ts = phase['throat.diffusive_conductance'] == 2.0
fig = tt.plot_connections(net, throats=net.throats()[blue_ts], c='b', fig=fig)
fig = tt.plot_connections(net, throats=net.throats()[red_ts], c='r', fig=fig)
alg.run()
deff = alg.calc_effective_diffusivity(domain_area=4.0, domain_length=4.0)
print('series', deff, 1 / deff)
phase['throat.diffusive_conductance'] = 1.0
phase['throat.diffusive_conductance'][:15] = 0.0
phase['throat.diffusive_conductance'][15:17] = 2.0
phase['throat.diffusive_conductance'][17:19] = 1.0
phase['throat.diffusive_conductance'][19:21] = 2.0
phase['throat.diffusive_conductance'][21:23] = 1.0
phase['throat.diffusive_conductance'][23:25] = 2.0
phase['throat.diffusive_conductance'][25:27] = 1.0
phase['throat.diffusive_conductance'][27:29] = 2.0
fig = tt.plot_coordinates(net, pores=net.pores(), c='b')
fig = tt.plot_connections(net, throats=net.throats()[blue_ts], c='b', fig=fig)
fig = tt.plot_connections(net, throats=net.throats()[red_ts], c='r', fig=fig)
alg.run()
deff = alg.calc_effective_diffusivity(domain_area=4.0, domain_length=4.0)
print('parallel', deff, 1 / deff)
