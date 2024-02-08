#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:42:02 2019

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt
import openpnm as op

plt.close('all')
wrk = op.Workspace()
spacing = 1e-5
L = 9e-3
Nx = int(L / spacing) + 1
net = op.network.Cubic(shape=[Nx, 1, 1], spacing=spacing)
# translate to origin
net['pore.coords'] -= np.array([spacing, spacing, spacing]) / 2
net.add_boundary_pores(labels=['left', 'right'], spacing=0.0)

geo = op.geometry.GenericGeometry(network=net, pores=net.Ps, throats=net.Ts)
geo['pore.diameter'] = spacing
geo['throat.diameter'] = spacing
geo['throat.length'] = spacing
geo['throat.area'] = (spacing)**2
geo['pore.area'] = (spacing)**2
geo['pore.volume'] = geo['pore.area'] * spacing
geo['throat.volume'] = 0.0

T0 = 303
K = 1
cp = 1399
rho = 2055
alpha = K / (cp * rho)
phase = op.phases.GenericPhase(network=net)
phase['pore.conductivity'] = alpha
phys = op.physics.GenericPhysics(network=net, geometry=geo, phase=phase)
conc = 1.0  # mol/m^3
phys['throat.conductance'] = conc * alpha * geo['throat.area'] / geo['throat.length']

Q = 25000 / (cp * rho)
heat_transfer_coefficient = 10
hc = heat_transfer_coefficient / (cp * rho)
bTs = net.throats('*boundary')
phys['throat.conductance'][bTs] = hc * geo['throat.area'][bTs]

Ps_x = net['pore.coords'][:, 0]
source = Q * net['pore.volume']
phys['pore.source.S1'] = 0.0
phys['pore.source.S2'] = source
phys['pore.source.rate'] = source


def run_transport(network, method='steady', t_initial=0,
                  t_final=60 * 60 * 10, t_step=60, t_output=60):
    if method == 'steady':
        alg = op.algorithms.ReactiveTransport(network=network)
        alg.setup(
            phase=phase,
            quantity="pore.temperature",
            conductance="throat.conductance",
            rxn_tolerance=1e-12,
            relaxation_source=0.9,
            relaxation_quantity=0.9,
        )

    else:
        alg = op.algorithms.TransientReactiveTransport(network=network)
        alg.setup(phase=phase,
                  conductance='throat.conductance',
                  quantity='pore.temperature',
                  t_initial=t_initial,
                  t_final=t_final,
                  t_step=t_step,
                  t_output=t_output,
                  t_tolerance=1e-9,
                  t_precision=12,
                  rxn_tolerance=1e-9,
                  t_scheme='implicit')
        alg.set_IC(values=T0)

    BP = net.pores('pore.right_boundary')
    alg.set_value_BC(pores=BP, values=T0)
    Ps = net.pores('internal')
    alg.set_source(propname='pore.source', pores=Ps)
    alg.run()
    return alg


alg = run_transport(network=net, method='transient')
res = alg.results()
times = list(res.keys())
plt.figure()
center = []
mid = []
mid_coord = int(Nx / 2)
end = []
for time in times[1:]:
    data = alg[time]
    plt.plot(data)
    center.append(data[0])
    mid.append(data[mid_coord])
    end.append(data[-3])
last_time = float(time.split('@')[-1]) / (60 * 60)

alg = run_transport(network=net, method='steady')
plt.plot(alg['pore.temperature'], 'k--')
hrs = np.linspace(0, last_time, len(center))
plt.figure()
plt.plot(hrs, center)
plt.plot(hrs, mid)
plt.plot(hrs, end)
