# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:13:35 2020

@author: Tom
"""

import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
import openpnm.topotools as tt

# >!----!''!---!>
spacing = 10.4e-6
area = 1.0
Nseries = 1+8+2+7+1
Nparallel = 1
net = op.network.Cubic([Nseries+1, 1, 1], spacing=10.4e-6)
a_al = 8.51e-2
a_lic6 = 1.42e-3
a_sep = 1.19e-3
a_lico = 6.88e-4
a_cu = 1.11e-1
net['throat.thermal_diffusivity'] = 1e-12
net['throat.thermal_diffusivity'][0] = a_al
net['throat.thermal_diffusivity'][1:9] = a_lic6
net['throat.thermal_diffusivity'][9:11] = a_sep
net['throat.thermal_diffusivity'][11:18] = a_lico
net['throat.thermal_diffusivity'][18] = a_cu
diffs = net['throat.thermal_diffusivity'].copy()
#print(diffs)
net['throat.length'] = spacing
net['throat.area'] = area
alg = op.algorithms.FickianDiffusion(network=net)
phase = op.phases.GenericPhase(network=net)
phase['throat.diffusive_conductance'] = net['throat.thermal_diffusivity']*net['throat.area']/net['throat.length']
alg.setup(phase=phase)
alg.set_value_BC(pores=[0], values=1.0)
alg.set_value_BC(pores=[-1], values=0.0)
alg.run()
a_series = alg.calc_effective_diffusivity(domain_area=area, domain_length=spacing*net.Nt)
print(a_series)

Nparallel = 4
net2 = op.network.Cubic([Nparallel, Nseries, 1], spacing=10.4e-6)
rows = np.indices([Nparallel, Nseries])[0]
net2['pore.row'] = rows.flatten()
conns = net2['throat.conns']
same_row = net2['pore.row'][conns[:, 0]] == net2['pore.row'][conns[:, 1]]
net2['throat.thermal_diffusivity'] = 1.0
diffs2 = np.hstack((diffs, diffs, diffs))
net2['throat.thermal_diffusivity'][~same_row] = diffs2
net2['throat.length'] = spacing
net2['throat.area'] = area
phase2 = op.phases.GenericPhase(network=net2)
phase2['throat.diffusive_conductance'] = net2['throat.thermal_diffusivity']*net2['throat.area']/net2['throat.length']
alg2 = op.algorithms.FickianDiffusion(network=net2)
alg2.setup(phase=phase)
alg2.set_value_BC(pores=net2.pores()[net2['pore.row']==0], values=1.0)
alg2.set_value_BC(pores=net2.pores()[net2['pore.row']==2], values=0.0)
alg2.run()
a_parallel = alg.calc_effective_diffusivity(domain_area=area*len(diffs), domain_length=spacing*(Nparallel-1))
print(a_parallel)
fig = tt.plot_coordinates(net2, pores=net2.Ps, c=net2['pore.row'])
col = phase2['throat.diffusive_conductance'][~same_row]
for c in np.unique(col):
    fig = tt.plot_connections(net2, throats=net2.throats()[~same_row][np.where(col == c)], fig=fig)
ax = fig.gca()
ax.set_xlim(net2['pore.coords'][:, 0].min()-spacing, net2['pore.coords'][:, 0].max()+spacing)
ax.set_ylim(net2['pore.coords'][:, 1].min()-spacing, net2['pore.coords'][:, 1].max()+spacing)
plt.show()
