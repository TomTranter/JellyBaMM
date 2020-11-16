# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:37:52 2020

@author: Tom
"""

import matplotlib.pyplot as plt
import numpy as np
import openpnm as op
from openpnm.topotools import plot_connections as pconn
from openpnm.topotools import plot_coordinates as pcord
plt.close('all')
net = op.network.Cubic(shape=[3, 3, 1], spacing=1.0, connectivity=18)
fig = pcord(net, net.Ps, c='b')
#fig = pconn(net, net.Ts, c='r', fig=fig)

conns = net['throat.conns']
coords = net['pore.coords']
v = coords[conns[:, 1]] - coords[conns[:, 0]]

z = np.array([0, 0, 1])
perp = np.cross(v, z)

v_lens = np.linalg.norm(v, axis=1)
v_unit = v / np.vstack((v_lens, v_lens, v_lens)).T
v_unit_perp = perp / np.vstack((v_lens, v_lens, v_lens)).T

zigzag = np.array([0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0])
segs = len(zigzag)
p_start = coords[conns[:, 0]]
for i in range(segs):
    p_end = p_start + v*(1/segs) + perp*(2/segs)*zigzag[i]

    for t in range(len(p_start)):
        plt.plot([p_start[t, 0], p_end[t, 0]], [p_start[t, 1], p_end[t, 1]])
    p_start = p_end