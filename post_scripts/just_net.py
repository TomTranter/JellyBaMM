# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:46:59 2020

@author: Tom
"""

import pybamm
import openpnm as op
import matplotlib.pyplot as plt
import ecm
import configparser
import os
import numpy as np
from string import ascii_lowercase as abc

plt.close("all")

#pybamm.set_logging_level("INFO")
wrk = op.Workspace()
wrk.clear()
save_im_path = 'D:\\pybamm_pnm_results\\46800\\figures'

#    save_root = sys.argv[-1]
save_root = 'D:\\pybamm_pnm_results\\46800\\test'
print(save_root)
config = configparser.ConfigParser()
config.read(os.path.join(save_root, 'config.txt'))
print(ecm.lump_thermal_props(config))

for sec in config.sections():
    print('='*67)
    print(sec)
    print('='*67)
    for key in config[sec]:
        print('!', key.ljust(30, ' '), '!', config.get(sec, key).ljust(30, ' '), '!')
        print('-'*67)

fig, axes = plt.subplots(2, 1, figsize=(6, 12))

for ax, [n, p] in enumerate([['0', '-1'], ['tesla', 'tesla']]):
    config.set('GEOMETRY', 'pos_tabs', p)
    config.set('GEOMETRY', 'neg_tabs', n)
    project, arc_edges = ecm.make_spiral_net(config)
    net = project.network
    plt.sca(axes[ax])
    fig = ecm.plot_topology(net, fig)
    axes[ax].axis('equal')
    t = axes[ax].text(-0.0, 1.0, abc[ax], transform=axes[ax].transAxes,
                fontsize=14, va='top')
    t.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='black'))
fig.tight_layout()
plt.savefig(os.path.join(save_im_path, 'fig0_40.png'), dpi=600)
#plt.axis('off')
#ecm.plot_topology(net)
#spm_map = ecm.interpolate_spm_number_model(project, dim=2000).astype(int)
#mask = spm_map == -1
#fig, ax1 = plt.subplots(1)
#arr = spm_map.astype(float)
#arr[mask] = np.nan
#ax1.imshow(arr)
#np.savez('im_spm_map_46800', spm_map)
#wrk.save_project(project=project, filename='46800')


