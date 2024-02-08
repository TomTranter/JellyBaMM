# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:46:59 2020

@author: Tom
"""

import openpnm as op
import matplotlib.pyplot as plt
import jellybamm
import configparser
import os
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
print(jellybamm.lump_thermal_props(config))

jellybamm.print_config(config)

fig, axes = plt.subplots(2, 1, figsize=(6, 12))

for ax, [n, p] in enumerate([['0', '-1'], ['tesla', 'tesla']]):
    config.set('GEOMETRY', 'pos_tabs', p)
    config.set('GEOMETRY', 'neg_tabs', n)
    project, arc_edges = jellybamm.make_spiral_net(config)
    net = project.network
    plt.sca(axes[ax])
    fig = jellybamm.plot_topology(net, fig)
    axes[ax].axis('equal')
    t = axes[ax].text(-0.0, 1.0, abc[ax], transform=axes[ax].transAxes,
                      fontsize=14, va='top')
    t.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='black'))
fig.tight_layout()
plt.savefig(os.path.join(save_im_path, 'fig0_40.png'), dpi=600)
