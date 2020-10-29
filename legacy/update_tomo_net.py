#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:49:29 2020

@author: thomas
"""

import openpnm as op
import openpnm.topotools as tt
import ecm
import os
import numpy as np
import matplotlib.pyplot as plt


plt.close('all')
wrk = op.Workspace()
cwd = os.getcwd()
input_dir = os.path.join(cwd, 'input')
wrk.load_project(os.path.join(input_dir, 'MJ141-mid-top_m_cc.pnm'))
sim_name = list(wrk.keys())[-1]
project = wrk[sim_name]
net = project.network


net['pore.pos_cc'] = net['pore.cc_a']
net['pore.neg_cc'] = net['pore.cc_b']
pos_cc_Ts = net.find_neighbor_throats(net.pores("pos_cc"), mode="xnor")
neg_cc_Ts = net.find_neighbor_throats(net.pores("neg_cc"), mode="xnor")
net['throat.pos_cc'] = False
net['throat.pos_cc'][pos_cc_Ts] = True
net['throat.neg_cc'] = False
net['throat.neg_cc'][neg_cc_Ts] = True
#pos_tab_nodes = net.pores()[net["pore.pos_cc"]][pos_tabs]
#neg_tab_nodes = net.pores()[net["pore.neg_cc"]][neg_tabs]
pos_tab_nodes = net.pores(['pos_cc', 'terminal'], mode='and')
neg_tab_nodes = net.pores(['neg_cc', 'terminal'], mode='and')
net["pore.pos_tab"] = False
net["pore.neg_tab"] = False
net["pore.pos_tab"][pos_tab_nodes[0]] = True
net["pore.neg_tab"][neg_tab_nodes[1]] = True
#fig = tt.plot_coordinates(net, net.Ps)
#fig = tt.plot_connections(net, net.Ts, fig=fig)
net['throat.spm_resistor'] = net['throat.interconnection']
net['throat.free_stream'] = net['throat.stitched']

del net['pore.cc_a']
del net['pore.cc_b']
del net['pore.terminal']
del net['pore.terminal_neighbor']
del net['pore.mirror']
del net['throat.stitched']
del net['throat.separator']
del net['throat.interconnection']

wrk.save_project(project=project, filename=os.path.join(input_dir, 'MJ141-mid-top_m_cc_ecm'))
ecm.plot_topology(net)
