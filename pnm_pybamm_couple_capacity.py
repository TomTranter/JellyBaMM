#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:14:20 2019

@author: thomas
"""

import pybamm
import openpnm as op
import matplotlib.pyplot as plt
import os
import jellysim as js


plt.close("all")
use_tomo = True
wrk = op.Workspace()
input_dir = os.path.join(os.getcwd(), 'input')
pybamm.set_logging_level(60)

# Simulation options
opt = {'domain': 'model',
       'Nlayers': 19,
       'cp': 1148,
       'rho': 5071.75,
       'K0': 1,
       'T0': 303,
       'heat_transfer_coefficient': 10,
       'length_3d': 0.065,
       'I_app_mag': 2.5,
       'cc_cond_neg': 3e7,
       'cc_cond_pos': 3e7,
       'dtheta': 10,
       'spacing': 1e-5}

sim = js.coupledSim()
sim.setup(opt)
spm = sim.runners['spm']
for I_app_mag in [3.0, 2.0, 1.0]:
    print('*'*30)
    print('I app', I_app_mag)
    spm.test_equivalent_capacity(I_app_mag=I_app_mag)
    print('*'*30)

