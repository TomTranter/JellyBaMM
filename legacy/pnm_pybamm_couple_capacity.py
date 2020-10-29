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
import numpy as np


plt.close("all")
use_tomo = True
wrk = op.Workspace()
input_dir = os.path.join(os.getcwd(), 'input')
#pybamm.set_logging_level(10)

# Simulation options
opt = {'domain': 'tomography',
       'Nlayers': 17,
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
pnm = sim.runners['pnm']
spm = sim.runners['spm']
#pnm.export_pnm(filename=opt['domain'])
for I_app_mag in [1.0, 2.0, 3.0]:
    print('*'*30)
    print('I app', I_app_mag)
    spm.test_equivalent_capacity(I_app_mag=I_app_mag)
    print('*'*30)


def specific_cap(diam, height, cap):
    a = np.pi * (diam / 2) ** 2
    v = a * height
    spec = cap / v
    print('Volume', v, 'cm-3', 'Specific Capacity', spec, 'mAh.cm-3')


print('State of Art 18650')
specific_cap(1.8, 6.5, 2500)
