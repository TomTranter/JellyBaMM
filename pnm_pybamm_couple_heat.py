#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:27:36 2019

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
pybamm.set_logging_level(10)
I_app = 1.0
# Simulation options
opt = {'domain': 'model',
       'Nlayers': 5,
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
spm = sim.runners['spm']
spm.Nunit
Q = np.ones(spm.Nunit)*25000
pnm.run_step_transient(heat_source=Q, time_step=1000, BC_value=303)
