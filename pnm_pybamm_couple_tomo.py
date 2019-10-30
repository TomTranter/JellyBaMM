#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Call OpenPNM to set up a global domain comprised of unit cells that have
electrochemistry calculated by PyBAMM.
Heat is generated in the unit cells which act as little batteries connected
in parallel.
The global heat equation is solved on the larger domain as well as the
global potentials which are used as boundary conditions for PyBAMM and a
lumped unit cell temperature is applied and updated over time
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
pybamm.set_logging_level(10)

# Simulation options
opt = {'domain': 'tomography',
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
#sim.run_thermal()
#sim.runners['spm'].test_equivalent_capacity()
sim.run(n_steps=2, time_step=0.01)
#sim.plots()
sim.save('test')
