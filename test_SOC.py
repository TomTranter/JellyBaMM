# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:05:43 2020

@author: Tom
"""

import pybamm
import ecm
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')
pybamm.set_logging_level("INFO")
e_height = 1.2115397404794837
length_3d = 0.065
pixel_size= 10.4e-6
T_non_dim = 0.0
fig, (ax1, ax2) = plt.subplots(1, 2)
fig2, (ax3, ax4) = plt.subplots(1, 2)

for I_app in [1.0, 2.0, 5.0, 10.0]:
    spm_sim = ecm.make_spm(I_app, thermal=False,
                           length_3d=length_3d, pixel_size=pixel_size)
    inputs = {"Current": I_app,
              'Electrode height [m]': e_height}
    Nsteps = 1000
    dt = 1e-3/I_app
    t = 0
    v_check = True
    
    while t < Nsteps and v_check:
        spm_sim.step(dt=dt,
                     inputs=inputs,
                     check_model=False)
        v_check = ecm.check_vlim(spm_sim, 4.7, 3.2)
    
    t = spm_sim.solution['Time [h]'](spm_sim.solution.t)
    c = t*I_app
    v = spm_sim.solution['Terminal voltage [V]'](spm_sim.solution.t)
    lithiation_neg = spm_sim.solution["Negative electrode average extent of lithiation"](spm_sim.solution.t)
    lithiation_pos = spm_sim.solution["Positive electrode average extent of lithiation"](spm_sim.solution.t)
    ax1.plot(c, v, label=str(I_app)+ ' [A]')
    ax2.plot(t, I_app*v, label=str(I_app)+ ' [A]')
    ax3.plot(c, lithiation_neg, label=str(I_app)+ ' [A]')
    ax4.plot(c, lithiation_pos, label=str(I_app)+ ' [A]')

ax1.set_xlabel('Capacity [Ah]')
ax1.set_ylabel('Terminal voltage [V]')
ax2.set_xlabel('Time [h]')
ax2.set_ylabel('Power [W]')
ax3.set_xlabel('Capacity [Ah]')
ax3.set_ylabel('Lithiation Neg')
ax4.set_xlabel('Capacity [Ah]')
ax4.set_ylabel('Lithiation Pos')

plt.legend()