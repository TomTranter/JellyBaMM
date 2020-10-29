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
e_height = 0.5
length_3d = 0.065
pixel_size= 10.4e-6
T_non_dim = 0.0
fig, (ax1, ax2) = plt.subplots(1, 2)
fig2, (ax3, ax4) = plt.subplots(1, 2)

config = ecm.load_test_config()

for I_app in [1.0, 2.0, 3.0]:
    spm_sim = ecm.make_spm(I_app, config)
    inputs = {"Current": I_app,
              'Electrode height [m]': e_height}

    dt = 100
    t = 0
    v_check = True
    t_eval = np.linspace(0, 2*3600, 201)
    # spm_sim.solve(t_eval, inputs=inputs)
    i = 0
    while v_check and i < 7200/dt:
        spm_sim.step(dt=dt,
                     inputs=inputs)
        v_check = ecm.check_vlim(spm_sim.solution, low=2.5, high=4.7)
        i += 1
        print(i)
    
    t = spm_sim.solution['Time [h]'].entries
    c = t*I_app
    v = spm_sim.solution['Terminal voltage [V]'].entries
    lithiation_neg = spm_sim.solution["X-averaged negative electrode extent of lithiation"].entries
    lithiation_pos = spm_sim.solution["X-averaged positive electrode extent of lithiation"].entries
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