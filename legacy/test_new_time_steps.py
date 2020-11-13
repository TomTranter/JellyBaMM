# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:15:00 2020

@author: Tom
"""

import pybamm
import ecm
import numpy as np
import matplotlib.pyplot as plt
import configparser
import os
plt.close('all')

save_root = 'C:\\Code\\pybamm_pnm_case1'
config = configparser.ConfigParser()
config.read(os.path.join(save_root, 'config.txt'))
I_typical = 1.0
e_height = 0.2
spm_sim = ecm.make_spm(I_typical, config)
T_non_dim = 0.0
inputs = {"Current": I_typical,
          'Electrode height [m]': e_height}
external_variables = {"X-averaged cell temperature": T_non_dim}
spm_sol = spm_sim.solver.solve(model=spm_sim.built_model, t_eval=np.linspace(0, 3600, 100),
                               external_variables=external_variables,
                               inputs=inputs)

plt.figure()
plt.plot(spm_sol["Negative electrode average extent of lithiation"](spm_sol.t))
overpotentials = [
        "X-averaged reaction overpotential [V]",
        "X-averaged concentration overpotential [V]",
        "X-averaged electrolyte ohmic losses [V]",
        "X-averaged solid phase ohmic losses [V]",
        "Change in measured open circuit voltage [V]",
        "Local ECM resistance [Ohm]"
    ]
#plot = pybamm.QuickPlot(spm_sol, output_variables=overpotentials)
#plot = pybamm.QuickPlot(spm_sol)
#plot.dynamic_plot()