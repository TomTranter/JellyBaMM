#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 12:16:25 2020

@author: thomas
"""

import pybamm
from pybamm import EvaluatorPython as ep
import numpy as np
import matplotlib.pyplot as plt
import ecm
import configparser
import os

save_root = os.getcwd()
config = ecm.load_test_config()
pybamm.set_logging_level('ERROR')
I_typical = 1.0
typical_height = 1.2
T_ref = 298.15
do_thermal = True
spm_sim = ecm.make_spm(I_typical, config)
dt = 50
voltage = []
spm_sol = None
key = 'Terminal voltage [V]'
param = spm_sim.parameter_values
temp_inputs = {"Current": I_typical,
               'Electrode height [m]': typical_height}
temp_parms = spm_sim.built_model.submodels["thermal"].param
Delta_T = param.process_symbol(temp_parms.Delta_T).evaluate(inputs=temp_inputs)
spm_temperature = 298.15
T_non_dim = (spm_temperature - T_ref) / Delta_T
for t in range(170):
    spm_sol = ecm.step_spm((spm_sim.built_model, spm_sim.solver,
                            spm_sol, I_typical, typical_height, dt, T_non_dim, False))
    voltage.append(spm_sol[key].entries[-1])

voltage = np.asarray(voltage)
plt.figure()
plt.plot(voltage)
