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
pybamm.set_logging_level('ERROR')
I_typical = 1.0
do_thermal = True
spm_sim = ecm.make_spm(I_typical, height=0.1, thermal=do_thermal)
spm_sol = ecm.step_spm((spm_sim, None, I_typical, 1e-6,  True))
variables_eval = {}
overpotentials_eval = {}
height = spm_sim.parameter_values["Electrode height [m]"]
width = spm_sim.parameter_values["Electrode width [m]"] 
t1 = spm_sim.parameter_values['Negative electrode thickness [m]']
t2 = spm_sim.parameter_values['Positive electrode thickness [m]']
t3 = spm_sim.parameter_values['Negative current collector thickness [m]']
t4 = spm_sim.parameter_values['Positive current collector thickness [m]']
t5 = spm_sim.parameter_values['Separator thickness [m]']
ttot = t1+t2+t3+t4+t5
A_cc = height * width
variables = [
    "Local ECM resistance [Ohm.m2]",
    "Local ECM voltage [V]",
    "Measured open circuit voltage [V]",
    "Local voltage [V]",
    "Change in measured open circuit voltage [V]",
    "X-averaged total heating [W.m-3]",
]
overpotentials = [
    "X-averaged reaction overpotential [V]",
    "X-averaged concentration overpotential [V]",
    "X-averaged electrolyte ohmic losses [V]",
    "X-averaged solid phase ohmic losses [V]",
    "Change in measured open circuit voltage [V]",
]
for var in variables:
    variables_eval[var] = ep(spm_sim.built_model.variables[var])
for var in overpotentials:
    overpotentials_eval[var] = ep(spm_sim.built_model.variables[var])

temp = ecm.evaluate_python(variables_eval, spm_sol, current=I_typical)
eta = ecm.evaluate_python(overpotentials_eval, spm_sol, current=I_typical)
R = temp[0] / A_cc
V_ecm = temp[1]
#print(temp)
#print(R)
R_max = R * 10

print('Thermal', do_thermal)
for i in range(len(overpotentials)):
    print(overpotentials[i], eta[i])
print('Total eta', np.sum(eta))