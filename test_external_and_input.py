#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:05:55 2020

@author: thomas
"""

import pybamm
from pybamm import EvaluatorPython as ep
import numpy as np
import ecm
import matplotlib.pyplot as plt
pybamm.logger.level=10
model_options = {
    "thermal": "x-lumped",
    "external submodels": ["thermal"],
}
def current_function(t):
    return pybamm.InputParameter("Current")

def evaluate_python(python_eval, solution, current):
    keys = list(python_eval.keys())
    out = np.zeros(len(keys))
    for i, key in enumerate(keys):
        temp = python_eval[key].evaluate(
                solution.t[-1], solution.y[:, -1], u={"Current": current}
                )
        out[i] = temp
    return out



model = pybamm.lithium_ion.SPMe(model_options)

solver = model.default_solver
sim = pybamm.Simulation(model, solver=solver)
param = sim.parameter_values
param.update(
    {
        "Current function": current_function,
        "Current": "[input]",
    }
)

Nsteps = 50
T_av = 303

def convert_temperature(T_dim):
    temp_parms = sim.model.submodels["thermal"].param
    Delta_T = sim.parameter_values.process_symbol(temp_parms.Delta_T).evaluate()
    T_ref = sim.parameter_values.process_symbol(temp_parms.T_ref).evaluate()
    return (T_dim - T_ref) / Delta_T

I_app = 1.0
external_variables = {"X-averaged cell temperature": convert_temperature(T_av)}
inputs={"Current": I_app}
variables_eval = {}
overpotentials_eval = {}
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

results = np.zeros([Nsteps,  len(variables)])
results_o = np.zeros([Nsteps,  len(overpotentials)])
# Init
dt = 1e-6
sim.step(dt, external_variables=external_variables, inputs=inputs)
dt = 1e-3
for var in variables:
    variables_eval[var] = ep(sim.built_model.variables[var])
for var in overpotentials:
    overpotentials_eval[var] = ep(sim.built_model.variables[var])

for i in range(Nsteps):
    print('*'*10)
    print(i)

#    T_av += 1.0
    sim.step(dt, external_variables=external_variables, inputs=inputs)
    results[i, :] = ecm.evaluate_python(variables_eval,
                                        sim.solution,
                                        I_app)
    results_o[i, :] = ecm.evaluate_python(overpotentials_eval,
                                          sim.solution,
                                          current=I_app)

plt.figure()
plt.plot(results[:, 0])