#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 09:05:37 2020

@author: thomas
"""

import pybamm
import numpy as np
import matplotlib.pyplot as plt
from pybamm import EvaluatorPython as ep
plt.close('all')

model_options = {
    "thermal": "x-lumped",
#    "external submodels": ["thermal"],
}
model = pybamm.lithium_ion.SPMe(model_options)
solver = pybamm.CasadiSolver()
#solver = model.default_solver
sim = pybamm.Simulation(model, solver=solver)
t_eval = np.linspace(0, 0.01, 10)
T_av = 0.0
for i in np.arange(1, len(t_eval) - 1):
    dt = t_eval[i + 1] - t_eval[i]
#    external_variables = {"X-averaged cell temperature": T_av}
    external_variables = None
    T_av += 1.0
    sim.step(dt, external_variables=external_variables)

eval_func_a = ep(sim.built_model.variables["Measured open circuit voltage [V]"])
eval_func_b = ep(sim.built_model.variables["X-averaged total heating [W.m-3]"])

t_last, y_last = sim.solution.t[-1], sim.solution.y[:, -1]
t_all, y_all = sim.solution.t, sim.solution.y

variables = {
    "Measured open circuit voltage [V]": sim.built_model.variables[
        "Measured open circuit voltage [V]"
    ],
    "X-averaged total heating [W.m-3]": sim.built_model.variables[
        "X-averaged total heating [W.m-3]"
    ]
}
var = pybamm.post_process_variables(
    variables, sim.solution.t, sim.solution.y, mesh=sim.mesh
)

plt.figure()
plt.plot(t_all, var["Measured open circuit voltage [V]"](t_all, y_all))
plt.figure()
plt.plot(t_all, var["X-averaged total heating [W.m-3]"](t_all, y_all))

# single eval fine for both
print(eval_func_a.evaluate(t, y))
print(eval_func_b.evaluate(t, y))
# Multi eval fine for a
print(eval_func_a.evaluate(t_all, y_all))
plt.figure()
plt.plot(t_all, eval_func_a.evaluate(t_all, y_all).flatten())
# Will fail here for b
print(eval_func_b.evaluate(t_all, y_all))
plt.figure()
plt.plot(t_all, eval_func_b.evaluate(t_all, y_all).flatten())
