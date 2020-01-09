#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:05:55 2020

@author: thomas
"""

import pybamm
import numpy as np
pybamm.logger.level=10
model_options = {
    "thermal": "x-lumped",
    "external submodels": ["thermal"],
}
def current_function(t):
    return pybamm.InputParameter("Current")
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
t_eval = np.linspace(0, 0.02, 51)
T_av = 0.0
for i in np.arange(1, len(t_eval) - 1):
    print('*'*10)
    print(i)
    dt = t_eval[i + 1] - t_eval[i]
    external_variables = {"X-averaged cell temperature": T_av}
    T_av += 1.0
    sim.step(dt, external_variables=external_variables, inputs={"Current": 1.0})
