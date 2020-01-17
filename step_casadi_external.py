#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 09:05:37 2020

@author: thomas
"""

import pybamm
import numpy as np
model_options = {
    "thermal": "x-lumped",
    "external submodels": ["thermal"],
}
model = pybamm.lithium_ion.SPMe(model_options)
solver = pybamm.CasadiSolver()
sim = pybamm.Simulation(model, solver=solver)
t_eval = np.linspace(0, 0.01, 10)
T_av = 0.0
for i in np.arange(1, len(t_eval) - 1):
    dt = t_eval[i + 1] - t_eval[i]
    external_variables = {"X-averaged cell temperature": T_av}
    T_av += 1.0
    sim.step(dt, external_variables=external_variables)