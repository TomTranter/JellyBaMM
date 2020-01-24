# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:09:33 2020

@author: Tom
"""
import pybamm
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
pybamm.set_logging_level('INFO')

I_typical = 1.0

def current_function(t):
    return pybamm.InputParameter("Current")

def run_model(increment_current=True):
    model = pybamm.lithium_ion.SPM()
    geometry = model.default_geometry
    param = model.default_parameter_values
    param.update(
        {
            "Typical current [A]": I_typical,
            "Current function [A]": current_function,
            "Current": "[input]",
#            "Electrode height [m]": "[input]",
        }
    )
    param.process_model(model)
    param.process_geometry(geometry)
    A_cc = param.process_symbol(pybamm.geometric_parameters.A_cc).evaluate()
    var = pybamm.standard_spatial_vars
    var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.r_n: 10, var.r_p: 10}
    spatial_methods = model.default_spatial_methods

    solver = pybamm.CasadiSolver()
    sim = pybamm.Simulation(
        model=model,
        geometry=geometry,
        parameter_values=param,
        var_pts=var_pts,
        spatial_methods=spatial_methods,
        solver=solver,
    )
    t_eval = np.linspace(0, 0.1, 100)
    dt = np.diff(t_eval)
    currents = []
    for i, t in enumerate(dt):
        if increment_current:
            I_app = I_typical+(i/100)
        else:
            I_app = I_typical
        sim.step(dt=t, inputs=  {
                                "Current": I_app,
#                                "Electrode height [m]": 0.1
                                }, save=True)
        currents.append(I_app)
    plt.figure()
    plt.plot(sim.solution["Measured open circuit voltage [V]"](sim.solution.t))
    plt.figure()
    plt.plot(currents)  
    plt.plot(sim.solution["Current collector current density [A.m-2]"](sim.solution.t)*A_cc, 'r--')
    print(sim.solution.inputs)

run_model(True)
run_model(False)

