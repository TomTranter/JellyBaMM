# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:00:15 2020

@author: Tom
"""
import pybamm
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
def current_function(t):
    return pybamm.InputParameter("Current")

e_height = 1e-3

model = pybamm.lithium_ion.SPMe()
geometry = model.default_geometry
param = model.default_parameter_values
pixel_size = 10.4e-6
t_neg_electrode = 8.0
t_pos_electrode = 7.0
t_neg_cc = 2.0
t_pos_cc = 2.0
t_sep = 2.0
neg_conc = 23800.0
pos_conc = 27300.0
param.update({"Electrode height [m]": "[input]",
              "Current function [A]": current_function,
              "Negative electrode thickness [m]": t_neg_electrode*pixel_size,
              "Positive electrode thickness [m]": t_pos_electrode*pixel_size,
              "Separator thickness [m]": t_sep*pixel_size,
              "Negative current collector thickness [m]": t_neg_cc*pixel_size,
              "Positive current collector thickness [m]": t_pos_cc*pixel_size,
              "Initial concentration in negative electrode [mol.m-3]": neg_conc,
              "Initial concentration in positive electrode [mol.m-3]": pos_conc,
              "Negative electrode conductivity [S.m-1]": 100,
              "Positive electrode conductivity [S.m-1]": 100,
              "Lower voltage cut-off [V]": 3.2,
              "Upper voltage cut-off [V]": 4.7,
              })
param.update({"Current": "[input]"}, check_already_exists=False)
param.process_model(model)
param.process_geometry(geometry)
inputs = {"Electrode height [m]": e_height,
          "Current": 1e-3}
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
sim.solve(t_eval=np.linspace(0, 3600, 100), inputs=inputs)
#plot = pybamm.QuickPlot(sim.solution)
#plot.dynamic_plot()
plt.figure()
plt.plot(sim.solution['X-averaged positive particle surface concentration [mol.m-3]'](sim.solution.t))
plt.plot(sim.solution['X-averaged negative particle surface concentration [mol.m-3]'](sim.solution.t))

#plt.figure()
result_template = None
variables = {
#    "Negative electrode average extent of lithiation": result_template,
#    "Positive electrode average extent of lithiation": result_template,
    "X-averaged negative particle surface concentration [mol.m-3]": result_template, 
    "X-averaged positive particle surface concentration [mol.m-3]": result_template,
    "Terminal voltage [V]": result_template,
    "X-averaged total heating [W.m-3]": result_template,
    "Time [h]": result_template,
    "Current collector current density [A.m-2]": result_template,
#        "Local ECM resistance [Ohm]": result_template,

    "X-averaged battery reaction overpotential [V]": result_template,
    "X-averaged battery concentration overpotential [V]": result_template,
    "X-averaged battery electrolyte ohmic losses [V]": result_template,
    "X-averaged battery solid phase ohmic losses [V]": result_template,
    "Change in measured open circuit voltage [V]": result_template,
}
lithiations = {
    "Negative electrode average extent of lithiation": result_template,
    "Positive electrode average extent of lithiation": result_template,
}
for key in variables.keys():
    print(key)
    variables[key] = sim.solution[key](sim.solution.t)
for key in lithiations.keys():
    print(key)
    if 'Negative' in key:
        x=0
    else:
        x=1
    lithiations[key] = sim.solution[key](sim.solution.t, x=x)