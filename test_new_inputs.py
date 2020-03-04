# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:00:15 2020

@author: Tom
"""
import pybamm
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
pybamm.set_logging_level("INFO")
def current_function(t):
    return pybamm.InputParameter("Current")

model = pybamm.lithium_ion.DFN(
                options={'thermal': 'lumped'}
                )
geometry = model.default_geometry
#param = model.default_parameter_values
param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.NCA_Kim2011)
param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen_2020)
pixel_size = 10.4e-6
t_neg_electrode = 8.0
t_pos_electrode = 7.0
t_neg_cc = 2.0
t_pos_cc = 2.0
t_sep = 2.0
neg_conc = 23800.0
pos_conc = 27300.0
e_brug = 1.5
param.update({"Electrode height [m]": "[input]",
              "Electrode width [m]": 0.065,
              "Current function [A]": current_function,
              "Negative electrode thickness [m]": t_neg_electrode*pixel_size,
              "Positive electrode thickness [m]": t_pos_electrode*pixel_size,
              "Separator thickness [m]": t_sep*pixel_size,
              "Negative current collector thickness [m]": t_neg_cc*pixel_size,
              "Positive current collector thickness [m]": t_pos_cc*pixel_size,
              "Separator porosity": 1.0,
#              'Electrolyte conductivity [S.m-1]': 400.0,
              "Negative electrode Bruggeman coefficient (electrolyte)": e_brug,
              "Negative electrode Bruggeman coefficient (electrode)": 2.0,
              "Positive electrode Bruggeman coefficient (electrolyte)": e_brug,
              "Positive electrode Bruggeman coefficient (electrode)": 2.0,
              "Separator Bruggeman coefficient (electrolyte)": e_brug,
              "Separator Bruggeman coefficient (electrode)": 2.0,
#              "Initial concentration in negative electrode [mol.m-3]": neg_conc,
#              "Initial concentration in positive electrode [mol.m-3]": pos_conc,
#              "Negative electrode conductivity [S.m-1]": 100,
#              "Positive electrode conductivity [S.m-1]": 100,
#              "Lower voltage cut-off [V]": 3.2,
#              "Upper voltage cut-off [V]": 4.7,
              })
nominal_1c_rate = 1.55 #  A
C_rate = 8.0
I_app = C_rate*nominal_1c_rate
param.update({"Current": "[input]"}, check_already_exists=False)
param.process_model(model)
param.process_geometry(geometry)
inputs = {"Electrode height [m]": 1.21,
          "Current": I_app}
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 10, var.r_p: 10}
spatial_methods = model.default_spatial_methods

solver = pybamm.CasadiSolver(mode='safe')
sim = pybamm.Simulation(
    model=model,
    geometry=geometry,
    parameter_values=param,
    var_pts=var_pts,
    spatial_methods=spatial_methods,
    solver=solver,
)
hours=1.0
sim.solve(t_eval=np.arange(0, hours*3600, np.int(np.ceil(100/C_rate))), inputs=inputs)
sim_time = sim.solution['Time [h]'](sim.solution.t[-1])
#plt.figure()
#plt.plot(sim.solution['X-averaged positive particle surface concentration [mol.m-3]'](sim.solution.t))
#plt.plot(sim.solution['X-averaged negative particle surface concentration [mol.m-3]'](sim.solution.t))

#plt.figure()
#result_template = None
#variables = {
##    "Negative electrode average extent of lithiation": result_template,
##    "Positive electrode average extent of lithiation": result_template,
#    "X-averaged negative particle surface concentration [mol.m-3]": result_template, 
#    "X-averaged positive particle surface concentration [mol.m-3]": result_template,
#    "Terminal voltage [V]": result_template,
#    "X-averaged total heating [W.m-3]": result_template,
#    "Time [h]": result_template,
#    "Current collector current density [A.m-2]": result_template,
##        "Local ECM resistance [Ohm]": result_template,
#}
#overpotentials = {
#    "X-averaged battery reaction overpotential [V]": result_template,
#    "X-averaged battery concentration overpotential [V]": result_template,
#    "X-averaged battery electrolyte ohmic losses [V]": result_template,
#    "X-averaged battery solid phase ohmic losses [V]": result_template,
#    "Change in measured open circuit voltage [V]": result_template,
#}
#lithiations = {
#    "Negative electrode average extent of lithiation": result_template,
#    "Positive electrode average extent of lithiation": result_template,
#}
#for key in variables.keys():
#    print(key)
#    variables[key] = sim.solution[key](sim.solution.t)
#for key in lithiations.keys():
#    print(key)
#    if 'Negative' in key:
#        x=0
#    else:
#        x=1
#    lithiations[key] = sim.solution[key](sim.solution.t, x=x)

print('Capacity', sim_time*I_app)

plot = pybamm.QuickPlot(sim.solution)
plot.dynamic_plot()

keys = list(param.keys())
keys.sort()
for k in keys:
    if 'porosity' in k.lower():
        print(k, param[k])

kappa_e = param['Electrolyte conductivity [S.m-1]']


#c_e = sim.solution['Electrolyte concentration [mol.m-3]'](sim.solution.t, x= np.linspace(0, 1.0, 101))
#k = []
#T = T_ref = 298.15
#for c in c_e.flatten():
#    k.append(kappa_e(c, T, T_ref, np.nan, 8.314).value)
#k = np.asarray(k)
#k = k.reshape(c_e.shape)


#plt.figure()
#c_e = np.arange(10, 1600, 10)
#for T in np.arange(-40, 60, 10):
#    plt.plot(c_e, [kappa_e(c, 298.15+T, 298.15, np.nan, 8.314).value for c in c_e], label=T)
#plt.legend()
