#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 08:50:15 2019

@author: thomas
"""

import pybamm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib.collections import LineCollection

#plt.close('all')
Nunit = 20
pixel_size = 10.4e-6
# load model
pybamm.set_logging_level("INFO")

options = {"thermal": "x-lumped",
           "dimensionality": 1,
           "current collector": "potential pair"}
model = pybamm.lithium_ion.SPM(options)
model.use_simplify = False
model.use_to_python = False

# load parameter values and process models and geometry
param = model.default_parameter_values
param.update({})

param.update(
    {
        "Typical current [A]": 1.0,
        "Initial temperature [K]": 303,
        "Negative current collector conductivity [S.m-1]": 1e7,
        "Positive current collector conductivity [S.m-1]": 1e7,
        "Electrode height [m]": 1.0,
        "Electrode width [m]": 0.065,
        "Negative electrode thickness [m]": 6.0*pixel_size,
        "Positive electrode thickness [m]": 9.0*pixel_size,
        "Separator thickness [m]": 1.0*pixel_size,
        "Positive current collector thickness [m]": 1e-5,
        "Negative current collector thickness [m]": 1e-5,
        "Negative tab centre z-coordinate [m]": 0.0,
        "Positive tab centre z-coordinate [m]": 1.0,
        "Positive electrode conductivity [S.m-1]": 0.1,
        "Negative electrode conductivity [S.m-1]": 0.1,
        "Lower voltage cut-off [V]": 3.45,
        "Upper voltage cut-off [V]": 4.7,
        "Heat transfer coefficient [W.m-2.K-1]": 0.1,
    }
)
param["Current function"] = pybamm.GetConstantCurrent()

param.process_model(model)
# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5,
           var.r_n: 5, var.r_p: 5, var.z: Nunit}

# create geometry
geometry = model.default_geometry
param.process_geometry(geometry)
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = np.linspace(0, 0.17, 100)
solver = pybamm.KLU()
solver.atol = 1e-8
solver.rtol = 1e-8
solution = solver.solve(model, t_eval)


def convert_time(non_dim_time, to="seconds"):
    s_parms = pybamm.standard_parameters_lithium_ion
    t_sec = param.process_symbol(s_parms.tau_discharge).evaluate()
    t = non_dim_time * t_sec
    if to == "hours":
        t *= 1 / 3600
    return t


def plot(concatenate=True):
    # Plotting
    z = np.linspace(0, 1, Nunit)
    pvs = {
        "X-averaged cell temperature [K]": None,
        "X-averaged reversible heating [A.V.m-3]": None,
        "X-averaged irreversible electrochemical heating [A.V.m-3]": None,
        "X-averaged Ohmic heating [A.V.m-3]": None,
        "X-averaged total heating [A.V.m-3]": None,
        "Current collector current density [A.m-2]": None,
#        "X-averaged positive particle " +
#        "surface concentration [mol.m-3]": None,
#        "X-averaged negative particle " +
#        "surface concentration [mol.m-3]": None,
#        "Negative current collector potential [V]": None,
#        "Positive current collector potential [V]": None,
    }
    for key in pvs.keys():
        proc = pybamm.ProcessedVariable(
            model.variables[key], solution.t, solution.y, mesh=mesh
        )
        pvs[key] = proc
    hrs = convert_time(solution.t, to="hours")
    for key in pvs.keys():
        fig, ax = plt.subplots()
        lines = []
        data = pvs[key](solution.t, z=z)
        for bat_id in range(Nunit):
            lines.append(np.column_stack((hrs, data[bat_id, :])))
        line_segments = LineCollection(lines)
        line_segments.set_array(z)
        ax.yaxis.set_major_formatter(
            mpl.ticker.ScalarFormatter(useMathText=True, useOffset=False)
        )
        ax.add_collection(line_segments)
        plt.xlabel("t [hrs]")
        plt.title(key)
        plt.xlim(hrs.min(), hrs.max())
        plt.ylim(data.min(), data.max())
        plt.show()

plot()
