# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:15:04 2020

@author: tom
"""


import pybamm
import numpy as np
import sys
import matplotlib.pyplot as plt


plt.close('all')

Nspm = 200

options = {
    "current collector": "potential pair",
    "dimensionality": 1,
    "thermal": "lumped",
}
model = pybamm.lithium_ion.SPM(options)
model.use_simplify = False
# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020)

I_app = 17.5
I_typical = I_app

pixel_size = 10.4e-6
t_neg_electrode = 8.0
t_pos_electrode = 7.0
t_neg_cc = 2.0
t_pos_cc = 2.0
t_sep = 2.0


dr = (2*t_neg_electrode + 2*t_pos_electrode + t_neg_cc + t_pos_cc + 2*t_sep)*pixel_size
inner_r = 185 * 1e-5
length_3d = 80e-3
dtheta = 10
Narc = np.int(360 / dtheta)
Narc = 36
Nlayers = 40

def spiral(r, dr, ntheta=36, n=10):
    theta = np.linspace(0, n * (2 * np.pi), (n * ntheta) + 1)
    pos = (np.linspace(0, n * ntheta, (n * ntheta) + 1) % ntheta)
    pos = pos.astype(int)
    rad = r + np.linspace(0, n * dr, (n * ntheta) + 1)
    x = rad * np.cos(theta)
    y = rad * np.sin(theta)
    return (x, y, rad, pos)
(x, y, rad, pos) = spiral(
    5*dr, dr, ntheta=Narc, n=Nlayers
        )
arc_edges = np.cumsum(np.deg2rad(dtheta) * rad)
arc_edges -= arc_edges[0]
e_height = arc_edges[-1]
cooled_surface_area = 2*np.pi*(rad[-1]*2)*length_3d
cell_volume = np.pi*(rad[-1]**2)*length_3d

param.update(
    {
        "Typical current [A]": I_typical,
        "Current function [A]": I_app,
        "Initial temperature [K]": 298.15,
        "Electrode height [m]": e_height,
        "Electrode width [m]": length_3d,
        "Negative electrode thickness [m]": t_neg_electrode*pixel_size,
        "Positive electrode thickness [m]": t_pos_electrode*pixel_size,
        "Separator thickness [m]": t_sep*pixel_size,
        "Negative current collector thickness [m]": t_neg_cc*pixel_size,
        "Positive current collector thickness [m]": t_pos_cc*pixel_size,
        "Negative tab centre z-coordinate [m]": 0.0,
        "Positive tab centre z-coordinate [m]": e_height,
        "Edge heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Negative current collector surface heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Negative tab heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Positive current collector surface heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Positive tab heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Total heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Cell cooling surface area [m2]": cooled_surface_area,
        "Cell volume [m3]": cell_volume,
        "Negative current collector density [kg.m-3]": 2702.0,
        "Negative electrode density [kg.m-3]": 1347.33,
        "Positive current collector density [kg.m-3]": 8933.0,
        "Positive electrode density [kg.m-3]": 2428.5,
        "Separator density [kg.m-3]": 1008.98,
        "Negative current collector specific heat capacity [J.kg-1.K-1]": 903.0,
        "Negative electrode specific heat capacity [J.kg-1.K-1]": 1437.4,
        "Positive current collector specific heat capacity [J.kg-1.K-1]": 385.0,
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 1269.21,
        "Separator specific heat capacity [J.kg-1.K-1]": 1978.16,
        "Negative current collector thermal conductivity [W.m-1.K-1]": 238.0,
        "Negative electrode thermal conductivity [W.m-1.K-1]": 1.04,
        "Positive current collector thermal conductivity [W.m-1.K-1]": 398.0,
        "Positive electrode thermal conductivity [W.m-1.K-1]": 1.58,
        "Separator thermal conductivity [W.m-1.K-1]": 0.334,  
        "Lower voltage cut-off [V]": 3.0,
        "Upper voltage cut-off [V]": 4.7,
    }, check_already_exists=False
)
def RKn_fit(x, U0, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9):
    A = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]
    R = 8.314
    T = 298.15
    F = 96485
    term1 = R*T/F*pybamm.log((1-x)/x)
    term2 = 0
    for k in range(len(A)):
        term2 += (A[k]/F)*((2*x-1)**(k+1) - (2*x*k*(1-x))/(2*x-1)**(1-k))
    return U0 + term1 + term2

def neg_OCP(sto):
    neg_popt = np.array([ 2.79099024e-01,  2.72347515e+04,  3.84107939e+04,  2.82700416e+04,
       -5.08764455e+03,  5.83084069e+04,  2.74900945e+05, -1.58889236e+05,
       -5.48361415e+05,  3.09910938e+05,  5.56788274e+05])
    return RKn_fit(sto, *neg_popt)

def pos_OCP(sto):
    c = [5.88523041,  -16.64427726,   65.89481612, -131.99750794,
        124.80902818,  -44.56278259]
    return c[0] + c[1]*sto + c[2]*sto**2 + c[3]*sto**3 + c[4]*sto**4 + c[5]*sto**5

def neg_dUdT(sto, c_n_max):
    c = [3.25182032e-04, -1.10405547e-03,  2.02525788e-02, -2.02055921e-01,
         7.09962540e-01, -1.13830746e+00,  8.59315741e-01, -2.48497618e-01]
    return c[0] + c[1]*sto + c[2]*sto**2 + c[3]*sto**3 + c[4]*sto**4 + c[5]*sto**5 + c[6]*sto**6 + c[7]*sto**7

def pos_dUdT(sto, c_p_max):
    c = [9.90601449e-06, -4.77219388e-04,  4.51317690e-03, -1.33763466e-02,
         1.55768635e-02, -6.33314715e-03]
    return c[0] + c[1]*sto + c[2]*sto**2 + c[3]*sto**3 + c[4]*sto**4 + c[5]*sto**5

param["Negative electrode OCP [V]"] = neg_OCP
param["Positive electrode OCP [V]"] = pos_OCP    
param["Negative electrode OCP entropic change [V.K-1]"] = neg_dUdT
param["Positive electrode OCP entropic change [V.K-1]"] = pos_dUdT


e_width = param["Electrode width [m]"]
z_edges = np.linspace(0, e_height, Nspm+1)
A_cc = param.evaluate(pybamm.GeometricParameters().A_cc)

param.process_model(model)
param.process_geometry(geometry)

sys.setrecursionlimit(10000)

var = pybamm.standard_spatial_vars
var_pts = {
    var.x_n: 5,
    var.x_s: 5,
    var.x_p: 5,
    var.r_n: 10,
    var.r_p: 10,
    var.z: Nspm,
}
submesh_types = model.default_submesh_types
pts = z_edges / z_edges[-1]
z = (pts[:-1] + pts[1:])/2
submesh_types["current collector"] = pybamm.MeshGenerator(
    pybamm.UserSupplied1DSubMesh, submesh_params={"edges": pts}
)

solver = pybamm.CasadiSolver(atol=1e-8, rtol=1e-8, mode='fast')

solver = pybamm.CasadiSolver()
sim = pybamm.Simulation(model=model,
                        geometry=geometry,
                        parameter_values=param,
                        submesh_types=submesh_types,
                        var_pts=var_pts,
                        spatial_methods=model.default_spatial_methods,
                        solver=solver)

t_eval = np.linspace(0, 3600, 101)
sim.solve(t_eval)

show_x = True
if show_x:
    output_variables = ['Terminal voltage [V]',
                        'Cell temperature [K]',
                        'X-averaged Ohmic heating [W.m-3]',
                        'X-averaged irreversible electrochemical heating [W.m-3]',
                        'X-averaged reversible heating [W.m-3]',
                        'X-averaged total heating [W.m-3]']
else:
    output_variables = ['Terminal voltage [V]',
                        'Volume-averaged cell temperature [K]',
                        'Volume-averaged Ohmic heating [W.m-3]',
                        'Volume-averaged irreversible electrochemical heating [W.m-3]',
                        'Volume-averaged reversible heating [W.m-3]',
                        'Volume-averaged total heating [W.m-3]']
sim.plot(output_variables)