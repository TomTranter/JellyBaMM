#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:58:48 2019

@author: thomas
"""

import openpnm as op
import openpnm.topotools as tt
from matplotlib import cm
import matplotlib as mpl
from matplotlib.collections import LineCollection
import pybamm
import numpy as np
import sys
import matplotlib.pyplot as plt

plt.close('all')
I_app = 1e-3
sigma = 3e7
spacing = 2e-5
Nunit = 10
length_3d = 0.065
pixel_size = 10.4e-6
net = op.network.Cubic(shape=[Nunit+1, 2, 1], spacing=spacing)
print(net.labels())
net['pore.cc'] = net['pore.left']
net['pore.sink'] = net['pore.right']
net['pore.terminal'] = net['pore.front']
net['pore.busbar'] = net['pore.cc'].copy()
net['pore.busbar'][net['pore.terminal']] = False

# Initial terminal voltages
Terminal_Voltage_P = 3.75
Terminal_Voltage_N = -0.001


del net['pore.left']
del net['pore.right']
del net['pore.front']
del net['pore.back']
del net['pore.internal']
del net['pore.surface']
del net['throat.internal']
del net['throat.surface']
print(net.labels())

Ts = net.find_neighbor_throats(net['pore.sink'], mode='xnor')
tt.trim(net, throats=Ts)
Ts = net.find_neighbor_throats(net['pore.cc'], mode='xnor')
net['throat.cc'] = False
net['throat.cc'][Ts] = True
net['throat.battery'] = ~net['throat.cc']


trim = np.logical_and(~net['pore.cc'], net['pore.terminal'])
tt.trim(net, pores=trim)
terminal = np.logical_and(net['pore.cc'], net['pore.terminal'])
Ps = net.Ps
Ts = net.Ts
geo = op.geometry.GenericGeometry(network=net, pores=Ps, throats=Ts)
geo['pore.volume'] = (spacing)**3
geo['throat.area'] = (spacing)**2
phase = op.phases.GenericPhase(network=net)
phase['throat.sigma'] = sigma
phase['throat.sigma'][net['throat.battery']] = 1/30
phase['throat.electrical_conductance'] = spacing/phase['throat.sigma']

#fig = tt.plot_coordinates(net, pores=net.pores('cc'), c='b')
#fig = tt.plot_coordinates(net, pores=net.pores('terminal'), c='k', fig=fig)
#fig = tt.plot_coordinates(net, pores=net.pores('sink'), c='r', fig=fig)
#fig = tt.plot_connections(net, throats=net.throats('cc'), c='b', fig=fig)
#fig = tt.plot_connections(net, throats=net.throats('battery'), c='r', fig=fig)

def get_potential(I_local, bc_val):
    q = geo['pore.volume']
    q[net['pore.sink']] *= I_local
    print('Q', q[net['pore.sink']])
    adj = 9.128361824146518476129467124e-2
    alg = op.algorithms.OhmicConduction(network=net)
    alg.setup(phase=phase,
              quantity="pore.potential",
              conductance="throat.electrical_conductance",
              )
    alg.set_rate_BC(net['pore.sink'], values=-q[net['pore.sink']])
    alg.set_value_BC(terminal, values=bc_val-adj)
    alg.run()
    alg['pore.potential'] += adj
    return alg['pore.potential'][net['pore.busbar']]


def non_dim_potential(param, phi_dim, domain):
    # Define a method which takes a dimensional potential [V] and converts
    # to the dimensionless potential used in pybamm
    pot_scale = param.process_symbol(
        pybamm.standard_parameters_lithium_ion.potential_scale
    ).evaluate()  # potential scaled on thermal voltage
    # positive potential measured with respect to reference OCV
    pot_ref = param.process_symbol(
        pybamm.standard_parameters_lithium_ion.U_p_ref
        - pybamm.standard_parameters_lithium_ion.U_n_ref
    ).evaluate()
    if domain == "negative":
        phi = phi_dim / pot_scale
    elif domain == "positive":
        phi = (phi_dim - pot_ref) / pot_scale
    return phi


def update_statevector(model, variables, statevector):
    "takes in a dict of variable name and vector of updated state"
    for name, new_vector in variables.items():
        var_slice = model.variables[name].y_slices
        statevector[var_slice] = new_vector
    return statevector


def update_external_potential(model, solution, phi_neg, phi_pos):
    current_state = solution.y[:, -1].copy()
    variables = {
        "Negative current collector potential": phi_neg,
        "Positive current collector potential": phi_pos,
    }
    new_state = update_statevector(model, variables, current_state)
    return new_state

#pybamm.set_logging_level("INFO")

options = {
    "current collector": "set external potential",
    "dimensionality": 1,
}
model = pybamm.lithium_ion.SPM(options)
model.use_simplify = False
# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values
param.update(
    {
        "Typical current [A]": I_app,
        "Initial temperature [K]": 303,
        "Negative current collector conductivity [S.m-1]": sigma,
        "Positive current collector conductivity [S.m-1]": sigma,
        "Electrode height [m]": Nunit*spacing,
        "Electrode width [m]": length_3d,
        "Negative electrode thickness [m]": 7.0*pixel_size,
        "Positive electrode thickness [m]": 7.0*pixel_size,
        "Separator thickness [m]": 1.0*pixel_size,
        "Positive current collector thickness [m]": 1.0*pixel_size,
        "Negative current collector thickness [m]": 1.0*pixel_size,
        "Negative tab centre z-coordinate [m]": 0.0,
        "Positive tab centre z-coordinate [m]": Nunit*spacing,
        "Positive electrode conductivity [S.m-1]": 0.1,
        "Negative electrode conductivity [S.m-1]": 0.1,
        "Lower voltage cut-off [V]": 3.45,
        "Upper voltage cut-off [V]": 4.7,
    }
)
#param["Current function"] = pybamm.GetConstantCurrent()
param.process_model(model)
param.process_geometry(geometry)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5,
           var.r_n: 10, var.r_p: 10, var.z: Nunit}
# depending on number of points in y-z plane may need to increase recursion depth...
sys.setrecursionlimit(10000)
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
unit_areas = length_3d*mesh["current collector"][0].d_edges.copy()
# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model -- simulate 1 s discharge
tau = param.process_symbol(pybamm.standard_parameters_lithium_ion.tau_discharge)
t_end = 3600 / tau.evaluate(0)
t_step = t_end / 100
t_eval = np.linspace(0, t_end, 101)
solver = pybamm.CasadiSolver(atol=1e-8, rtol=1e-8, mode='fast')
#solution = pybamm.KLU().solve(model, t_eval)


def convert_time(param, non_dim_time, to="seconds"):
    s_parms = pybamm.standard_parameters_lithium_ion
    t_sec = param.process_symbol(s_parms.tau_discharge).evaluate()
    t = non_dim_time * t_sec
    if to == "hours":
        t *= 1 / 3600
    return t


def run_step(model=None, mesh=None, solver=None, solution=None, time_step=0.01, n_subs=5):
    # Step model for one global time interval
    # Note: In order to make the solver converge, we need to compute
    # consistent initial values for the algebraic part of the model.
    # Since the (dummy) equation for the external temperature is an ODE
    # the imposed change in temperature is unaffected by this process
    # i.e. the temperature is exactly that provided by the pnm model
    if solution is not None:
        solver.y0 = solver.calculate_consistent_initial_conditions(
                solver.rhs, solver.algebraic, solution.y[:, -1]
                )
    current_solution = solver.step(model, time_step, npts=n_subs)

    return current_solution


z = np.linspace(0, 1, Nunit)
solution = None
for outer_i in range(3):
    current_solution = run_step(model, mesh, solver, solution, t_step)
    key = "Current collector current density [A.m-2]"
    j = pybamm.ProcessedVariable(model.variables[key],
                                 current_solution.t, current_solution.y, mesh=mesh)
    key = "Negative current collector potential [V]"
    phi_n = pybamm.ProcessedVariable(model.variables[key],
                                     current_solution.t, current_solution.y, mesh=mesh)
    key = "Positive current collector potential [V]"
    phi_p = pybamm.ProcessedVariable(model.variables[key],
                                     current_solution.t, current_solution.y, mesh=mesh)
#    over = ['X-averaged battery concentration overpotential [V]',
#            'X-averaged battery reaction overpotential [V]',
#            'X-averaged battery electrolyte ohmic losses [V]',
#            'X-averaged battery solid phase ohmic losses [V]']
#    local_overpotentials = np.zeros(Nunit)
#    for op_key in over:
#        func = pybamm.ProcessedVariable(model.variables[op_key],
#                                        current_solution.t, current_solution.y, mesh=mesh)
#        local_overpotentials += func(current_solution.t[-1], z=z).flatten()

    J = j(current_solution.t[-1], z=z).flatten()
    I_local = J*unit_areas
    print('I__local', I_local)
    phi_neg_pnm = get_potential(I_local, Terminal_Voltage_N)
    print('Phi Neg', phi_neg_pnm)
    phi_neg_nd = non_dim_potential(param, phi_neg_pnm, 'negative')
    phi_pos_pnm = get_potential(I_local, Terminal_Voltage_P)
    print('Phi Pos', phi_pos_pnm)
    phi_pos_nd = non_dim_potential(param, phi_pos_pnm, 'positive')
    new_state = update_external_potential(model, current_solution, phi_neg_nd, phi_pos_nd)
    P_n = phi_n(current_solution.t[-1], z=z).flatten()
    P_p = phi_p(current_solution.t[-1], z=z).flatten()
    V_local = phi_pos_pnm - phi_neg_pnm
    R_local = V_local/I_local
#    print('Local overpotentials', local_overpotentials)
    print('R local', R_local)
    print('*'*30)
    phase['throat.sigma'][net['throat.battery']] = 1/R_local
    phase['throat.electrical_conductance'] = spacing/phase['throat.sigma']
    if solution is None:
        solution = current_solution
    else:
        solution.append(current_solution)
#    print(J)


def plot(model, mesh, param, solution, Nunit, concatenate=True):
    # Plotting
    z = np.linspace(0, 1, Nunit)
    sol = solution
    pvs = {
#        "Current collector current density [A.m-2]": None,
#        "X-averaged positive particle " +
#        "surface concentration [mol.m-3]": None,
#        "X-averaged negative particle " +
#        "surface concentration [mol.m-3]": None,
#        "Negative current collector potential [V]": None,
#        "Positive current collector potential [V]": None,
        
#        "X-averaged battery concentration overpotential [V]": None,
#        "X-averaged battery reaction overpotential [V]": None,
#        "X-averaged battery electrolyte ohmic losses [V]": None,
        "X-averaged battery solid phase ohmic losses [V]": None,
            }
    for key in pvs.keys():
        proc = pybamm.ProcessedVariable(
            model.variables[key], sol.t, sol.y, mesh=mesh
        )
        pvs[key] = proc
    hrs = convert_time(param, sol.t, to="hours")
    for key in pvs.keys():
        fig, ax = plt.subplots()
        lines = []
        data = pvs[key](sol.t, z=z)
        for bat_id in range(Nunit):
            lines.append(np.column_stack((hrs, data[bat_id, :])))
        line_segments = LineCollection(lines)
        line_segments.set_array(z)
        ax.yaxis.set_major_formatter(
            mpl.ticker.ScalarFormatter(useMathText=True, useOffset=False)
        )
        ax.add_collection(line_segments)
        plt.xlabel("t [hrs]")
        plt.ylabel(key)
        plt.xlim(hrs.min(), hrs.max())
        plt.ylim(data.min(), data.max())
        #            plt.ticklabel_format(axis='y', style='sci')
        plt.show()

plot(model, mesh, param, solution, Nunit)