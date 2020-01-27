#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:11:13 2019

@author: thomas
"""
import numpy as np
import openpnm as op
import openpnm.topotools as tt
from openpnm.topotools import plot_connections as pconn
from openpnm.topotools import plot_coordinates as pcoord
from openpnm.models.physics.generic_source_term import linear
import pybamm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import sys
import time
import os
from scipy import io


def plot_topology(net):
    inner = net["pore.inner"]
    outer = net["pore.outer"]
    fig = pconn(net, throats=net.throats("throat.neg_cc"), c="blue")
    fig = pconn(net, throats=net.throats("throat.pos_cc"), c="red", fig=fig)
    fig = pcoord(net, pores=net["pore.neg_cc"], c="blue", fig=fig)
    fig = pcoord(net, pores=net["pore.pos_cc"], c="red", fig=fig)
    fig = pcoord(net, pores=net["pore.neg_tab"], c="blue", s=100, fig=fig)
    fig = pcoord(net, pores=net["pore.pos_tab"], c="red", s=100, fig=fig)
    fig = pcoord(net, pores=inner, c="pink", fig=fig)
    fig = pcoord(net, pores=outer, c="yellow", fig=fig)
    fig = pcoord(net, pores=net.pores('free_stream'), c="green", fig=fig)
    fig = pconn(net, throats=net.throats("throat.free_stream"), c="green",
                fig=fig)
    t_sep = net.throats("spm_resistor")
    if len(t_sep) > 0:
        fig = pconn(
            net, throats=net.throats("spm_resistor"),
            c="k", fig=fig
        )


def plot_phase_data(project, data='pore.temperature'):
    net = project.network
    phase = project.phases()['phase_01']
    Ps = net.pores('free_stream', mode='not')
    coords = net['pore.coords']
    x = coords[:, 0][Ps]
    y = coords[:, 1][Ps]
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.scatter(x, y, c=phase[data][Ps])
    ax = fig.gca()
    ax.set_xlim(x.min()*1.05,
                x.max()*1.05)
    ax.set_ylim(y.min()*1.05,
                y.max()*1.05)


def spiral(r, dr, ntheta=36, n=10):
    theta = np.linspace(0, n * (2 * np.pi), (n * ntheta) + 1)
    pos = (np.linspace(0, n * ntheta, (n * ntheta) + 1) % ntheta)
    pos = pos.astype(int)
    rad = r + np.linspace(0, n * dr, (n * ntheta) + 1)
    x = rad * np.cos(theta)
    y = rad * np.sin(theta)
    return (x, y, rad, pos)


def make_spiral_net(Nlayers=3, dtheta=10, spacing=190e-6,
                    pos_tabs=[0], neg_tabs=[-1],
                    R=1.0, length_3d=0.065):
    Narc = np.int(360 / dtheta)  # number of nodes in a wind/layer
    Nunit = np.int(Nlayers * Narc)  # total number of unit cells
    N1d = 2
    # 2D assembly
    assembly = np.zeros([Nunit, N1d], dtype=int)

    assembly[:, 0] = 0
    assembly[:, 1] = 1
    unit_id = np.tile(np.arange(0, Nunit), (N1d, 1)).T
    prj = op.Project()
    net = op.network.Cubic(shape=[Nunit, N1d, 1],
                           spacing=spacing, project=prj)
    net["pore.pos_cc"] = net["pore.right"]
    net["pore.neg_cc"] = net["pore.left"]

    net["pore.region_id"] = assembly.flatten()
    net["pore.cell_id"] = unit_id.flatten()
    # Extend the connections in the cell repetition direction
    net["pore.coords"][:, 0] *= 10
    inner_r = 185 * 1e-5
    # Update coords
    net["pore.radial_position"] = 0.0
    net["pore.arc_index"] = 0
    r_start = net["pore.coords"][net["pore.cell_id"] == 0][:, 1]
    dr = spacing * N1d
    for i in range(N1d):
        (x, y, rad, pos) = spiral(
            r_start[i] + inner_r, dr, ntheta=Narc, n=Nlayers
        )
        mask = net["pore.coords"][:, 1] == r_start[i]
        coords = net["pore.coords"][mask]
        coords[:, 0] = x[:-1]
        coords[:, 1] = y[:-1]
        net["pore.coords"][mask] = coords
        net["pore.radial_position"][mask] = rad[:-1]
        net["pore.arc_index"][mask] = pos[:-1]
        if i == 0:
            arc_edges = np.cumsum(np.deg2rad(dtheta) * rad)
            arc_edges -= arc_edges[0]
    # Make interlayer connections after rolling
    Ps_neg_cc = net.pores("neg_cc")
    Ps_pos_cc = net.pores("pos_cc")
    coords_left = net["pore.coords"][Ps_neg_cc]
    coords_right = net["pore.coords"][Ps_pos_cc]

    pos_cc_Ts = net.find_neighbor_throats(net.pores("pos_cc"), mode="xnor")
    neg_cc_Ts = net.find_neighbor_throats(net.pores("neg_cc"), mode="xnor")
    pos_tab_nodes = net.pores()[net["pore.pos_cc"]][pos_tabs]
    neg_tab_nodes = net.pores()[net["pore.neg_cc"]][neg_tabs]
    net["pore.pos_tab"] = False
    net["pore.neg_tab"] = False
    net["pore.pos_tab"][pos_tab_nodes] = True
    net["pore.neg_tab"][neg_tab_nodes] = True

    conns = []
    # Identify pores in rolled layers that need new connections
    # This represents the separator layer which is not explicitly resolved
    for i_left, cl in enumerate(coords_left):
        vec = coords_right - cl
        dist = np.linalg.norm(vec, axis=1)
        if np.any(dist < 2 * spacing):
            i_right = np.argwhere(dist < 1.5 * spacing)[0][0]
            conns.append([Ps_neg_cc[i_left], Ps_pos_cc[i_right]])
    # Create new throats
    op.topotools.extend(network=net, throat_conns=conns,
                        labels=["separator"])
    h = net.check_network_health()
    if len(h['duplicate_throats']) > 0:
        trim_Ts = np.asarray(h['duplicate_throats'])[:, 1]
        op.topotools.trim(network=net, throats=trim_Ts)
    Ts = net.find_neighbor_throats(pores=net.pores('pos_cc'), mode='xor')

    net["throat.pos_cc"] = False
    net["throat.neg_cc"] = False
    net["throat.pos_cc"][pos_cc_Ts] = True
    net["throat.neg_cc"][neg_cc_Ts] = True
    net["throat.spm_resistor"] = True
    net["throat.spm_resistor"][pos_cc_Ts] = False
    net["throat.spm_resistor"][neg_cc_Ts] = False

    Ps = net['throat.conns'][Ts].flatten()
    Ps, counts = np.unique(Ps.flatten(), return_counts=True)
    boundary = Ps[counts == 1]
    net["pore.inner"] = False
    net["pore.outer"] = False
    net["pore.inner"][boundary] = True
    net["pore.outer"][boundary] = True
    net["pore.inner"][net.pores('pos_cc')] = False
    net["pore.outer"][net.pores('neg_cc')] = False

    # Free stream convection boundary nodes
    free_rad = inner_r + (Nlayers + 0.5) * dr
    (x, y, rad, pos) = spiral(free_rad, dr, ntheta=Narc, n=1)
    net_free = op.network.Cubic(shape=[Narc, 1, 1], spacing=spacing)

    net_free["throat.trimmers"] = True
    net_free["pore.free_stream"] = True
    net_free["pore.coords"][:, 0] = x[:-1]
    net_free["pore.coords"][:, 1] = y[:-1]
    op.topotools.stitch(
        network=net,
        donor=net_free,
        P_network=net.pores(),
        P_donor=net_free.Ps,
        len_max=1.0*dr,
        method="nearest",
    )
    net['throat.free_stream'] = net['throat.stitched']
    del net['throat.stitched']
    del net["pore.left"]
    del net["pore.right"]
    del net["pore.front"]
    del net["pore.back"]
    del net["pore.internal"]
    del net["pore.surface"]
    del net["throat.internal"]
    del net["throat.surface"]
    del net["throat.separator"]

    free_pores = net.pores("free_stream")
    net["pore.radial_position"][free_pores] = rad[:-1]
    net["pore.arc_index"][free_pores] = pos[:-1]
    op.topotools.trim(network=net,
                      throats=net.throats("trimmers"))

    net["pore.region_id"][net["pore.free_stream"]] = -1
    net["pore.cell_id"][net["pore.free_stream"]] = -1
    plot_topology(net)
    print('N SPM', net.num_throats('spm_resistor'))
    geo = setup_geometry(net, dtheta, spacing, length_3d=length_3d)
    phase = op.phases.GenericPhase(network=net)
    phys = op.physics.GenericPhysics(network=net,
                                     phase=phase,
                                     geometry=geo)
    return prj, arc_edges


def make_tomo_net(dtheta=10, spacing=190e-6, length_3d=0.065):
    wrk = op.Workspace()
    cwd = os.getcwd()
    input_dir = os.path.join(cwd, 'input')
    wrk.load_project(os.path.join(input_dir, 'MJ141-mid-top_m_cc_ecm.pnm'))
    sim_name = list(wrk.keys())[-1]
    project = wrk[sim_name]
    net = project.network
    arc_edges = [0.0]
    Ps = net.pores('neg_cc')
    Nunit = net['pore.cell_id'][Ps].max() + 1
    old_coord = None
    for cell_id in range(Nunit):
        P = Ps[net['pore.cell_id'][Ps] == cell_id]
        coord = net['pore.coords'][P]
        if old_coord is not None:
            d = np.linalg.norm(coord-old_coord)
            arc_edges.append(arc_edges[-1] + d)
        old_coord = coord
    # Add 1 more
    arc_edges.append(arc_edges[-1] + d)
    arc_edges = np.asarray(arc_edges)
    geo = setup_geometry(net, dtheta, spacing, length_3d=0.065)
    phase = op.phases.GenericPhase(network=net)
    phys = op.physics.GenericPhysics(network=net,
                                     phase=phase,
                                     geometry=geo)
    return project, arc_edges


def setup_ecm_alg(project, spacing, R, cc_cond=3e7):
    net = project.network
    phase = project.phases()['phase_01']
    phys = project.physics()['phys_01']
#    phase = op.phases.GenericPhase(network=net)
#    cc_cond = 3e7
    cc_unit_len = spacing
    cc_unit_area = 10.4e-6 * 0.207
    econd = cc_cond * cc_unit_area / cc_unit_len
    phys["throat.electrical_conductance"] = econd
    res_Ts = net.throats("spm_resistor")
    phys["throat.electrical_conductance"][res_Ts] = 1 / R
    alg = op.algorithms.OhmicConduction(network=net)
    alg.setup(
        phase=phase,
        quantity="pore.potential",
        conductance="throat.electrical_conductance",
    )
    alg.settings["rxn_tolerance"] = 1e-8
    return alg


#def _pool_eval_wrapper(args):
#    return args[0].evaluate(args[1], args[2], args[3])
#
#
#def evaluate_python_pool(python_eval, solutions, currents, max_workers):
#    pool = ProcessPoolExecutor(max_workers=max_workers)
#    keys = list(python_eval.keys())
#    out = np.zeros([len(solutions), len(keys)])
#
#    for i, key in enumerate(keys):
#        pool_in = []
#        for j, solution in enumerate(solutions):
#            pool_in.append([python_eval[key],
#                            solution.t[-1],
#                            solution.y[:, -1],
#                            {"Current": currents[j]}])
#        out[:, i] = list(pool.map(_pool_eval_wrapper, pool_in))
#    return out


def evaluate_python(python_eval, solution, inputs):
    keys = list(python_eval.keys())
    out = np.zeros(len(keys))
    for i, key in enumerate(keys):
        temp = python_eval[key].evaluate(
                solution.t[-1], solution.y[:, -1], u=inputs
                )
        out[i] = temp
    return out


def evaluate_solution(python_eval, solution, current):
    keys = list(python_eval.keys())
    out = np.zeros(len(keys))
    for i, key in enumerate(keys):
        temp = solution[key](
                solution.t[-1]
#                solution.t[-1], solution.y[:, -1], u={"Current": current}
                )
        out[i] = temp
    return out


def spm_1p1D(Nunit, Nsteps, I_app, total_length):
    st = time.time()
    # set logging level
    pybamm.set_logging_level("INFO")

    # load (1+1D) SPMe model
    options = {
        "current collector": "potential pair",
        "dimensionality": 1,
    }
    model = pybamm.lithium_ion.SPM(options)
    # create geometry
    geometry = model.default_geometry
    # load parameter values and process model and geometry
    param = model.default_parameter_values
    param.update(
        {
            "Typical current [A]": I_app,
            "Current function": "[constant]",
            "Initial temperature [K]": 298.15,
            "Negative current collector conductivity [S.m-1]": 3e7,
            "Positive current collector conductivity [S.m-1]": 3e7,
            "Heat transfer coefficient [W.m-2.K-1]": 1,
            "Electrode height [m]": total_length,
            "Positive tab centre z-coordinate [m]": total_length,
            "Negative tab centre z-coordinate [m]": total_length,
        }
    )
    param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    var = pybamm.standard_spatial_vars
    var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5,
               var.r_n: 10, var.r_p: 10, var.z: Nunit}
    sys.setrecursionlimit(10000)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)
    # solve model -- simulate one hour discharge
    stp_liion = pybamm.standard_parameters_lithium_ion
    tau = param.process_symbol(stp_liion.tau_discharge)
    t_end = 3600 / tau.evaluate(0)
    t_eval = np.linspace(0, t_end, Nsteps)

    solver = pybamm.CasadiSolver(mode="fast")
    solution = solver.solve(model, t_eval)
    var = "Current collector current density [A.m-2]"
    J_local = model.variables[var].evaluate(solution.t, solution.y)
    u_len = mesh["current collector"][0].d_edges
    w = param['Electrode width [m]']
    h = param['Electrode height [m]']
    A = u_len * w * h
    I_local = A[:, np.newaxis] * J_local
    print('*'*30)
    print('1+1D time', time.time()-st)
    print('*'*30)
    return model, param, solution, mesh, t_eval, I_local.T


def convert_time(param, non_dim_time, to="seconds", inputs=None):
    s_parms = pybamm.standard_parameters_lithium_ion
    t_sec = param.process_symbol(s_parms.tau_discharge).evaluate(u=inputs)
    t = non_dim_time * t_sec
    if to == "hours":
        t *= 1 / 3600
    return t


def current_function(t):
    return pybamm.InputParameter("Current")


def make_spm(I_typical, thermal=True, length_3d=0.065, pixel_size=10.4e-6):
    if thermal:
        model_options = {
                "thermal": "x-lumped",
                "external submodels": ["thermal"],
            }
        model = pybamm.lithium_ion.SPM(model_options)
    else:
        model = pybamm.lithium_ion.SPM()
    geometry = model.default_geometry
    param = model.default_parameter_values
    param.update(
        {
            "Typical current [A]": I_typical,
            "Current function [A]": current_function,
            "Current": "[input]",
            "Electrode height [m]": "[input]",
            "Electrode width [m]": length_3d,
            "Negative electrode thickness [m]": 6.0*pixel_size,
            "Positive electrode thickness [m]": 9.0*pixel_size,
            "Separator thickness [m]": 1.0*pixel_size,
            "Positive current collector thickness [m]": 1.0*pixel_size,
            "Negative current collector thickness [m]": 1.0*pixel_size,
#            "Negative tab centre z-coordinate [m]": z_edges[0],
#            "Positive tab centre z-coordinate [m]": z_edges[-1],
            "Positive electrode conductivity [S.m-1]": 0.1,
            "Negative electrode conductivity [S.m-1]": 0.1,
            "Lower voltage cut-off [V]": 3.45,
            "Upper voltage cut-off [V]": 4.7,
        }
    )
    param.process_model(model)
    param.process_geometry(geometry)
    var = pybamm.standard_spatial_vars
    var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.r_n: 10, var.r_p: 10}
    spatial_methods = model.default_spatial_methods
    solver = pybamm.CasadiSolver()
#    solver = model.default_solver
    sim = pybamm.Simulation(
        model=model,
        geometry=geometry,
        parameter_values=param,
        var_pts=var_pts,
        spatial_methods=spatial_methods,
        solver=solver,
    )
    sim.build(check_model=True)
    return sim


def calc_R(sim, current):
    overpotentials = [
        "X-averaged reaction overpotential [V]",
        "X-averaged concentration overpotential [V]",
        "X-averaged electrolyte ohmic losses [V]",
        "X-averaged solid phase ohmic losses [V]",
    ]
    initial_ocv = 3.8518206633137266
    ocv = evaluate(sim, "X-averaged battery open circuit voltage [V]", current)
    totdV = initial_ocv - ocv
    for overpotential in overpotentials:
        totdV -= evaluate(sim, overpotential, current)
    return totdV / current


def calc_R_new(overpotentials, current):
    totdV = -np.sum(overpotentials, axis=1)

    return totdV/current


def evaluate(sim, var="Current collector current density [A.m-2]",
             current=0.0):
    model = sim.built_model
    #    mesh = sim.mesh
    solution = sim.solution
    #    proc = pybamm.ProcessedVariable(
    #        model.variables[var], solution.t, solution.y, mesh=mesh,
    #        inputs={"Current": current}
    #    )
    value = model.variables[var].evaluate(
        solution.t[-1], solution.y[:, -1], u={"Current": current}
    )
    # should move this definition to the main script...
#    python_eval = pybamm.EvaluatorPython(model.variables[var])
#    python_value = python_eval.evaluate(
#        solution.t[-1], solution.y[:, -1], u={"Current": current}
#    )

    #    return proc(solution.t[-1])
    return value


def convert_temperature(built_model, param, T_dim, inputs):
    temp_parms = built_model.submodels["thermal"].param
#    param = sim.parameter_values
    Delta_T = param.process_symbol(temp_parms.Delta_T).evaluate(u=inputs)
    T_ref = param.process_symbol(temp_parms.T_ref).evaluate(u=inputs)
    return (T_dim - T_ref) / Delta_T


def step_spm(zipped):
    built_model, solver, solution, I_app, e_height, dt, T_av, dead = zipped
    inputs = {"Current": I_app,
              'Electrode height [m]': e_height}
#    T_av_non_dim = convert_temperature(built_model, param, T_av, inputs)
#    T_av_non_dim = 0.0
    if len(built_model.external_variables) > 0:
        external_variables = {"X-averaged cell temperature": T_av}
    else:
        external_variables = None
    if ~dead:
#        print(inputs)
        if solution is not None:
            solved_len = solver.y0.shape[0]
            solver.y0 = solution.y[:solved_len, -1]
            solver.t = solution.t[-1]
        solution = solver.step(
            built_model, dt, external_variables=external_variables, inputs=inputs
        )
#        sim.step(dt=dt, inputs=inputs,
#                 external_variables=external_variables,
#                 save=False)
    return solution

def step_spm_old(zipped):
    sim, solution, I_app, e_height, dt, T_av, dead = zipped
    inputs = {"Current": I_app,
              'Electrode height [m]': e_height}
    T_av_non_dim = convert_temperature(sim, T_av, inputs)
    if len(sim.model.external_variables) > 0:
        external_variables = {"X-averaged cell temperature": T_av_non_dim}
    else:
        external_variables = None
    if ~dead:
#        print(inputs)
        if solution is not None:
            solved_len = sim.solver.y0.shape[0]
            sim.solver.y0 = solution.y[:solved_len, -1]
            sim.solver.t = solution.t[-1]
        sim.step(dt=dt, inputs=inputs,
                 external_variables=external_variables,
                 save=False)
    return sim.solution


def make_1D_net(Nunit, R, spacing, pos_tabs, neg_tabs):
#    net = op.network.Cubic([Nunit + 2, 2, 1], spacing)
    net = op.network.Cubic([Nunit+2, 2, 1], spacing)
    net["pore.pos_cc"] = net["pore.right"]
    net["pore.neg_cc"] = net["pore.left"]

    T = net.find_neighbor_throats(net.pores("front"), mode="xnor")
    tt.trim(net, throats=T)
    T = net.find_neighbor_throats(net.pores("back"), mode="xnor")
    tt.trim(net, throats=T)
    pos_cc_Ts = net.find_neighbor_throats(net.pores("pos_cc"), mode="xnor")
    neg_cc_Ts = net.find_neighbor_throats(net.pores("neg_cc"), mode="xnor")

#    P_pos_a = net.pores(["pos_cc", "front"], "and")
#    P_neg_a = net.pores(["neg_cc", "front"], "and")
#    P_pos_b = net.pores(["pos_cc", "back"], "and")
#    P_neg_b = net.pores(["neg_cc", "back"], "and")
    pos_tab_nodes = net.pores()[net["pore.pos_cc"]][pos_tabs]
    neg_tab_nodes = net.pores()[net["pore.neg_cc"]][neg_tabs]

    net["pore.pos_tab"] = False
    net["pore.neg_tab"] = False
    net["pore.pos_tab"][pos_tab_nodes] = True
    net["pore.neg_tab"][neg_tab_nodes] = True
#    net["pore.pos_terminal_b"] = False
#    net["pore.neg_terminal_b"] = False
#    net["pore.pos_terminal_b"][P_pos_b] = True
#    net["pore.neg_terminal_b"][P_neg_b] = True
    net["throat.pos_cc"] = False
    net["throat.neg_cc"] = False
    net["throat.pos_cc"][pos_cc_Ts] = True
    net["throat.neg_cc"][neg_cc_Ts] = True
    net["throat.spm_resistor"] = True
    net["throat.spm_resistor"][pos_cc_Ts] = False
    net["throat.spm_resistor"][neg_cc_Ts] = False

    del net["pore.left"]
    del net["pore.right"]
    del net["pore.front"]
    del net["pore.back"]
    del net["pore.internal"]
    del net["pore.surface"]
    del net["throat.internal"]
    del net["throat.surface"]

    fig = tt.plot_coordinates(net, net.pores("pos_cc"), c="b")
    fig = tt.plot_coordinates(net, net.pores("pos_tab"), c="y", fig=fig)
    fig = tt.plot_coordinates(net, net.pores("neg_cc"), c="r", fig=fig)
    fig = tt.plot_coordinates(net, net.pores("neg_tab"), c="g", fig=fig)
    fig = tt.plot_connections(net, net.throats("pos_cc"), c="b", fig=fig)
    fig = tt.plot_connections(net, net.throats("neg_cc"), c="r", fig=fig)
    fig = tt.plot_connections(net, net.throats("spm_resistor"), c="k", fig=fig)

    phase = op.phases.GenericPhase(network=net)
#    cc_cond = 3e7
#    cc_unit_len = spacing
#    cc_unit_area = 25e-6 * 0.207
#    phase["throat.electrical_conductance"] = cc_cond * cc_unit_area / cc_unit_len
#    phase["throat.electrical_conductance"][net.throats("spm_resistor")] = 1 / R
    geo = op.geometry.GenericGeometry(
            network=net, pores=net.Ps, throats=net.Ts
            )
    phys = op.physics.GenericPhysics(network=net,
                                     phase=phase,
                                     geometry=geo)
#    alg = op.algorithms.OhmicConduction(network=net)
#    alg.setup(
#        phase=phase,
#        quantity="pore.potential",
#        conductance="throat.electrical_conductance",
#    )
#    alg.settings["rxn_tolerance"] = 1e-8
    return net.project


def run_ecm(net, alg, V_terminal, plot=False):
    potential_pairs = net["throat.conns"][net.throats("spm_resistor")]
    P1 = potential_pairs[:, 0]
    P2 = potential_pairs[:, 1]
    adj = np.random.random(1) / 1e3
    alg.set_value_BC(net.pores("pos_tab"), values=V_terminal + adj)
#    alg.set_value_BC(net.pores("pos_tab"), values=V_terminal)
    alg.set_value_BC(net.pores("neg_tab"), values=adj)
    #    alg['pore.potential'] -= adj
    alg.run()
    V_local_pnm = np.abs(alg["pore.potential"][P2] - alg["pore.potential"][P1])
    I_local_pnm = alg.rate(throats=net.throats("spm_resistor"), mode="single")
    R_local_pnm = V_local_pnm / I_local_pnm
    if plot:
        plt.figure()
        plt.plot(alg["pore.potential"][net.pores('pos_cc')])
        plt.plot(alg["pore.potential"][net.pores('neg_cc')])

    return (V_local_pnm, I_local_pnm, R_local_pnm)


def setup_geometry(net, dtheta, spacing, length_3d):
    # Create Geometry based on circular arc segment
    drad = 2 * np.pi * dtheta / 360
    geo = op.geometry.GenericGeometry(
            network=net, pores=net.Ps, throats=net.Ts
            )
    geo["throat.radial_position"] = net.interpolate_data(
            "pore.radial_position"
            )
    geo["pore.volume"] = (
            net["pore.radial_position"] * drad * spacing * length_3d
            )
    cn = net["throat.conns"]
    C1 = net["pore.coords"][cn[:, 0]]
    C2 = net["pore.coords"][cn[:, 1]]
    D = np.sqrt(((C1 - C2) ** 2).sum(axis=1))
    geo["throat.length"] = D
    # Work out if throat connects pores in same radial position
    rPs = geo["pore.arc_index"][net["throat.conns"]]
    sameR = rPs[:, 0] == rPs[:, 1]
    geo["throat.area"] = spacing * length_3d
    geo['throat.electrode_height'] = geo["throat.radial_position"] * drad
    geo["throat.area"][sameR] = geo['throat.electrode_height'][sameR] * length_3d
    geo["throat.volume"] = 0.0
    geo["throat.volume"][sameR] = geo["throat.area"][sameR]*spacing
    return geo


def setup_thermal(project, options):
    T0 = options['T0']
    cp = options['cp']
    rho = options['rho']
    K0 = options['K0']
    heat_transfer_coefficient = options['heat_transfer_coefficient']
    net = project.network
    geo = project.geometries()['geo_01']
#    phase = op.phases.GenericPhase(network=net)
    phase = project.phases()['phase_01']
    phys = project.physics()['phys_01']
    alpha = K0 / (cp * rho)
    hc = heat_transfer_coefficient / (cp * rho)
    # Set up Phase and Physics
    phase["pore.temperature"] = T0
    phase["pore.thermal_conductivity"] = alpha  # [W/(m.K)]
    phys["throat.conductance"] = (
        alpha * geo["throat.area"] / geo["throat.length"]
    )
    # Reduce separator conductance
    Ts = net.throats("spm_resistor")
    phys["throat.conductance"][Ts] *= 0.1
    # Free stream convective flux
    Ts = net.throats("free_stream")
    phys["throat.conductance"][Ts] = geo["throat.area"][Ts] * hc

    print('Mean throat conductance',
          np.mean(phys['throat.conductance']))
    print('Mean throat conductance Boundary',
          np.mean(phys['throat.conductance'][Ts]))


def apply_heat_source(project, Q):
    # The SPMs are defined at the throat but the pores represent the
    # Actual electrode volume so need to interpolate for heat sources
    net = project.network
    phys = project.physics()['phys_01']
    spm_Ts = net.throats('spm_resistor')
    phys['throat.heat_source'] = 0.0
    phys['throat.heat_source'][spm_Ts] = Q
    phys.add_model(propname='pore.heat_source',
                   model=op.models.misc.from_neighbor_throats,
                   throat_prop='throat.heat_source',
                   mode='max')


def run_step_transient(project, time_step, BC_value):
    # To Do - test whether this needs to be transient
    net = project.network
    phase = project.phases()['phase_01']
    phys = project.physics()['phys_01']
    Q_scaled = phys['pore.heat_source']
    phys["pore.A1"] = 0.0
    phys["pore.A2"] = Q_scaled * net["pore.volume"]
    # Heat Source
    T0 = phase['pore.temperature']
    t_step = float(time_step/10)
    phys.add_model(
        "pore.source",
        model=linear,
        X="pore.temperature",
        A1="pore.A1",
        A2="pore.A2",
    )
    # Run Transient Heat Transport Algorithm
    alg = op.algorithms.TransientReactiveTransport(network=net)
    alg.setup(phase=phase,
              conductance='throat.conductance',
              quantity='pore.temperature',
              t_initial=0.0,
              t_final=time_step,
              t_step=t_step,
              t_output=t_step,
              t_tolerance=1e-9,
              t_precision=12,
              rxn_tolerance=1e-9,
              t_scheme='implicit')
    alg.set_IC(values=T0)
    bulk_Ps = net.pores("free_stream", mode="not")
    alg.set_source("pore.source", bulk_Ps)
    alg.set_value_BC(net.pores("free_stream"), values=BC_value)
    alg.run()
    print(
        "Max Temp",
        alg["pore.temperature"].max(),
        "Min Temp",
        alg["pore.temperature"].min(),
    )
    phase["pore.temperature"] = alg["pore.temperature"]
    project.purge_object(alg)


def setup_pool(max_workers, pool_type='Process'):
#    if pool_type == 'Process':
#        pool = ProcessPoolExecutor(max_workers=max_workers)
#    else:
#        pool = ThreadPoolExecutor(max_workers=max_workers)
#    return pool
    if pool_type == 'Process':
        pool = ProcessPoolExecutor()
    else:
        pool = ThreadPoolExecutor()
    return pool


def _regroup_models(spm_models, max_workers):
    unpack = list(spm_models)
    num_models = len(unpack)
    num_chunk = np.int(np.ceil(num_models/max_workers))
    split = []
    mod_num = 0
    for i in range(max_workers):
        temp = []
        for j in range(num_chunk):
            if mod_num < num_models:
                temp.append(unpack[mod_num])
                mod_num += 1
        split.append(temp)
    return split


def pool_spm(spm_models, pool, max_workers):
    split_models = _regroup_models(spm_models, max_workers)
    split_data = list(pool.map(serial_spm, split_models))
    data = []
    for temp in split_data:
        data = data + temp
    return data


def shutdown_pool(pool):
    pool.shutdown()
    del pool


def serial_spm(inputs):
    outputs = []
    for bundle in inputs:
        outputs.append(step_spm(bundle))
    return outputs


def collect_solutions(solutions):
    temp_y = []
    temp_t = []
    for sol in solutions:
        temp_y.append(sol.y[:, -1])
        temp_t.append(sol.t[-1])
    temp_y = np.asarray(temp_y)
    temp_t = np.asarray(temp_t)
    return temp_t, temp_y.T


def _format_key(key):
    key = [word+'_' for word in key.split() if '[' not in word]
    return ''.join(key)[:-1]


def export(save_dir=None, export_dict=None, prefix=''):
    if save_dir is None:
        save_dir = os.getcwd()
    else:
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
    for key in export_dict.keys():
        data = export_dict[key]
        save_path = os.path.join(save_dir, prefix+_format_key(key))
        io.savemat(file_name=save_path,
                   mdict={'data': data},
                   long_field_names=True)
