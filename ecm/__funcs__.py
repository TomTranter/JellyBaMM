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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import sys
import time
import os
from scipy import io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import griddata
from scipy.interpolate import NearestNDInterpolator
from matplotlib import gridspec
import json
import ecm


def plot_topology(net, fig=None):
    # inner = net["pore.inner"]
    # outer = net["pore.outer"]
    if fig is None:
        fig = plt.figure(figsize=(10, 10))
    fig = ecm.plot_resistors(net, throats=net.throats("throat.neg_cc"),
                             c="blue", fig=fig)
    fig = ecm.plot_resistors(net, throats=net.throats("throat.pos_cc"),
                             c="red", fig=fig)
    fig = pcoord(net, pores=net.pores("neg_cc"), c="blue", s=25, fig=fig)
    fig = pcoord(net, pores=net.pores("pos_cc"), c="red", s=25, fig=fig)
    fig = pcoord(net, pores=net["pore.neg_tab"], c="blue", s=75, fig=fig)
    fig = pcoord(net, pores=net["pore.pos_tab"], c="red", s=75, fig=fig)
    try:
        fig = pcoord(net, pores=net.pores('free_stream'), c="green", fig=fig)
        fig = pconn(net, throats=net.throats("throat.free_stream"), c="green",
                    fig=fig)
    except KeyError:
        pass

    t_sep = net.throats("spm_resistor")
    if len(t_sep) > 0:
        fig = pconn(
            net, throats=net.throats("spm_resistor"),
            c="k", fig=fig
        )
    return fig


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
    ax.set_xlim(x.min() * 1.05,
                x.max() * 1.05)
    ax.set_ylim(y.min() * 1.05,
                y.max() * 1.05)


def spiral(r, dr, ntheta=36, n=10):
    theta = np.linspace(0, n * (2 * np.pi), (n * ntheta) + 1)
    pos = (np.linspace(0, n * ntheta, (n * ntheta) + 1) % ntheta)
    pos = pos.astype(int)
    rad = r + np.linspace(0, n * dr, (n * ntheta) + 1)
    x = rad * np.cos(theta)
    y = rad * np.sin(theta)
    return (x, y, rad, pos)


def make_spiral_net(config):
    sub = 'GEOMETRY'
    Nlayers = config.getint(sub, 'Nlayers')
    dtheta = config.getint(sub, 'dtheta')
    spacing = config.getfloat(sub, 'layer_spacing')
    tesla_tabs = False
    try:
        pos_tabs = config.getint(sub, 'pos_tabs')
        neg_tabs = config.getint(sub, 'neg_tabs')
    except ValueError:
        print('Tesla tabs')
        tesla_tabs = True
    length_3d = config.getfloat(sub, 'length_3d')
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
    net["pore.pos_cc"] = net["pore.front"]
    net["pore.neg_cc"] = net["pore.back"]

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
    no_tab = np.array([4, 5, 13, 14, 22, 23, 31, 32])
    if tesla_tabs:
        pos_tabs = net["pore.arc_index"][Ps_pos_cc]
        neg_tabs = net["pore.arc_index"][Ps_neg_cc]
        pos_tabs = ~np.in1d(pos_tabs, no_tab)
        neg_tabs = ~np.in1d(neg_tabs, no_tab)

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
    net['throat.spm_resistor_order'] = -1
    spm_res = net["throat.spm_resistor"]
    net['throat.spm_resistor_order'][spm_res] = np.arange(np.sum(spm_res))
    p1 = net['throat.conns'][:, 0]
    p1_neg = net['pore.neg_cc'][p1]
    net['throat.spm_neg_inner'] = p1_neg * net["throat.spm_resistor"]
    net['throat.spm_pos_inner'] = (~p1_neg) * net["throat.spm_resistor"]
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
        len_max=1.0 * dr,
        method="nearest",
    )

    net['throat.free_stream'] = net['throat.stitched']
    del net['throat.stitched']

    free_pores = net.pores("free_stream")
    net["pore.radial_position"][free_pores] = rad[:-1]
    net["pore.arc_index"][free_pores] = pos[:-1]
    op.topotools.trim(network=net,
                      throats=net.throats("trimmers"))

    net["pore.region_id"][net["pore.free_stream"]] = -1
    net["pore.cell_id"][net["pore.free_stream"]] = -1

    # Inner boundary nodes
    inner_rad = inner_r - 0.5 * dr
    (x, y, rad, pos) = spiral(inner_rad, dr, ntheta=Narc, n=1)
    net_inner = op.network.Cubic(shape=[Narc, 1, 1], spacing=spacing)

    net_inner["throat.trimmers"] = True
    net_inner["pore.inner_boundary"] = True
    net_inner["pore.coords"][:, 0] = x[:-1]
    net_inner["pore.coords"][:, 1] = y[:-1]
    op.topotools.stitch(
        network=net,
        donor=net_inner,
        P_network=net.pores(),
        P_donor=net_inner.Ps,
        len_max=0.95 * dr,
        method="nearest",
    )
    inner_pores = net.pores("inner_boundary")
    net["pore.radial_position"][inner_pores] = rad[:-1]
    net["pore.arc_index"][inner_pores] = pos[:-1]
    net["pore.region_id"][net["pore.inner_boundary"]] = -1
    net["pore.cell_id"][net["pore.inner_boundary"]] = -1
    P1 = net['throat.conns'][:, 0]
    P2 = net['throat.conns'][:, 1]
    same_arc = net["pore.arc_index"][P1] == net["pore.arc_index"][P2]
    cross_stitch = np.logical_and(net['throat.stitched'], ~same_arc)
    net['throat.inner_boundary'] = net['throat.stitched']
    net['throat.trimmers'][cross_stitch] = True
    del net['throat.stitched']
    del net["pore.left"]
    del net["pore.right"]
    del net["pore.front"]
    del net["pore.back"]
    del net["pore.internal"]
    del net["pore.surface"]
    del net["throat.internal"]
    del net["throat.surface"]

    op.topotools.trim(network=net,
                      throats=net.throats("trimmers"))

    print('N SPM', net.num_throats('spm_resistor'))
    geo = setup_geometry(net, dtheta, spacing, length_3d=length_3d)
    net['throat.arc_length'] = np.deg2rad(dtheta) * net["throat.radial_position"]
    phase = op.phases.GenericPhase(network=net)
    op.physics.GenericPhysics(network=net,
                              phase=phase,
                              geometry=geo)
    return prj, arc_edges


def _get_spm_order(project):
    net = project.network
    # SPM resitor throats mixture of connecting cc's upper and lower
    res_Ts = net.throats("spm_resistor")
    # Connecting cc pores - should always be 1 neg and 1 pos
    conns = net['throat.conns'][res_Ts]
    neg_Ps = net['pore.neg_cc']  # label
    pos_Ps = net['pore.pos_cc']  # label
    # The pore numbers in current resistor order
    neg_Ps_res_order = conns[neg_Ps[conns]]
    pos_Ps_res_order = conns[pos_Ps[conns]]
    # The pore order along cc
    neg_order = net['pore.neg_cc_order']
    pos_order = net['pore.pos_cc_order']
    # CC order as found by indexing in the throat resistor order
    neg_Ps_cc_res_order = neg_order[neg_Ps_res_order]
    pos_Ps_cc_res_order = pos_order[pos_Ps_res_order]
    # Is the order of the negative node lower than the positive node
    # True for about half
    same_order = neg_Ps_cc_res_order == pos_Ps_cc_res_order
    print(np.sum(same_order))
    res_order = np.zeros(len(res_Ts))
    neg_filter = neg_Ps_cc_res_order[same_order]
    pos_filter = pos_Ps_cc_res_order[~same_order]
    res_order[same_order] = neg_filter
    res_order[~same_order] = pos_filter + neg_filter.max()
    res_order = res_order - res_order.min()
    res_order = res_order.astype(int)
    net['throat.spm_resistor_same_order'] = False
    net['throat.spm_resistor_same_order'][res_Ts[same_order]] = True
    net['throat.spm_resistor_order'] = -1
    net['throat.spm_resistor_order'][res_Ts] = res_order


def _get_cc_order(project):
    net = project.network
    phase = project.phases()['phase_01']
    for dom in ['neg', 'pos']:
        phase['throat.entry_pressure'] = 1e6
        phase['throat.entry_pressure'][net.throats(dom + '_cc')] = 1.0
        ip = op.algorithms.InvasionPercolation(network=net)
        ip.setup(phase=phase, entry_pressure='throat.entry_pressure')
        ip.set_inlets(pores=net.pores(dom + '_tab'))
        ip.run()
        inv_seq = ip['pore.invasion_sequence'].copy()
        inv_seq += 1
        inv_seq[net.pores(dom + '_tab')] = 0
        order = inv_seq[net.pores(dom + '_cc')]
        if dom == 'pos':
            order = order.max() - order
        net['pore.' + dom + '_cc_order'] = -1
        net['pore.' + dom + '_cc_order'][net.pores(dom + '_cc')] = order
    _get_spm_order(project)


def make_tomo_net(config):
    sub = 'GEOMETRY'
    dtheta = config.getint(sub, 'dtheta')
    spacing = config.getfloat(sub, 'layer_spacing')
    length_3d = config.getfloat(sub, 'length_3d')
    wrk = op.Workspace()
    input_dir = ecm.INPUT_DIR
    tomo_pnm = config.get('TOMOGRAPHY', 'filename')
    wrk.load_project(os.path.join(input_dir, tomo_pnm))
    sim_name = list(wrk.keys())[-1]
    project = wrk[sim_name]
    net = project.network
    update_tabs(project, config)
    arc_edges = [0.0]
    Ps = net.pores('neg_cc')
    Nunit = net['pore.cell_id'][Ps].max() + 1
    old_coord = None
    for cell_id in range(Nunit):
        P = Ps[net['pore.cell_id'][Ps] == cell_id]
        coord = net['pore.coords'][P]
        if old_coord is not None:
            d = np.linalg.norm(coord - old_coord)
            arc_edges.append(arc_edges[-1] + d)
        old_coord = coord
    # Add 1 more
    arc_edges.append(arc_edges[-1] + d)
    arc_edges = np.asarray(arc_edges)
    geo = setup_geometry(net, dtheta, spacing, length_3d=length_3d)
    phase = op.phases.GenericPhase(network=net)
    op.physics.GenericPhysics(network=net,
                              phase=phase,
                              geometry=geo)
    return project, arc_edges


def setup_ecm_alg(project, config, R):
    net = project.network
    phase = project.phases()['phase_01']
    phys = project.physics()['phys_01']
    length_3d = config.getfloat('GEOMETRY', 'length_3d')
    neg_cc_econd = config.getfloat('PHYSICS', 'neg_cc_econd')
    pos_cc_econd = config.getfloat('PHYSICS', 'pos_cc_econd')
    pixel_size = config.getfloat('THICKNESS', 'pixel_size')
    t_neg_cc = config.getfloat('THICKNESS', 'neg_cc')
    t_pos_cc = config.getfloat('THICKNESS', 'pos_cc')
    cc_unit_len = net['throat.arc_length']
    neg_econd = neg_cc_econd * (pixel_size * t_neg_cc * length_3d)
    pos_econd = pos_cc_econd * (pixel_size * t_pos_cc * length_3d)

    phys["throat.electrical_conductance"] = 1.0
    neg_Ts = net.throats('neg_cc')
    phys["throat.electrical_conductance"][neg_Ts] = neg_econd / cc_unit_len[neg_Ts]
    pos_Ts = net.throats('pos_cc')
    phys["throat.electrical_conductance"][pos_Ts] = pos_econd / cc_unit_len[pos_Ts]
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


def evaluate_python(python_eval, solution, inputs):
    keys = list(python_eval.keys())
    out = np.zeros(len(keys))
    for i, key in enumerate(keys):
        temp = python_eval[key].evaluate(
            solution.t[-1], solution.y[:, -1], inputs=inputs
        )
        out[i] = temp
    return out


def evaluate_solution(python_eval, solution, current):
    keys = list(python_eval.keys())
    out = np.zeros(len(keys))
    for i, key in enumerate(keys):
        temp = solution[key](solution.t[-1])
        out[i] = temp
    return out


def spm_1p1D(Nunit, Nsteps, I_app, total_length):
    st = time.time()
    # set logging level
    pybamm.set_logging_level("INFO")

    # load (1+1D) model
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
    stp_liion = pybamm.LithiumIonParameters()
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
    print('*' * 30)
    print('1+1D time', time.time() - st)
    print('*' * 30)
    return model, param, solution, mesh, t_eval, I_local.T


def convert_time(param, non_dim_time, to="seconds", inputs=None):
    s_parms = pybamm.LithiumIonParameters()
    t_sec = param.process_symbol(s_parms.tau_discharge).evaluate(inputs=inputs)
    t = non_dim_time * t_sec
    if to == "hours":
        t *= 1 / 3600
    return t


def current_function(t):
    return pybamm.InputParameter("Current")


def RKn_fit(x, U0, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9):
    A = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]
    R = 8.314
    T = 298.15
    F = 96485
    term1 = R * T / F * pybamm.log((1 - x) / x)
    term2 = 0
    for k in range(len(A)):
        a = (2 * x - 1)**(k + 1)
        b = (2 * x * k * (1 - x))
        c = (2 * x - 1)**(1 - k)
        term2 += (A[k] / F) * (a - b / c)
    return U0 + term1 + term2


def neg_OCP(sto):
    neg_popt = np.array([2.79099024e-01, 2.72347515e+04, 3.84107939e+04,
                         2.82700416e+04, -5.08764455e+03, 5.83084069e+04,
                         2.74900945e+05, -1.58889236e+05, -5.48361415e+05,
                         3.09910938e+05, 5.56788274e+05])
    return RKn_fit(sto, *neg_popt)


def pos_OCP(sto):
    c = [5.88523041, -16.64427726,
         65.89481612, -131.99750794,
         124.80902818, -44.56278259]
    return (c[0] + c[1] * sto + c[2] * sto**2 +
            c[3] * sto**3 + c[4] * sto**4 + c[5] * sto**5)


def neg_dUdT(sto, c_n_max):
    c = [3.25182032e-04, -1.10405547e-03, 2.02525788e-02, -2.02055921e-01,
         7.09962540e-01, -1.13830746e+00, 8.59315741e-01, -2.48497618e-01]
    return (c[0] + c[1] * sto + c[2] * sto**2 + c[3] * sto**3 + c[4] * sto**4 +
            c[5] * sto**5 + c[6] * sto**6 + c[7] * sto**7)


def pos_dUdT(sto, c_p_max):
    c = [9.90601449e-06, -4.77219388e-04, 4.51317690e-03, -1.33763466e-02,
         1.55768635e-02, -6.33314715e-03]
    return (c[0] + c[1] * sto + c[2] * sto**2 +
            c[3] * sto**3 + c[4] * sto**4 + c[5] * sto**5)


def make_spm(I_typical, config):
    thermal = config.getboolean('PHYSICS', 'do_thermal')
    length_3d = config.getfloat('GEOMETRY', 'length_3d')
    sub = 'THICKNESS'
    pixel_size = config.getfloat(sub, 'pixel_size')
    t_neg_electrode = config.getfloat(sub, 'neg_electrode')
    t_pos_electrode = config.getfloat(sub, 'pos_electrode')
    t_sep = config.getfloat(sub, 'sep')
    t_neg_cc = config.getfloat(sub, 'neg_cc')
    t_pos_cc = config.getfloat(sub, 'pos_cc')
    # sub = 'INIT'
    # neg_conc = config.getfloat(sub, 'neg_conc')
    # pos_conc = config.getfloat(sub, 'pos_conc')
    # sub = 'PHYSICS'
    # neg_elec_econd = config.getfloat(sub, 'neg_elec_econd')
    # pos_elec_econd = config.getfloat(sub, 'pos_elec_econd')
    # vlim_lower = config.getfloat('RUN', 'vlim_lower')
    # vlim_upper = config.getfloat('RUN', 'vlim_upper')
    model_cfg = config.get('RUN', 'model')

    if model_cfg == 'SPM':
        model_class = pybamm.lithium_ion.SPM
    elif model_cfg == 'SPMe':
        model_class = pybamm.lithium_ion.SPMe
    else:
        model_class = pybamm.lithium_ion.DFN
    if thermal:
        model_options = {
            "thermal": "x-lumped",
            "external submodels": ["thermal"],
        }
        model = model_class(model_options)
    else:
        model = model_class()
    geometry = model.default_geometry
    param = model.default_parameter_values
    param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020)
    param.update(
        {
            "Typical current [A]": I_typical,
            "Current function [A]": current_function,
            "Electrode height [m]": "[input]",
            "Electrode width [m]": length_3d,
            "Negative electrode thickness [m]": t_neg_electrode * pixel_size,
            "Positive electrode thickness [m]": t_pos_electrode * pixel_size,
            "Separator thickness [m]": t_sep * pixel_size,
            "Negative current collector thickness [m]": t_neg_cc * pixel_size,
            "Positive current collector thickness [m]": t_pos_cc * pixel_size,
        }
    )
    # param.update(
    #     {
#            "Initial concentration in negative electrode [mol.m-3]": neg_conc,
#            "Initial concentration in positive electrode [mol.m-3]": pos_conc,
#            "Negative electrode conductivity [S.m-1]": neg_elec_econd,
#            "Positive electrode conductivity [S.m-1]": pos_elec_econd,
#            "Lower voltage cut-off [V]": vlim_lower,
#            "Upper voltage cut-off [V]": vlim_upper,
    #     }
    # )
    # Dummy wrappers to get round the c_n_max term

    param["Negative electrode OCP [V]"] = neg_OCP
    param["Positive electrode OCP [V]"] = pos_OCP
    param["Negative electrode OCP entropic change [V.K-1]"] = neg_dUdT
    param["Positive electrode OCP entropic change [V.K-1]"] = pos_dUdT
    param.update({"Current": "[input]"}, check_already_exists=False)
    param.process_model(model)
    param.process_geometry(geometry)
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
    return totdV / current


def evaluate(sim, var="Current collector current density [A.m-2]",
             current=0.0):
    model = sim.built_model
    #    mesh = sim.mesh
    solution = sim.solution
    value = model.variables[var].evaluate(
        solution.t[-1], solution.y[:, -1], inputs={"Current": current}
    )
    return value


def convert_temperature(built_model, param, T_dim, inputs):
    temp_parms = built_model.submodels["thermal"].param
    Delta_T = param.process_symbol(temp_parms.Delta_T).evaluate(inputs=inputs)
    T_ref = param.process_symbol(temp_parms.T_ref).evaluate(inputs=inputs)
    return (T_dim - T_ref) / Delta_T


def step_spm(zipped):
    built_model, solver, solution, I_app, e_height, dt, T_av, dead = zipped
    inputs = {"Current": I_app,
              'Electrode height [m]': e_height}
    if len(built_model.external_variables) > 0:
        external_variables = {"Volume-averaged cell temperature": T_av}
    else:
        external_variables = None
    if ~dead:
        built_model.timescale_eval = built_model.timescale.evaluate(inputs=inputs)
        if solution is not None:
            pass

        solution = solver.step(old_solution=solution,
                               model=built_model,
                               dt=dt,
                               external_variables=external_variables,
                               inputs=inputs,
                               npts=2,
                               save=False
                               )

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
        if solution is not None:
            solved_len = sim.solver.y0.shape[0]
            sim.solver.y0 = solution.y[:solved_len, -1]
            sim.solver.t = solution.t[-1]
        sim.step(dt=dt, inputs=inputs,
                 external_variables=external_variables,
                 save=False)
    return sim.solution


def make_1D_net(config):
    sub = 'GEOMETRY'
    Nunit = config.getint(sub, 'nunit_OneD')
    spacing = config.getfloat(sub, 'spacing_OneD')
    pos_tabs = config.getint(sub, 'pos_tabs')
    neg_tabs = config.getint(sub, 'neg_tabs')
    net = op.network.Cubic([Nunit + 2, 2, 1], spacing)
    net["pore.pos_cc"] = net["pore.front"]
    net["pore.neg_cc"] = net["pore.back"]

    T = net.find_neighbor_throats(net.pores("left"), mode="xnor")
    tt.trim(net, throats=T)
    T = net.find_neighbor_throats(net.pores("right"), mode="xnor")
    tt.trim(net, throats=T)
    pos_cc_Ts = net.find_neighbor_throats(net.pores("pos_cc"), mode="xnor")
    neg_cc_Ts = net.find_neighbor_throats(net.pores("neg_cc"), mode="xnor")

    pos_tab_nodes = net.pores()[net["pore.pos_cc"]][pos_tabs]
    neg_tab_nodes = net.pores()[net["pore.neg_cc"]][neg_tabs]

    net["pore.pos_tab"] = False
    net["pore.neg_tab"] = False
    net["pore.pos_tab"][pos_tab_nodes] = True
    net["pore.neg_tab"][neg_tab_nodes] = True
    net["throat.pos_cc"] = False
    net["throat.neg_cc"] = False
    net["throat.pos_cc"][pos_cc_Ts] = True
    net["throat.neg_cc"][neg_cc_Ts] = True
    net["throat.spm_resistor"] = True
    net["throat.spm_resistor"][pos_cc_Ts] = False
    net["throat.spm_resistor"][neg_cc_Ts] = False
    net['throat.spm_resistor_order'] = -1
    net['throat.spm_resistor_order'][net["throat.spm_resistor"]] = np.arange(Nunit)
    net['throat.spm_neg_inner'] = net["throat.spm_resistor"]
    net['pore.free_stream'] = False
    del net["pore.left"]
    del net["pore.right"]
    del net["pore.front"]
    del net["pore.back"]
    del net["pore.internal"]
    del net["pore.surface"]
    del net["throat.internal"]
    del net["throat.surface"]

    phase = op.phases.GenericPhase(network=net)

    geo = op.geometry.GenericGeometry(
        network=net, pores=net.Ps, throats=net.Ts
    )
    op.physics.GenericPhysics(network=net,
                              phase=phase,
                              geometry=geo)

    net["pore.radial_position"] = net['pore.coords'][:, 0]
    net["pore.arc_index"] = np.indices([Nunit + 2, 2, 1])[0].flatten()
    net["pore.region_id"] = -1
    net["pore.cell_id"] = -1
    net['throat.arc_length'] = spacing
    net['throat.electrode_height'] = spacing
    # placeholder
    net['pore.volume'] = 1.0
    net['throat.area'] = 1.0
    net['throat.length'] = 1.0
    plot_topology(net)
    return net.project, np.cumsum(net['throat.arc_length'])


def get_cc_heat(net, alg, V_terminal):
    neg_Ts = net["throat.conns"][net.throats("neg_cc")]
    nP1 = neg_Ts[:, 0]
    nP2 = neg_Ts[:, 1]
    pos_Ts = net["throat.conns"][net.throats("pos_cc")]
    pP1 = pos_Ts[:, 0]
    pP2 = pos_Ts[:, 1]
    adj = np.random.random(1) / 1e3
    alg.set_value_BC(net.pores("pos_tab"), values=V_terminal + adj)
    alg.set_value_BC(net.pores("neg_tab"), values=adj)
    alg.run()
    dV_neg = alg["pore.potential"][nP2] - alg["pore.potential"][nP1]
    dV_pos = alg["pore.potential"][pP2] - alg["pore.potential"][pP1]
    I_neg = alg.rate(throats=net.throats("neg_cc"), mode="single")
    I_pos = alg.rate(throats=net.throats("pos_cc"), mode="single")
    Pow_neg = np.abs(dV_neg * I_neg)
    Pow_pos = np.abs(dV_pos * I_pos)
    net['throat.cc_power_loss'] = 0.0
    net['throat.cc_power_loss'][net.throats("neg_cc")] = Pow_neg
    net['throat.cc_power_loss'][net.throats("pos_cc")] = Pow_pos
    net.add_model(propname='pore.cc_power_loss',
                  model=op.models.misc.from_neighbor_throats,
                  throat_prop='throat.cc_power_loss',
                  mode='max')


def run_ecm(net, alg, V_terminal, plot=False):
    potential_pairs = net["throat.conns"][net.throats("spm_resistor")]
    P1 = potential_pairs[:, 0]
    P2 = potential_pairs[:, 1]
    adj = np.random.random(1) / 1e3
    alg.set_value_BC(net.pores("pos_tab"), values=V_terminal + adj)
    alg.set_value_BC(net.pores("neg_tab"), values=adj)
    alg.run()
    V_local_pnm = alg["pore.potential"][P2] - alg["pore.potential"][P1]
    V_local_pnm[net['pore.pos_cc'][P1]] *= -1
    I_local_pnm = alg.rate(throats=net.throats("spm_resistor"), mode="single")
    I_local_pnm *= np.sign(V_terminal.flatten())
    R_local_pnm = V_local_pnm / I_local_pnm
    if plot:
        pos_mask = net.pores('pos_cc')
        neg_mask = net.pores('neg_cc')
        plt.figure()
        plt.plot(alg["pore.potential"][pos_mask])
        plt.plot(alg["pore.potential"][neg_mask])

    return (V_local_pnm, I_local_pnm, R_local_pnm)


def setup_geometry(net, dtheta, spacing, length_3d):
    # Create Geometry based on circular arc segment
    drad = np.deg2rad(dtheta)
    geo = op.geometry.GenericGeometry(
        network=net, pores=net.Ps, throats=net.Ts
    )
    if "throat.radial_position" not in net.props():
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
    geo["throat.volume"][sameR] = geo["throat.area"][sameR] * spacing
    return geo


def setup_thermal(project, config):
    sub = 'PHYSICS'
    T0 = config.getfloat(sub, 'T0')
    lumpy_therm = lump_thermal_props(config)
    cp = lumpy_therm['lump_Cp']
    rho = lumpy_therm['lump_rho']

    heat_transfer_coefficient = config.getfloat(sub, 'heat_transfer_coefficient')
    net = project.network
    geo = project.geometries()['geo_01']
    phase = project.phases()['phase_01']
    phys = project.physics()['phys_01']
    hc = heat_transfer_coefficient / (cp * rho)
    # Set up Phase and Physics
    phase["pore.temperature"] = T0
    alpha_spiral = lumpy_therm['alpha_spiral']
    alpha_radial = lumpy_therm['alpha_radial']
    phys["throat.conductance"] = (
        1.0 * geo["throat.area"] / geo["throat.length"]
    )
    # Apply anisotropic heat conduction
    Ts = net.throats("spm_resistor")
    phys["throat.conductance"][Ts] *= alpha_radial
    Ts = net.throats("spm_resistor", mode='not')
    phys["throat.conductance"][Ts] *= alpha_spiral
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


def run_step_transient(project, time_step, BC_value, cp, rho, third=False):
    # To Do - test whether this needs to be transient
    net = project.network
    phase = project.phases()['phase_01']
    phys = project.physics()['phys_01']
    phys["pore.A1"] = 0.0
    Q_spm = phys['pore.heat_source'] * net["pore.volume"]
    Q_cc = net['pore.cc_power_loss']
    print('Q_spm', np.around(np.sum(Q_spm), 2), '\n',
          'Q_cc', np.around(np.sum(Q_cc), 2), '\n',
          'ratio Q_cc/Q_spm', np.around(np.sum(Q_cc) / np.sum(Q_spm), 2))
    phys["pore.A2"] = (Q_spm + Q_cc) / (cp * rho)
    # Heat Source
    T0 = phase['pore.temperature']
    t_step = float(time_step / 10)
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
    if third:
        # To do - 12 only works if detheta is 10
        free_pores = net.pores('free_stream')
        Ps = free_pores[net['pore.arc_index'][free_pores] < 12]
    else:
        Ps = net.pores('free_stream')
    alg.set_value_BC(Ps, values=BC_value)
    alg.run()
    print(
        "Max Temp",
        np.around(alg["pore.temperature"].max(), 3),
        "Min Temp",
        np.around(alg["pore.temperature"].min(), 3),
    )
    phase["pore.temperature"] = alg["pore.temperature"]
    project.purge_object(alg)


def setup_pool(max_workers, pool_type='Process'):
    if pool_type == 'Process':
        pool = ProcessPoolExecutor()
    else:
        pool = ThreadPoolExecutor()
    return pool


def _regroup_models(spm_models, max_workers):
    unpack = list(spm_models)
    num_models = len(unpack)
    num_chunk = np.int(np.ceil(num_models / max_workers))
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
    key = [word + '_' for word in key.split() if '[' not in word]
    return ''.join(key)[:-1]


def export(project, save_dir=None, export_dict=None, prefix='', lower_mask=None,
           save_animation=False):
    if save_dir is None:
        save_dir = os.getcwd()
    else:
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
    for key in export_dict.keys():
        for suffix in ['lower', 'upper']:
            if suffix == 'lower':
                mask = lower_mask
            else:
                mask = ~lower_mask
            data = export_dict[key][:, mask]
            save_path = os.path.join(save_dir, prefix + _format_key(key) + '_' + suffix)
            io.savemat(file_name=save_path,
                       mdict={'data': data},
                       long_field_names=True)


def polar_transform(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def cartesian_transform(r, t):
    x = r * np.cos(t)
    y = r * np.sin(t)
    return x, y


def animate_data(project, data, filename):
    cwd = os.getcwd()
    input_dir = os.path.join(cwd, 'input')
    im_soft = np.load(os.path.join(input_dir, 'im_soft.npz'))['arr_0']
    x_len, y_len = im_soft.shape
    net = project.network
    res_Ts = net.throats('spm_resistor')
    sorted_res_Ts = net['throat.spm_resistor_order'][res_Ts].argsort()
    res_pores = net['pore.coords'][net['throat.conns'][res_Ts[sorted_res_Ts]]]
    res_Ts_coords = np.mean(res_pores, axis=1)
    x = res_Ts_coords[:, 0]
    y = res_Ts_coords[:, 1]
    coords = np.vstack((x, y)).T
    X, Y = np.meshgrid(x, y)
    f = 1.05
    grid_x, grid_y = np.mgrid[x.min() * f:x.max() * f:np.complex(x_len, 0),
                              y.min() * f:y.max() * f:np.complex(y_len, 0)]
    fig = plt.figure()
    ims = []
    print('Saving Animation', filename)
    for t in range(data.shape[0]):
        print('Processing time step', t)
        t_data = data[t, :]
        grid_z0 = griddata(coords, t_data, (grid_x, grid_y), method='nearest')
        grid_z0[np.isnan(im_soft)] = np.nan
        ims.append([plt.imshow(grid_z0, vmin=data.min(), vmax=data.max())])
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)

    im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000,
                                       blit=True)
    if '.mp4' not in filename:
        filename = filename + '.mp4'
    im_ani.save(filename, writer=writer)


def animate_init():
    pass


def plot_subplots(grid_x, grid_y, interp_func, data, t):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[4, 1], width_ratios=[8, 1])
    ax1 = plt.subplot(gs[0, 0])
    ax1c = plt.subplot(gs[0, 1])
    ax2 = plt.subplot(gs[1, :])
    im = ax1.imshow(interp_func(grid_x, grid_y, t))
    plt.colorbar(im, cax=ax1c)
    ax2.plot(np.max(data, axis=1), 'k--')
    ax2.plot(np.min(data, axis=1), 'k--')
    ax2.plot(np.mean(data, axis=1), 'b')
    ax2.plot([t, t], [np.min(data), np.max(data[t, :])], 'r--')
    plt.tight_layout()
    plt.show()
    return fig


def interpolate_spm_number(project):
    cwd = os.getcwd()
    input_dir = os.path.join(cwd, 'input')
    im_soft = np.load(os.path.join(input_dir, 'im_soft.npz'))['arr_0']
    x_len, y_len = im_soft.shape
    net = project.network
    res_Ts = net.throats('spm_resistor')
    sorted_res_Ts = net['throat.spm_resistor_order'][res_Ts].argsort()
    res_pores = net['pore.coords'][net['throat.conns'][res_Ts[sorted_res_Ts]]]
    res_Ts_coords = np.mean(res_pores, axis=1)
    x = res_Ts_coords[:, 0]
    y = res_Ts_coords[:, 1]
    all_x = []
    all_y = []
    all_t = []
    all_data = []
    data = np.arange(0, len(res_Ts))[np.newaxis, :]
    data = data.astype(float)
    for t in range(data.shape[0]):
        all_x = all_x + x.tolist()
        all_y = all_y + y.tolist()
        all_t = all_t + (np.ones(len(x)) * t).tolist()
        all_data = all_data + data[t, :].tolist()
    all_x = np.asarray(all_x)
    all_y = np.asarray(all_y)
    all_t = np.asarray(all_t)
    all_data = np.asarray(all_data)
    points = np.vstack((all_x, all_y, all_t)).T
    myInterpolator = NearestNDInterpolator(points, all_data)
    f = 1.05
    grid_x, grid_y = np.mgrid[x.min() * f:x.max() * f:np.complex(x_len, 0),
                              y.min() * f:y.max() * f:np.complex(y_len, 0)]
    arr = myInterpolator(grid_x, grid_y, 0)
    arr[np.isnan(im_soft)] = np.nan
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(im_soft)
    ax2.imshow(arr)
    np.savez('im_spm_map', arr)


def interpolate_spm_number_model(project, dim=1000):
    x_len = y_len = dim
    net = project.network
    all_x = []
    all_y = []
    all_t = []
    all_data = []
    # Inner boundary
    inner_Ts = net.throats('inner_boundary')
    inner_Ts_coords = np.mean(net['pore.coords'][net['throat.conns'][inner_Ts]], axis=1)
    x = inner_Ts_coords[:, 0]
    y = inner_Ts_coords[:, 1]
    data = np.ones(len(inner_Ts))[np.newaxis, :] * -1
    data = data.astype(float)
    for t in range(data.shape[0]):
        all_x = all_x + x.tolist()
        all_y = all_y + y.tolist()
        all_t = all_t + (np.ones(len(x)) * t).tolist()
        all_data = all_data + data[t, :].tolist()
    # Resistor Ts
    res_Ts = net.throats('spm_resistor')
    sorted_res_Ts = net['throat.spm_resistor_order'][res_Ts].argsort()
    res_pores = net['pore.coords'][net['throat.conns'][res_Ts[sorted_res_Ts]]]
    res_Ts_coords = np.mean(res_pores, axis=1)
    x = res_Ts_coords[:, 0]
    y = res_Ts_coords[:, 1]
    data = np.arange(0, len(res_Ts))[np.newaxis, :]
    data = data.astype(float)
    for t in range(data.shape[0]):
        all_x = all_x + x.tolist()
        all_y = all_y + y.tolist()
        all_t = all_t + (np.ones(len(x)) * t).tolist()
        all_data = all_data + data[t, :].tolist()
    # Outer boundary
    free_Ts = net.throats('free_stream')
    free_Ts_coords = np.mean(net['pore.coords'][net['throat.conns'][free_Ts]], axis=1)
    x = free_Ts_coords[:, 0]
    y = free_Ts_coords[:, 1]
    data = np.ones(len(free_Ts))[np.newaxis, :] * -1
    data = data.astype(float)
    for t in range(data.shape[0]):
        all_x = all_x + x.tolist()
        all_y = all_y + y.tolist()
        all_t = all_t + (np.ones(len(x)) * t).tolist()
        all_data = all_data + data[t, :].tolist()

    all_x = np.asarray(all_x)
    all_y = np.asarray(all_y)
    all_t = np.asarray(all_t)
    all_data = np.asarray(all_data)
    points = np.vstack((all_x, all_y, all_t)).T
    myInterpolator = NearestNDInterpolator(points, all_data)
    f = 1.05
    grid_x, grid_y = np.mgrid[x.min() * f:x.max() * f:np.complex(x_len, 0),
                              y.min() * f:y.max() * f:np.complex(y_len, 0)]
    arr = myInterpolator(grid_x, grid_y, 0)
    return arr


def interpolate_timeseries(project, data):
    cwd = os.getcwd()
    input_dir = os.path.join(cwd, 'input')
    im_soft = np.load(os.path.join(input_dir, 'im_soft.npz'))['arr_0']
    x_len, y_len = im_soft.shape
    net = project.network
    res_Ts = net.throats('spm_resistor')
    sorted_res_Ts = net['throat.spm_resistor_order'][res_Ts].argsort()
    res_pores = net['pore.coords'][net['throat.conns'][res_Ts[sorted_res_Ts]]]
    res_Ts_coords = np.mean(res_pores, axis=1)
    x = res_Ts_coords[:, 0]
    y = res_Ts_coords[:, 1]
    all_x = []
    all_y = []
    all_t = []
    all_data = []
    for t in range(data.shape[0]):
        all_x = all_x + x.tolist()
        all_y = all_y + y.tolist()
        all_t = all_t + (np.ones(len(x)) * t).tolist()
        all_data = all_data + data[t, :].tolist()
    all_x = np.asarray(all_x)
    all_y = np.asarray(all_y)
    all_t = np.asarray(all_t)
    all_data = np.asarray(all_data)
    points = np.vstack((all_x, all_y, all_t)).T
    myInterpolator = NearestNDInterpolator(points, all_data)
    return myInterpolator


def check_vlim(solution, low, high):
    l_check = solution['Terminal voltage [V]'].entries[-1] > low
    h_check = solution['Terminal voltage [V]'].entries[-1] < high
    return l_check * h_check


def run_simulation(I_app, save_path, config):
    max_workers = int(os.cpu_count() / 2)
    hours = config.getfloat('RUN', 'hours')
    Nsteps = np.int(hours * 60 * I_app) + 1  # number of time steps
    V_over_max = 2.0
    if config.get('GEOMETRY', 'domain') == 'model':
        project, arc_edges = make_spiral_net(config)
    elif config.get('GEOMETRY', 'domain') == '1d':
        project, arc_edges = make_1D_net(config)
    else:
        project, arc_edges = make_tomo_net(config)

    net = project.network
    if config.get('GEOMETRY', 'domain') != '1d':
        plot_topology(net)
    # The jellyroll layers are double sided around the cc except for the inner
    # and outer layers the number of spm models is the number of throat
    # connections between cc layers
    Nspm = net.num_throats('spm_resistor')
    res_Ts = net.throats("spm_resistor")
    electrode_heights = net['throat.electrode_height'][res_Ts]
    print('Total Electrode Height', np.around(np.sum(electrode_heights), 2), 'm')
    typical_height = np.mean(electrode_heights)
    I_typical = I_app / Nspm
    temp_inputs = {"Current": I_typical,
                   'Electrode height [m]': typical_height}
    total_length = arc_edges[-1]  # m
    print('Total cc length', total_length)
    print('Total pore volume', np.sum(net['pore.volume']))
    print('Mean throat area', np.mean(net['throat.area']))
    print('Num throats', net.num_throats())
    print('Num throats SPM', Nspm)
    print('Num throats pos_cc', net.num_throats('pos_cc'))
    print('Num throats neg_cc', net.num_throats('neg_cc'))
    print('Typical height', typical_height)
    print('Typical current', I_typical)
    ###########################################################################
    spm_sim = make_spm(I_typical, config)
    width = spm_sim.parameter_values["Electrode width [m]"]
    t1 = spm_sim.parameter_values['Negative electrode thickness [m]']
    t2 = spm_sim.parameter_values['Positive electrode thickness [m]']
    t3 = spm_sim.parameter_values['Negative current collector thickness [m]']
    t4 = spm_sim.parameter_values['Positive current collector thickness [m]']
    t5 = spm_sim.parameter_values['Separator thickness [m]']
    ttot = t1 + t2 + t3 + t4 + t5
    A_cc = electrode_heights * width
    bat_vol = np.sum(A_cc * ttot)
    print('BATTERY ELECTRODE VOLUME', bat_vol)
    print('18650 VOLUME', 0.065 * np.pi * ((8.75e-3)**2 - (2.0e-3)**2))
    ###########################################################################
    temp = np.ones([Nsteps, Nspm])
    temp.fill(np.nan)
    lithiations = {
        "X-averaged negative electrode extent of lithiation": temp.copy(),
        "X-averaged positive electrode extent of lithiation": temp.copy(),
    }
    variables = {
        "X-averaged negative particle surface concentration [mol.m-3]": temp.copy(),
        "X-averaged positive particle surface concentration [mol.m-3]": temp.copy(),
        "Terminal voltage [V]": temp.copy(),
        "Time [h]": temp.copy(),
        "Current collector current density [A.m-2]": temp.copy(),
    }
    overpotentials = {
        "X-averaged battery reaction overpotential [V]": temp.copy(),
        "X-averaged battery concentration overpotential [V]": temp.copy(),
        "X-averaged battery electrolyte ohmic losses [V]": temp.copy(),
        "X-averaged battery solid phase ohmic losses [V]": temp.copy(),
        "Change in measured open circuit voltage [V]": temp.copy(),
    }
    variables_heating = {
        "Volume-averaged Ohmic heating [W.m-3]": temp.copy(),
        "Volume-averaged irreversible electrochemical heating [W.m-3]": temp.copy(),
        "Volume-averaged reversible heating [W.m-3]": temp.copy(),
        "Volume-averaged total heating [W.m-3]": temp.copy(),
        "Volume-averaged Ohmic heating CC [W.m-3]": temp.copy(),
    }
    param = spm_sim.parameter_values
    temp_parms = spm_sim.built_model.submodels["thermal"].param
    Delta_T = param.process_symbol(temp_parms.Delta_T).evaluate(inputs=temp_inputs)
    Delta_T_spm = Delta_T * (typical_height / electrode_heights)
    T_ref = param.process_symbol(temp_parms.T_ref).evaluate()
    T0 = config.getfloat('PHYSICS', 'T0')
    lumpy_therm = lump_thermal_props(config)
    cp = lumpy_therm['lump_Cp']
    rho = lumpy_therm['lump_rho']
    T_non_dim = (T0 - T_ref) / Delta_T
    spm_sol = step_spm((spm_sim.built_model,
                        spm_sim.solver,
                        None, I_typical, typical_height, 1e-6,
                        T_non_dim, False))
    # Create dictionaries of evaluator functions from the discretized model
    variable_keys = list(variables.keys())
    overpotential_keys = list(overpotentials.keys())
    heating_keys = list(variables_heating.keys())
    heating_keys.pop(-1)
    temp = 0.0
    for j, key in enumerate(overpotential_keys):
        temp -= spm_sol[key].entries[-1]
    R = temp / I_typical
    guess_R = R * typical_height / electrode_heights
    V_ecm = temp.flatten()
    print(R)
    R_max = R * 1e6
    # Initialize with a guess for the terminal voltage
    alg = setup_ecm_alg(project, config, guess_R)
    phys = project.physics()['phys_01']
    phase = project.phases()['phase_01']
    (V_local_pnm, I_local_pnm, R_local_pnm) = run_ecm(net, alg, V_ecm)
    print("*" * 30)
    print("V local pnm", V_local_pnm, "[V]")
    print("I local pnm", I_local_pnm, "[A]")
    print("R local pnm", R_local_pnm, "[Ohm]")

    spm_models = [spm_sim.built_model for i in range(Nspm)]

    spm_solvers = [pybamm.CasadiSolver() for i in range(Nspm)]
    spm_params = [spm_sim.parameter_values for i in range(Nspm)]

    solutions = [
        None for i in range(Nspm)
    ]
    terminal_voltages = np.ones(Nsteps) * np.nan
    V_test = V_ecm
    tol = 1e-5
    local_R = np.zeros([Nspm, Nsteps])
    st = time.time()
    all_time_I_local = np.zeros([Nsteps, Nspm])
    all_time_temperature = np.zeros([Nsteps, Nspm])

    sym_tau = pybamm.LithiumIonParameters().tau_discharge
    t_end = hours * 3600
    dt = t_end / (Nsteps - 1)
    tau_spm = []
    for i in range(Nspm):
        temp_tau = spm_params[i].process_symbol(sym_tau)
        tau_input = {'Electrode height [m]': electrode_heights[i]}
        tau_spm.append(temp_tau.evaluate(inputs=tau_input))
    tau_spm = np.asarray(tau_spm)
    dim_time_step = dt
    dead = np.zeros(Nspm, dtype=bool)
    if config.getboolean('RUN', 'parallel'):
        pool = setup_pool(max_workers, pool_type='Process')
    outer_step = 0
    if config.getboolean('PHYSICS', 'do_thermal'):
        setup_thermal(project, config)
    T_non_dim_spm = np.ones(len(res_Ts)) * T_non_dim
    max_temperatures = []
    sorted_res_Ts = net['throat.spm_resistor_order'][res_Ts].argsort()
    try:
        thermal_third = config.getboolean('RUN', 'third')
    except KeyError:
        thermal_third = False
    while np.any(~dead) and outer_step < Nsteps and V_test < V_over_max:
        print("*" * 30)
        print("Outer", outer_step)
        print("Elapsed Simulation Time", np.around((outer_step) * dt, 2), 's')
        # Find terminal voltage that satisfy ecm total currents for R
        current_match = False
        max_inner_steps = 1000
        inner_step = 0
        damping = 0.66
        # Iterate the ecm until the currents match
        t_ecm_start = time.time()
        while (inner_step < max_inner_steps) and (not current_match):
            (V_local_pnm, I_local_pnm, R_local_pnm) = run_ecm(net,
                                                              alg,
                                                              V_test)
            tot_I_local_pnm = np.sum(I_local_pnm)
            diff = (I_app - tot_I_local_pnm) / I_app
            if np.absolute(diff) < tol:
                current_match = True
            else:
                V_test *= 1 + (diff * damping)
            inner_step += 1
        get_cc_heat(net, alg, V_test)
        if V_test < V_over_max:
            print("N inner", inner_step, 'time per step',
                  (time.time() - t_ecm_start) / inner_step)
            print("Over-voltage", np.around(V_test, 2), 'V')
            all_time_I_local[outer_step, :] = I_local_pnm
            terminal_voltages[outer_step] = V_test
            # I_local_pnm should now sum to match the total applied current
            # Run the spms for the the new I_locals for the next time interval
            time_steps = np.ones(Nspm) * dt
            bundle_inputs = zip(spm_models, spm_solvers,
                                solutions, I_local_pnm, electrode_heights,
                                time_steps, T_non_dim_spm, dead)
            t_spm_start = time.time()
            if config.getboolean('RUN', 'parallel'):
                solutions = pool_spm(
                    bundle_inputs,
                    pool,
                    max_workers
                )
            else:
                solutions = serial_spm(
                    bundle_inputs
                )
            print('Finished stepping SPMs in ',
                  np.around((time.time() - t_spm_start), 2), 's')
            print('Solution size', solutions[0].t.shape)
            # Gather the results for this time step
            results_o = np.ones([Nspm, len(overpotential_keys)]) * np.nan
            t_eval_start = time.time()
            for si, i in enumerate(sorted_res_Ts):
                if solutions[i].termination != 'final time':
                    dead[i] = True
                else:
                    temp_inputs = {"Current": I_local_pnm[i],
                                   'Electrode height [m]': electrode_heights[i]}
                    for key in lithiations.keys():
                        temp = solutions[i][key].entries[-1]
                        lithiations[key][outer_step, si] = temp
                    for key in variable_keys:
                        # temp = solutions[i][key](solutions[i].t[-1])
                        temp = solutions[i][key].entries[-1]
                        variables[key][outer_step, si] = temp
                    for j, key in enumerate(overpotential_keys):
                        temp = solutions[i][key].entries[-1]
                        overpotentials[key][outer_step, si] = temp
                        results_o[i, j] = temp
                    for j, key in enumerate(heating_keys):
                        temp = solutions[i][key].entries[-1]
                        variables_heating[key][outer_step, si] = temp
            print('Finished evaluating SPMs in ',
                  np.around((time.time() - t_eval_start), 2), 's')
            if config.getboolean('PHYSICS', 'do_thermal'):
                # Apply Heat Sources
                # To Do: make this better
                vh = variables_heating
                Q_tot = vh["Volume-averaged total heating [W.m-3]"][outer_step, :]
                Q_ohm_cc = net.interpolate_data('pore.cc_power_loss')[res_Ts]
                Q_ohm_cc /= net['throat.volume'][res_Ts]
                key = "Volume-averaged Ohmic heating CC [W.m-3]"
                vh[key][outer_step, :] = Q_ohm_cc[sorted_res_Ts]
                Q = Q_tot
                Q[np.isnan(Q)] = 0.0
                apply_heat_source(project, Q)
                # Calculate Global Temperature
                run_step_transient(project, dim_time_step, T0, cp, rho, thermal_third)
                # Interpolate the node temperatures for the SPMs
                spm_temperature = phase.interpolate_data('pore.temperature')[res_Ts]
                all_time_temperature[outer_step, :] = spm_temperature
                max_temperatures.append(spm_temperature.max())
                T_non_dim_spm = (spm_temperature - T_ref) / Delta_T_spm
            # Get new equivalent resistances
            temp_R = calc_R_new(results_o, I_local_pnm)
            # Update ecm conductivities for the spm_resistor throats
            sig = 1 / temp_R
            if np.any(temp_R > R_max):
                print('Max R found')
                print(I_local_pnm[temp_R > R_max])
                dead[temp_R > R_max] = True
                sig[temp_R > R_max] = 1 / R_max
            if np.any(np.isnan(temp_R)):
                print('Nans found')
                print(I_local_pnm[np.isnan(temp_R)])
                dead[np.isnan(temp_R)] = True
                sig[np.isnan(temp_R)] = 1 / R_max
            phys["throat.electrical_conductance"][res_Ts] = sig
            local_R[:, outer_step] = temp_R
            if solutions[0].t.shape[0] > 1:
                if not check_vlim(solutions[0],
                                  config.getfloat('RUN', 'vlim_lower'),
                                  config.getfloat('RUN', 'vlim_upper')):
                    dead.fill(True)
                    print('VOLTAGE LIMITS EXCEEDED')
            else:
                dead.fill(True)
                print(solutions[0].termination)

            outer_step += 1

    if config.getboolean('RUN', 'parallel'):
        shutdown_pool(pool)

    variables['ECM R local'] = local_R[sorted_res_Ts, :outer_step].T
    variables['ECM I Local'] = all_time_I_local[:outer_step, sorted_res_Ts]
    variables['Temperature [K]'] = all_time_temperature[:outer_step, sorted_res_Ts]

    variables.update(lithiations)
    if config.getboolean('PHYSICS', 'do_thermal'):
        variables.update(variables_heating)
    if outer_step < Nsteps:
        for key in variables.keys():
            variables[key] = variables[key][:outer_step - 1, :]
        for key in overpotentials.keys():
            overpotentials[key] = overpotentials[key][:outer_step - 1, :]

    if config.getboolean('OUTPUT', 'plot'):
        run_ecm(net, alg, V_test, plot=True)
        for key in variables.keys():
            fig, ax = plt.subplots()
            ax.plot(variables[key][:, sorted_res_Ts])
            plt.title(key)
            plt.show()

        plot_phase_data(project, 'pore.temperature')
        fig, ax = plt.subplots()
        ax.plot(max_temperatures)
        ax.set_xlabel('Discharge Time [h]')
        ax.set_ylabel('Maximum Temperature [K]')

    if config.getboolean('OUTPUT', 'save'):
        print('Saving to', save_path)
        lower_mask = net['throat.spm_neg_inner'][res_Ts[sorted_res_Ts]]
        export(project, save_path, variables, 'var_',
               lower_mask=lower_mask, save_animation=False)
        export(project, save_path, overpotentials, 'eta_',
               lower_mask=lower_mask, save_animation=False)
        project.export_data(phases=[phase], filename='ecm.vtp')

    print("*" * 30)
    print("ECM Sim time", time.time() - st)
    print("*" * 30)
    return project, variables, solutions


def update_tabs(project, config):
    net = project.network
    sec = 'GEOMETRY'
    pos_Ps = net.pores('pos_cc')
    neg_Ps = net.pores('neg_cc')
    pos_ints = json.loads(config.get(sec, 'pos_tabs'))
    neg_ints = json.loads(config.get(sec, 'neg_tabs'))
    pos_tabs = pos_Ps[pos_ints]
    neg_tabs = neg_Ps[neg_ints]
    net['pore.pos_tab'] = False
    net['pore.neg_tab'] = False
    net['pore.pos_tab'][pos_tabs] = True
    net['pore.neg_tab'][neg_tabs] = True


def lump_thermal_props(config):
    sec = 'THICKNESS'
    pixel_size = config.getfloat(sec, 'pixel_size')
    lens = np.array([config.getfloat(sec, 'neg_electrode'),
                     config.getfloat(sec, 'pos_electrode'),
                     config.getfloat(sec, 'neg_cc') / 2,
                     config.getfloat(sec, 'pos_cc') / 2,
                     config.getfloat(sec, 'sep')])
    lens *= pixel_size
    sec = 'MATERIAL'
    rhos = np.array([config.getfloat(sec, 'neg_rho'),
                     config.getfloat(sec, 'pos_rho'),
                     config.getfloat(sec, 'neg_cc_rho'),
                     config.getfloat(sec, 'pos_cc_rho'),
                     config.getfloat(sec, 'sep_rho')])
    rho_lump = np.sum(lens * rhos) / np.sum(lens)
    Cps = np.array([config.getfloat(sec, 'neg_cp'),
                    config.getfloat(sec, 'pos_cp'),
                    config.getfloat(sec, 'neg_cc_cp'),
                    config.getfloat(sec, 'pos_cc_cp'),
                    config.getfloat(sec, 'sep_cp')])
    Cp_lump = np.sum(lens * rhos * Cps) / np.sum(lens * rhos)
    ks = np.array([config.getfloat(sec, 'neg_k'),
                   config.getfloat(sec, 'pos_k'),
                   config.getfloat(sec, 'neg_cc_k'),
                   config.getfloat(sec, 'pos_cc_k'),
                   config.getfloat(sec, 'sep_k')])
    alphas = ks / (rhos * Cps)
    res = 1 / alphas
    print(res)
    R_radial = np.sum(lens * res) / np.sum(lens)
    R_spiral = np.sum(lens) / np.sum(lens / res)
    out = {'alpha_radial': 1 / R_radial,
           'alpha_spiral': 1 / R_spiral,
           'lump_rho': rho_lump,
           'lump_Cp': Cp_lump}
    return out
