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
import pybamm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import sys
import time

plt.close("all")
# set logging level
pybamm.set_logging_level("INFO")


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
    fig = pconn(net, throats=net.throats("throat.free_stream"), c="green", fig=fig)
    t_sep = net.throats("spm_resistor")
    if len(t_sep) > 0:
        fig = pconn(
            net, throats=net.throats("spm_resistor"),
            c="k", fig=fig
        )


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
                    R=1.0):
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
    #        self.plot()
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
    #        self.plot()
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
    return net, arc_edges


def setup_ecm_alg(net, spacing, R):
    phase = op.phases.GenericPhase(network=net)
    cc_cond = 3e7
    cc_unit_len = spacing
    cc_unit_area = 25e-6 * 0.207
    phase["throat.electrical_conductance"] = cc_cond * cc_unit_area / cc_unit_len
    phase["throat.electrical_conductance"][net.throats("spm_resistor")] = 1 / R
    alg = op.algorithms.OhmicConduction(network=net)
    alg.setup(
        phase=phase,
        quantity="pore.potential",
        conductance="throat.electrical_conductance",
    )
    alg.settings["rxn_tolerance"] = 1e-8
    return alg, phase


def evaluate_python(python_eval, solution, current):
    keys = list(python_eval.keys())
    out = np.zeros(len(keys))
    for i, key in enumerate(keys):
        temp = python_eval[key].evaluate(
                solution.t[-1], solution.y[:, -1], u={"Current": current}
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
    tau = param.process_symbol(pybamm.standard_parameters_lithium_ion.tau_discharge)
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


def convert_time(param, non_dim_time, to="seconds"):
    s_parms = pybamm.standard_parameters_lithium_ion
    t_sec = param.process_symbol(s_parms.tau_discharge).evaluate()
    t = non_dim_time * t_sec
    if to == "hours":
        t *= 1 / 3600
    return t


def current_function(t):
    return pybamm.InputParameter("Current")


def make_spm(I_typical, height):
    model_options = {
            "thermal": "x-lumped",
            "external submodels": ["thermal"],
        }
    model = pybamm.lithium_ion.SPM(model_options)
    geometry = model.default_geometry
    param = model.default_parameter_values
    param.update(
        {
            "Typical current [A]": I_typical,
            "Current function": current_function,
            "Current": "[input]",
            "Electrode height [m]": height,
        }
    )
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
#    initial_ocv = 3.8518206633137266
#    totdV = initial_ocv - overpotentials[:, -1]
#    l = overpotentials.shape[1]-1
#    totdV -= np.sum(overpotentials[:, :l], axis=1)
    totdV = -np.sum(overpotentials, axis=1)

    return totdV/current


def evaluate(sim, var="Current collector current density [A.m-2]", current=0.0):
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
    python_eval = pybamm.EvaluatorPython(model.variables[var])
    python_value = python_eval.evaluate(
        solution.t[-1], solution.y[:, -1], u={"Current": current}
    )

    #    return proc(solution.t[-1])
    return value


def step_spm(zipped):
    sim, solution, I_app, dt, dead = zipped
    #    h = sim.parameter_values['Electrode height [m]']
    #    w = sim.parameter_values['Electrode width [m]']
    #    A_cc = h*w
#    results = np.zeros(len(variables) + 1)
    T_av = 303.15
    if ~dead:
        if solution is not None:
            sim.solver.y0 = solution.y[:, -1]
            sim.solver.t = solution.t[-1]
        sim.step(dt=dt, inputs={"Current": I_app},
                 external_variables={"X-averaged cell temperature": T_av},
                 save=False)
#        for i, key in enumerate(variables):
#            results[i] = evaluate(sim, key, I_app)
#        results[-1] = calc_R(sim, I_app)

#    else:
#        results.fill(np.nan)
    return sim.solution


def make_net(Nunit, R, spacing, pos_tabs, neg_tabs):
#    net = op.network.Cubic([Nunit + 2, 2, 1], spacing)
    net = op.network.Cubic([Nunit, 2, 1], spacing)
    net["pore.pos_cc"] = net["pore.right"]
    net["pore.neg_cc"] = net["pore.left"]

#    T = net.find_neighbor_throats(net.pores("front"), mode="xnor")
#    tt.trim(net, throats=T)
#    T = net.find_neighbor_throats(net.pores("back"), mode="xnor")
#    tt.trim(net, throats=T)
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
    cc_cond = 3e7
    cc_unit_len = spacing
    cc_unit_area = 25e-6 * 0.207
    phase["throat.electrical_conductance"] = cc_cond * cc_unit_area / cc_unit_len
    phase["throat.electrical_conductance"][net.throats("spm_resistor")] = 1 / R
    alg = op.algorithms.OhmicConduction(network=net)
    alg.setup(
        phase=phase,
        quantity="pore.potential",
        conductance="throat.electrical_conductance",
    )
    alg.settings["rxn_tolerance"] = 1e-8
    return net, alg, phase


def run_ecm(net, alg, V_terminal, plot=False):
    potential_pairs = net["throat.conns"][net.throats("spm_resistor")]
    P1 = potential_pairs[:, 0]
    P2 = potential_pairs[:, 1]
    adj = np.random.random(1) / 1e3
    alg.set_value_BC(net.pores("pos_tab"), values=V_terminal + adj)
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


def setup_pool(max_workers):
    pool = ThreadPoolExecutor(max_workers=max_workers)
    return pool


def pool_spm(spm_models, pool):
    data = list(pool.map(step_spm, spm_models))
    return data


def shutdown_pool(pool):
    pool.shutdown()
    del pool


def serial_spm(inputs):
    outputs = []
    for bundle in inputs:
        outputs.append(step_spm(bundle))
    return outputs
