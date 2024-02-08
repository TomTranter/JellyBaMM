#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:11:13 2019

@author: thomas
"""
import numpy as np
import openpnm as op
from openpnm.models.physics.source_terms import linear
import pybamm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os
from scipy import io
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator
import json

wrk = op.Workspace()


def cc_cond(project, param):
    net = project.network
    length_3d = param["Electrode width [m]"]
    neg_cc_econd = param["Negative current collector conductivity [S.m-1]"]
    pos_cc_econd = param["Positive current collector conductivity [S.m-1]"]
    t_neg_cc = param["Negative current collector thickness [m]"]
    t_pos_cc = param["Positive current collector thickness [m]"]
    cc_len = net["throat.arc_length"]
    neg_econd = neg_cc_econd * (t_neg_cc * length_3d)
    pos_econd = pos_cc_econd * (t_pos_cc * length_3d)
    neg_Ts = net.throats("neg_cc")
    neg_econd = neg_econd / cc_len[neg_Ts]
    pos_Ts = net.throats("pos_cc")
    pos_econd = pos_econd / cc_len[pos_Ts]
    net["throat.electrical_conductance"] = 0.0
    net["throat.electrical_conductance"][neg_Ts] = neg_econd
    net["throat.electrical_conductance"][pos_Ts] = pos_econd
    return neg_econd, pos_econd


def current_function(t):
    return pybamm.InputParameter("Current")


def adjust_parameters(parameter_values, I_typical):
    parameter_values.update(
        {
            # "Typical current [A]": I_typical,
            "Current function [A]": current_function,
            "Electrode height [m]": "[input]",
        }
    )
    parameter_values.update({"Current": "[input]"}, check_already_exists=False)
    return parameter_values


def output_variables():
    return [
        "Terminal voltage [V]",
        "Volume-averaged cell temperature [K]",
        "Current collector current density [A.m-2]",
        "X-averaged negative electrode extent of lithiation",
        "X-averaged positive electrode extent of lithiation",
        "Volume-averaged total heating [W.m-3]",
        "X-averaged battery reaction overpotential [V]",
        "X-averaged battery concentration overpotential [V]",
        "X-averaged battery electrolyte ohmic losses [V]",
        "X-averaged battery solid phase ohmic losses [V]",
        "Battery open-circuit voltage [V]",
    ]


def calc_R(overpotentials, current):
    totdV = -np.sum(overpotentials, axis=1)
    return totdV / current


def step_spm(zipped):
    built_model, solver, solution, I_app, e_height, dt, T_av, dead = zipped
    inputs = {"Current": I_app, "Electrode height [m]": e_height}
    if len(built_model.external_variables) > 0:
        external_variables = {"Volume-averaged cell temperature": T_av}
    else:
        external_variables = None
    if ~dead:
        built_model.timescale_eval = built_model.timescale.evaluate(inputs=inputs)
        if solution is not None:
            pass

        solution = solver.step(
            old_solution=solution,
            model=built_model,
            dt=dt,
            external_variables=external_variables,
            inputs=inputs,
            npts=2,
            save=False,
        )

    return solution


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
    net["throat.cc_power_loss"] = 0.0
    net["throat.cc_power_loss"][net.throats("neg_cc")] = Pow_neg
    net["throat.cc_power_loss"][net.throats("pos_cc")] = Pow_pos
    net.add_model(
        propname="pore.cc_power_loss",
        model=op.models.misc.from_neighbor_throats,
        prop="throat.cc_power_loss",
        mode="max",
    )


def run_jellybamm(net, alg, V_terminal, plot=False):
    potential_pairs = net["throat.conns"][net.throats("spm_resistor")]
    P1 = potential_pairs[:, 0]
    P2 = potential_pairs[:, 1]
    adj = np.random.random(1) / 1e3
    alg.set_value_BC(net.pores("pos_tab"), values=V_terminal + adj)
    alg.set_value_BC(net.pores("neg_tab"), values=adj)
    alg.run()
    V_local_pnm = alg["pore.potential"][P2] - alg["pore.potential"][P1]
    V_local_pnm[net["pore.pos_cc"][P1]] *= -1
    I_local_pnm = alg.rate(throats=net.throats("spm_resistor"), mode="single")
    I_local_pnm *= np.sign(V_terminal.flatten())
    R_local_pnm = V_local_pnm / I_local_pnm
    if plot:
        pos_mask = net.pores("pos_cc")
        neg_mask = net.pores("neg_cc")
        plt.figure()
        plt.plot(alg["pore.potential"][pos_mask])
        plt.plot(alg["pore.potential"][neg_mask])

    return (V_local_pnm, I_local_pnm, R_local_pnm)


def setup_geometry(net, dtheta, spacing, length_3d):
    # Create Geometry based on circular arc segment
    drad = np.deg2rad(dtheta)
    geo = op.geometry.GenericGeometry(network=net, pores=net.Ps, throats=net.Ts)
    if "throat.radial_position" not in net.props():
        geo["throat.radial_position"] = net.interpolate_data("pore.radial_position")
    geo["pore.volume"] = net["pore.radial_position"] * drad * spacing * length_3d
    cn = net["throat.conns"]
    C1 = net["pore.coords"][cn[:, 0]]
    C2 = net["pore.coords"][cn[:, 1]]
    D = np.sqrt(((C1 - C2) ** 2).sum(axis=1))
    geo["throat.length"] = D
    # Work out if throat connects pores in same radial position
    rPs = geo["pore.arc_index"][net["throat.conns"]]
    sameR = rPs[:, 0] == rPs[:, 1]
    geo["throat.area"] = spacing * length_3d
    geo["throat.electrode_height"] = geo["throat.radial_position"] * drad
    geo["throat.area"][sameR] = geo["throat.electrode_height"][sameR] * length_3d
    geo["throat.volume"] = 0.0
    geo["throat.volume"][sameR] = geo["throat.area"][sameR] * spacing
    return geo


def setup_thermal(project, parameter_values):
    T0 = parameter_values["Initial temperature [K]"]
    lumpy_therm = lump_thermal_props(parameter_values)
    cp = lumpy_therm["lump_Cp"]
    rho = lumpy_therm["lump_rho"]
    total_htc = parameter_values["Total heat transfer coefficient [W.m-2.K-1]"]
    net = project.network
    geo = project.geometries()["geo_01"]
    phase = project.phases()["phase_01"]
    phys = project.physics()["phys_01"]
    hc = total_htc / (cp * rho)
    # Set up Phase and Physics
    phase["pore.temperature"] = T0
    alpha_spiral = lumpy_therm["alpha_spiral"]
    alpha_radial = lumpy_therm["alpha_radial"]
    phys["throat.conductance"] = 1.0 * geo["throat.area"] / geo["throat.length"]
    # Apply anisotropic heat conduction
    Ts = net.throats("spm_resistor")
    phys["throat.conductance"][Ts] *= alpha_radial
    Ts = net.throats("spm_resistor", mode="not")
    phys["throat.conductance"][Ts] *= alpha_spiral
    # Free stream convective flux
    Ts = net.throats("free_stream")
    phys["throat.conductance"][Ts] = geo["throat.area"][Ts] * hc

    # print("Mean throat conductance", np.mean(phys["throat.conductance"]))
    # print("Mean throat conductance Boundary", np.mean(phys["throat.conductance"][Ts]))


def apply_heat_source(project, Q):
    # The SPMs are defined at the throat but the pores represent the
    # Actual electrode volume so need to interpolate for heat sources
    net = project.network
    phys = project.physics()["phys_01"]
    spm_Ts = net.throats("spm_resistor")
    phys["throat.heat_source"] = 0.0
    phys["throat.heat_source"][spm_Ts] = Q
    phys.add_model(
        propname="pore.heat_source",
        model=op.models.misc.from_neighbor_throats,
        prop="throat.heat_source",
        mode="max",
    )


def apply_heat_source_lp(project, Q):
    # The SPMs are defined at the throat but the pores represent the
    # Actual electrode volume so need to interpolate for heat sources
    phys = project.physics()["phys_01"]
    phys["throat.heat_source"] = Q
    phys.add_model(
        propname="pore.heat_source",
        model=op.models.misc.from_neighbor_throats,
        prop="throat.heat_source",
        mode="mean",
    )


def run_step_transient(project, time_step, BC_value, cp, rho, third=False):
    # To Do - test whether this needs to be transient
    net = project.network
    phase = project.phases()["phase_01"]
    phys = project.physics()["phys_01"]
    phys["pore.A1"] = 0.0
    Q_spm = phys["pore.heat_source"] * net["pore.volume"]
    # Q_cc = net["pore.cc_power_loss"]
    # print(
    #     "Q_spm",
    #     np.around(np.sum(Q_spm), 2),
    #     "\n",
    #     "Q_cc",
    #     np.around(np.sum(Q_cc), 2),
    #     "\n",
    #     "ratio Q_cc/Q_spm",
    #     np.around(np.sum(Q_cc) / np.sum(Q_spm), 2),
    # )
    # phys["pore.A2"] = (Q_spm + Q_cc) / (cp * rho)
    phys["pore.A2"] = (Q_spm) / (cp * rho)
    # Heat Source
    T0 = phase["pore.temperature"]
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
    alg.setup(
        phase=phase,
        conductance="throat.conductance",
        quantity="pore.temperature",
        t_initial=0.0,
        t_final=time_step,
        t_step=t_step,
        t_output=t_step,
        t_tolerance=1e-9,
        t_precision=12,
        rxn_tolerance=1e-9,
        t_scheme="implicit",
    )
    alg.set_IC(values=T0)
    bulk_Ps = net.pores("free_stream", mode="not")
    alg.set_source("pore.source", bulk_Ps)
    if third:
        # To do - 12 only works if detheta is 10
        free_pores = net.pores("free_stream")
        Ps = free_pores[net["pore.arc_index"][free_pores] < 12]
    else:
        Ps = net.pores("free_stream")
    alg.set_value_BC(Ps, values=BC_value)
    alg.run()
    # print(
    #     "Max Temp",
    #     np.around(alg["pore.temperature"].max(), 3),
    #     "Min Temp",
    #     np.around(alg["pore.temperature"].min(), 3),
    # )
    phase["pore.temperature"] = alg["pore.temperature"]
    project.purge_object(alg)


def setup_pool(max_workers, pool_type="Process"):
    if pool_type == "Process":
        pool = ProcessPoolExecutor()
    else:
        pool = ThreadPoolExecutor()
    return pool


def _regroup_models(spm_models, max_workers):
    unpack = list(spm_models)
    num_models = len(unpack)
    num_chunk = int(np.ceil(num_models / max_workers))
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


def _format_key(key):
    key = [word + "_" for word in key.split() if "[" not in word]
    return "".join(key)[:-1]


def export(
    project,
    save_dir=None,
    export_dict=None,
    prefix="",
    lower_mask=None,
    save_animation=False,
):
    if save_dir is None:
        save_dir = os.getcwd()
    else:
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
    for key in export_dict.keys():
        for suffix in ["lower", "upper"]:
            if suffix == "lower":
                mask = lower_mask
            else:
                mask = ~lower_mask
            data = export_dict[key][:, mask]
            save_path = os.path.join(save_dir, prefix + _format_key(key) + "_" + suffix)
            io.savemat(file_name=save_path, mdict={"data": data}, long_field_names=True)


def polar_transform(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def cartesian_transform(r, t):
    x = r * np.cos(t)
    y = r * np.sin(t)
    return x, y


def interpolate_spm_number(project, x_len=2000, y_len=2000):
    net = project.network
    all_x = []
    all_y = []
    all_t = []
    all_data = []
    for label in ["inner_boundary", "spm_resistor", "free_stream"]:
        if "throat." + label in net.labels():
            res_Ts = net.throats(label)
            sorted_res_Ts = net["throat.spm_resistor_order"][res_Ts].argsort()
            res_pores = net["pore.coords"][net["throat.conns"][res_Ts[sorted_res_Ts]]]
            res_Ts_coords = np.mean(res_pores, axis=1)
            x = res_Ts_coords[:, 0]
            y = res_Ts_coords[:, 1]
            if label == "spm_resistor":
                data = np.arange(0, len(res_Ts))[np.newaxis, :]
            else:
                data = np.ones(len(res_Ts))[np.newaxis, :] * -1
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
    grid_x, grid_y = np.mgrid[
        x.min() * f : x.max() * f : complex(x_len, 0),
        y.min() * f : y.max() * f : complex(y_len, 0),
    ]
    arr = myInterpolator(grid_x, grid_y, 0)
    # arr[arr == -1] = np.nan
    fig, (ax1) = plt.subplots(1, 1)
    ax1.imshow(arr)
    return arr


def interpolate_spm_number_model(project, dim=1000):
    x_len = y_len = dim
    net = project.network
    all_x = []
    all_y = []
    all_t = []
    all_data = []
    # Inner boundary
    inner_Ts = net.throats("inner_boundary")
    inner_Ts_coords = np.mean(net["pore.coords"][net["throat.conns"][inner_Ts]], axis=1)
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
    res_Ts = net.throats("spm_resistor")
    sorted_res_Ts = net["throat.spm_resistor_order"][res_Ts].argsort()
    res_pores = net["pore.coords"][net["throat.conns"][res_Ts[sorted_res_Ts]]]
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
    free_Ts = net.throats("free_stream")
    free_Ts_coords = np.mean(net["pore.coords"][net["throat.conns"][free_Ts]], axis=1)
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
    grid_x, grid_y = np.mgrid[
        x.min() * f : x.max() * f : complex(x_len, 0),
        y.min() * f : y.max() * f : complex(y_len, 0),
    ]
    arr = myInterpolator(grid_x, grid_y, 0)
    return arr


def check_vlim(solution, low, high):
    l_check = solution["Terminal voltage [V]"].entries[-1] > low
    h_check = solution["Terminal voltage [V]"].entries[-1] < high
    return l_check * h_check


def update_tabs(project, config):
    net = project.network
    sec = "GEOMETRY"
    pos_Ps = net.pores("pos_cc")
    neg_Ps = net.pores("neg_cc")
    pos_ints = json.loads(config.get(sec, "pos_tabs"))
    neg_ints = json.loads(config.get(sec, "neg_tabs"))
    pos_tabs = pos_Ps[pos_ints]
    neg_tabs = neg_Ps[neg_ints]
    net["pore.pos_tab"] = False
    net["pore.neg_tab"] = False
    net["pore.pos_tab"][pos_tabs] = True
    net["pore.neg_tab"][neg_tabs] = True


def lump_thermal_props(param):
    layers = [
        "Negative electrode",
        "Positive electrode",
        "Negative current collector",
        "Positive current collector",
        "Separator",
    ]
    props = [
        "thickness [m]",
        "density [kg.m-3]",
        "specific heat capacity [J.kg-1.K-1]",
        "thermal conductivity [W.m-1.K-1]",
    ]
    all_props = np.zeros([len(props), len(layers)])
    for i, prop in enumerate(props):
        for j, l in enumerate(layers):
            all_props[i][j] = param[l + " " + prop]
    # Break them up
    lens = all_props[:, 0]
    rhos = all_props[:, 1]
    Cps = all_props[:, 2]
    ks = all_props[:, 3]
    # Lumped props
    rho_lump = np.sum(lens * rhos) / np.sum(lens)
    Cp_lump = np.sum(lens * rhos * Cps) / np.sum(lens * rhos)
    # Thermal diffusivity alpha = k / (rho * Cp)
    alphas = ks / (rhos * Cps)
    # Thermal resistance
    res = 1 / alphas
    R_radial = np.sum(lens * res) / np.sum(lens)
    R_spiral = np.sum(lens) / np.sum(lens / res)
    out = {
        "alpha_radial": 1 / R_radial,
        "alpha_spiral": 1 / R_spiral,
        "lump_rho": rho_lump,
        "lump_Cp": Cp_lump,
    }
    return out
