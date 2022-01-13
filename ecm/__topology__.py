#
# Topology Functions
#

import numpy as np
import openpnm as op
import openpnm.topotools as tt
from openpnm.topotools import plot_connections as pconn
from openpnm.topotools import plot_coordinates as pcoord
import os
import matplotlib.pyplot as plt
import ecm


def plot_topology(net, ax=None):
    # inner = net["pore.inner"]
    # outer = net["pore.outer"]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax = ecm.plot_resistors(net, throats=net.throats("throat.neg_cc"), c="blue", ax=ax)
    ax = ecm.plot_resistors(net, throats=net.throats("throat.pos_cc"), c="red", ax=ax)
    ax = pcoord(net, pores=net.pores("neg_cc"), c="blue", s=25, ax=ax)
    ax = pcoord(net, pores=net.pores("pos_cc"), c="red", s=25, ax=ax)
    ax = pcoord(net, pores=net["pore.neg_tab"], c="blue", s=75, ax=ax)
    ax = pcoord(net, pores=net["pore.pos_tab"], c="red", s=75, ax=ax)
    try:
        ax = pcoord(net, pores=net.pores("free_stream"), c="green", ax=ax)
        ax = pconn(net, throats=net.throats("throat.free_stream"), c="green", ax=ax)
    except KeyError:
        pass

    t_sep = net.throats("spm_resistor")
    if len(t_sep) > 0:
        ax = pconn(net, throats=net.throats("spm_resistor"), c="k", ax=ax)
    return ax


def spiral(r, dr, ntheta=36, n=10):
    theta = np.linspace(0, n * (2 * np.pi), (n * ntheta) + 1)
    pos = np.linspace(0, n * ntheta, (n * ntheta) + 1) % ntheta
    pos = pos.astype(int)
    rad = r + np.linspace(0, n * dr, (n * ntheta) + 1)
    x = rad * np.cos(theta)
    y = rad * np.sin(theta)
    return (x, y, rad, pos)


def make_spiral_net(config):
    sub = "GEOMETRY"
    Nlayers = config.getint(sub, "Nlayers")
    dtheta = config.getint(sub, "dtheta")
    spacing = config.getfloat(sub, "layer_spacing")
    tesla_tabs = False
    try:
        pos_tabs = config.getint(sub, "pos_tabs")
        neg_tabs = config.getint(sub, "neg_tabs")
    except ValueError:
        print("Tesla tabs")
        tesla_tabs = True
    length_3d = config.getfloat(sub, "length_3d")
    Narc = np.int(360 / dtheta)  # number of nodes in a wind/layer
    Nunit = np.int(Nlayers * Narc)  # total number of unit cells
    N1d = 2
    # 2D assembly
    assembly = np.zeros([Nunit, N1d], dtype=int)

    assembly[:, 0] = 0
    assembly[:, 1] = 1
    unit_id = np.tile(np.arange(0, Nunit), (N1d, 1)).T
    prj = op.Project()
    net = op.network.Cubic(shape=[Nunit, N1d, 1], spacing=spacing, project=prj)
    net["pore.pos_cc"] = net["pore.back"]
    net["pore.neg_cc"] = net["pore.front"]

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
        (x, y, rad, pos) = spiral(r_start[i] + inner_r, dr, ntheta=Narc, n=Nlayers)
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
    op.topotools.extend(network=net, throat_conns=conns, labels=["separator"])
    h = net.check_network_health()
    if len(h["duplicate_throats"]) > 0:
        trim_Ts = np.asarray(h["duplicate_throats"])[:, 1]
        op.topotools.trim(network=net, throats=trim_Ts)
    Ts = net.find_neighbor_throats(pores=net.pores("pos_cc"), mode="xor")

    net["throat.pos_cc"] = False
    net["throat.neg_cc"] = False
    net["throat.pos_cc"][pos_cc_Ts] = True
    net["throat.neg_cc"][neg_cc_Ts] = True
    net["throat.spm_resistor"] = True
    net["throat.spm_resistor"][pos_cc_Ts] = False
    net["throat.spm_resistor"][neg_cc_Ts] = False
    net["throat.spm_resistor_order"] = -1
    spm_res = net["throat.spm_resistor"]
    net["throat.spm_resistor_order"][spm_res] = np.arange(np.sum(spm_res))
    p1 = net["throat.conns"][:, 0]
    p1_neg = net["pore.neg_cc"][p1]
    net["throat.spm_neg_inner"] = p1_neg * net["throat.spm_resistor"]
    net["throat.spm_pos_inner"] = (~p1_neg) * net["throat.spm_resistor"]
    Ps = net["throat.conns"][Ts].flatten()
    Ps, counts = np.unique(Ps.flatten(), return_counts=True)
    boundary = Ps[counts == 1]
    net["pore.inner"] = False
    net["pore.outer"] = False
    net["pore.inner"][boundary] = True
    net["pore.outer"][boundary] = True
    net["pore.inner"][net.pores("pos_cc")] = False
    net["pore.outer"][net.pores("neg_cc")] = False

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

    net["throat.free_stream"] = net["throat.stitched"]
    del net["throat.stitched"]

    free_pores = net.pores("free_stream")
    net["pore.radial_position"][free_pores] = rad[:-1]
    net["pore.arc_index"][free_pores] = pos[:-1]
    op.topotools.trim(network=net, throats=net.throats("trimmers"))

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
    P1 = net["throat.conns"][:, 0]
    P2 = net["throat.conns"][:, 1]
    same_arc = net["pore.arc_index"][P1] == net["pore.arc_index"][P2]
    cross_stitch = np.logical_and(net["throat.stitched"], ~same_arc)
    net["throat.inner_boundary"] = net["throat.stitched"]
    net["throat.trimmers"][cross_stitch] = True
    del net["throat.stitched"]
    del net["pore.left"]
    del net["pore.right"]
    del net["pore.front"]
    del net["pore.back"]
    del net["pore.internal"]
    del net["pore.surface"]
    del net["throat.internal"]
    del net["throat.surface"]

    op.topotools.trim(network=net, throats=net.throats("trimmers"))

    print("N SPM", net.num_throats("spm_resistor"))
    geo = ecm.setup_geometry(net, dtheta, spacing, length_3d=length_3d)
    net["throat.arc_length"] = np.deg2rad(dtheta) * net["throat.radial_position"]
    phase = op.phases.GenericPhase(network=net)
    op.physics.GenericPhysics(network=net, phase=phase, geometry=geo)
    return prj, arc_edges


def make_tomo_net(config):
    sub = "GEOMETRY"
    dtheta = config.getint(sub, "dtheta")
    spacing = config.getfloat(sub, "layer_spacing")
    length_3d = config.getfloat(sub, "length_3d")
    wrk = op.Workspace()
    input_dir = ecm.INPUT_DIR
    tomo_pnm = config.get("TOMOGRAPHY", "filename")
    wrk.load_project(os.path.join(input_dir, tomo_pnm))
    sim_name = list(wrk.keys())[-1]
    project = wrk[sim_name]
    net = project.network
    ecm.update_tabs(project, config)
    arc_edges = [0.0]
    Ps = net.pores("neg_cc")
    Nunit = net["pore.cell_id"][Ps].max() + 1
    old_coord = None
    for cell_id in range(Nunit):
        P = Ps[net["pore.cell_id"][Ps] == cell_id]
        coord = net["pore.coords"][P]
        if old_coord is not None:
            d = np.linalg.norm(coord - old_coord)
            arc_edges.append(arc_edges[-1] + d)
        old_coord = coord
    # Add 1 more
    arc_edges.append(arc_edges[-1] + d)
    arc_edges = np.asarray(arc_edges)
    geo = ecm.setup_geometry(net, dtheta, spacing, length_3d=length_3d)
    phase = op.phases.GenericPhase(network=net)
    op.physics.GenericPhysics(network=net, phase=phase, geometry=geo)
    return project, arc_edges


def make_1D_net(config):
    sub = "GEOMETRY"
    Nunit = config.getint(sub, "nunit_OneD")
    spacing = config.getfloat(sub, "spacing_OneD")
    pos_tabs = config.getint(sub, "pos_tabs")
    neg_tabs = config.getint(sub, "neg_tabs")
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
    net["throat.spm_resistor_order"] = -1
    net["throat.spm_resistor_order"][net["throat.spm_resistor"]] = np.arange(Nunit)
    net["throat.spm_neg_inner"] = net["throat.spm_resistor"]
    net["pore.free_stream"] = False
    del net["pore.left"]
    del net["pore.right"]
    del net["pore.front"]
    del net["pore.back"]
    del net["pore.internal"]
    del net["pore.surface"]
    del net["throat.internal"]
    del net["throat.surface"]

    phase = op.phases.GenericPhase(network=net)

    geo = op.geometry.GenericGeometry(network=net, pores=net.Ps, throats=net.Ts)
    op.physics.GenericPhysics(network=net, phase=phase, geometry=geo)

    net["pore.radial_position"] = net["pore.coords"][:, 0]
    net["pore.arc_index"] = np.indices([Nunit + 2, 2, 1])[0].flatten()
    net["pore.region_id"] = -1
    net["pore.cell_id"] = -1
    net["throat.arc_length"] = spacing
    net["throat.electrode_height"] = spacing
    # placeholder
    net["pore.volume"] = 1.0
    net["throat.area"] = 1.0
    net["throat.length"] = 1.0
    ecm.plot_topology(net)
    return net.project, np.cumsum(net["throat.arc_length"])