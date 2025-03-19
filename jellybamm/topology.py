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
import jellybamm
import pandas as pd


def plot_topology(net, ax=None):
    # inner = net["pore.inner"]
    # outer = net["pore.outer"]
    c1 = np.array([[75 / 255, 139 / 255, 190 / 255, 1]])  # Cyan-Blue Azure
    c1 = np.array([[48 / 255, 105 / 255, 152 / 255, 1]])  # Lapis Lazuli
    c2 = np.array([[1, 232 / 255, 115 / 255, 1]])  # Shandy
    c2 = np.array([[1, 212 / 255, 59 / 255, 1]])  # Sunglow
    c3 = np.array([[100 / 255, 100 / 255, 100 / 255, 1]])  # Granite Gray
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax = jellybamm.plot_resistors(
        net, throats=net.throats("throat.neg_cc"), color=c1, ax=ax
    )
    ax = jellybamm.plot_resistors(
        net, throats=net.throats("throat.pos_cc"), color=c2, ax=ax
    )
    ax = pcoord(net, pores=net.pores("neg_cc"), color=c1, s=25, ax=ax)
    ax = pcoord(net, pores=net.pores("pos_cc"), color=c2, s=25, ax=ax)
    ax = pcoord(net, pores=net["pore.neg_tab"], color=c1, s=75, ax=ax)
    ax = pcoord(net, pores=net["pore.pos_tab"], color=c2, s=75, ax=ax)

    for label in ["inner_boundary", "free_stream"]:
        try:
            ax = pcoord(net, pores=net.pores(label), color=c3, ax=ax)
            ax = pconn(net, throats=net.throats(label), color=c3, ax=ax)
        except KeyError:
            pass

    t_sep = net.throats("spm_resistor")
    if len(t_sep) > 0:
        ax = pconn(net, throats=net.throats("spm_resistor"), color="k", ax=ax)
    return ax


def spiral(r, dr, ntheta=36, n=10):
    theta = np.linspace(0, n * (2 * np.pi), (n * ntheta) + 1)
    pos = np.linspace(0, n * ntheta, (n * ntheta) + 1) % ntheta
    pos = pos.astype(int)
    rad = r + np.linspace(0, n * dr, (n * ntheta) + 1)
    x = rad * np.cos(theta)
    y = rad * np.sin(theta)
    return (x, y, rad, pos)


def make_spiral_net(
    Nlayers, dtheta, spacing, inner_r, pos_tabs, neg_tabs, length_3d, tesla_tabs
):
    r"""
    Generate a perfect spiral network

    Parameters
    ----------
    Nlayers : TYPE
        DESCRIPTION.
    dtheta : TYPE
        DESCRIPTION.
    spacing : TYPE
        DESCRIPTION.
    pos_tabs : TYPE
        DESCRIPTION.
    neg_tabs : TYPE
        DESCRIPTION.
    length_3d : TYPE
        DESCRIPTION.
    tesla_tabs : TYPE
        DESCRIPTION.

    Returns
    -------
    prj : TYPE
        DESCRIPTION.
    arc_edges : TYPE
        DESCRIPTION.

    """
    Narc = int(360 / dtheta)  # number of nodes in a wind/layer
    Nunit = int(Nlayers * Narc)  # total number of unit cells
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

    # print("N SPM", net.num_throats("spm_resistor"))
    geo = jellybamm.setup_geometry(net, dtheta, spacing, length_3d=length_3d)
    net["throat.arc_length"] = np.deg2rad(dtheta) * net["throat.radial_position"]
    phase = op.phases.GenericPhase(network=net)
    op.physics.GenericPhysics(network=net, phase=phase, geometry=geo)
    return prj, arc_edges


def make_tomo_net(tomo_pnm, dtheta, spacing, length_3d, pos_tabs, neg_tabs):
    wrk = op.Workspace()
    input_dir = jellybamm.INPUT_DIR
    wrk.load_project(os.path.join(input_dir, tomo_pnm))
    sim_name = list(wrk.keys())[-1]
    project = wrk[sim_name]
    net = project.network
    pos_Ps = net.pores("pos_cc")
    neg_Ps = net.pores("neg_cc")
    # Translate relative indices into absolute indices
    pos_tabs = pos_Ps[pos_tabs]
    neg_tabs = neg_Ps[neg_tabs]
    net["pore.pos_tab"] = False
    net["pore.neg_tab"] = False
    net["pore.pos_tab"][pos_tabs] = True
    net["pore.neg_tab"][neg_tabs] = True

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
    geo = jellybamm.setup_geometry(net, dtheta, spacing, length_3d=length_3d)
    phase = op.phases.GenericPhase(network=net)
    op.physics.GenericPhysics(network=net, phase=phase, geometry=geo)
    return project, arc_edges


def make_1D_net(Nunit, spacing, pos_tabs, neg_tabs):
    r"""
    Generate a 1D network of batteries connected in parallel

    Parameters
    ----------
    Nunit : TYPE
        DESCRIPTION.
    spacing : TYPE
        DESCRIPTION.
    pos_tabs : TYPE
        DESCRIPTION.
    neg_tabs : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    net = op.network.Cubic([Nunit + 2, 2, 1], spacing)
    net["pore.pos_cc"] = net["pore.front"]
    net["pore.neg_cc"] = net["pore.back"]

    T = net.find_neighbor_throats(net.pores("left"), mode="xnor")
    tt.trim(net, throats=T)
    T = net.find_neighbor_throats(net.pores("right"), mode="xnor")
    tt.trim(net, throats=T)

    pos_cc_Ts = net.find_neighbor_throats(net.pores("pos_cc"), mode="xnor")
    neg_cc_Ts = net.find_neighbor_throats(net.pores("neg_cc"), mode="xnor")

    net.add_boundary_pores(labels=["front", "back"])
    net["pore.free_stream"] = np.logical_or(
        net["pore.front_boundary"], net["pore.back_boundary"]
    )
    net["throat.free_stream"] = np.logical_or(
        net["throat.front_boundary"], net["throat.back_boundary"]
    )

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
    net["throat.spm_resistor"][net["throat.free_stream"]] = False
    net["throat.spm_resistor_order"] = -1
    net["throat.spm_resistor_order"][net["throat.spm_resistor"]] = np.arange(Nunit)
    net["throat.spm_neg_inner"] = net["throat.spm_resistor"]

    del net["pore.left"]
    del net["pore.right"]
    del net["pore.front"]
    del net["pore.front_boundary"]
    del net["pore.back"]
    del net["pore.back_boundary"]
    del net["pore.internal"]
    del net["pore.surface"]
    del net["throat.internal"]
    del net["throat.surface"]

    phase = op.phases.GenericPhase(network=net)

    geo = op.geometry.GenericGeometry(network=net, pores=net.Ps, throats=net.Ts)
    op.physics.GenericPhysics(network=net, phase=phase, geometry=geo)

    net["pore.radial_position"] = net["pore.coords"][:, 0]
    net["pore.arc_index"] = np.indices([Nunit + 2, 4, 1])[0].flatten()
    net["pore.region_id"] = -1
    net["pore.cell_id"] = -1
    net["throat.arc_length"] = spacing
    net["throat.electrode_height"] = spacing
    # placeholder
    net["pore.volume"] = 1.0
    net["throat.area"] = 1.0
    net["throat.length"] = 1.0
    return net.project, np.cumsum(net["throat.arc_length"])


def network_to_netlist(network, Rs=1e-5, Ri=60, V=3.6, I_app=-5.0, Re_cc_pos=1.213e-05, Re_cc_neg=9.649e-06):
    r"""
    Make a liionpack netlist from a network

    Parameters
    ----------
    network : TYPE
        DESCRIPTION.
    Rs : TYPE, optional
        DESCRIPTION. The default is 1e-5.
    Ri : TYPE, optional
        DESCRIPTION. The default is 60.
    V : TYPE, optional
        DESCRIPTION. The default is 3.6.
    I_app : TYPE, optional
        DESCRIPTION. The default is -5.0.

    Returns
    -------
    netlist : TYPE
        DESCRIPTION.

    """
    desc = []
    node1 = []
    node2 = []
    value = []
    node1_x = []
    node1_y = []
    node2_x = []
    node2_y = []
    xs = network["pore.coords"][:, 0]
    ys = network["pore.coords"][:, 1]
    Tid = []

    # Negative current collector
    for t in network.throats("neg_cc"):
        desc.append("Rbn" + str(t))
        n1, n2 = network["throat.conns"][t]
        node1.append(n1)
        node2.append(n2)
        value.append(1 / network["throat.electrical_conductance"][t])
        node1_x.append(xs[n1])
        node1_y.append(ys[n1])
        node2_x.append(xs[n2])
        node2_y.append(ys[n2])
        Tid.append(t)

    # Positive current collector
    for t in network.throats("pos_cc"):
        desc.append("Rbp" + str(t))
        n1, n2 = network["throat.conns"][t]
        node1.append(n1)
        node2.append(n2)
        value.append(1 / network["throat.electrical_conductance"][t])
        node1_x.append(xs[n1])
        node1_y.append(ys[n1])
        node2_x.append(xs[n2])
        node2_y.append(ys[n2])
        Tid.append(t)

    # check contiguous
    node_max = max((max(node1), max(node2)))
    for i in range(node_max):
        if i not in node1:
            if i not in node2:
                print("Missing", i)
    add_res = True
    nn = node_max
    # Battery Segment
    for t in network.throats("throat.spm_resistor"):
        n1, n2 = network["throat.conns"][t]
        # swap node if n1 is negative
        n1_neg = network["pore.neg_cc"][n1]
        if n1_neg:
            n1, n2 = network["throat.conns"][t][::-1]
        vx = xs[n2] - xs[n1]
        vy = ys[n2] - ys[n1]
        vax = xs[n1] + vx / 3
        vbx = xs[n1] + vx * 2 / 3
        vay = ys[n1] + vy / 3
        vby = ys[n1] + vy * 2 / 3
        if add_res:
            # Make a new connection resistor from neg to V
            nn += 1
            desc.append("Rs" + str(t))
            node1.append(n1)
            node2.append(nn)
            value.append(Rs)
            node1_x.append(xs[n1])
            node1_y.append(ys[n1])
            node2_x.append(vax)
            node2_y.append(vay)
            Tid.append(t)
            # Make a battery node Va to Vb
            nn += 1
            desc.append("V" + str(t))
            node1.append(nn - 1)
            node2.append(nn)
            value.append(V)
            node1_x.append(vax)
            node1_y.append(vay)
            node2_x.append(vbx)
            node2_y.append(vby)
            Tid.append(t)
            # Make an intenal resistor from Vb to pos
            desc.append("Ri" + str(t))
            node1.append(nn)
            node2.append(n2)
            value.append(Ri)
            node1_x.append(vbx)
            node1_y.append(vby)
            node2_x.append(xs[n2])
            node2_y.append(ys[n2])
            Tid.append(t)
        else:
            desc.append("V" + str(t))
            node1.append(n1)
            node2.append(n2)
            value.append(V)
            node1_x.append(xs[n1])
            node1_y.append(ys[n1])
            node2_x.append(xs[n2])
            node2_y.append(ys[n2])
            Tid.append(t)

    # Terminals
    n1 = network.pores('pos_cc')[-1]
    for i, p in enumerate(network.pores('pos_tab')):
        n2 = network.pores('pos_tab')[i]
        if n1 != n2:
            desc.append("Rtp" + str(n2))
            node1.append(n1)
            node2.append(n2)
            value.append(Re_cc_pos)
            node1_x.append(xs[n1])
            node1_y.append(ys[n1])
            node2_x.append(xs[n2])
            node2_y.append(ys[n2])
            Tid.append(-1)
    n1 = network.pores('neg_cc')[0]
    for i, p in enumerate(network.pores('neg_tab')):
        n2 = p
        if n1 != n2:
            desc.append("Rtn" + str(n2))
            node1.append(n1)
            node2.append(n2)
            value.append(Re_cc_neg)
            node1_x.append(xs[n1])
            node1_y.append(ys[n1])
            node2_x.append(xs[n2])
            node2_y.append(ys[n2])
            Tid.append(-1)
            

    n1 = network.pores("pos_cc")[-1]
    n2 = network.pores("neg_cc")[0]
    desc.append("I0")
    node1.append(n1)
    node2.append(n2)
    value.append(I_app)
    node1_x.append(xs[n1])
    node1_y.append(ys[n1])
    node2_x.append(xs[n2])
    node2_y.append(ys[n2])
    Tid.append(-1)

    netlist_data = {
        "desc": desc,
        "node1": node1,
        "node2": node2,
        "value": value,
        "node1_x": node1_x,
        "node1_y": node1_y,
        "node2_x": node2_x,
        "node2_y": node2_y,
        "pnm_throat_id": Tid,
    }
    # add internal resistors
    netlist = pd.DataFrame(netlist_data)
    return netlist
