#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:10:52 2020

@author: thomas
"""
import numpy as np
import openpnm as op
from openpnm.topotools import plot_connections as pconn
from openpnm.topotools import plot_coordinates as pcoord
import matplotlib.pyplot as plt

plt.close('all')

def plot_topology(net):
#    an = net["pore.region_id"] == 1
#    an_cc = net["pore.region_id"] == 2
#    cat = net["pore.region_id"] == 3
#    cat_cc = net["pore.region_id"] == 4
#    sep = net["pore.region_id"] == 5
    inner = net["pore.inner"]
    outer = net["pore.outer"]
    fig = pconn(net, throats=net.throats("throat.neg_cc"), c="blue")
    fig = pconn(net, throats=net.throats("throat.pos_cc"), c="red", fig=fig)
#    fig = pcoord(net, pores=an, c="r", fig=fig)
#    fig = pcoord(net, pores=an_cc, c="y", fig=fig)
#    fig = pcoord(net, pores=cat, c="g", fig=fig)
#    fig = pcoord(net, pores=cat_cc, c="y", fig=fig)
#    fig = pcoord(net, pores=sep, c="k", fig=fig)
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


def make_spiral_net(Nlayers=3, dtheta=10, spacing=190e-6, pos_tabs = [0], neg_tabs = [-1]):
    
    # Number of nodes in each layer
    #    Nan = 9  # anode
    #    Ncat = 6  # cathode
    #    Ncc = 2  # current collector
    #    Nsep = 3  # separator
    Narc = np.int(360 / dtheta)  # number of nodes in a wind/layer
    Nunit = np.int(Nlayers * Narc)  # total number of unit cells
    print('Nunit', Nunit)
    # Number of nodes in the unit cell
    #    N1d = (Nan + Ncat + Ncc + Nsep) * 2
    N1d = 2
    # 2D assembly
    assembly = np.zeros([Nunit, N1d], dtype=int)
    #    p1 = Nan
    #    p2 = Nan + Ncc
    #    p3 = Nan + Ncc + Nan
    #    p4 = Nan + Ncc + Nan + Nsep
    #    p5 = Nan + Ncc + Nan + Nsep + Ncat
    #    p6 = Nan + Ncc + Nan + Nsep + Ncat + Ncc
    #    p7 = Nan + Ncc + Nan + Nsep + Ncat + Ncc + Ncat
    #    p8 = Nan + Ncc + Nan + Nsep + Ncat + Ncc + Ncat + Nsep
    #    assembly[:, :p1] = 1
    #    assembly[:, p1:p2] = 2
    #    assembly[:, p2:p3] = 1
    #    assembly[:, p3:p4] = 5
    #    assembly[:, p4:p5] = 3
    #    assembly[:, p5:p6] = 4
    #    assembly[:, p6:p7] = 3
    #    assembly[:, p7:p8] = 5
    assembly[:, 0] = 0
    assembly[:, 1] = 1
    unit_id = np.tile(np.arange(0, Nunit), (N1d, 1)).T
    #        (fig, (ax1, ax2)) = plt.subplots(1, 2)
    #        ax1.imshow(assembly)
    #        ax2.imshow(unit_id)
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
    
    
    #    sep_Ps = net["pore.region_id"] == 5
    #    Ts = net.find_neighbor_throats(pores=sep_Ps)
    #    net["throat.separator"][Ts] = True
    # Identify inner boundary pores and outer boundary pores
    # they are left and right, but not connected to each other
    #nbrs_left = net.find_neighbor_pores(
    #    pores=net.pores("left"), mode="union"
    #)
    #nbrs_right = net.find_neighbor_pores(
    #    pores=net.pores("right"), mode="union"
    #)
    #set_left = set(Ps_left)
    #set_right = set(Ps_right)
    #set_neighbors_left = set(nbrs_left)
    #set_neighors_right = set(nbrs_right)
    #inner_Ps = list(set_left.difference(set_neighors_right))
    #outer_Ps = list(set_right.difference(set_neighbors_left))
    Ps = net['throat.conns'][Ts].flatten()
    Ps, counts = np.unique(Ps.flatten(), return_counts=True)
    boundary = Ps[counts == 1]
    net["pore.inner"] = False
    net["pore.outer"] = False
    net["pore.inner"][boundary] = True
    net["pore.outer"][boundary] = True
    net["pore.inner"][net.pores('pos_cc')] = False
    net["pore.outer"][net.pores('neg_cc')] = False
    #        fig = pconn(self.net, throats=self.net.Ts)
    #        fig = pcoord(self.net, pores=self.net.pores("inner"), c="r", fig=fig)
    #        fig = pcoord(self.net, pores=self.net.pores("outer"), c="g", fig=fig)
    # Free stream convection boundary nodes
    free_rad = inner_r + (Nlayers + 0.5) * dr
    (x, y, rad, pos) = spiral(free_rad, dr, ntheta=Narc, n=1)
    net_free = op.network.Cubic(shape=[Narc, 1, 1], spacing=spacing)
    #        net_free['pore.radial_position'] = rad[:-1]
    #        net_free['pore.arc_index'] = pos[:-1]
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
    print('N nodes', net.num_pores('neg_cc'))
    print('N SPM', net.num_throats('spm_resistor'))

make_spiral_net()