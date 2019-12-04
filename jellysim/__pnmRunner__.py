#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:55:42 2019

@author: thomas
"""
import openpnm as op
import matplotlib.pyplot as plt
import numpy as np
from openpnm.topotools import plot_connections as pconn
from openpnm.topotools import plot_coordinates as pcoord
from openpnm.models.physics.generic_source_term import linear
import os

wrk = op.Workspace()
input_dir = os.path.join(os.getcwd(), 'input')


class pnm_runner(object):
    def __init__(self):
        pass

    def setup(self, options):
        if options['domain'] == 'tomography':
            self.setup_tomo()
        else:
            self.setup_jelly(Nlayers=options['Nlayers'],
                             dtheta=options['dtheta'],
                             spacing=options['spacing'])
        self.setup_geometry(dtheta=options['dtheta'],
                            spacing=options['spacing'],
                            length_3d=options['length_3d'])
        self.setup_thermal(options['T0'],
                           options['cp'],
                           options['rho'],
                           options['K0'],
                           options['heat_transfer_coefficient'])

    def setup_jelly(self, Nlayers=19, dtheta=10, spacing=1e-5):
        # Number of nodes in each layer
        Nan = 9  # anode
        Ncat = 6  # cathode
        Ncc = 2  # current collector
        Nsep = 3  # separator
        Narc = np.int(360 / dtheta)  # number of nodes in a wind/layer
        self.Nunit = np.int(Nlayers * Narc)  # total number of unit cells
        # Number of nodes in the unit cell
        N1d = (Nan + Ncat + Ncc + Nsep) * 2
        # 2D assembly
        assembly = np.zeros([self.Nunit, N1d], dtype=int)
        # %% Layer labels
        p1 = Nan
        p2 = Nan + Ncc
        p3 = Nan + Ncc + Nan
        p4 = Nan + Ncc + Nan + Nsep
        p5 = Nan + Ncc + Nan + Nsep + Ncat
        p6 = Nan + Ncc + Nan + Nsep + Ncat + Ncc
        p7 = Nan + Ncc + Nan + Nsep + Ncat + Ncc + Ncat
        p8 = Nan + Ncc + Nan + Nsep + Ncat + Ncc + Ncat + Nsep
        assembly[:, :p1] = 1
        assembly[:, p1:p2] = 2
        assembly[:, p2:p3] = 1
        assembly[:, p3:p4] = 5
        assembly[:, p4:p5] = 3
        assembly[:, p5:p6] = 4
        assembly[:, p6:p7] = 3
        assembly[:, p7:p8] = 5
        unit_id = np.tile(np.arange(0, self.Nunit), (N1d, 1)).T
#        (fig, (ax1, ax2)) = plt.subplots(1, 2)
#        ax1.imshow(assembly)
#        ax2.imshow(unit_id)

        self.net = op.network.Cubic(shape=[self.Nunit, N1d, 1],
                                    spacing=spacing)
        self.project = self.net.project
        self.net["pore.region_id"] = assembly.flatten()
        self.net["pore.cell_id"] = unit_id.flatten()
        # Extend the connections in the cell repetition direction
        self.net["pore.coords"][:, 0] *= 10
        #        self.plot()
        inner_r = 185 * spacing
        # Update coords
        self.net["pore.radial_position"] = 0.0
        self.net["pore.arc_index"] = 0
        r_start = self.net["pore.coords"][self.net["pore.cell_id"] == 0][:, 1]
        dr = spacing * N1d
        middle = np.int(np.floor(N1d / 2))
        for i in range(N1d):
            (x, y, rad, pos) = self.spiral(
                r_start[i] + inner_r, dr, ntheta=Narc, n=Nlayers
            )
            mask = self.net["pore.coords"][:, 1] == r_start[i]
            coords = self.net["pore.coords"][mask]
            coords[:, 0] = x[:-1]
            coords[:, 1] = y[:-1]
            self.net["pore.coords"][mask] = coords
            self.net["pore.radial_position"][mask] = rad[:-1]
            self.net["pore.arc_index"][mask] = pos[:-1]
            if i == middle:
                self.arc_edges = np.cumsum(np.deg2rad(dtheta) * rad)
                self.arc_edges -= self.arc_edges[0]
        #        self.plot()
        # Make interlayer connections after rolling
        Ps_left = self.net.pores("left")
        Ps_right = self.net.pores("right")
        coords_left = self.net["pore.coords"][Ps_left]
        coords_right = self.net["pore.coords"][Ps_right]
        conns = []
        # Identify pores in rolled layers that need new connections
        # This represents the separator layer which is not explicitly resolved
        for i_left, cl in enumerate(coords_left):
            vec = coords_right - cl
            dist = np.linalg.norm(vec, axis=1)
            if np.any(dist < 2 * spacing):
                i_right = np.argwhere(dist < 2 * spacing)[0][0]
                conns.append([Ps_left[i_left], Ps_right[i_right]])
        # Create new throats
        op.topotools.extend(network=self.net, throat_conns=conns,
                            labels=["separator"])
        sep_Ps = self.net["pore.region_id"] == 5
        Ts = self.net.find_neighbor_throats(pores=sep_Ps)
        self.net["throat.separator"][Ts] = True
        # Identify inner boundary pores and outer boundary pores
        # they are left and right, but not connected to each other
        nbrs_left = self.net.find_neighbor_pores(
            pores=self.net.pores("left"), mode="union"
        )
        nbrs_right = self.net.find_neighbor_pores(
            pores=self.net.pores("right"), mode="union"
        )
        set_left = set(Ps_left)
        set_right = set(Ps_right)
        set_neighbors_left = set(nbrs_left)
        set_neighors_right = set(nbrs_right)
        inner_Ps = list(set_left.difference(set_neighors_right))
        outer_Ps = list(set_right.difference(set_neighbors_left))
        self.net["pore.inner"] = False
        self.net["pore.outer"] = False
        self.net["pore.inner"][inner_Ps] = True
        self.net["pore.outer"][outer_Ps] = True
#        fig = pconn(self.net, throats=self.net.Ts)
#        fig = pcoord(self.net, pores=self.net.pores("inner"), c="r", fig=fig)
#        fig = pcoord(self.net, pores=self.net.pores("outer"), c="g", fig=fig)
        # Free stream convection boundary nodes
        free_rad = inner_r + (Nlayers + 0.5) * dr
        (x, y, rad, pos) = self.spiral(free_rad, dr, ntheta=Narc, n=1)
        net_free = op.network.Cubic(shape=[Narc, 1, 1], spacing=spacing)
        #        net_free['pore.radial_position'] = rad[:-1]
        #        net_free['pore.arc_index'] = pos[:-1]
        net_free["throat.trimmers"] = True
        net_free["pore.free_stream"] = True
        net_free["pore.coords"][:, 0] = x[:-1]
        net_free["pore.coords"][:, 1] = y[:-1]
        op.topotools.stitch(
            network=self.net,
            donor=net_free,
            P_network=self.net.pores("outer"),
            P_donor=net_free.Ps,
            len_max=dr,
            method="nearest",
        )
        free_pores = self.net.pores("free_stream")
        self.net["pore.radial_position"][free_pores] = rad[:-1]
        self.net["pore.arc_index"][free_pores] = pos[:-1]
        op.topotools.trim(network=self.net,
                          throats=self.net.throats("trimmers"))
#        self.plot_topology()
        self.net["pore.region_id"][self.net["pore.free_stream"]] = -1
        self.net["pore.cell_id"][self.net["pore.free_stream"]] = -1

    def setup_tomo(self):
        wrk.load_project(os.path.join(input_dir, 'MJ141-mid-top_m.pnm'))
        sim_name = list(wrk.keys())[0]
        self.project = wrk[sim_name]
        self.net = self.project.network
        self.arc_edges = [0.0]
        Ps = self.net.pores('cc_b')
        self.Nunit = self.net['pore.cell_id'][Ps].max() + 1
        old_coord = None
        for cell_id in range(self.Nunit):
            P = Ps[self.net['pore.cell_id'][Ps] == cell_id]
            coord = self.net['pore.coords'][P]
            if old_coord is not None:
                d = np.linalg.norm(coord-old_coord)
                self.arc_edges.append(self.arc_edges[-1] + d)
            old_coord = coord
        # Add 1 more
        self.arc_edges.append(self.arc_edges[-1] + d)
        self.arc_edges = np.asarray(self.arc_edges)

    def setup_geometry(self, dtheta, spacing, length_3d):
        # Create Geometry based on circular arc segment
        drad = 2 * np.pi * dtheta / 360
        self.geo = op.geometry.GenericGeometry(
                network=self.net, pores=self.net.Ps, throats=self.net.Ts
                )
        self.geo["throat.radial_position"] = self.net.interpolate_data(
                "pore.radial_position"
                )
        self.geo["pore.volume"] = (
                self.net["pore.radial_position"] * drad * spacing * length_3d
                )
        cn = self.net["throat.conns"]
        C1 = self.net["pore.coords"][cn[:, 0]]
        C2 = self.net["pore.coords"][cn[:, 1]]
        D = np.sqrt(((C1 - C2) ** 2).sum(axis=1))
        self.geo["throat.length"] = D
        # Work out if throat connects pores in same radial position
        rPs = self.geo["pore.arc_index"][self.net["throat.conns"]]
        sameR = rPs[:, 0] == rPs[:, 1]
        self.geo["throat.area"] = spacing * length_3d
        self.geo["throat.area"][sameR] = (
                self.geo["throat.radial_position"][sameR] * drad * length_3d
                )
        self.geo["throat.volume"] = 0.0
#        fig, (ax1, ax2) = plt.subplots(2, 2)
#        ax1[0].hist(self.geo["throat.area"])
#        ax1[1].hist(self.geo["throat.length"])
#        ax2[0].hist(self.geo["pore.radial_position"])
#        ax2[1].hist(self.geo["pore.volume"])

    def setup_thermal(self, T0, cp, rho, K0, heat_transfer_coefficient):
        self.phase = op.phases.GenericPhase(network=self.net)
        self.cp = cp
        self.rho = rho
        alpha = K0 / (cp * rho)
        hc = heat_transfer_coefficient / (cp * rho)
        # Set up Phase and Physics
        self.phase["pore.temperature"] = T0
        self.phase["pore.thermal_conductivity"] = alpha  # [W/(m.K)]
        self.phase["throat.conductance"] = (
            alpha * self.geo["throat.area"] / self.geo["throat.length"]
        )
        # Reduce separator conductance
        Ts = self.net.throats("separator")
        self.phase["throat.conductance"][Ts] *= 0.1
        # Free stream convective flux
        Ts = self.net.throats("stitched")
        self.phase["throat.conductance"][Ts] = self.geo["throat.area"][Ts] * hc
        self.phys = op.physics.GenericPhysics(
            network=self.net, geometry=self.geo, phase=self.phase
        )
        print('Mean throat conductance',
              np.mean(self.phase['throat.conductance']))
        print('Mean throat conductance Boundary',
              np.mean(self.phase['throat.conductance'][Ts]))

    def plot_topology(self):
        an = self.net["pore.region_id"] == 1
        an_cc = self.net["pore.region_id"] == 2
        cat = self.net["pore.region_id"] == 3
        cat_cc = self.net["pore.region_id"] == 4
        sep = self.net["pore.region_id"] == 5
        inner = self.net["pore.left"]
        outer = self.net["pore.right"]
        fig = pconn(self.net, throats=self.net.Ts)
        fig = pcoord(self.net, pores=an, c="r", fig=fig)
        fig = pcoord(self.net, pores=an_cc, c="y", fig=fig)
        fig = pcoord(self.net, pores=cat, c="g", fig=fig)
        fig = pcoord(self.net, pores=cat_cc, c="y", fig=fig)
        fig = pcoord(self.net, pores=sep, c="k", fig=fig)
        fig = pcoord(self.net, pores=inner, c="pink", fig=fig)
        fig = pcoord(self.net, pores=outer, c="purple", fig=fig)
        t_sep = self.net.throats("separator*")
        if len(t_sep) > 0:
            fig = pconn(
                self.net, throats=self.net.throats("separator*"),
                c="k", fig=fig
            )

    def spiral(self, r, dr, ntheta=36, n=10):
        theta = np.linspace(0, n * (2 * np.pi), (n * ntheta) + 1)
        pos = (np.linspace(0, n * ntheta, (n * ntheta) + 1) % ntheta)
        pos = pos.astype(int)
        rad = r + np.linspace(0, n * dr, (n * ntheta) + 1)
        x = rad * np.cos(theta)
        y = rad * np.sin(theta)
        return (x, y, rad, pos)

    def convert_spm_data(self, spm_data):
        out_data = np.ones(self.net.Np)
        out_data = spm_data[self.net["pore.cell_id"]]
        return out_data

    def run_step(self, heat_source, time_step, BC_value):
        # To Do - test whether this needs to be transient
        # Set Heat Source
        Q_scaled = self.convert_spm_data(heat_source) / (self.cp * self.rho)
        self.phys["pore.A1"] = 0.0
        self.phys["pore.A2"] = Q_scaled * self.net["pore.volume"]
        # Heat Source
        self.phys.add_model(
            "pore.source",
            model=linear,
            X="pore.temperature",
            A1="pore.A1",
            A2="pore.A2",
        )
        # Run Heat Transport Algorithm
        alg = op.algorithms.ReactiveTransport(network=self.net)
        alg.setup(
            phase=self.phase,
            quantity="pore.temperature",
            conductance="throat.conductance",
            rxn_tolerance=1e-12,
            relaxation_source=0.9,
            relaxation_quantity=0.9,
        )
        bulk_Ps = self.net.pores("free_stream", mode="not")
        alg.set_source("pore.source", bulk_Ps)
        alg.set_value_BC(self.net.pores("free_stream"), values=BC_value)
        alg.run()
        print(
            "Max Temp",
            alg["pore.temperature"].max(),
            "Min Temp",
            alg["pore.temperature"].min(),
        )

        self.phase.update(alg.results())

    def run_step_transient(self, heat_source, time_step, BC_value):
        # To Do - test whether this needs to be transient
        # Set Heat Source
        Q_scaled = self.convert_spm_data(heat_source) / (self.cp * self.rho)
        self.phys["pore.A1"] = 0.0
        self.phys["pore.A2"] = Q_scaled * self.net["pore.volume"]
        # Heat Source
        T0 = self.phase['pore.temperature']
        t_step = float(time_step/10)
        self.phys.add_model(
            "pore.source",
            model=linear,
            X="pore.temperature",
            A1="pore.A1",
            A2="pore.A2",
        )
        # Run Transient Heat Transport Algorithm
        alg = op.algorithms.TransientReactiveTransport(network=self.net)
        alg.setup(phase=self.phase,
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
        bulk_Ps = self.net.pores("free_stream", mode="not")
        alg.set_source("pore.source", bulk_Ps)
        alg.set_value_BC(self.net.pores("free_stream"), values=BC_value)
        alg.run()
        print(
            "Max Temp",
            alg["pore.temperature"].max(),
            "Min Temp",
            alg["pore.temperature"].min(),
        )
        self.phase["pore.temperature"] = alg["pore.temperature"]

    def plot_temperature_profile(self):
        data = self.phase["pore.temperature"]
        self.plot_pore_data(data, title="Temperature [K]")

    def plot_pore_data(self, data, title=None):
        fig, ax = plt.subplots(1)
        bulk_Ps = self.net.pores("free_stream", mode="not")
        coords = self.net["pore.coords"][bulk_Ps]
        xmin = coords[:, 0].min() * 1.05
        ymin = coords[:, 1].min() * 1.05
        xmax = coords[:, 0].max() * 1.05
        ymax = coords[:, 1].max() * 1.05
        mappable = ax.scatter(coords[:, 0], coords[:, 1], c=data[bulk_Ps])
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.colorbar(mappable)
        if title is not None:
            plt.title(title)

    def get_average_temperature(self):
        temp = np.zeros(self.Nunit)
        for i in range(self.Nunit):
            cell = self.net["pore.cell_id"] == i
            temp[i] = np.mean(self.phase["pore.temperature"][cell])
        return temp

    def export_pnm(self, filename='jelly_pnm'):
        self.project.export_data(filename=filename)
