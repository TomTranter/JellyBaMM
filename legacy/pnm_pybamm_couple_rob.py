#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:28:34 2019

@author: thomas
"""

import pybamm
import sys
import openpnm as op
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider
import numpy as np
from openpnm.topotools import plot_connections as pconn
from openpnm.topotools import plot_coordinates as pcoord
from openpnm.models.physics.generic_source_term import linear
import time


# %% Set up domain in OpenPNM

# Thermal Parameters
Q_Pybamm_averaged = 2e6  # [W/m^3]
cp = 1148
rho = 5071.75
K0 = 1  # 132.58
T0 = 303
alpha = K0 / (cp * rho)
heat_transfer_coefficient = 10
hc = heat_transfer_coefficient / (cp * rho)

# Number of nodes in each layer
Nan = 9  # anode
Ncat = 6  # cathode
Ncc = 2  # current collector
Nsep = 3  # separator
# Number of unit cells
Nlayers = 10  # number of windings
dtheta = 20  # arc angle between nodes
Narc = np.int(360 / dtheta)  # number of nodes in a wind/layer
Nunit = np.int(Nlayers * Narc)  # total number of unit cells
# Number of nodes in the unit cell
N1d = (Nan + Ncat + Ncc + Nsep) * 2
# 2D assembly
assembly = np.zeros([Nunit, N1d], dtype=int)
# Network spacing
spacing = 1e-5  # 10 microns
length_3d = 0.065
I_app_mag = 2.5 * (Nlayers / 20)  # A

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

unit_id = np.tile(np.arange(0, Nunit), (N1d, 1)).T

(fig, (ax1, ax2)) = plt.subplots(1, 2)
ax1.imshow(assembly)
ax2.imshow(unit_id)

# %% Start OPENPNM


class pnm_runner:
    def __init__(self):
        pass

    def setup(self):
        self.net = op.network.Cubic(shape=[Nunit, N1d, 1], spacing=spacing)
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
        op.topotools.extend(network=self.net, throat_conns=conns, labels=["separator"])
        #        self.plot()
        Ts = self.net.find_neighbor_throats(pores=self.net["pore.region_id"] == 5)
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
        fig = pconn(self.net, throats=self.net.Ts)
        fig = pcoord(self.net, pores=self.net.pores("inner"), c="r", fig=fig)
        fig = pcoord(self.net, pores=self.net.pores("outer"), c="g", fig=fig)
        # Free stream convection boundary nodes
        # Make new network wrapping around the original domain and
        # stitch together
        #        free_rad = inner_r + (Nlayers+1.0)*dr
        #        (x, y, rad, pos) = self.spiral(free_rad, 0.0, ntheta=Narc, n=1)
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
        op.topotools.trim(network=self.net, throats=self.net.throats("trimmers"))
        self.plot_topology()
        self.net["pore.region_id"][self.net["pore.free_stream"]] = -1
        self.net["pore.cell_id"][self.net["pore.free_stream"]] = -1

        # Create Geometry based on circular arc segment
        drad = 2 * np.pi * dtheta / 360
        geo = op.geometry.GenericGeometry(
            network=self.net, pores=self.net.Ps, throats=self.net.Ts
        )
        geo["throat.radial_position"] = self.net.interpolate_data(
            "pore.radial_position"
        )
        geo["pore.volume"] = (
            self.net["pore.radial_position"] * drad * spacing * length_3d
        )
        cn = self.net["throat.conns"]
        C1 = self.net["pore.coords"][cn[:, 0]]
        C2 = self.net["pore.coords"][cn[:, 1]]
        D = np.sqrt(((C1 - C2) ** 2).sum(axis=1))
        geo["throat.length"] = D
        # Work out if throat connects pores in same radial position
        rPs = geo["pore.arc_index"][self.net["throat.conns"]]
        sameR = rPs[:, 0] == rPs[:, 1]
        geo["throat.area"] = spacing * length_3d
        geo["throat.area"][sameR] = (
            geo["throat.radial_position"][sameR] * drad * length_3d
        )
        geo["throat.volume"] = 0.0
        self.phase = op.phases.GenericPhase(network=self.net)
        # Set up Phase and Physics
        self.phase["pore.temperature"] = T0
        self.phase["pore.thermal_conductivity"] = alpha  # [W/(m.K)]
        self.phase["throat.conductance"] = (
            alpha * geo["throat.area"] / geo["throat.length"]
        )
        # Reduce separator conductance
        self.phase["throat.conductance"][self.net.throats("separator")] *= 0.1
        # Free stream convective flux
        Ts = self.net.throats("stitched")
        self.phase["throat.conductance"][Ts] = geo["throat.area"][Ts] * hc
        self.phys = op.physics.GenericPhysics(
            network=self.net, geometry=geo, phase=self.phase
        )

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
                self.net, throats=self.net.throats("separator*"), c="k", fig=fig
            )

    def spiral(self, r, dr, ntheta=36, n=10):
        theta = np.linspace(0, n * (2 * np.pi), (n * ntheta) + 1)
        pos = (np.linspace(0, n * ntheta, (n * ntheta) + 1) % ntheta).astype(int)
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
        Q_scaled = self.convert_spm_data(heat_source) / (cp * rho)
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
        temp = np.zeros(Nunit)
        for i in range(Nunit):
            cell = self.net["pore.cell_id"] == i
            temp[i] = np.mean(self.phase["pore.temperature"][cell])
        return temp


# %% FINISH OPENPNM
# %% START PYBAMM
# %%


class spm_runner:
    def __init__(self):
        pass

    def setup(self, I_app, T0, cc_cond_neg, cc_cond_pos, z_edges):
        # set logging level
#        pybamm.set_logging_level("INFO")
        # load (1+1D) SPM model
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "set external temperature",
        }
        self.model = pybamm.lithium_ion.SPM(options)
        # create geometry
        self.geometry = self.model.default_geometry
        # load parameter values and process model and geometry
        self.param = self.model.default_parameter_values
        self.param.update(
            {
                "Typical current [A]": I_app,
                "Initial temperature [K]": T0,
                "Negative current collector conductivity [S.m-1]": cc_cond_neg,
                "Positive current collector conductivity [S.m-1]": cc_cond_pos,
                "Electrode height [m]": z_edges[-1],
                "Electrode width [m]": length_3d,
                "Negative electrode thickness [m]": 1.5e-4,
                "Negative tab centre z-coordinate [m]": z_edges[0],
                "Positive tab centre z-coordinate [m]": z_edges[-1],
                "Positive electrode conductivity [S.m-1]": 0.1,
                "Negative electrode conductivity [S.m-1]": 0.1,
            }
        )
        self.param.process_model(self.model)
        self.param.process_geometry(self.geometry)
        # set mesh
        self.var = pybamm.standard_spatial_vars
        self.var_pts = {
            self.var.x_n: 5,
            self.var.x_s: 5,
            self.var.x_p: 5,
            self.var.r_n: 5,
            self.var.r_p: 5,
            self.var.z: Nunit,
        }
        # depending on number of points in z direction
        # may need to increase recursion depth...
        sys.setrecursionlimit(10000)
        submesh_types = self.model.default_submesh_types
        pts = z_edges / z_edges[-1]
        submesh_types["current collector"] = pybamm.MeshGenerator(
            pybamm.UserSupplied1DSubMesh, submesh_params={"edges": pts}
        )
        self.mesh = pybamm.Mesh(self.geometry, submesh_types, self.var_pts)
        # discretise model
        self.disc = pybamm.Discretisation(self.mesh, self.model.default_spatial_methods)
        self.disc.process_model(self.model)
        # set up solver
        self.model.convert_to_format = (
            "casadi"
        )  # Use casadi for fast jacobian calculation
        self.solver = pybamm.IDAKLUSolver(atol=1e-8, rtol=1e-8)
        # self.solver = self.model.default_solver
        self.last_time = 0.0
        self.solution = None

    def convert_time(self, non_dim_time, to="seconds"):
        s_parms = pybamm.standard_parameters_lithium_ion
        t_sec = self.param.process_symbol(s_parms.tau_discharge).evaluate()
        t = non_dim_time * t_sec
        if to == "hours":
            t *= 1 / 3600
        return t

    def update_statevector(self, variables, statevector):
        "takes in a dict of variable name and vector of updated state"
        for name, new_vector in variables.items():
            var_slice = self.model.variables[name].y_slices
            statevector[var_slice] = new_vector
        return statevector

    def non_dim_potential(self, phi_dim, domain):
        # Define a method which takes a dimensional potential [V] and converts
        # to the dimensionless potential used in pybamm
        pot_scale = self.param.process_symbol(
            pybamm.standard_parameters_lithium_ion.potential_scale
        ).evaluate()  # potential scaled on thermal voltage
        pot_ref = self.param.process_symbol(
            pybamm.standard_parameters_lithium_ion.U_p_ref
            - pybamm.standard_parameters_lithium_ion.U_n_ref
        ).evaluate()  # positive potential measured with respect to reference OCV
        if domain == "negative":
            phi = phi_dim / pot_scale
        elif domain == "positive":
            phi = (phi_dim - pot_ref) / pot_scale
        return phi

    def non_dim_temperature(self, temperature):
        temp_parms = self.model.submodels["thermal"].param
        Delta_T = self.param.process_symbol(temp_parms.Delta_T).evaluate()
        T_ref = self.param.process_symbol(temp_parms.T_ref).evaluate()
        return (temperature - T_ref) / Delta_T

    def run_step(self, time_step, n_subs=20):
        # Step model for one global time interval
        # Note: In order to make the solver converge, we need to compute consistent
        # initial values for the algebraic part of the model. Since the
        # (dummy) equation for the external temperature is an ODE, the imposed
        # change in temperature is unaffected by this process (i.e. the
        # temperature is exactly that provided by the pnm model)
        if self.last_time > 0.0:
            self.solver.y0 = self.solver.calculate_consistent_initial_conditions(
                self.solver.rhs, self.solver.algebraic, self.current_state
            )
        current_solution = self.solver.step(self.model, time_step, npts=n_subs)
        if self.solution is None:
            self.solution = current_solution
        else:
            self.solution.append(current_solution)
        # Save Current State and update external variables from
        # global calculation
        self.current_state = self.solution.y[:, -1]
        #        self.plot()
        self.last_time += time_step

    def update_external_potential(self, phi_neg, phi_pos):
        sf_cn = 1.0
        vname = "Negative current collector potential"
        phi_s_cn_dim_new = (
            self.current_state[self.model.variables[vname].y_slices] * sf_cn
        )
        sf_cp = 5e-2
        adj = sf_cp * np.linspace(0, 1, self.var_pts[self.var.z])
        vname = "Positive current collector potential"
        phi_s_cp_dim_new = (
            self.current_state[self.model.variables[vname].y_slices] - adj
        )
        #        phi_s_cn_dim_new = self.non_dim_potential(phi_neg, domain='negative')
        #        phi_s_cp_dim_new = self.non_dim_potential(phi_pos, domain='positive')
        variables = {
            "Negative current collector potential": phi_s_cn_dim_new,
            "Positive current collector potential": phi_s_cp_dim_new,
        }
        new_state = self.update_statevector(variables, self.current_state)
        self.current_state = new_state

    def update_external_temperature(self, temperature):

        non_dim_t_external = self.non_dim_temperature(temperature)
        # Note: All of the variables "X-averaged ... temperature" point to the
        # same y_slice of the statevector, so only need to update one.
        variables = {"X-averaged cell temperature": non_dim_t_external}
        new_state = self.update_statevector(variables, self.current_state)
        self.current_state = new_state

    def plot(self, concatenate=True):
        # Plotting
        z = np.linspace(0, 1, Nunit)
        sol = self.solution
        pvs = {
            "X-averaged reversible heating [W.m-3]": None,
            "X-averaged irreversible electrochemical heating [W.m-3]": None,
            "X-averaged Ohmic heating [W.m-3]": None,
            "X-averaged total heating [W.m-3]": None,
            "Current collector current density [A.m-2]": None,
            "X-averaged positive particle surface concentration [mol.m-3]": None,
            "X-averaged negative particle surface concentration [mol.m-3]": None,
            #               "X-averaged positive particle surface concentration": None,
            #               "X-averaged negative particle surface concentration": None,
            "X-averaged cell temperature [K]": None,
            "Negative current collector potential [V]": None,
            "Positive current collector potential [V]": None,
        }
        for key in pvs.keys():
            proc = pybamm.ProcessedVariable(
                self.model.variables[key], sol.t, sol.y, mesh=self.mesh
            )
            pvs[key] = proc
        hrs = self.convert_time(sol.t, to="hours")
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
            #            axcb = fig.colorbar(line_segments)
            #            axcb.set_label('Normalized Arc Position')
            plt.xlabel("t [hrs]")
            plt.ylabel(key)
            plt.xlim(hrs.min(), hrs.max())
            plt.ylim(data.min(), data.max())
            #            plt.ticklabel_format(axis='y', style='sci')
            plt.show()

    def get_processed_variable(self, var, time_index=None):
        z = np.linspace(0, 1, Nunit)
        proc = pybamm.ProcessedVariable(
            self.model.variables[var], self.solution.t, self.solution.y, mesh=self.mesh
        )
        data = proc(self.solution.t, z=z)
        if time_index is None:
            return data
        else:
            return data[:, time_index]

    def get_heat_source(self):
        var = "X-averaged total heating [W.m-3]"
        return self.get_processed_variable(var, time_index=-1)

    def get_potentials(self):
        z = np.linspace(0, 1, Nunit)
        potential_vars = [
            "Negative current collector potential [V]",
            "Positive current collector potential [V]",
        ]
        out = []
        for var in potential_vars:
            proc = pybamm.ProcessedVariable(
                self.model.variables[var],
                self.solution.t,
                self.solution.y,
                mesh=self.mesh,
            )
            data = proc(self.solution.t, z=z)
            out.append(data[:, -1])
        return out

    def get_cell_volumes(self):
        cc_lens = self.mesh["current collector"][0].d_edges
        cc_lens *= self.param["Electrode height [m]"]
        len_3d = self.param["Electrode width [m]"]
        l_x_p = self.param["Positive electrode thickness [m]"]
        l_x_n = self.param["Negative electrode thickness [m]"]
        l_x_s = self.param["Separator thickness [m]"]
        l_x_ccp = self.param["Positive current collector thickness [m]"]
        l_x_ccn = self.param["Negative current collector thickness [m]"]
        l_x = l_x_p + l_x_n + l_x_s + l_x_ccp + l_x_ccn
        vols = cc_lens * l_x * len_3d
        return vols

    def test_equivalent_capacity(self):
        tot_cap = 0.0
        for I_app in [-1.0, 1.0]:
            I_app *= I_app_mag
            model = pybamm.lithium_ion.SPM()
            geometry = model.default_geometry
            param = model.default_parameter_values
            save_ps = [
                "Negative tab centre z-coordinate [m]",
                "Positive tab centre z-coordinate [m]",
            ]
            save_dict = {k: param[k] for k in save_ps}
            param.update(self.param)
            param.update(save_dict)
            param["Typical current [A]"] = I_app
            param.process_model(model)
            param.process_geometry(geometry)
            s_var = pybamm.standard_spatial_vars
            var_pts = {
                s_var.x_n: 5,
                s_var.x_s: 5,
                s_var.x_p: 5,
                s_var.r_n: 5,
                s_var.r_p: 10,
            }
            # set mesh
            mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
            # discretise model
            disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
            disc.process_model(model)
            # solve model
            t_eval = np.linspace(0, 1.0, 101)
            solver = model.default_solver
            sol = solver.solve(model, t_eval)
            time = pybamm.ProcessedVariable(
                model.variables["Time [h]"], sol.t, sol.y, mesh=mesh
            )
            var = "X-averaged positive particle surface concentration [mol.m-3]"
            xpsurf = pybamm.ProcessedVariable(
                model.variables[var], sol.t, sol.y, mesh=mesh
            )
            time_hours = time(sol.t)
            dc_time = np.around(time_hours[-1], 3)
            # Capacity mAh
            tot_cap += np.absolute(I_app * 1000 * dc_time)
            plt.figure()
            plt.plot(time(sol.t), xpsurf(sol.t))
        vol_a = np.sum(self.get_cell_volumes())
        l_y = param["Electrode width [m]"]
        l_z = param["Electrode height [m]"]
        l_x_p = param["Positive electrode thickness [m]"]
        l_x_n = param["Negative electrode thickness [m]"]
        l_x_s = param["Separator thickness [m]"]
        l_x_ccp = param["Positive current collector thickness [m]"]
        l_x_ccn = param["Negative current collector thickness [m]"]
        l_x = l_x_p + l_x_n + l_x_s + l_x_ccp + l_x_ccn
        vol_b = l_x * l_y * l_z
        print("vols", vol_a, vol_b)
        print("Total Capacity", tot_cap, "mAh")
        print("Total Volume", (vol_b * 1e6), "cm3")
        print("Specific Capacity", tot_cap / (vol_b * 1e6), "mAh.cm-3")

    def plot_3d(var="Current collector current density [A.m-2]"):
        data = spm.get_processed_variable(var)
        X, Y = np.meshgrid(
            np.arange(0, data.shape[1], 1), np.arange(0, data.shape[0], 1)
        )
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        surf = ax.plot_surface(
            X, Y, data, cmap=cm.viridis, linewidth=0, antialiased=False
        )
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel("$Time$", rotation=150)
        ax.set_ylabel("$Arc Position$")
        ax.set_zlabel(var, rotation=60)
        plt.show()


# %% Main Loop
# def main():
r"""
Call OpenPNM to set up a global domain comprised of unit cells that have
electrochemistry calculated by PyBAMM.
Heat is generated in the unit cells which act as little batteries connected
in parallel.
The global heat equation is solved on the larger domain as well as the
global potentials which are used as boundary conditions for PyBAMM and a
lumped unit cell temperature is applied and updated over time
"""
start_time = time.time()
plt.close("all")
pnm = pnm_runner()
pnm.setup()
do_just_heat = False
if do_just_heat:
    C_rate = 2.0
    heat_source = np.ones(Nunit) * 25e3 * C_rate
    time_step = 0.01
    pnm.run_step(heat_source, time_step, BC_value=T0)
    pnm.plot_temperature_profile()
else:
    spm = spm_runner()
    spm.setup(
        I_app=I_app_mag, T0=T0, cc_cond_neg=3e7, cc_cond_pos=3e7, z_edges=pnm.arc_edges
    )
    spm.test_equivalent_capacity()
    full = True
    if full:
        t_final = 0.01  # non-dim
        n_steps = 2
        time_step = t_final / n_steps
        jelly_potentials = []

        # Initialize - Run through loop to get temperature then discard solution with small
        #              time step
        print("*" * 30)
        print("Initializing")
        print("*" * 30)
        spm.run_step(time_step / 1000)
        heat_source = spm.get_heat_source()
        print("Heat Source", np.mean(heat_source))
        pnm.run_step(heat_source, time_step, BC_value=T0)
        global_temperature = pnm.get_average_temperature()
        #    global_temperature = np.ones_like(global_temperature)*303.5
        print("Global Temperature", np.mean(global_temperature))
        T_diff = global_temperature.max() - global_temperature.min()
        print("Temperature Range", T_diff)
        spm.update_external_temperature(global_temperature)
        spm.solution = None
        print("*" * 30)
        print("Running Steps")
        print("*" * 30)
        for i in range(n_steps):
            spm.run_step(time_step)
            heat_source = spm.get_heat_source()
            print("Heat Source", np.mean(heat_source))
            pnm.run_step(heat_source, time_step, BC_value=T0)
            global_temperature = pnm.get_average_temperature()
            print("Global Temperature", np.mean(global_temperature))
            T_diff = global_temperature.max() - global_temperature.min()
            print("Temperature Range", T_diff)
            spm.update_external_temperature(global_temperature)
            jelly_potentials.append(spm.get_potentials()[-1])

        spm.plot()
        # spm.quick_plot()
        plt.figure()
        for i in range(len(jelly_potentials)):
            plt.plot(jelly_potentials[i])
        plt.figure()
        plt.plot(jelly_potentials[0])
        end_time = time.time()
        print("*" * 30)
        print("Simulation Time", np.around(end_time - start_time, 2), "s")
        print("*" * 30)
        vars = [
            "X-averaged total heating [W.m-3]",
            "X-averaged cell temperature [K]",
            "X-averaged positive particle surface concentration [mol.m-3]",
            "X-averaged negative particle surface concentration [mol.m-3]",
            "Negative current collector potential [V]",
            "Positive current collector potential [V]",
            "Local current collector potential difference [V]",
        ]

        tind = -1
        for var in vars:
            data = pnm.convert_spm_data(
                spm.get_processed_variable(var, time_index=tind)
            )
            pnm.plot_pore_data(data, title=var + " @ time " + str(spm.solution.t[tind]))

        def plot_time_series(var):
            all_time_data = spm.get_processed_variable(var, time_index=None)
            data = pnm.convert_spm_data(all_time_data[:, 0])
            fig, ax = plt.subplots(1)
            fig.subplots_adjust(bottom=0.25)
            bulk_Ps = pnm.net.pores("free_stream", mode="not")
            coords = pnm.net["pore.coords"][bulk_Ps]
            xmin = coords[:, 0].min() * 1.05
            ymin = coords[:, 1].min() * 1.05
            xmax = coords[:, 0].max() * 1.05
            ymax = coords[:, 1].max() * 1.05
            mappable = ax.scatter(coords[:, 0], coords[:, 1], c=data[bulk_Ps])
            mappable.set_clim([all_time_data.min(), all_time_data.max()])
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.colorbar(mappable)
            plt.title(var)
            ax1_pos = fig.add_axes([0.2, 0.1, 0.65, 0.03])
            #    s1 = Slider(ax1_pos, 'time', valmin=0, valmax=len(spm.solution.t),
            #                valinit=0, valfmt='%i')
            s1 = Slider(ax1_pos, "Time", 0, spm.solution.t.max(), valinit=0)

            def update1(v):
                tind = np.argwhere(spm.solution.t > s1.val).min()
                data = pnm.convert_spm_data(all_time_data[:, tind])
                mappable.set_array(data[bulk_Ps])
                fig.canvas.draw_idle()

            s1.on_changed(update1)
            plt.show()

        # plot_time_series(var="X-averaged positive particle surface concentration [mol.m-3]")

        # var = "X-averaged cell temperature [K]"
        var = vars[2]
        tind = -1
        data = pnm.convert_spm_data(spm.get_processed_variable(var, time_index=tind))
        pnm.plot_pore_data(data, title=var + " @ time " + str(spm.solution.t[tind]))
        pnm.plot_temperature_profile()


def specific_cap(diam, height, cap):
    a = np.pi * (diam / 2) ** 2
    v = a * height
    spec = cap / v
    return spec


spm.plot_3d()
print("State of Art 18650", specific_cap(1.8, 6.5, 2500), "mAh.cm-3")
