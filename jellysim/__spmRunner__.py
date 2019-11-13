#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:11:13 2019

@author: thomas
"""
import pybamm
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib.collections import LineCollection
import numpy as np
from scipy import io


class spm_runner(object):
    def __init__(self):
        pass

    def setup(self, I_app, T0, cc_cond_neg, cc_cond_pos, z_edges, length_3d):
        self.Nunit = len(z_edges)-1
        # set logging level
        pybamm.set_logging_level("INFO")
        # load (1+1D) SPM model
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "set external temperature",
        }
        self.model = pybamm.lithium_ion.SPM(options)
        self.model.use_simplify = False
        # create geometry
        self.geometry = self.model.default_geometry
        # load parameter values and process model and geometry
        self.param = self.model.default_parameter_values
        pixel_size = 10.4e-6
        self.param.update(
            {
                "Typical current [A]": I_app,
                "Initial temperature [K]": T0,
                "Negative current collector conductivity [S.m-1]": cc_cond_neg,
                "Positive current collector conductivity [S.m-1]": cc_cond_pos,
                "Electrode height [m]": z_edges[-1],
                "Electrode width [m]": length_3d,
                "Negative electrode thickness [m]": 6.0*pixel_size,
                "Positive electrode thickness [m]": 9.0*pixel_size,
                "Separator thickness [m]": 1.0*pixel_size,
                "Positive current collector thickness [m]": 1.0*pixel_size,
                "Negative current collector thickness [m]": 1.0*pixel_size,
                "Negative tab centre z-coordinate [m]": z_edges[0],
                "Positive tab centre z-coordinate [m]": z_edges[-1],
                "Positive electrode conductivity [S.m-1]": 0.1,
                "Negative electrode conductivity [S.m-1]": 0.1,
                "Lower voltage cut-off [V]": 3.45,
                "Upper voltage cut-off [V]": 4.7,
            }
        )
#        self.param["Current function"] = pybamm.GetConstantCurrent()
        # set mesh
        self.var = pybamm.standard_spatial_vars
        self.var_pts = {
            self.var.x_n: 5,
            self.var.x_s: 5,
            self.var.x_p: 5,
            self.var.r_n: 5,
            self.var.r_p: 5,
            self.var.z: self.Nunit,
        }
        self.param.process_model(self.model)
        self.param.process_geometry(self.geometry)
        # depending on number of points in z direction
        # may need to increase recursion depth...
        sys.setrecursionlimit(10000)
        submesh_types = self.model.default_submesh_types
        pts = z_edges / z_edges[-1]
        submesh_types["current collector"] = pybamm.MeshGenerator(
            pybamm.UserSupplied1DSubMesh, submesh_params={"edges": pts}
        )
        self.mesh = pybamm.Mesh(self.geometry, submesh_types, self.var_pts)
        # set up solver
#        self.solver = pybamm.IDAKLUSolver(atol=1e-8, root_tol=1e-8)
        self.solver = pybamm.CasadiSolver(atol=1e-8, rtol=1e-8)
#        self.solver = pybamm.KLU()
#        self.solver.atol = 1e-8
#        self.solver.rtol = 1e-8
        self.last_time = 0.0
        self.solution = None
        self.disc = None

    def _setup_discretization(self):
        self.disc = pybamm.Discretisation(self.mesh,
                                          self.model.default_spatial_methods)
        self.disc.process_model(self.model)

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
        # positive potential measured with respect to reference OCV
        pot_ref = self.param.process_symbol(
            pybamm.standard_parameters_lithium_ion.U_p_ref
            - pybamm.standard_parameters_lithium_ion.U_n_ref
        ).evaluate()
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
        if self.disc is None:
            self._setup_discretization()
        # Step model for one global time interval
        # Note: In order to make the solver converge, we need to compute
        # consistent initial values for the algebraic part of the model.
        # Since the (dummy) equation for the external temperature is an ODE
        # the imposed change in temperature is unaffected by this process
        # i.e. the temperature is exactly that provided by the pnm model
        sol = self.solver
        if self.last_time > 0.0:
            sol.y0 = sol.calculate_consistent_initial_conditions(
                    sol.rhs, sol.algebraic, self.current_state
                    )
        current_solution = sol.step(self.model, time_step, npts=n_subs)
        if self.solution is None:
            self.solution = current_solution
        else:
            self.solution.append(current_solution)
        # Save Current State and update external variables from
        # global calculation
        self.current_state = self.solution.y[:, -1]
        #        self.plot()
        self.last_time += time_step
        return current_solution

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
        z = np.linspace(0, 1, self.Nunit)
        sol = self.solution
        pvs = {
            "X-averaged reversible heating [W.m-3]": None,
            "X-averaged irreversible electrochemical heating [W.m-3]": None,
            "X-averaged Ohmic heating [W.m-3]": None,
            "X-averaged total heating [W.m-3]": None,
            "Current collector current density [A.m-2]": None,
            "X-averaged positive particle " +
            "surface concentration [mol.m-3]": None,
            "X-averaged negative particle " +
            "surface concentration [mol.m-3]": None,
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
            for bat_id in range(self.Nunit):
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

    def get_processed_variable(self, var, time_index=None):
        z = np.linspace(0, 1, self.Nunit)
        proc = pybamm.ProcessedVariable(
            self.model.variables[var], self.solution.t,
            self.solution.y, mesh=self.mesh
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
        z = np.linspace(0, 1, self.Nunit)
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
        cc_lens = self.mesh["current collector"][0].d_edges.copy()
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

    def test_equivalent_capacity(self, I_app_mag=1.0):
        tot_cap = 0.0
        sim_time_hrs = 0.0
        for I_app in [-1.0, 1.0]:
            I_app *= I_app_mag
            model = pybamm.lithium_ion.SPM()
            model.use_simplify = False
            geometry = model.default_geometry
            param = model.default_parameter_values
            save_ps = [
                "Negative tab centre z-coordinate [m]",
                "Positive tab centre z-coordinate [m]",
            ]
            save_dict = {k: param[k] for k in save_ps}
            param.update(self.param.copy())
            param.update(save_dict)
            param.update({"Typical current [A]": I_app})
            param["Current function"] = pybamm.GetConstantCurrent()
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
#            solver = pybamm.IDAKLUSolver(atol=1e-8, rtol=1e-8)
            solver = pybamm.CasadiSolver(atol=1e-8, rtol=1e-8)
            sol = solver.solve(model, t_eval)
            time_h = pybamm.ProcessedVariable(
                model.variables["Time [h]"], sol.t, sol.y, mesh=mesh
            )
            var = ("X-averaged positive particle " +
                   "surface concentration [mol.m-3]")
            xpsurf = pybamm.ProcessedVariable(
                model.variables[var], sol.t, sol.y, mesh=mesh
            )
            time_hours = time_h(sol.t)
            dc_time = np.around(time_hours[-1], 5)
#            dc_time = model.variables["Time [h]"].evaluate(t=1.0)
            # Capacity mAh
            tot_cap += np.absolute(I_app * 1000 * dc_time)
            plt.figure()
            plt.plot(time_h(sol.t), xpsurf(sol.t))
            sim_time_hrs += dc_time
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
        print("Total Charge/Discharge Time [hrs]", sim_time_hrs)
        print("Total Capacity", tot_cap, "mAh")
        print("Total Volume", (vol_b * 1e6), "cm3")
        print("Specific Capacity", tot_cap / (vol_b * 1e6), "mAh.cm-3")

    def plot_3d(self, var='Current collector current density [A.m-2]'):
        data = self.get_processed_variable(var)
        X, Y = np.meshgrid(np.arange(0, data.shape[1], 1),
                           np.arange(0, data.shape[0], 1))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, data, cmap=cm.viridis,
                               linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel('$Time$', rotation=150)
        ax.set_ylabel('$Arc Position$')
        ax.set_zlabel(var, rotation=60)
        plt.show()

    def export_3d_mat(self, var='Current collector current density [A.m-2]',
                      fname='data.mat'):
        data = self.get_processed_variable(var)
        if '.mat' not in fname:
            fname = fname + '.mat'
        io.savemat(fname, mdict={'data': data})
