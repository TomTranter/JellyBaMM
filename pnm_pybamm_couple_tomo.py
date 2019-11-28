#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Call OpenPNM to set up a global domain comprised of unit cells that have
electrochemistry calculated by PyBAMM.
Heat is generated in the unit cells which act as little batteries connected
in parallel.
The global heat equation is solved on the larger domain as well as the
global potentials which are used as boundary conditions for PyBAMM and a
lumped unit cell temperature is applied and updated over time
"""

import pybamm
import openpnm as op
import matplotlib.pyplot as plt
import os
import jellysim as js


plt.close("all")
use_tomo = True
wrk = op.Workspace()
input_dir = os.path.join(os.getcwd(), 'input')
#pybamm.set_logging_level(10)
I_app = 0.5
# Simulation options
opt = {'domain': 'model',
       'Nlayers': 3,
       'cp': 1399.0,
       'rho': 2055.0,
       'K0': 1.0,
       'T0': 303,
       'heat_transfer_coefficient': 10,
       'length_3d': 0.065,
       'I_app_mag': I_app*1.0,
       'cc_cond_neg': 3e7,
       'cc_cond_pos': 3e7,
       'dtheta': 10,
       'spacing': 1e-5}

sim = js.coupledSim()
sim.setup(opt)
j_dir = opt['domain']+'_journal_tomo_'+str(I_app)+'amp'
#sim.run_thermal()
#sim.runners['spm'].test_equivalent_capacity()
sim.run(n_steps=5, time_step=0.005, n_subs=5, journal=j_dir)
sim.plots()
#sim.save('test')
spm = sim.runners['spm']
spm.export_3d_mat(var='Current collector current density [A.m-2]',
                  fname='./'+j_dir+'/current_density.mat')
var = "X-averaged negative particle surface concentration [mol.m-3]"
spm.export_3d_mat(var=var,
                  fname='./'+j_dir+'/neg_particle_conc.mat')
var = "X-averaged positive particle surface concentration [mol.m-3]"
spm.export_3d_mat(var=var,
                  fname='./'+j_dir+'/pos_particle_conc.mat')
var = "Positive current collector potential [V]"
spm.export_3d_mat(var=var,
                  fname='./'+j_dir+'/pos_cc_potential.mat')
var = "Negative current collector potential [V]"
spm.export_3d_mat(var=var,
                  fname='./'+j_dir+'/neg_cc_potential.mat')

#post = pybamm.post_process_variables(variables=spm.model.variables,
#                              t_sol=spm.solution.t,
#                              u_sol=spm.solution.y,
#                              mesh=spm.mesh)
#js.save_obj('test_save_spm', spm)  # N Can't pickle local object 'primitive.<locals>.f_wrapped'
#js.save_obj('test_save_spm_param', spm.param)  # N Can't pickle local object 'primitive.<locals>.f_wrapped'
#js.save_obj('test_save_spm_model', spm.model)  # N Can't pickle local object 'primitive.<locals>.f_wrapped'
#js.save_obj('test_save_spm_geometry', spm.geometry)  # Y
#js.save_obj('test_save_spm_var', spm.var)  # N can't pickle module objects
#js.save_obj('test_save_spm_mesh', spm.mesh)  # Y
#js.save_obj('test_save_spm_disc', spm.disc)  # N Can't pickle local object 'primitive.<locals>.f_wrapped'
#js.save_obj('test_save_spm_solver', spm.solver)  # N Can't pickle local object 'DaeSolver.set_up.<locals>.rhs'
#js.save_obj('test_save_spm_solution', spm.solution)  # Y
