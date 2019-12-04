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
import numpy as np


plt.close("all")
use_tomo = True
wrk = op.Workspace()
# Options
I_app = 1.0
# Simulation options
model_name = 'model_nlayer3_dtheta20'
opt = {'domain': 'model',
       'Nlayers': 3,
       'cp': 1399.0,
       'rho': 2055.0,
       'K0': 1.0,
       'T0': 303,
       'heat_transfer_coefficient': 5,
       'length_3d': 0.065,
       'I_app': I_app,
       'cc_cond_neg': 3e7,
       'cc_cond_pos': 3e7,
       'dtheta': 20,
       'spacing': 1e-5,
       'model_name': model_name}
# Directories
cwd = os.getcwd()
input_dir = os.path.join(cwd, 'input')
parent_dir = os.path.dirname(cwd)
model_dir = os.path.join(parent_dir, 'models')
j_dir = opt['domain']+str(I_app)+'amp'
out_dir = os.path.join(parent_dir, 'pybamm_pnm_data')
out_sub_dir = os.path.join(out_dir, j_dir)
save_path = os.path.join(out_sub_dir, model_name)
# Simulation
sim = js.coupledSim()
sim.setup(opt)
spm = sim['spm']
#spm.sim.save(save_path)
#j_dir = opt['domain']+'_journal_tomo_'+str(I_app)+'amp'
#sim.run_thermal()
#sim.runners['spm'].test_equivalent_capacity()

sim.run(n_steps=2, time_step=0.005, n_subs=5)



#sim.plots()
#sim.save('test')


#spm.plot_3d()

#print('1', sim['spm'].solution.y.shape)

#print('1', sim['spm'].solution.y.shape)
sim.save(save_path)
sim2 = js.coupledSim()
sim2.load(save_path)
#print('1', sim['spm'].solution.y.shape)
#print('2', sim2['spm'].solution.y.shape)
sim.run(n_steps=1, time_step=0.005, n_subs=5)
#print('1', sim['spm'].solution.y.shape)
sim2.run(n_steps=1, time_step=0.005, n_subs=5)
# local sols
print(np.allclose(sim['spm'].sim.solution.y[:, -1], sim2['spm'].sim.solution.y[:, -1]))
# Amalgamated solutons
print(np.allclose(sim['spm'].solution.y[:, -1], sim2['spm'].solution.y[:, -1]))
#print('2', sim2['spm'].solution.y.shape)
#spm.sim.save(save_path+'_post')

#sim2 = pybamm.load_sim(save_path)
#spm.export_3d_mat(var='Current collector current density [A.m-2]',
#                  fname='./'+j_dir+'/current_density.mat')
#var = "X-averaged negative particle surface concentration [mol.m-3]"
#spm.export_3d_mat(var=var,
#                  fname='./'+j_dir+'/neg_particle_conc.mat')
#var = "X-averaged positive particle surface concentration [mol.m-3]"
#spm.export_3d_mat(var=var,
#                  fname='./'+j_dir+'/pos_particle_conc.mat')
#var = "Positive current collector potential [V]"
#spm.export_3d_mat(var=var,
#                  fname='./'+j_dir+'/pos_cc_potential.mat')
#var = "Negative current collector potential [V]"
#spm.export_3d_mat(var=var,
#                  fname='./'+j_dir+'/neg_cc_potential.mat')

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
