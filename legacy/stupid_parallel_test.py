# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:50:45 2020

@author: Tom
"""
import pybamm
from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np
import warnings
warnings.simplefilter("ignore")


def current_function(t):
    return pybamm.InputParameter("Current")

def make_spm(I_typical, thermal=True):
    if thermal:
        model_options = {
                "thermal": "x-lumped",
                "external submodels": ["thermal"],
            }
        model = pybamm.lithium_ion.SPM(model_options)
    else:
        model = pybamm.lithium_ion.SPM()
    geometry = model.default_geometry
    param = model.default_parameter_values
    param.update(
        {
            "Typical current [A]": I_typical,
            "Current function [A]": current_function,
            "Electrode height [m]": "[input]",
        }
    )
    param.update({"Current": "[input]"}, check_already_exists=False)
    param.process_model(model)
    param.process_geometry(geometry)
    var = pybamm.standard_spatial_vars
    var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.r_n: 10, var.r_p: 10}
    spatial_methods = model.default_spatial_methods
    solver = pybamm.CasadiSolver()
#    solver = model.default_solver
    sim = pybamm.Simulation(
        model=model,
        geometry=geometry,
        parameter_values=param,
        var_pts=var_pts,
        spatial_methods=spatial_methods,
        solver=solver,
    )
    sim.build(False)
    return sim

def convert_temperature(sim, T_dim, inputs):
    temp_parms = sim.model.submodels["thermal"].param
    param = sim.parameter_values
    Delta_T = param.process_symbol(temp_parms.Delta_T).evaluate(u=inputs)
    T_ref = sim.parameter_values.process_symbol(temp_parms.T_ref).evaluate(u=inputs)
    return (T_dim - T_ref) / Delta_T


def step_spm(zipped):
    built_model, solver, param, solution, I_app, e_height, dt, T_av, dead = zipped
    inputs = {"Current": I_app,
              'Electrode height [m]': e_height}
#    T_av_non_dim = convert_temperature(sim, T_av, inputs)
    T_av_non_dim = 0.0
    if len(built_model.external_variables) > 0:
        external_variables = {"X-averaged cell temperature": T_av_non_dim}
    else:
        external_variables = None
    if ~dead:
#        print(inputs)
        if solution is not None:
            solved_len = solver.y0.shape[0]
            solver.y0 = solution.y[:solved_len, -1]
            solver.t = solution.t[-1]
        solution = solver.step(old_solution=solution,
            model=built_model, dt=dt, external_variables=external_variables, inputs=inputs
        )

    return solution

def pool_func(inputs):
    model, solver, param, dt = inputs
    solution = step_spm((model, solver, param, None,
                         1.0, 1.0, dt, 303, False))
    return solution

def main():
    pass_param = True
    Nspm = 10
    max_workers = int(os.cpu_count() / 2)
    pool = ProcessPoolExecutor(max_workers=max_workers)
    sim = make_spm(I_typical=1.0, thermal=False)
    models = [sim.built_model for i in range(Nspm)]
    solvers = [sim.solver for i in range(Nspm)]
    if pass_param:
        params = [sim.parameter_values for i in range(Nspm)]
    else:
        params = [None for i in range(Nspm)]
    time_steps = np.linspace(0.1, 1, Nspm)*1e-6
    solutions = list(pool.map(pool_func, zip(models, solvers, params, time_steps)))
#    solutions = []
#    for args in zip(models, solvers, params, time_steps):
#        solutions.append(pool_func(args))
    print([sol.t[-1] for sol in solutions])
    pool.shutdown()
    del pool
    
if __name__ == '__main__':
    main()