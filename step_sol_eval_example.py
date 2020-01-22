#
# Example to compare solving for all times against stepping individually
#
import pybamm
from pybamm import EvaluatorPython as ep
import numpy as np
import matplotlib.pyplot as plt
import time as record_time

pybamm.set_logging_level("INFO")

# load model
model = pybamm.lithium_ion.SPMe()

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values
param.process_model(model)
param.process_geometry(geometry)

# set mesh
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
#t_eval = np.linspace(0, 0.2, 100)

var = "Terminal voltage [V]"
# step model

time = 0

N_SPM = 20
N_steps = 10
dt = 0.1/N_steps
func = ep(model.variables[var])

solutions = [None]*N_SPM
solvers = [pybamm.CasadiSolver() for i in range(N_SPM)]

results_a = np.zeros([N_steps, N_SPM])
results_b = np.zeros([N_steps, N_SPM])
results_c = np.zeros([N_steps, N_SPM])
times = []

eval_time_a = 0.0
eval_time_b = 0.0
eval_time_c = 0.0
solve_time = 0.0

for t in range(N_steps):
    for i in range(N_SPM):
        # Solve
        st = record_time.time()
        if not solutions[i]:
            # create solution object on first step
            solutions[i] = solvers[i].step(model, dt=dt, npts=10)
        else:
            # append solution from the current step to step_solution
            solved_len = solvers[i].y0.shape[0]
            solvers[i].y0 = solutions[i].y[:solved_len, -1]
            solvers[i].t = solutions[i].t[-1]
            solutions[i].append(solvers[i].step(model, dt=dt, npts=10))
        solve_time += record_time.time() - st
        # Eval in place
        st = record_time.time()
        results_a[t, i] = func.evaluate(solutions[i].t[-1],
                                        solutions[i].y[:, -1])
        eval_time_a += record_time.time() - st
        # From solution in place
        st = record_time.time()
        results_c[t, i] = solutions[i][var](solutions[i].t[-1])
        eval_time_c += record_time.time() - st
    # Eval at end
    st = record_time.time()
    temp_t = np.asarray([solutions[i].t[-1] for i in range(N_SPM)])
    temp_y = np.asarray([solutions[i].y[:, -1] for i in range(N_SPM)]).T
    results_b[t, :] = func.evaluate(temp_t, temp_y)
    eval_time_b += record_time.time() - st
    time += dt
    times.append(time)
times = np.asarray(times)

step_voltage_a = solutions[0]["Terminal voltage [V]"]
step_voltage_b = solutions[1]["Terminal voltage [V]"]
print(np.allclose(step_voltage_a(times),
                  step_voltage_b(times)))
print(np.allclose(step_voltage_a(times),
                  results_a[:, 0]))
print(np.allclose(results_a, results_b))
print(np.allclose(results_b, results_c))
print('Solve time', solve_time)
print('Eval time individual', eval_time_a)
print('Eval time concatenate', eval_time_b)
print('Solution eval time', eval_time_c)
