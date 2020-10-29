import pybamm
import numpy as np
import matplotlib.pyplot as plt

plt.close("all")
# set logging level
pybamm.set_logging_level("INFO")
# def spm_init(height=None, width=None, I_app=None):
# create the model
model = pybamm.lithium_ion.SPM()
# set the default model geometry
geometry = model.default_geometry
# set the default model parameters
param = model.default_parameter_values
# change the typical current and set a constant discharge using the typical current value
def current_function(t):
    return pybamm.InputParameter("Current")


param.update({"Current function": current_function, "Current": "[input]"})
# set the parameters for the model and the geometry
param.process_model(model)
param.process_geometry(geometry)
# mesh the domains
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
# discretise the model equations
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)
# Solve the model at the given time points
I_app = 1.0
solver = pybamm.CasadiSolver()
dt = 1e-4


def step_spm(model, solver, dt, I_app):
    step_sol = solver.step(model=model, dt=dt, inputs={"Current": I_app})
    return step_sol


solution = None
current_state = np.array([])
key = "Current collector current density [A.m-2]"
for i in range(3):
    current = I_app - i * 0.1
    step_sol = step_spm(model, solver, dt, current)
    step_sol.t += i * dt
    if solution is None:
        solution = step_sol
    else:
        solution.append(step_sol)
    proc = pybamm.ProcessedVariable(
        model.variables[key],
        solution.t,
        solution.y,
        mesh=mesh,
        inputs={"Current": current},
    )
    current_state = np.concatenate([current_state, proc(step_sol.t[-1:])])

plt.figure()
plt.plot(solution.t[1:], current_state, "*")
plt.show()