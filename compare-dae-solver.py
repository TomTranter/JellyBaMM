import pybamm
import numpy as np


# load (1+1D) SPMe model
options = {
    "current collector": "potential pair",
    "dimensionality": 1,
    "thermal": "set external temperature",
}
model = pybamm.lithium_ion.SPM(options)

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values
param.process_model(model)
param.process_geometry(geometry)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.r_n: 5, var.r_p: 5, var.z: 400}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = np.linspace(0, 0.17, 100)

pybamm.set_logging_level("INFO")
names = [None] * 7
solutions = [None] * 7

# CasadiSolver
#names[1] = "Casadi"
#solutions[1] = pybamm.CasadiSolver(atol=1e-8, rtol=1e-8).solve(model, t_eval)
# IDAKLUSolver
#names[6] = "IDAKLU. convert to casadi (default)"
#solutions[6] = pybamm.IDAKLUSolver(atol=1e-8, rtol=1e-8).solve(model, t_eval)
## IDAKLUSolver, convert to python
#names[2] = "IDAKLU. convert to python"
#model.convert_to_format = "python"
#solutions[2] = pybamm.IDAKLUSolver(atol=1e-8, rtol=1e-8).solve(model, t_eval)
## IDAKLUSolver, convert to python no simplify
#names[3] = "IDAKLU. convert to python, no simplify"
#model.convert_to_format = "python"
#model.use_simplify = False
#solutions[3] = pybamm.IDAKLUSolver(atol=1e-8, rtol=1e-8).solve(model, t_eval)
## IDAKLUSolver, no convert
#names[4] = "IDAKLU. convert to None"
#model.convert_to_format = None
#model.use_simplify = True
#solutions[4] = pybamm.IDAKLUSolver(atol=1e-8, rtol=1e-8).solve(model, t_eval)
## IDAKLUSolver, no convert no simplify
#names[5] = "IDAKLU. convert to None, no simplify"
#model.use_simplify = False
#solutions[5] = pybamm.IDAKLUSolver(atol=1e-8, rtol=1e-8).solve(model, t_eval)
# CasadiSolver fast mode
names[0] = "Casadi fast"
solutions[0] = pybamm.CasadiSolver(atol=1e-8, rtol=1e-8, mode="fast").solve(
    model, t_eval
)

for i, solution in enumerate(solutions):
    if solution is not None:
        print(names[i])
        print("set up time")
        print(solution.set_up_time)
        print("solve time")
        print(solution.solve_time)
