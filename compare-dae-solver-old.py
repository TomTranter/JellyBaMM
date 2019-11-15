import pybamm
import numpy as np


# load (1+1D) SPMe model
options = {
    "current collector": "potential pair",
    "dimensionality": 1,
    "thermal": "lumped",
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
var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.r_n: 5, var.r_p: 5, var.z: 100}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = np.linspace(0, 0.17, 100)

pybamm.set_logging_level("INFO")
names = [None] * 3
solutions = [None] * 3

# IDAKLUSolver, to python (default on old klu branch)
names[0] = "KLU. convert to python"
solutions[0] = pybamm.KLU(atol=1e-8, rtol=1e-8).solve(model, t_eval)
# IDAKLUSolver, no convert
names[1] = "KLU. convert to None"
model.convert_to_format = None
model.use_simplify = True
solutions[1] = pybamm.KLU(atol=1e-8, rtol=1e-8).solve(model, t_eval)
# IDAKLUSolver, no convert no simplify
names[2] = "KLU. convert to None, no simplify"
model.use_simplify = False
solutions[2] = pybamm.KLU(atol=1e-8, rtol=1e-8).solve(model, t_eval)


for i, solution in enumerate(solutions):
    print(names[i])
    print("set up time")
    print(solution.set_up_time)
    print("solve time")
    print(solution.solve_time)
