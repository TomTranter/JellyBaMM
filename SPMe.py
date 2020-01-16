#
# Example showing how to load and solve the SPMe
#

import pybamm
import numpy as np
import matplotlib.pyplot as plt
#plt.close('all')


pybamm.set_logging_level("INFO")

def current_function(t):
    return pybamm.InputParameter("Current")

# load model
model = pybamm.lithium_ion.SPMe()
I_typical = 1.0

# load parameter values and process model and geometry
param = model.default_parameter_values
param.update(
    {
        "Typical current [A]": I_typical,
        "Current function": current_function,
        "Current function [A]": I_typical,
        "Current": "[input]",
        "Lower voltage cut-off [V]": 3.5,
    }
)


sim = pybamm.Simulation(
    model=model,
    parameter_values=param,

)
# solve model
t_eval = np.linspace(0, 1.0, 1000)
#solver = pybamm.CasadiSolver()
solver = model.default_solver

overpotentials = {
    "X-averaged reaction overpotential [V]": [],
    "X-averaged concentration overpotential [V]": [],
    "X-averaged electrolyte ohmic losses [V]": [],
    "X-averaged solid phase ohmic losses [V]": [],
    "X-averaged battery open circuit voltage [V]": [],
}
#plot = pybamm.QuickPlot(sim.built_model, sim.mesh, sim.solution,
#                        output_variables=overpotentials)
#plot.dynamic_plot()
tot_R = []
time = []

def calc_R(sim, current):
    initial_ocv = 3.8518206633137266
    totdV = initial_ocv
    t = evaluate(sim, 'Time [h]', current)
    for key in overpotentials.keys():
        eta = evaluate(sim, key, current)[0][0]
        overpotentials[key].append(eta)
#        print(key, eta)
        totdV -= eta
    R = totdV / current
    tot_R.append(R)
    time.append(t)

def evaluate(sim, var="Current collector current density [A.m-2]", current=0.0):
    model = sim.built_model
    solution = sim.solution
    value = model.variables[var].evaluate(
        solution.t[-1], solution.y[:, -1], u={"Current": current}
    )
    return value

dt = 1e-2
i = 0
terminated = False
while i < 16:
    try:
        sim.step(dt=dt, inputs={"Current": I_typical}, save=True)
        calc_R(sim, I_typical)
    except:
        pass
    if sim.solution.termination is not 'final time':
        terminated = True
    i += 1
# plot
plt.figure()
plt.plot(time, tot_R)

sim.solver.get_termination_reason(sim.solution, sim.solver.events)

