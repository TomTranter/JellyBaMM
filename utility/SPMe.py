#
# Example showing how to load and solve the SPMe
#

import pybamm
import numpy as np
import matplotlib.pyplot as plt


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
        # "Typical current [A]": I_typical,
        "Current function": current_function,
        "Current function [A]": I_typical,
        "Current": "[input]",
        "Lower voltage cut-off [V]": 3.5,
    }, check_already_exists=False
)

solver = pybamm.CasadiSolver()
sim = pybamm.Simulation(
    model=model,
    parameter_values=param,
    solver=solver

)
dt = 100
t_eval = np.arange(0, 3610, dt)

overpotentials = {
    "X-averaged reaction overpotential [V]": [],
    "X-averaged concentration overpotential [V]": [],
    "X-averaged electrolyte ohmic losses [V]": [],
    "X-averaged solid phase ohmic losses [V]": [],
    "Change in measured open circuit voltage [V]": [],
}
tot_R = []
time = []


def calc_R(sim, current):
    # initial_ocv = 3.8518206633137266
    totdV = 0.0
    t = evaluate(sim, 'Time [h]', current)
    for key in overpotentials.keys():
        eta = evaluate(sim, key, current)[0][0]
        overpotentials[key].append(eta)
        totdV -= eta
    R = totdV / current
    tot_R.append(R)
    time.append(t)


def evaluate(sim, var="Current collector current density [A.m-2]", current=0.0):
    model = sim.built_model
    solution = sim.solution
    value = model.variables[var].evaluate(
        solution.t[-1], solution.y[:, -1], inputs={"Current": current}
    )
    return value


terminated = False
for i in range(len(t_eval)):
    sim.step(dt=dt, inputs={"Current": I_typical}, save=True)
    calc_R(sim, I_typical)
    if sim.solution.termination != 'final time':
        terminated = True
# plot
plt.figure()
plt.plot(time, tot_R)
