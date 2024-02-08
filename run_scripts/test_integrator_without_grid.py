#
# Test electrode height input parameter
#

import pybamm
import numpy as np

model = pybamm.lithium_ion.SPM()
# Cycling experiment, using PyBaMM
# experiment = pybamm.Experiment([
#     "Discharge at 0.5 A for 10 minutes",
#     ],
#     period="10 seconds")

# PyBaMM battery parameters
chemistry = pybamm.parameter_sets.Chen2020
param = pybamm.ParameterValues(chemistry=chemistry)

# Make the electrode width an input parameter and use arc edges
param.update({'Electrode height [m]': "[input]",
               'Electrode width [m]': 0.065,
               'Current function [A]': "[input]"}, check_already_exists=False)
e_heights = np.array([5e-4, 1.0, 2.0])
solver = pybamm.CasadiSolver(mode='safe without grid')
sim = pybamm.Simulation(model=model,
                        parameter_values=param,
                        # experiment=experiment,
                        solver=solver)
solution = sim.solve(t_eval=np.linspace(0, 100, 11), inputs={'Electrode height [m]': e_heights[0],
                                                             'Current function [A]': e_heights[0]})
# solution = sim.solve()
# sim.plot()


ocp = solution['Measured open circuit voltage [V]'].entries

sym_tau = pybamm.LithiumIonParameters().tau_discharge
tau_spm = []
for i in range(len(e_heights)):
    temp_tau = param.process_symbol(sym_tau)
    tau_input = {"Electrode height [m]": e_heights[i]}
    tau_spm.append(temp_tau.evaluate(inputs=tau_input))
tau_spm = np.asarray(tau_spm)
print(tau_spm)