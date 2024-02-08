#
# Example running a 1D parallel string
#

import openpnm as op
import matplotlib.pyplot as plt
import jellybamm
import liionpack as lp
import pybamm


plt.close("all")

# pybamm.set_logging_level("INFO")
wrk = op.Workspace()
wrk.clear()


if __name__ == "__main__":
    # Geometry of 1D mesh
    Nunit = 10
    spacing = 0.1
    pos_tabs = [-1]
    neg_tabs = [0]
    length_3d = 1.0

    # Experiment
    I_app = 1.0
    dt = 30
    Nsteps = 12
    hours = dt * Nsteps / 3600
    experiment = pybamm.Experiment(
        [
            f"Discharge at {I_app} A for {hours} hours",
        ],
        period=f"{dt} seconds",
    )

    # OpenPNM project
    project, arc_edges = jellybamm.make_1D_net(Nunit, spacing, pos_tabs, neg_tabs)

    # Parameter set
    param = pybamm.ParameterValues("Chen2020")
    # JellyBaMM discretises the spiral using the electrode height for spiral length
    # This parameter set has the longer length set to the Electrode width
    # We want to swap this round
    param["Electrode width [m]"] = length_3d
    initial_soc = None

    # Run simulation
    project, output = jellybamm.run_simulation_lp(
        parameter_values=param,
        experiment=experiment,
        initial_soc=initial_soc,
        project=project,
    )
    lp.plot_output(output)
