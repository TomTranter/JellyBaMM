<p align="center">
  <a href="https://github.com/TomTranter/JellyBaMM/actions/workflows/ci.yml"><img src="https://github.com/TomTranter/JellyBaMM/actions/workflows/ci.yml/badge.svg?branch=main" alt="Scheduled"></a>
  <a href="https://jellybamm.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/jellybamm/badge/?version=latest" alt="Documentation Status"></a>
  <a href="https://codecov.io/gh/TomTranter/JellyBaMM" > 
 <img src="https://codecov.io/gh/TomTranter/JellyBaMM/graph/badge.svg?token=U8IN5ZME8E"/> 
 </a>
</p>

<p align="center">
<img src="docs\logo.jpg" width="500" height="500" alt="Logo">
</p>

# Welcome to JellyBamm

The lithium-ion battery cell simulator powered by [PyBaMM](https://www.pybamm.org/). JellyBamm allows you to specify cyclindrical jellyroll cell designs parametrically or extract them from images.

Leverage the experiments and parameter sets from PyBaMM and scale up your simulations from 1D representations to detailed 2D (and 3D - in future) cell simulations. Include thermal effects and account for inhomogenteities introduced by tabs or variable inputs for heat transfer.

Include statistical distributions in the battery parameters for sections of the cell.

### Installing

Execute the following command to install `JellyBamm` with pip after navigating to the root folder:

```bash
pip install -e .
```

## Getting Started

Example notebooks can be found in the `docs\examples` folder. Example scripts can be found in the `run_scripts` folder. Here is an example:

```
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
```

## Running the tests

```bash
pytest
```


## Contributing

Please read the [docs](https://jellybamm.readthedocs.io/en/latest/?badge=latest) for details on our code of conduct, and the process for submitting pull requests to us.


## Authors

- **Tom Tranter** - _Initial work_ - [Tom Tranter](https://github.com/TomTranter)

See also the list of [contributors](https://github.com/TomTranter/JellyBaMM/contributors) who participated in this project.

## License

This project is licensed under the BSD-3-Clause license - see the [LICENSE](LICENSE) file for details

## Acknowledgments

- Great thanks to the PyBaMM and OpenPNM teams
