#
# Unit test for funcs
#

import ecm
import openpnm as op
import pybamm
import unittest


wrk = op.Workspace()


class funcsTest(unittest.TestCase):
    def _teardown(self, fpath):
        # Delete Data files
        pass

    def _ecm_general(self, project):
        # Experiment
        I_app = 0.35
        dt = 5
        Nsteps = 3
        hours = dt * Nsteps / 3600
        experiment = pybamm.Experiment(
            [
                f"Discharge at {I_app} A for {hours} hours",
            ],
            period=f"{dt} seconds",
        )

        # Parameter set
        param = pybamm.ParameterValues("Chen2020")
        # JellyBaMM discretises the spiral using the electrode height for spiral length
        # This parameter set has the longer length set to the Electrode width
        # We want to swap this round
        param["Electrode width [m]"] = 0.08
        # Passing None as initial_soc will take values from Parameter set and apply
        # uniformly everywhere
        initial_soc = None

        # Run simulation
        project, output = ecm.run_simulation_lp(
            parameter_values=param,
            experiment=experiment,
            initial_soc=initial_soc,
            project=project,
        )
        assert output is not None

    def test_ecm_spiral(self):
        wrk.clear()
        Nlayers = 2
        dtheta = 10
        spacing = 195e-6  # To do should come from params
        inner_r = 10 * spacing
        pos_tabs = [-1]
        neg_tabs = [0]
        length_3d = 0.08
        tesla_tabs = False
        # OpenPNM project
        project, arc_edges = ecm.make_spiral_net(
            Nlayers, dtheta, spacing, inner_r,
            pos_tabs, neg_tabs, length_3d, tesla_tabs
        )
        self._ecm_general(project)

    def test_ecm_spiral_tesla(self):
        wrk.clear()
        Nlayers = 2
        dtheta = 10
        spacing = 195e-6  # To do should come from params
        inner_r = 10 * spacing
        pos_tabs = [-1]
        neg_tabs = [0]
        length_3d = 0.08
        tesla_tabs = True
        # OpenPNM project
        project, arc_edges = ecm.make_spiral_net(
            Nlayers, dtheta, spacing, inner_r,
            pos_tabs, neg_tabs, length_3d, tesla_tabs
        )
        self._ecm_general(project)

    def test_ecm_tomo(self):
        wrk.clear()
        # Geometry of spiral
        tomo_pnm = "spider_net.pnm"
        dtheta = 10
        spacing = 195e-6
        length_3d = 0.08
        pos_tabs = [-1]
        neg_tabs = [0]
        # OpenPNM project
        project, arc_edges = ecm.make_tomo_net(
            tomo_pnm, dtheta, spacing, length_3d, pos_tabs, neg_tabs
        )
        self._ecm_general(project)

    def test_ecm_1d(self):
        wrk.clear()
        # Geometry of 1D mesh
        Nunit = 100
        spacing = 0.01
        pos_tabs = [-1]
        neg_tabs = [0]
        # OpenPNM project
        project, arc_edges = ecm.make_1D_net(Nunit, spacing, pos_tabs, neg_tabs)
        self._ecm_general(project)


if __name__ == "__main__":
    unittest.main()
