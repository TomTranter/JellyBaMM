#
# Unit test for postprocessing funcs
#

import jellybamm
import matplotlib.pyplot as plt
import os
import openpnm as op
import unittest


wrk = op.Workspace()
close_plots = True
root = jellybamm.TEST_CASES_DIR
children = []
for file in os.listdir(root):
    child = os.path.join(root, file)
    children.append(child)


class postprocessTest(unittest.TestCase):
    def _load(self, filepath):
        data = jellybamm.load_cases(filepath)
        cases = list(data.keys())
        amps = jellybamm.get_amp_cases(os.path.join(filepath, cases[0]))
        return data, amps, cases

    # def test_jellyroll_subplot(self):
    #     data = jellybamm.load_cases(root)
    #     case_index = 0
    #     amp_index = 0
    #     var_index = 0
    #     case_folder = children[case_index]
    #     amps = jellybamm.get_amp_cases(case_folder)
    #     case = list(data.keys())[case_index]
    #     jellybamm.jellyroll_subplot(
    #         data,
    #         case,
    #         amps[amp_index],
    #         var=var_index,
    #         soc_list=[[1.0, 0.99], [0.98, 0.97]],
    #         global_range=False,
    #         dp=1,
    #     )
    #     if close_plots:
    #         plt.close("all")
    #     wrk.clear()
    #     assert 1 == 1

    def test_multivar_subplot(self):
        data, amps, cases = self._load(root)
        jellybamm.multi_var_subplot(data, [cases[0]], amps, [2, 0], landscape=False)
        if close_plots:
            plt.close("all")
        wrk.clear()
        assert 1 == 1

    def test_spacetime(self):
        data, amps, cases = self._load(root)
        jellybamm.spacetime(data, cases, amps, var=0, group="neg", normed=True)
        if close_plots:
            plt.close("all")
        wrk.clear()
        assert 1 == 1

    def test_chargeogram(self):
        data, amps, cases = self._load(root)
        jellybamm.chargeogram(data, cases, amps, group="neg")
        if close_plots:
            plt.close("all")
        wrk.clear()
        assert 1 == 1

    # def test_animate(self):
    #     data, amps, cases = self._load(root)
    #     fname = os.path.join(children[0], "test.mp4")
    #     jellybamm.animate_data4(data, cases[0], amps[0], variables=[0, 1], filename=fname)
    #     os.remove(fname)
    #     wrk.clear()

    def test_super_subplot(self):
        filepath = os.path.join(jellybamm.FIXTURES_DIR, "model")
        data, amps, cases = self._load(filepath)
        jellybamm.super_subplot(data, [cases[0]], [cases[0]], amps[0])
        if close_plots:
            plt.close("all")
        wrk.clear()
        assert 1 == 1


if __name__ == "__main__":
    unittest.main()
