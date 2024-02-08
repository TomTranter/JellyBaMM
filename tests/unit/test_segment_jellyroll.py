#
# Unit test for jellyroll segmentation
#

import jellybamm
import matplotlib.pyplot as plt
import os
import unittest


class postprocessTest(unittest.TestCase):
    def test_segment_jellyroll(self):
        # Return the average of the input images, the mid-half-span and a distance
        # transform of the image with centre = 0 to act as radial map
        im, mhs, dt = jellybamm.average_images()
        step = 1  # step to take when binning for averaging
        deg = 6  # degree of polynomial fitting when fitting intensity profile
        im_soft = jellybamm.remove_beam_hardening(im, dt, step, deg)

        # Label Layers
        im_soft, cc_im = jellybamm.label_layers(
            im_soft, dt, mhs, can_width=30, im_thresh=19000, small_feature_size=20000
        )
        # Make the spider web network
        filename = "_test_network.pnm"
        net = jellybamm.spider_web_network(
            im_soft, mhs, cc_im, dtheta=20, path=jellybamm.INPUT_DIR, filename=filename
        )
        prj = net.project
        # make smp map
        jellybamm.interpolate_spm_number(prj)
        os.remove(os.path.join(jellybamm.INPUT_DIR, filename))
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
