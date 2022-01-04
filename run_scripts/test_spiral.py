#
# Test the spiral net
#

import openpnm as op
import matplotlib.pyplot as plt
import ecm
import configparser
import os

plt.close("all")

#pybamm.set_logging_level("INFO")
wrk = op.Workspace()
wrk.clear()


if __name__ == "__main__":
    save_root = os.path.join(ecm.OUTPUT_DIR, 'spiral')
    print(save_root)
    config = configparser.ConfigParser()
    config.read(os.path.join(save_root, 'config.txt'))
    print(ecm.lump_thermal_props(config))
    I_apps = [config.get('RUN', key) for key in config['RUN'] if 'i_app' in key]
    prj, arc_edges = ecm.make_spiral_net(config)
    ecm.plot_topology(prj.network)
