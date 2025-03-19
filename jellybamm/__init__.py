"""
JellyBaMM: a package for simulating Li-ion electrochemistry on jellyroll structures with PyBaMM
"""

# __all__ = [
#     'plot_topology',
#     'spiral',
#     'topology.make_spiral_net',
#     'topology.make_tomo_net',
#     'topology.make_1D_net',
#     'topology.network_to_netlist'
# ]

from .funcs import *
from .postprocess import *
from .definitions import *
from .segment_jellyroll import *
from .utilities import *
from .topology import plot_topology, spiral, make_spiral_net, make_1D_net, network_to_netlist
from .battery import *
from .liionsolve import *

__version__ = "0.0.3"
