import matplotlib.pyplot as plt
import streamlit as st
import ecm
from openpnm.topotools import plot_connections as pconn
from openpnm.topotools import plot_coordinates as pcoord


def plot_topology(net):
    inner = net["pore.inner"]
    outer = net["pore.outer"]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = pconn(net, throats=net.throats("throat.neg_cc"), c="blue", ax=ax)
    ax = pconn(net, throats=net.throats("throat.pos_cc"), c="red", ax=ax)
    ax = pcoord(net, pores=net["pore.neg_cc"], c="blue", ax=ax)
    ax = pcoord(net, pores=net["pore.pos_cc"], c="red", ax=ax)
    ax = pcoord(net, pores=net["pore.neg_tab"], c="blue", s=300, ax=ax)
    ax = pcoord(net, pores=net["pore.pos_tab"], c="red", s=300, ax=ax)
    ax = pcoord(net, pores=inner, c="pink", ax=ax)
    ax = pcoord(net, pores=outer, c="yellow", ax=ax)
    ax = pcoord(net, pores=net.pores("free_stream"), c="green", ax=ax)
    ax = pconn(net, throats=net.throats("throat.free_stream"), c="green", ax=ax)
    t_sep = net.throats("spm_resistor")
    if len(t_sep) > 0:
        ax = pconn(net, throats=net.throats("spm_resistor"), c="k", ax=ax)
    plt.tight_layout()
    st.pyplot(fig)


def update_plot(pos_ints, neg_ints, net):
    pos_Ps = net.pores("pos_cc")
    neg_Ps = net.pores("neg_cc")
    # pos_ints = [p1, p2, p3, p4, p5]
    # neg_ints = [n1, n2, n3, n4, n5]
    pos_tabs = pos_Ps[pos_ints]
    neg_tabs = neg_Ps[neg_ints]
    net["pore.pos_tab"] = False
    net["pore.neg_tab"] = False
    net["pore.pos_tab"][pos_tabs] = True
    net["pore.neg_tab"][neg_tabs] = True
    plot_topology(net)


with st.sidebar:
    Nlayers = st.slider("Number of layers", 0, 20, 10)
    # st.write('No. of layers:', Nlayers)

    width = st.slider("Layer width [um]", 100, 200, 150)
    # st.write('Layer width [um]:', width)

    dtheta = st.slider("dtheta [deg]", 1, 90, 10)
    # st.write('dtheta:', dtheta)

    tesla_tabs = st.checkbox("Tesla tabs")
    if not tesla_tabs:
        num_tabs = st.slider("Number of tabs", 1, 5, 1)
        tab_spacing = int(Nlayers / num_tabs)
        nodes_per_layer = int(360 / dtheta)
        positive_offset = st.slider("Positive tab offset", 0, nodes_per_layer, 0)
        negative_offset = st.slider("Negative tab offset", 0, nodes_per_layer, 0)

# Geometry of spiral
# Nlayers = 2
# dtheta = 10
# spacing = 195e-6  # To do should come from params
pos_tabs = [-1]
neg_tabs = [0]
length_3d = 0.08
# tesla_tabs = False


# OpenPNM project
project, arc_edges = ecm.make_spiral_net(
    Nlayers, dtheta, width * 1e-6, pos_tabs, neg_tabs, length_3d, tesla_tabs
)

net = project.network
Npcc = net.num_pores("pos_cc")
Nncc = net.num_pores("neg_cc")
print("Num pos", Npcc, "Num neg", Nncc)
if tesla_tabs:
    plot_topology(net)
else:

    pos_tabs = [
        positive_offset + (tab_spacing * nodes_per_layer * i) for i in range(num_tabs)
    ]
    neg_tabs = [
        (Nncc - 1 - negative_offset) - (tab_spacing * nodes_per_layer * i)
        for i in range(num_tabs)
    ]
    update_plot(pos_tabs, neg_tabs, net)
