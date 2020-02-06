# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:49:46 2019

@author: Tom
"""

from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops as rp
import scipy.ndimage as spim
from skimage.morphology import binary_erosion, binary_dilation, medial_axis, disk, square
from skimage.measure import label
from skimage.exposure import equalize_adapthist, rescale_intensity
import skimage.filters.rank as rank
from skimage.segmentation import flood_fill
from skimage.feature import canny
from skimage.restoration import denoise_nl_means, estimate_sigma
import os
from scipy.stats import itemfreq
plt.close('all')
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
import openpnm as op
import openpnm.topotools as tt
import pandas as pd
import ecm

pixel_size=10.4e-6
wrk = op.Workspace()
path = os.path.join(os.getcwd(), 'input')

# Get File list
#path = os.path.join(input_dir, 'recon-manual-Mid-top')
files = []
for file in os.listdir(path):
    if file.split('.')[-1] == 'tiff':
        files.append(file)
# Pick subset
#files = np.asarray(files)
#start = 800
#finish = 801
#files = files[start:finish]
# Load files
ims = [io.imread(os.path.join(path, file)) for file in files]
masks = []
boxes = []
mids = []
# Find mid points of can 
for i in range(len(files)):
    can = ims[i] < 25000
    sel_ero = np.ones([5, 5])
    sel_dil = np.ones([2, 2])
    can = binary_dilation(binary_erosion(can, selem=sel_ero), sel_dil)
    labels = label(can)
    mask = labels != 1
    masks.append(mask)
    props = rp(mask.astype(int))[0]
    boxes.append(props.bbox)
    mids.append(props.centroid)
# Find span of image
mids = np.floor(np.asarray(mids)).astype(int)
boxes = np.asarray(boxes)
x_span = boxes[:, 2] - boxes[:, 0]
y_span = boxes[:, 3] - boxes[:, 1]
mhs = np.int(np.ceil(np.min([x_span.min(), y_span.min()])/2)) # min half span
# Factorize image size - used for adaptive histogran
factors = []
for i in np.arange(2, mhs*2):
    if mhs*2 % i == 0.0:
        factors.append(i)
# Crop images
crops = []
for i, (x_min, y_min, x_max, y_max) in enumerate(boxes):
    crops.append(ims[i][mids[i, 0]-mhs:mids[i, 0]+mhs, mids[i, 1]-mhs:mids[i, 1]+mhs]*masks[i][mids[i, 0]-mhs:mids[i, 0]+mhs, mids[i, 1]-mhs:mids[i, 1]+mhs])
    print(i, crops[-1].shape)

# Get new mids from crops
#fig, (ax1, ax2) = plt.subplots(2, 1)
#mids = []
#for i in range(len(crops)):
#    mid = np.floor(np.array(crops[i].shape)/2).astype(int)
#    ax1.plot(crops[i][mid[0], :])
#    ax2.plot(crops[i][:, mid[1]])
#    mids.append(mid)
#
#mids = np.asarray(mids)
    
im = crops[0].copy()
dt = np.ones(im.shape)
dt[mhs:mhs+1, mhs:mhs+1] = 0
dt = spim.distance_transform_edt(dt)
plt.figure()
plt.imshow(dt)

def get_radial_average(im, step):
    mid = np.floor(np.array(im.shape)/2).astype(int)
    bands = np.arange(300, mid[0]-25, step)
    band_avs = []
    r = []
    for i in range(len(bands)-1):
        l = bands[i]
        u = bands[i+1]
        mask = (dt >= l) * (dt < u)
        band_avs.append(np.mean(im[mask]))
        r.append((l+u)/2)
    return np.vstack((r, band_avs)).T

def adjust_radial_average(im, step, deg):
    mid = np.floor(np.array(im.shape)/2).astype(int)
    bands = np.arange(300, mid[0]-25, step)
    band_avs = []
    r = []
    for i in range(len(bands)-1):
        l = bands[i]
        u = bands[i+1]
        mask = (dt >= l) * (dt < u)
        band_avs.append(np.mean(im[mask]))
        r.append((l+u)/2)
    dat = np.vstack((r, band_avs)).T
    p = np.polyfit(dat[:, 0], dat[:, 1], deg=deg)
    pfunc = np.poly1d(p)
    adj = pfunc(dat[:, 0])
    adj = adj - adj[0]
    adj_im = im.copy().astype(float)
    for i in range(len(bands)-1):
        l = bands[i]
        u = bands[i+1]
        mask = (dt >= l) * (dt < u)
        adj_im[mask] -= adj[i]
    return adj_im

step = 1
deg = 6

dat_im = get_radial_average(im, step)
im_soft = adjust_radial_average(im, step, deg)
dat_soft = get_radial_average(im_soft, step)
fig, axes  = plt.subplots(2, 1)
axes[0].imshow(im_soft - im.astype(float))
axes[1].plot(dat_im[:, 0], dat_im[:, 1])
axes[1].plot(dat_soft[:, 0], dat_soft[:, 1])

# Remove outer
outer = dt > mhs
# Remove Can
can = dt > mhs - 30
im_soft[outer] = 0
im_soft[can] = 0
# Final layer labels
layers = np.zeros(can.shape, dtype=int)
layers[can] = layers.max() + 1
# Threshold image 
mask = im_soft > 19000
plt.figure()
plt.imshow(mask)
# Label threshold - Should result in 2 or 3 regions - if not look at threshold
lab = label(mask)
plt.imshow(lab)
# Count Labels and remove any small ones
itmfrq = itemfreq(lab)
itmfrq = itmfrq[1:]

# Get rid of small and medium size edge features
itmfrq = itmfrq[itmfrq[:, 1] > 20000]

cc_im = np.zeros(lab.shape, dtype=np.int8)

for l, _ in itmfrq:
    if l > 0:
        tmp = lab == l
        layers[tmp] = layers.max() + 1
        tmp = binary_erosion(binary_dilation(tmp, disk(6)), disk(4))
        skel = medial_axis(tmp, return_distance=False)
        skel = skel.astype(np.uint8)
        # sum the values in the kernel to identify joints and ends
        skel_rank_sum = rank.sum(skel, selem=square(3))
        # only keep sums in the skeleton
        skel_rank_sum *= skel
        tmp2 = skel_rank_sum.copy()
        # temporarily remove joints
        tmp2[skel_rank_sum > 3] = 0
        (seeds_x, seeds_y) = np.where(tmp2 == 2)
        for i in range(len(seeds_x)):
            tmp2[seeds_x[i], seeds_y[i]] = 3
            tmp2 = flood_fill(tmp2, (seeds_x[i], seeds_y[i]), 2)
        tmp2[tmp2 == 2] = 0
        tmp2[skel_rank_sum > 3] = 1
        cc_im[tmp2 > 0] = l
        layers[tmp2 > 0] = layers.max() + 1
        plt.figure()
        plt.imshow(tmp2)

# get inner section
cc_all = cc_im > 0
cc_all = binary_dilation(cc_all, disk(8))
plt.figure()
plt.imshow(cc_all)
lab = label(1-cc_all)
plt.figure()
plt.imshow(lab)
inner_lab = lab[mhs, mhs]
mask_nan = lab == inner_lab
im_soft[mask_nan] = 0
plt.figure()
im_soft[im_soft==0] = np.nan
plt.imshow(im_soft)

# Make spiderweb dividing lines
(inds_x, inds_y) = np.indices(im_soft.shape)
inds_x -= mhs
inds_y -= mhs
theta = np.around(np.arctan2(inds_y, inds_x)*360/(2*np.pi), 2)
dtheta = 10
remainder = theta % dtheta
lines = np.logical_or((remainder < 1.0), (remainder > 9.0))
med_ax = medial_axis(lines)
plt.figure()
plt.imshow(med_ax)
med_ax = binary_dilation(med_ax, selem=square(2))
# Multiply cc image by -1 along the dividing lines to define nodes
cc_im[med_ax] *= -1
# Extract coordinates of nodes on the current collectors
cc_coords = []
for i in np.unique(cc_im)[np.unique(cc_im) < 0]:
    lab, N = label(cc_im == i, return_num=True)    
    coords = np.zeros([N, 2], dtype=int)
    for l in np.arange(N):
        x, y = np.where(lab==l+1)
        if len(x) > 0:
            coords[l] = [np.int(np.around(np.mean(x),0)),
                         np.int(np.around(np.mean(y),0))]
    cc_coords.append(coords)
plt.figure()
plt.imshow(cc_im)
plt.scatter(cc_coords[0][:, 1], cc_coords[0][:, 0], c='r', s=50)
plt.scatter(cc_coords[1][:, 1], cc_coords[1][:, 0], c='b', s=50)

# Collect coords together
coords = np.vstack((cc_coords[0], cc_coords[1]))
# Record which belongs to same cc
first_cc = np.zeros(len(coords), dtype=bool)
first_cc[:len(cc_coords[0])] = True
# Find which theta bin coords are in
point_thetas = theta[coords[:, 0], coords[:, 1]]
bins = np.arange(-185, 195, 10)
point_group_theta = np.zeros(len(coords[:, 0]), dtype=int)
for i in range(len(bins)-1):
    l = bins[i]
    u = bins[i+1]
    sel = (point_thetas > l) * (point_thetas < u)
    point_group_theta[sel] = i
# Get radial position of coords
rads = np.linalg.norm(coords - mhs, axis=1)


###############################################################################
groups, counts = np.unique(point_group_theta, return_counts=True)
sorted_groups = np.zeros([len(groups), counts.max()], dtype=int)
sorted_groups.fill(-1)
for g in range(len(groups)):
    Ps = np.arange(0, len(point_group_theta), 1)[point_group_theta == g]
    group_rads = rads[Ps]
    sorted_rads = group_rads.argsort()
    sorted_Ps = Ps[sorted_rads]
    sorted_groups[g, :len(sorted_Ps)] = sorted_Ps
plt.figure()
cc_groups = first_cc[sorted_groups].astype(int)
cc_groups[sorted_groups == -1] = -1
plt.imshow(cc_groups)
inner_Ps = sorted_groups[:, 0]
inner_cc = first_cc[inner_Ps]
change_cc = inner_cc != np.roll(inner_cc, -1)
sorted_cc =  first_cc[sorted_groups].astype(int)
#sorted_cc[change_cc, 0] += 3
#plt.imshow(sorted_cc)
arc_indices = []
cc_roll = [[], []]
for cc_num, this_cc in enumerate([True, False]):
    start_group = np.argwhere(np.logical_and(inner_cc==this_cc, change_cc)).flatten()[0]
    layer_group = np.arange(0, len(groups), 1)
    layer_group = np.roll(layer_group, -start_group)
    i = 0
    layer = 0

    #this_cc = True
    numbered_cc =  first_cc[sorted_groups].astype(int)
    while layer < counts.max():
        g = layer_group[0]
        if sorted_cc[g, layer] == this_cc:
            pass
        else:
            layer += 1
        if layer < counts.max():
            if sorted_groups[g, layer] > -1:
                cc_roll[cc_num].append(sorted_groups[g, layer])
                numbered_cc[g, layer] = i
                arc_indices.append(g)
                i += 1
            else:
                pass
        layer_group = np.roll(layer_group, 1)

    plt.figure()
    plt.imshow(numbered_cc)
print(len(np.unique(cc_roll[0]+cc_roll[1])))
ordered_neg_Ps = np.asarray(cc_roll[0])
ordered_pos_Ps = np.asarray(cc_roll[1])
neg_coords = coords[ordered_neg_Ps]
neg_conns = np.vstack((np.arange(0, len(ordered_neg_Ps)-1, 1),
                       np.arange(0, len(ordered_neg_Ps)-1, 1)+1)).T
pos_coords = coords[ordered_pos_Ps]
pos_conns = np.vstack((np.arange(0, len(ordered_pos_Ps)-1, 1),
                       np.arange(0, len(ordered_pos_Ps)-1, 1)+1)).T
pos_conns += len(ordered_neg_Ps)
neg_inner_conns = []
for i, p_neg in enumerate(ordered_neg_Ps):
    g, layer = np.argwhere(sorted_groups == p_neg).flatten()
    if layer < counts.max() - 1:
        neighbor = sorted_groups[g, layer+1]
        if neighbor > -1:
            sorted_neighbor = np.argwhere(ordered_pos_Ps == neighbor).flatten()[0]
            neg_inner_conns.append([i, sorted_neighbor+len(ordered_neg_Ps)])
pos_inner_conns = []
for i, p_pos in enumerate(ordered_pos_Ps):
    g, layer = np.argwhere(sorted_groups == p_pos).flatten()
    if layer < counts.max() - 1:
        neighbor = sorted_groups[g, layer+1]
        if neighbor > -1:
            sorted_neighbor = np.argwhere(ordered_neg_Ps == neighbor).flatten()[0]
            pos_inner_conns.append([i+len(ordered_neg_Ps), sorted_neighbor])
neg_inner_conns = np.asarray(neg_inner_conns)
pos_inner_conns = np.asarray(pos_inner_conns)

new_coords = np.vstack((neg_coords, pos_coords))
coords_3d = np.vstack((new_coords[:, 0], new_coords[:, 1], np.zeros(new_coords.shape[0]))).T
new_conns = np.vstack((neg_conns, pos_conns, neg_inner_conns, pos_inner_conns))
net = op.network.GenericNetwork(conns=new_conns, coords=coords_3d)

Ps, counts = np.unique(np.hstack((net['throat.conns'][:, 0],
                                  net['throat.conns'][:, 1])), return_counts=True)
net['pore.surface'] = False
net['pore.terminal'] = False
net['pore.neg_cc'] = False
net['pore.pos_cc'] = False
net['pore.inner'] = False
net['pore.outer'] = False
net['pore.surface'][Ps[counts < 4]] = True
net['pore.terminal'][Ps[counts == 2]] = True
net['pore.neg_cc'][:len(ordered_neg_Ps)] = True
net['pore.pos_cc'][len(ordered_neg_Ps):] = True
net['pore.cell_id'] = np.arange(0, net.Np, 1).astype(int)
net['pore.arc_index'] = np.asarray(arc_indices)
#net['pore.cell_id'][:len(ordered_neg_Ps)] = np.arange(0, len(ordered_neg_Ps), 1).astype(int)
#net['pore.cell_id'][:len(ordered_neg_Ps)] = np.arange(0, len(ordered_neg_Ps), 1).astype(int)
net['pore.radial_position'] = np.linalg.norm(new_coords - mhs, axis=1)
inner_mask = np.logical_and(net['pore.surface'], net['pore.radial_position'] < mhs/2)
outer_mask = np.logical_and(net['pore.surface'], net['pore.radial_position'] > mhs/2)
net['pore.inner'][inner_mask] = True
net['pore.outer'][outer_mask] = True
net['throat.neg_cc'] = False
net['throat.pos_cc'] = False

net['throat.spm_resistor'] = True
pos_cc_Ts = net.find_neighbor_throats(net.pores("pos_cc"), mode="xnor")
neg_cc_Ts = net.find_neighbor_throats(net.pores("neg_cc"), mode="xnor")
net['throat.neg_cc'][neg_cc_Ts] = True
net['throat.pos_cc'][pos_cc_Ts] = True
net['throat.spm_resistor'][neg_cc_Ts] = False
net['throat.spm_resistor'][pos_cc_Ts] = False
net['throat.spm_resistor_order'] = -1
net['throat.spm_resistor_order'][net['throat.spm_resistor']] = np.arange(0, net.num_throats('spm_resistor'), 1).astype(int)
net['throat.spm_neg_inner'] = False
net['throat.spm_pos_inner'] = False
net['throat.spm_neg_inner'][(net['throat.spm_resistor_order'] > -1) * (net['throat.spm_resistor_order'] < len(neg_inner_conns))] = True
net['throat.spm_pos_inner'][(net['throat.spm_resistor_order'] > -1) * (net['throat.spm_resistor_order'] >= len(neg_inner_conns))] = True


net['pore.neg_tab'] = False
net['pore.pos_tab'] = False
net['pore.neg_tab'][net.pores(['inner', 'neg_cc', 'terminal'], mode='and')] = True
net['pore.pos_tab'][net.pores(['outer', 'pos_cc', 'terminal'], mode='and')] = True
num_free = net.num_pores('outer')
outer_pos = net['pore.coords'][net.pores('outer')]
x = outer_pos[:, 0] - mhs
y = outer_pos[:, 1] - mhs
r, t = ecm.polar_transform(x, y)
#r_new = np.ones(num_free)*(r.max() + 50)
r_new = np.ones(num_free)*(r + 50)
new_x, new_y = ecm.cartesian_transform(r_new, t)
free_coords = outer_pos.copy()
free_coords[:, 0] = new_x + mhs
free_coords[:, 1] = new_y + mhs
start = len(ordered_neg_Ps) + len(ordered_pos_Ps) - num_free
free_conns = np.vstack((np.arange(0, num_free, 1),
                        np.arange(0, num_free, 1)+num_free)).T
free_conns += start
tt.extend(net, pore_coords=free_coords, throat_conns=free_conns, labels=['free_stream'])
net['pore.arc_index'][net['pore.free_stream']] = net['pore.arc_index'][net['pore.outer']]


fig = plt.figure()
plt.imshow(im_soft.T)
fig=tt.plot_connections(net, throats=net.throats('neg_cc'), c='r', fig=fig)
fig=tt.plot_connections(net, throats=net.throats('pos_cc'), c='b', fig=fig)
fig=tt.plot_connections(net, throats=net.throats('spm_resistor'), c='k', fig=fig)
fig=tt.plot_coordinates(net, pores=net.pores('neg_cc'), c='r', fig=fig)
fig=tt.plot_coordinates(net, pores=net.pores('pos_cc'), c='b', fig=fig)
fig=tt.plot_coordinates(net, pores=net.pores('surface'), c='k', fig=fig)
fig=tt.plot_coordinates(net, pores=net.pores('outer'), c='pink', fig=fig)
fig=tt.plot_coordinates(net, pores=net.pores('inner'), c='purple', fig=fig)
fig=tt.plot_coordinates(net, pores=net.pores('terminal'), c='y', s=100, fig=fig)
fig=tt.plot_coordinates(net, pores=net.pores('free_stream'), c='g', fig=fig)
fig=tt.plot_connections(net, throats=net.throats('free_stream'), c='g', fig=fig)

ecm.plot_topology(net)

fig=tt.plot_connections(net, throats=net.Ts)
fig=tt.plot_coordinates(net, pores=net.Ps, c=net['pore.arc_index'], fig=fig)
###############################################################################
if 1 == 2:
    # Perform triangulation of coordinates to define connections
    tri = Delaunay(coords)
    
    plt.figure()
    plt.imshow(im_soft)
    dtheta_rad = 2*np.pi/36
    plt.scatter(coords[:, 1], coords[:, 0], c=point_group_theta)
    conns = []
    for simplex in tri.simplices:
        for roll in range(3):
            ps = np.roll(simplex, roll)[:2]
            # If two coords are in same theta bin they have a radial connection
            if len(np.unique(point_group_theta[ps])) == 1:
                plt.plot(coords[ps][:, 1], coords[ps][:, 0], 'r')
                conns.append(list(ps))
            elif first_cc[ps[0]] == first_cc[ps[1]]:
                # If they are not radial then they should be in same cc
                # Check the distance to make sure not connecting across center
                # Should be around arc length r*dtheta
                d = np.linalg.norm(coords[ps[1]]-coords[ps[0]])
                l = rads[ps[0]]*dtheta_rad*1.5
                if d < l:
                    #find which cc they are in
                    if first_cc[ps[0]]:
                        plt.plot(coords[ps][:, 1], coords[ps][:, 0], 'b')
                        conns.append(list(ps))
                    else:
                        plt.plot(coords[ps][:, 1], coords[ps][:, 0], 'y')
                        conns.append(list(ps))
    coords3d = np.vstack((coords[:, 0], coords[:, 1], np.zeros(coords.shape[0]))).T
    net = op.network.GenericNetwork(coords=coords3d, conns=conns)
    net['pore.arc_index'] = point_group_theta
    net['pore.theta'] = point_thetas
    net['pore.radial_position'] = rads
    net['pore.cc_a'] = first_cc
    net['pore.cc_b'] = ~first_cc
    
    def plot_domain(net):
        fig = plt.figure()
        plt.imshow(np.swapaxes(im_soft, 0, 1))
        fig = tt.plot_coordinates(net, pores=net.pores('cc_a'), c='r', fig=fig)
        fig = tt.plot_coordinates(net, pores=net.pores('cc_b'), c='b', fig=fig)
        cc_a_Ts = net.find_neighbor_throats(pores=net.pores('cc_a'), mode='xnor')
        cc_b_Ts = net.find_neighbor_throats(pores=net.pores('cc_b'), mode='xnor')
        fig = tt.plot_connections(net, throats=cc_a_Ts, c='r', fig=fig)
        fig = tt.plot_connections(net, throats=cc_b_Ts, c='b', fig=fig)
        fig = tt.plot_connections(net, throats=net.throats('interconnection'), c='k', fig=fig)
        fig = tt.plot_coordinates(net, pores=net.pores('terminal'), s=100, c='k', fig=fig)
        if 'pore.inner' in net.labels():
            fig = tt.plot_coordinates(net, pores=net.pores('inner'), s=100, c='purple', fig=fig)
            fig = tt.plot_coordinates(net, pores=net.pores('outer'), s=100, c='pink', fig=fig)
        if 'pore.free_stream' in net.labels():
            fig = tt.plot_coordinates(net, pores=net.pores('free_stream'), s=100, c='green', fig=fig)
        for i in range(20):
            if 'pore.layer_'+str(i) in net.labels():
                fig = tt.plot_coordinates(net, pores=net.pores('layer_'+str(i)), s=10, c='orange', fig=fig)
                fig = tt.plot_connections(net, throats=net.throats('layer_'+str(i)), c='orange', fig=fig)
        if 'throat.stitched' in net.labels():
            fig = tt.plot_connections(net, throats=net.throats('stitched'), c='g', fig=fig)
    
    cc_a_Ts = net.find_neighbor_throats(pores=net.pores('cc_a'), mode='xnor')
    cc_b_Ts = net.find_neighbor_throats(pores=net.pores('cc_b'), mode='xnor')
    net['throat.interconnection'] = True
    net['throat.interconnection'][cc_a_Ts] = False
    net['throat.interconnection'][cc_b_Ts] = False
    
    h = net.check_network_health()
    dupes = h['duplicate_throats']
    if len(dupes) > 0:
        dupes = np.asarray(dupes)
        tt.trim(net, throats=dupes[:, 1])
    # Find ends
    net['pore.terminal'] = False
    net['pore.terminal_neighbor'] = False
    all_nbrs = net.find_neighbor_throats(pores=net.Ps, flatten=False)
    for p, nbrs in enumerate(all_nbrs):
        if len(nbrs) == 2:
            net['pore.terminal'][p] = True
            inter_throat = nbrs[net['throat.interconnection'][nbrs]]
            inter_nbrs = net.find_connected_pores(throats=inter_throat)
            net['pore.terminal_neighbor'][inter_nbrs[inter_nbrs != p]] = True
    
    links = []
    for start in net.pores('terminal'):
        end_found = False
        link = [start]
        while not end_found:
            nbrs = net.find_neighbor_pores(pores=start)
            same_cc = net['pore.cc_a'][nbrs] == net['pore.cc_a'][start]
            nbrs = nbrs[same_cc]
            nbr = nbrs[~np.in1d(nbrs, link)]
            if net['pore.terminal_neighbor'][nbr]:
                end_found = True
            else:
                link.append(nbr[0])
                start = nbr
        print(len(link), net['pore.theta'][link], net['pore.cc_a'][start])
        links.append(link)
    
    # Pick shortest link to get rid of
    trim = []
    shortest = 1e9
    shortest_link = 0
    for i, link in enumerate(links):
        if len(link) < shortest:
            shortest = len(link)
            shortest_link = i
    trim = trim + links[shortest_link]
    # find if theta counts increase or not
    clockwise = np.zeros(len(links), dtype=bool)
    for i, link in enumerate(links):
        tmp = net['pore.theta'][link]
        comp = tmp < np.roll(tmp, -1)
        if np.sum(comp) < len(comp)/2:
            clockwise[i] = True
    # Find which current colletor shortest link belongs to
    other_trim = net['pore.cc_a'][net.pores('terminal')]
    # Other trim needs to be going the opposite way and be on the opposite cc
    other_trim = other_trim != other_trim[shortest_link]
    other_trim *= clockwise != clockwise[shortest_link]
    # Get index into links for the other trim
    other_link = np.arange(0, len(links))[other_trim][0]
    # Add the trimmed links together
    trim = trim + links[other_link]
    tt.trim(net, pores=trim)
    
    # Now relabel the terminals - two will have been deleted
    new_terminals = []
    for tp in net.pores('terminal'):
        nbrs = net.find_neighbor_pores(tp)
        nbr = nbrs[net['pore.cc_a'][nbrs] != net['pore.cc_a'][tp]]
        new_terminals.append(nbr[0])
    net['pore.terminal'][new_terminals] = True
    plot_domain(net)
    
    # Start from inner terminals and label neighboring pores with increasing #cell 
    inner_terminals = net.pores('terminal')[net['pore.radial_position'][net.pores('terminal')] < mhs/2]
    net['pore.cell_id'] = 0
    for start in inner_terminals:
        end_found = False
        link = [start]
        cell_id = 0
        while not end_found:
            nbrs = net.find_neighbor_pores(pores=start)
            same_cc = net['pore.cc_a'][nbrs] == net['pore.cc_a'][start]
            nbrs = nbrs[same_cc]
            nbr = nbrs[~np.in1d(nbrs, link)]
            if net['pore.terminal'][nbr]:
                end_found = True
            link.append(nbr[0])
            start = nbr
            cell_id +=1
            net['pore.cell_id'][nbr[0]] = cell_id
    
    
    def split_interconnection(net, layer=0):
        print('Np', net.Np, 'Nt', net.Nt)
        Ts = net.throats('interconnection')
        P1 = net['throat.conns'][Ts][:, 0]
        P2 = net['throat.conns'][Ts][:, 1]
        new_coords = (net['pore.coords'][P1] + net['pore.coords'][P2])/2
        new_pore_index = np.arange(0, len(P1)) + net.Np
        new_conns1 = np.vstack((P1, new_pore_index)).T
        new_conns2 = np.vstack((P2, new_pore_index)).T
        tt.extend(net, pore_coords=new_coords, labels='layer_'+str(layer))
        tt.extend(net, throat_conns=new_conns1, labels='interconnection')
        tt.extend(net, throat_conns=new_conns2, labels='interconnection')
        tt.trim(net, throats=Ts)
        print('Np', net.Np, 'Nt', net.Nt)
    
    #for i in range(4):
    #    split_interconnection(net, layer=i)
    
    
    
    #net_coords_2d = np.vstack((net['pore.coords'][:, 0], net['pore.coords'][:, 1])).T
    #vor = Voronoi(net_coords_2d)
    #voronoi_plot_2d(vor)
    
    #tmp_layers = layers.copy()
    #for num in [3, 5]:
    #    skel_rank_sum = rank.sum((layers==num).astype(int), selem=square(3))
    #    skel_rank_sum[layers!=num] = 0
    #    end_points = skel_rank_sum == 2
    #    end_x, end_y = np.where(end_points)
    #    # Pick one closest to middle
    #    k = np.argmin(np.abs(end_x - mhs))
    #    tmp = end_points.copy()
    #    tmp[end_x[k]-20:end_x[k]+20, end_y[k]] = True
    #    tmp[end_x[k], end_y[k]-20:end_y[k]+20] = True
    #    tmp_layers[(layers == 0) * tmp] = -1
    #
    #lab = label(tmp_layers==0)
    #mid_lab = lab[mhs, mhs]
    #tmp = lab == mid_lab
    #middle = binary_erosion(binary_dilation(tmp, disk(1)), disk(1))
    #layers[middle] = layers.max() + 1
    #
    #plt.figure()
    #plt.imshow(layers)
    
    net['pore.inner'] = False
    net['pore.outer'] = False
    for ai in np.unique(net['pore.arc_index']):
        Ps = net.pores()[net['pore.arc_index'] == ai]
        inner = Ps[np.argmin(net['pore.radial_position'][Ps])]
        outer = Ps[np.argmax(net['pore.radial_position'][Ps])]
        net['pore.inner'][inner] = True
        net['pore.outer'][outer] = True
    
    free_theta = np.linspace(0, 2*np.pi, 37)
    free_x = mhs*(1+0.975*np.sin(free_theta))
    free_y = mhs*(1+0.975*np.cos(free_theta))
    plt.scatter(free_x, free_y, c='pink', s=100)
    free_coords = np.vstack((free_x, free_y, np.zeros(len(free_x)))).T
    free_conns = np.asarray([[i, i+1] for i in range(36)])
    net_free = op.network.GenericNetwork(coords=free_coords, conns=free_conns)
    net_free['throat.trim'] = True
    net_free['pore.free_stream'] = True
    net_free['pore.radial_position'] = mhs
    net_free['pore.theta'] = free_theta
    tt.stitch(net, net_free, P_network=net.pores('outer'), P_donor=net_free.Ps, len_max=80, method='nearest')
    #net['throat.interconnection'][-37:] = True
    tt.trim(network=net, throats=net.throats('trim'))
    Ps = net.pores('free_stream')
    nPs = np.asarray(net.find_neighbor_pores(Ps, flatten=False)).flatten()
    net['pore.arc_index'][Ps] = net['pore.arc_index'][nPs]
    plot_domain(net)
    net['pore.cell_id'][net.pores('free_stream')] = -1
    
    #wrk.clear()
    #wrk.load_project(filename=os.path.join(path, 'MJ141-mid-top'))
    #prj = wrk['sim_01']
    #net = prj.network
    max_id = net['pore.cell_id'][net.pores('cc_b')].max()
    data_a = pd.DataFrame({'cell_id': net['pore.cell_id'][net.pores('cc_a')],
                           'pore_index': net.pores('cc_a')})
    data_b = pd.DataFrame({'cell_id': net['pore.cell_id'][net.pores('cc_b')],
                           'pore_index': net.pores('cc_b')})
    data_a = data_a.sort_values(by=['cell_id'])
    data_b = data_b.sort_values(by=['cell_id'])
    #
    #interconns = np.vstack((list(data_a.pore_index), list(data_b.pore_index))).T
    #new_coords = (net['pore.coords'][interconns[:, 0]] + net['pore.coords'][interconns[:, 1]])/2
    #all_Ts = net.find_neighbor_throats(net.Ps, flatten=False)
    #for i, pore in enumerate(data_b.pore_index[:37]):
    #    Ts = all_Ts[pore]
    #    inter_Ts = Ts[net['throat.interconnection'][Ts]]
    #    print(net['pore.cell_id'][pore], net['pore.cell_id'][net['throat.conns'][Ts]] - i)

make_new_layers = False
if make_new_layers:
    Ts = net.throats('interconnection')
    tt.trim(network=net, throats=Ts)
    vec = net['pore.coords'][data_a.pore_index] - net['pore.coords'][data_b.pore_index]
    cc2cc = np.linalg.norm(vec, axis=1)
    mean_cc2cc = np.mean(cc2cc)
    num_new_layers = 17
    for nlayer in range(num_new_layers):
        new_layer = []
        frac = (nlayer+1)/(num_new_layers+1)
        Np0 = net.Np
        vec = net['pore.coords'][data_a.pore_index] - net['pore.coords'][data_b.pore_index]
        new_coords = net['pore.coords'][data_b.pore_index] + vec*frac
        tt.extend(net, pore_coords=new_coords, labels='layer_'+str(nlayer))
        net['pore.cell_id'][-len(new_coords):] = data_b.cell_id
        net['pore.arc_index'][-len(new_coords):] = net['pore.arc_index'][data_b.pore_index]
        net['pore.theta'][-len(new_coords):] = net['pore.theta'][data_b.pore_index]
    #    net['pore.radial_position'][-len(new_coords):] = np.linalg.norm(net['pore.coords'][-len(new_coords):] - mhs, axis=1)
        # Same cell conns
        if nlayer == num_new_layers - 1:
            new_conns = np.vstack((data_a.pore_index, np.arange(0, len(new_coords), dtype=int)+Np0)).T
            tt.extend(net, throat_conns=new_conns, labels='interconnection')
        new_layer = new_layer + list(np.arange(0, len(new_coords), dtype=int)+Np0)
        # Back-layer conns
        if nlayer == 0:
            new_conns = np.vstack((data_b.pore_index, np.arange(0, len(new_coords), dtype=int)+Np0)).T
            tt.extend(net, throat_conns=new_conns, labels='interconnection')
        new_conns = np.vstack((np.arange(0, len(new_coords)-1, dtype=int)+Np0, np.arange(1, len(new_coords), dtype=int)+Np0)).T
        tt.extend(net, throat_conns=new_conns, labels='layer_'+str(nlayer))
        # Mirror
        tmp_b = data_b.pore_index[36:]
        tmp_a = np.roll(data_a.pore_index, 36)[36:]
        Np0 = net.Np
        vec = net['pore.coords'][tmp_a] - net['pore.coords'][tmp_b]
        vnorm = np.linalg.norm(vec, axis=1)
        new_coords = net['pore.coords'][tmp_b] + vec*frac
        tt.extend(net, pore_coords=new_coords, labels='layer_'+str(nlayer)+'_m')
        net['pore.cell_id'][-len(new_coords):] = data_b.cell_id[36:]
        net['pore.arc_index'][-len(new_coords):] = net['pore.arc_index'][tmp_b]
        net['pore.theta'][-len(new_coords):] = net['pore.theta'][tmp_b]
    #    net['pore.radial_position'][-len(new_coords):] = np.linalg.norm(net['pore.coords'][-len(new_coords):] - mhs, axis=1)
        # Same cell conns
        if nlayer == num_new_layers - 1:
            new_conns = np.vstack((tmp_a, np.arange(0, len(new_coords), dtype=int)+Np0)).T
            tt.extend(net, throat_conns=new_conns, labels='interconnection')
        # Back-layer conns
        if nlayer == 0:
            new_conns = np.vstack((tmp_b, np.arange(0, len(new_coords), dtype=int)+Np0)).T
            tt.extend(net, throat_conns=new_conns, labels='interconnection')
        new_conns = np.vstack((np.arange(0, len(new_coords)-1, dtype=int)+Np0, np.arange(1, len(new_coords), dtype=int)+Np0)).T
        tt.extend(net, throat_conns=new_conns, labels='layer_'+str(nlayer)+'_m')
        new_layer = new_layer + list(np.arange(0, len(new_coords), dtype=int)+Np0)
        # Inter layer conns
        if nlayer > 0:
            new_layer = np.asarray(new_layer)
            new_conns = np.vstack((new_layer, new_layer-len(new_layer))).T
            tt.extend(net, throat_conns=new_conns, labels='interconnection')
    
#plot_domain(net)
prj = wrk['sim_01']
net['pore.coords'] *= pixel_size
mean = mhs * pixel_size
net['pore.coords'][:, 0] -= mean
net['pore.coords'][:, 1] -= mean
net['pore.radial_position'] = np.linalg.norm(net['pore.coords'], axis=1)
net['throat.radial_position'] = net.interpolate_data('pore.radial_position')
net['throat.arc_length'] = net['throat.radial_position']*np.deg2rad(dtheta)
#net['pore.mirror'] = False
#net['pore.mirror'][net.pores('*_m')] = True
#net['throat.separator'] = False
#Ts = net.throats(["layer_7", "layer_7_m", "layer_8", "layer_8_m"])
#net['throat.separator'][Ts] = True
prj.export_data(filename='tomography_m_new')
wrk.save_project(project=prj, filename=os.path.join(path, 'MJ141-mid-top_m_cc_new'))
#tt.plot_coordinates(net, net.Ps, c=net['pore.cell_id'])

#show_Ps = np.in1d(net['pore.cell_id'], np.arange(0, 650, 10))
#tt.plot_coordinates(net, net.Ps[show_Ps], c=net['pore.cell_id'][show_Ps])
