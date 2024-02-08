# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:09:24 2020

@author: Tom
"""
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops as rp
import scipy.ndimage as spim
from skimage.morphology import binary_erosion, binary_dilation
from skimage.morphology import medial_axis, disk, square
from skimage.measure import label
import skimage.filters.rank as rank
from skimage.segmentation import flood_fill
import os
import openpnm as op
import openpnm.topotools as tt
import jellybamm


wrk = op.Workspace()


def average_images(path=None):
    if path is None:
        path = jellybamm.INPUT_DIR
    # Get File list
    files = []
    for file in os.listdir(path):
        if file.split(".")[-1] == "tiff":
            files.append(file)
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
        can = binary_dilation(binary_erosion(can, footprint=sel_ero), sel_dil)
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
    mhs = int(np.ceil(np.min([x_span.min(), y_span.min()]) / 2))  # min half span
    # Factorize image size - used for adaptive histogran
    factors = []
    for i in np.arange(2, mhs * 2):
        if mhs * 2 % i == 0.0:
            factors.append(i)
    # Crop images
    crops = []
    for i, (x_min, y_min, x_max, y_max) in enumerate(boxes):
        xl = mids[i, 0] - mhs
        xh = mids[i, 0] + mhs
        yl = mids[i, 1] - mhs
        yh = mids[i, 1] + mhs
        crops.append(ims[i][xl:xh, yl:yh] * masks[i][xl:xh, yl:yh])
        # print(i, crops[-1].shape)
    # Pick image - replace with average
    im = crops[0].copy()
    dt = np.ones(im.shape)
    dt[mhs : mhs + 1, mhs : mhs + 1] = 0
    dt = spim.distance_transform_edt(dt)
    return im, mhs, dt


def get_radial_average(im, step, dt):
    mid = np.floor(np.array(im.shape) / 2).astype(int)
    bands = np.arange(300, mid[0] - 25, step)
    band_avs = []
    r = []
    for i in range(len(bands) - 1):
        lower = bands[i]
        upper = bands[i + 1]
        mask = (dt >= lower) * (dt < upper)
        band_avs.append(np.mean(im[mask]))
        r.append((lower + upper) / 2)
    return np.vstack((r, band_avs)).T


def adjust_radial_average(im, step, deg, dt):
    mid = np.floor(np.array(im.shape) / 2).astype(int)
    bands = np.arange(300, mid[0] - 25, step)
    band_avs = []
    r = []
    for i in range(len(bands) - 1):
        lower = bands[i]
        upper = bands[i + 1]
        mask = (dt >= lower) * (dt < upper)
        band_avs.append(np.mean(im[mask]))
        r.append((lower + upper) / 2)
    dat = np.vstack((r, band_avs)).T
    p = np.polyfit(dat[:, 0], dat[:, 1], deg=deg)
    pfunc = np.poly1d(p)
    adj = pfunc(dat[:, 0])
    adj = adj - adj[0]
    adj_im = im.copy().astype(float)
    for i in range(len(bands) - 1):
        lower = bands[i]
        upper = bands[i + 1]
        mask = (dt >= lower) * (dt < upper)
        adj_im[mask] -= adj[i]
    return adj_im


def remove_beam_hardening(im, dt, step, deg):
    # Compute radial averages and adjustments for beam hardening - soften image
    dat_im = jellybamm.get_radial_average(im, step, dt)
    im_soft = jellybamm.adjust_radial_average(im, step, deg, dt)
    dat_soft = jellybamm.get_radial_average(im_soft, step, dt)
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].imshow(im_soft - im.astype(float))
    axes[1].plot(dat_im[:, 0], dat_im[:, 1])
    axes[1].plot(dat_soft[:, 0], dat_soft[:, 1])
    return im_soft


def label_layers(im, dt, mhs, can_width=30, im_thresh=19000, small_feature_size=20000):
    # Remove outer
    outer = dt > mhs
    # Remove Can
    can = dt > mhs - can_width
    im[outer] = 0
    im[can] = 0
    # Final layer labels
    layers = np.zeros(can.shape, dtype=int)
    layers[can] = layers.max() + 1
    # Threshold image
    mask = im > im_thresh
    plt.figure()
    plt.imshow(mask)
    # Label threshold - Should result in 2 or 3 regions - if not look at threshold
    lab = label(mask)
    plt.imshow(lab)
    # Count Labels and remove any small ones
    itmfrq = np.unique(lab, return_counts=True)
    itmfrq = np.vstack(itmfrq).T
    itmfrq = itmfrq[1:]
    # Get rid of small and medium size edge features
    itmfrq = itmfrq[itmfrq[:, 1] > small_feature_size]
    cc_im = np.zeros(lab.shape, dtype=np.int8)

    for l_int, _ in itmfrq:
        if l_int > 0:
            tmp = lab == l_int
            layers[tmp] = layers.max() + 1
            tmp = binary_erosion(binary_dilation(tmp, disk(6)), disk(4))
            skel = medial_axis(tmp, return_distance=False)
            skel = skel.astype(np.uint8)
            # sum the values in the kernel to identify joints and ends
            skel_rank_sum = rank.sum(skel, footprint=square(3))
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
            cc_im[tmp2 > 0] = l_int
            layers[tmp2 > 0] = layers.max() + 1
            plt.figure()
            plt.imshow(tmp2)

    # get inner section
    cc_all = cc_im > 0
    cc_all = binary_dilation(cc_all, disk(8))
    plt.figure()
    plt.imshow(cc_all)
    lab = label(1 - cc_all)
    plt.figure()
    plt.imshow(lab)
    inner_lab = lab[mhs, mhs]
    mask_nan = lab == inner_lab
    im[mask_nan] = 0
    plt.figure()
    im[im == 0] = np.nan
    plt.imshow(im)
    return im, cc_im


def spider_web_network(
    im_soft, mhs, cc_im, dtheta=10, pixel_size=10.4e-6, path=None, filename="spider_net"
):
    # Make spiderweb dividing lines
    (inds_x, inds_y) = np.indices(im_soft.shape)
    inds_x -= mhs
    inds_y -= mhs
    theta = np.around(np.arctan2(inds_y, inds_x) * 360 / (2 * np.pi), 2)
    remainder = theta % dtheta
    lines = np.logical_or((remainder < 1.0), (remainder > (dtheta - 1.0)))
    med_ax = medial_axis(lines)
    plt.figure()
    plt.imshow(med_ax)
    med_ax = binary_dilation(med_ax, footprint=square(2))
    # Multiply cc image by -1 along the dividing lines to define nodes
    cc_im[med_ax] *= -1
    # Extract coordinates of nodes on the current collectors
    cc_coords = []
    for i in np.unique(cc_im)[np.unique(cc_im) < 0]:
        lab, N = label(cc_im == i, return_num=True)
        coords = np.zeros([N, 2], dtype=int)
        for l_int in np.arange(N):
            x, y = np.where(lab == l_int + 1)
            if len(x) > 0:
                coords[l_int] = [
                    int(np.around(np.mean(x), 0)),
                    int(np.around(np.mean(y), 0)),
                ]
        cc_coords.append(coords)
    plt.figure()
    plt.imshow(cc_im)
    plt.scatter(cc_coords[0][:, 1], cc_coords[0][:, 0], c="r", s=50)
    plt.scatter(cc_coords[1][:, 1], cc_coords[1][:, 0], c="b", s=50)
    # Collect coords together
    coords = np.vstack((cc_coords[0], cc_coords[1]))
    # Record which belongs to same cc
    first_cc = np.zeros(len(coords), dtype=bool)
    first_cc[: len(cc_coords[0])] = True
    # Find which theta bin coords are in
    point_thetas = theta[coords[:, 0], coords[:, 1]]
    bins = np.arange(-180, 180 + 2 * dtheta, dtheta) - dtheta / 2
    point_group_theta = np.zeros(len(coords[:, 0]), dtype=int)
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]
        sel = (point_thetas > lower) * (point_thetas < upper)
        point_group_theta[sel] = i
    # Get radial position of coords
    rads = np.linalg.norm(coords - mhs, axis=1)

    groups, counts = np.unique(point_group_theta, return_counts=True)
    sorted_groups = np.zeros([len(groups), counts.max()], dtype=int)
    sorted_groups.fill(-1)
    for g in range(len(groups)):
        Ps = np.arange(0, len(point_group_theta), 1)[point_group_theta == g]
        group_rads = rads[Ps]
        sorted_rads = group_rads.argsort()
        sorted_Ps = Ps[sorted_rads]
        sorted_groups[g, : len(sorted_Ps)] = sorted_Ps
    plt.figure()
    cc_groups = first_cc[sorted_groups].astype(int)
    cc_groups[sorted_groups == -1] = -1
    plt.imshow(cc_groups)
    inner_Ps = sorted_groups[:, 0]
    inner_cc = first_cc[inner_Ps]
    change_cc = inner_cc != np.roll(inner_cc, -1)
    sorted_cc = first_cc[sorted_groups].astype(int)

    arc_indices = []
    cc_roll = [[], []]
    for cc_num, this_cc in enumerate([True, False]):
        start_group = np.argwhere(
            np.logical_and(inner_cc == this_cc, change_cc)
        ).flatten()[0]
        layer_group = np.arange(0, len(groups), 1)
        layer_group = np.roll(layer_group, -start_group)
        i = 0
        layer = 0
        numbered_cc = first_cc[sorted_groups].astype(int)
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
    print(len(np.unique(cc_roll[0] + cc_roll[1])))
    ordered_neg_Ps = np.asarray(cc_roll[0])
    ordered_pos_Ps = np.asarray(cc_roll[1])
    neg_coords = coords[ordered_neg_Ps]
    neg_conns = np.vstack(
        (
            np.arange(0, len(ordered_neg_Ps) - 1, 1),
            np.arange(0, len(ordered_neg_Ps) - 1, 1) + 1,
        )
    ).T
    pos_coords = coords[ordered_pos_Ps]
    pos_conns = np.vstack(
        (
            np.arange(0, len(ordered_pos_Ps) - 1, 1),
            np.arange(0, len(ordered_pos_Ps) - 1, 1) + 1,
        )
    ).T
    pos_conns += len(ordered_neg_Ps)
    neg_inner_conns = []
    for i, p_neg in enumerate(ordered_neg_Ps):
        g, layer = np.argwhere(sorted_groups == p_neg).flatten()
        if layer < counts.max() - 1:
            neighbor = sorted_groups[g, layer + 1]
            if neighbor > -1:
                sorted_neighbor = np.argwhere(ordered_pos_Ps == neighbor).flatten()
                if len(sorted_neighbor) > 0:
                    sorted_neighbor = sorted_neighbor[0]
                    neg_inner_conns.append([i, sorted_neighbor + len(ordered_neg_Ps)])
    pos_inner_conns = []
    for i, p_pos in enumerate(ordered_pos_Ps):
        g, layer = np.argwhere(sorted_groups == p_pos).flatten()
        if layer < counts.max() - 1:
            neighbor = sorted_groups[g, layer + 1]
            if neighbor > -1:
                sorted_neighbor = np.argwhere(ordered_neg_Ps == neighbor).flatten()
                if len(sorted_neighbor) > 0:
                    sorted_neighbor = sorted_neighbor[0]
                    pos_inner_conns.append([i + len(ordered_neg_Ps), sorted_neighbor])
    neg_inner_conns = np.asarray(neg_inner_conns)
    pos_inner_conns = np.asarray(pos_inner_conns)

    new_coords = np.vstack((neg_coords, pos_coords))
    coords_3d = np.vstack(
        (new_coords[:, 0], new_coords[:, 1], np.zeros(new_coords.shape[0]))
    ).T
    new_conns = np.vstack((neg_conns, pos_conns, neg_inner_conns, pos_inner_conns))
    net = op.network.GenericNetwork(conns=new_conns, coords=coords_3d)

    Ps, counts = np.unique(
        np.hstack((net["throat.conns"][:, 0], net["throat.conns"][:, 1])),
        return_counts=True,
    )
    net["pore.surface"] = False
    net["pore.terminal"] = False
    net["pore.neg_cc"] = False
    net["pore.pos_cc"] = False
    net["pore.inner"] = False
    net["pore.outer"] = False
    net["pore.surface"][Ps[counts < 4]] = True
    net["pore.terminal"][Ps[counts == 2]] = True
    net["pore.neg_cc"][: len(ordered_neg_Ps)] = True
    net["pore.pos_cc"][len(ordered_neg_Ps) :] = True
    net["pore.cell_id"] = np.arange(0, net.Np, 1).astype(int)
    net["pore.arc_index"] = np.asarray(arc_indices)
    net["pore.radial_position"] = np.linalg.norm(new_coords - mhs, axis=1)
    rad_pos = net["pore.radial_position"]
    inner_mask = np.logical_and(net["pore.surface"], rad_pos < mhs / 2)
    outer_mask = np.logical_and(net["pore.surface"], rad_pos > mhs / 2)
    net["pore.inner"][inner_mask] = True
    net["pore.outer"][outer_mask] = True
    net["throat.neg_cc"] = False
    net["throat.pos_cc"] = False

    net["throat.spm_resistor"] = True
    pos_cc_Ts = net.find_neighbor_throats(net.pores("pos_cc"), mode="xnor")
    neg_cc_Ts = net.find_neighbor_throats(net.pores("neg_cc"), mode="xnor")
    net["throat.neg_cc"][neg_cc_Ts] = True
    net["throat.pos_cc"][pos_cc_Ts] = True
    net["throat.spm_resistor"][neg_cc_Ts] = False
    net["throat.spm_resistor"][pos_cc_Ts] = False
    net["throat.spm_resistor_order"] = -1
    spm_res = net["throat.spm_resistor"]
    n_spm = np.sum(spm_res)
    net["throat.spm_resistor_order"][spm_res] = np.arange(0, n_spm, 1, dtype=int)
    net["throat.spm_neg_inner"] = False
    net["throat.spm_pos_inner"] = False
    res_order = net["throat.spm_resistor_order"]
    net["throat.spm_neg_inner"][spm_res * (res_order < len(neg_inner_conns))] = True
    net["throat.spm_pos_inner"][spm_res * (res_order >= len(neg_inner_conns))] = True
    net["pore.neg_tab"] = False
    net["pore.pos_tab"] = False
    net["pore.neg_tab"][net.pores(["inner", "neg_cc", "terminal"], mode="and")] = True
    net["pore.pos_tab"][net.pores(["outer", "pos_cc", "terminal"], mode="and")] = True

    # Add Free Stream Pores
    num_free = net.num_pores("outer")
    outer_pos = net["pore.coords"][net.pores("outer")]
    x = outer_pos[:, 0] - mhs
    y = outer_pos[:, 1] - mhs
    r, t = jellybamm.polar_transform(x, y)
    r_new = np.ones(num_free) * (r + 25)
    new_x, new_y = jellybamm.cartesian_transform(r_new, t)
    free_coords = outer_pos.copy()
    free_coords[:, 0] = new_x + mhs
    free_coords[:, 1] = new_y + mhs
    free_conns = np.vstack((net.pores("outer"), np.arange(0, num_free, 1) + net.Np)).T
    tt.extend(
        net, pore_coords=free_coords, throat_conns=free_conns, labels=["free_stream"]
    )
    free_Ps = net["pore.free_stream"]
    net["pore.arc_index"][free_Ps] = net["pore.arc_index"][net["pore.outer"]]
    net["pore.cell_id"][net.pores("free_stream")] = -1
    net["pore.cell_id"] = net["pore.cell_id"].astype(int)

    # Add Inner Boundary Pores
    num_inner = net.num_pores("inner")
    inner_pos = net["pore.coords"][net.pores("inner")]
    x = inner_pos[:, 0] - mhs
    y = inner_pos[:, 1] - mhs
    r, t = jellybamm.polar_transform(x, y)
    r_new = np.ones(num_inner) * (r - 25)
    new_x, new_y = jellybamm.cartesian_transform(r_new, t)
    inner_coords = inner_pos.copy()
    inner_coords[:, 0] = new_x + mhs
    inner_coords[:, 1] = new_y + mhs
    inner_conns = np.vstack((net.pores("inner"), np.arange(0, num_inner, 1) + net.Np)).T
    tt.extend(
        net,
        pore_coords=inner_coords,
        throat_conns=inner_conns,
        labels=["inner_boundary"],
    )
    inner_Ps = net["pore.inner_boundary"]
    net["pore.arc_index"][inner_Ps] = net["pore.arc_index"][net["pore.inner"]]
    net["pore.cell_id"][net.pores("inner_boundary")] = -1
    net["pore.cell_id"] = net["pore.cell_id"].astype(int)

    # Scale and save net
    prj = wrk["proj_01"]
    net["pore.coords"] *= pixel_size
    mean = mhs * pixel_size
    net["pore.coords"][:, 0] -= mean
    net["pore.coords"][:, 1] -= mean
    net["pore.radial_position"] = np.linalg.norm(net["pore.coords"], axis=1)
    net["throat.radial_position"] = net.interpolate_data("pore.radial_position")
    net["throat.arc_length"] = net["throat.radial_position"] * np.deg2rad(dtheta)

    if path is None:
        path = jellybamm.INPUT_DIR
    wrk.save_project(project=prj, filename=os.path.join(path, filename))
    return net
