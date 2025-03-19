# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:32:23 2020

@author: Tom
"""
import os
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import openpnm as op
import math
from matplotlib import cm
import configparser
import matplotlib.animation as animation
from matplotlib import gridspec
import matplotlib.ticker as mtick
from string import ascii_lowercase as abc
import jellybamm


prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]


wrk = op.Workspace()
cmap = cm.inferno


def get_saved_var_names():
    saved_vars = [
        "var_Current_collector_current_density",
        "var_Temperature",
        "var_Terminal_voltage",
        "var_X-averaged_negative_electrode_extent_of_lithiation",
        "var_X-averaged_positive_electrode_extent_of_lithiation",
        "var_X-averaged_negative_particle_surface_concentration",
        "var_X-averaged_positive_particle_surface_concentration",
        "var_Volume-averaged_total_heating",
        "var_ECM_I_Local",
        "var_ECM_R_local",
        "var_Time",
        "eta_Change_in_measured_open_circuit_voltage",
        "eta_X-averaged_battery_reaction_overpotential",
        "eta_X-averaged_battery_concentration_overpotential",
        "eta_X-averaged_battery_electrolyte_ohmic_losses",
        "eta_X-averaged_battery_solid_phase_ohmic_losses",
        "var_Volume-averaged_Ohmic_heating",
        "var_Volume-averaged_irreversible_electrochemical_heating",
        "var_Volume-averaged_reversible_heating",
        "var_Volume-averaged_Ohmic_heating_CC",
    ]
    return saved_vars


def get_saved_var_units():
    units = [
        "A.m-2",
        "K",
        "V",
        "-",
        "-",
        "mol.m-3",
        "mol.m-3",
        "W.m-3",
        "A",
        "Ohm",
        "h",
        "V",
        "V",
        "V",
        "V",
        "V",
        "W.m-3",
        "W.m-3",
        "W.m-3",
        "W.m-3",
    ]
    return units


def plot_phase_data(project, data="pore.temperature"):
    net = project.network
    phase = project.phases()["phase_01"]
    Ps = net.pores("free_stream", mode="not")
    coords = net["pore.coords"]
    x = coords[:, 0][Ps]
    y = coords[:, 1][Ps]
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.scatter(x, y, c=phase[data][Ps])
    ax = fig.gca()
    ax.set_xlim(x.min() * 1.05, x.max() * 1.05)
    ax.set_ylim(y.min() * 1.05, y.max() * 1.05)


def get_amp_cases(filepath):
    amps = [float(file.strip("A")) for file in os.listdir(filepath) if "A" in file]
    return amps


def load_and_amalgamate(save_root, var_name):
    try:
        file_lower = os.path.join(save_root, var_name + "_lower")
        file_upper = os.path.join(save_root, var_name + "_upper")
        data_lower = io.loadmat(file_lower)["data"]
        data_upper = io.loadmat(file_upper)["data"]
        data_amalg = np.hstack((data_lower, data_upper))
    except KeyError:
        data_amalg = None
    return data_amalg


def format_label(i):
    saved_vars = get_saved_var_names()
    units = get_saved_var_units()
    label = saved_vars[i]
    var_axis_name = label.replace("var_", "")
    var_axis_name = var_axis_name.replace("eta_", "")
    var_axis_name = var_axis_name.replace("_", " ")
    var_axis_name = var_axis_name + " [" + units[i] + "]"
    return var_axis_name


def config2dict(config):
    temp = {}
    for sec in config.sections():
        temp[sec] = {}
        for key in config[sec]:
            temp[sec][key] = config.get(sec, key)
    return temp


def compare_config(config_a, config_b):
    for sec in config_a.keys():
        for option in config_a[sec].keys():
            if "i_app" not in option:
                opt_a = config_a[sec][option]
                opt_b = config_b[sec][option]
                if opt_a != opt_b:
                    print(sec, option, opt_a, opt_b)


def load_cases(filepath):
    d = {}
    for file in os.listdir(filepath):
        d[file] = load_data(os.path.join(filepath, file))
    return d


def load_data(filepath):
    config = configparser.ConfigParser()
    net = get_net(filepath=filepath, filename="net.pnm")
    weights = get_weights(net)
    amps = get_amp_cases(filepath)
    variables = get_saved_var_names()
    data = {}
    config.read(os.path.join(filepath, "config.txt"))
    data["config"] = config2dict(config)
    data["network"] = net
    for amp in amps:
        amp_folder = os.path.join(filepath, str(amp) + "A")
        data[amp] = {}
        for vi, v in enumerate(variables):
            data[amp][vi] = {}
            temp = load_and_amalgamate(amp_folder, v)
            if temp is not None:
                if vi == 0:
                    check_nans = np.any(np.isnan(temp), axis=1)
                    if np.any(check_nans):
                        print("Nans removed from", amp_folder)
                if np.any(check_nans):
                    temp = temp[~check_nans, :]
                data[amp][vi]["data"] = temp
                means = np.zeros(temp.shape[0])
                for t in range(temp.shape[0]):
                    (mean, std_dev) = weighted_avg_and_std(temp[t, :], weights)
                    means[t] = mean
                data[amp][vi]["mean"] = means
                data[amp][vi]["min"] = np.min(temp, axis=1)
                data[amp][vi]["max"] = np.max(temp, axis=1)
        if temp is not None:
            t_hrs = data[amp][10]["data"][:, 0]
            cap = t_hrs * amp
            data[amp]["capacity"] = cap
    return data


def get_net(filepath=None, filename="spider_net.pnm"):
    if filepath is None:
        filepath = jellybamm.INPUT_DIR
    wrk.load_project(os.path.join(filepath, filename))
    sim_name = list(wrk.keys())[-1]
    project = wrk[sim_name]
    net = project.network
    return net


def get_weights(net):
    weights = net["throat.arc_length"][net.throats("spm_resistor")]
    return weights


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, math.sqrt(variance))


def min_mean_max_subplot(
    data,
    case=0,
    amp=4,
    var=0,
    normed=False,
    c="k",
    ax=None,
    print_amps=False,
    show="all",
    time_cap="Time",
):
    if ax is None:
        fig, ax = plt.subplots()
    cap = data[case][amp]["capacity"].copy()
    if time_cap == "Time":
        cap /= amp
    dmin = data[case][amp][var]["min"]
    dmean = data[case][amp][var]["mean"]
    dmax = data[case][amp][var]["max"]
    lab = "Case " + case
    if print_amps:
        lab += ": I = " + str(amp) + "[A]"
    if normed:
        if show == "all" or show == "min":
            ax.plot(cap, dmin / dmean, c=c, linestyle="dashed")
        if show == "all" or show == "mean":
            ax.plot(cap, dmean / dmean, c=c, label=lab)
        if show == "all" or show == "max":
            ax.plot(cap, dmax / dmean, c=c, linestyle="dashed")
    else:
        if show == "all" or show == "min":
            ax.plot(cap, dmin, c=c, linestyle="dashed")
        if show == "all" or show == "mean":
            ax.plot(cap, dmean, c=c, label=lab)
        if show == "all" or show == "max":
            ax.plot(cap, dmax, c=c, linestyle="dashed")
    ax.set
    return ax


def chargeogram(data, case_list, amp_list, group="neg"):
    wrk.clear()
    net = data[case_list[0]]["network"]
    nrows = len(case_list)
    ncols = len(amp_list)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(int(5 * ncols), int(5 * nrows)),
        sharex=True,
        sharey=False,
    )
    var = 0  # Current density
    Ts = net.throats("spm_" + group + "_inner")
    roll_pos = np.cumsum(net["throat.arc_length"][Ts])
    norm_roll_pos = roll_pos / roll_pos[-1]
    Nspm = net.num_throats("spm_resistor")
    if group == "neg":
        spm_ids = np.arange(Nspm)[: len(Ts)]
    else:
        spm_ids = np.arange(Nspm)[-len(Ts) :]
    for ci, case in enumerate(case_list):
        for ai, amp in enumerate(amp_list):
            ax = axes[ci][ai]
            data_amalg = data[case][amp][var]["data"].copy()
            mean = data[case][amp][var]["mean"][0]
            data_amalg /= mean
            data_amalg *= 100
            filtered_data = data_amalg[:, spm_ids]
            fmin = int(np.floor(filtered_data.min())) - 1
            fmax = int(np.ceil(filtered_data.max())) + 1
            nbins = fmax - fmin
            data_2d = np.zeros([len(spm_ids), nbins], dtype=float)
            for i in range(len(spm_ids)):
                hdata, bins = np.histogram(
                    filtered_data[:, i], bins=nbins, range=(fmin, fmax), density=True
                )
                data_2d[i, :] = hdata * 100

            x_data, y_data = np.meshgrid(norm_roll_pos, (bins[:-1] + bins[1:]) / 2)
            heatmap = data_2d.astype(float)
            heatmap[heatmap == 0.0] = np.nan
            im = ax.pcolormesh(x_data, y_data - 100, heatmap.T, cmap=cm.inferno)
            ax.set_title(case + ": " + str(amp) + "[A]")
            if ci == len(case_list) - 1:
                ax.set_xlabel("Normalized roll position")
            plt.colorbar(im, ax=ax)
            ax.grid(True)

    return fig


def spacetime(data, case_list, amp_list, var=0, group="neg", normed=False):
    wrk.clear()
    net = data[case_list[0]]["network"]
    nrows = len(amp_list)
    ncols = len(case_list)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(int(5 * ncols), int(5 * nrows)), sharex=True, sharey=True
    )
    Ts = net.throats("spm_" + group + "_inner")
    roll_pos = np.cumsum(net["throat.arc_length"][Ts])
    norm_roll_pos = roll_pos / roll_pos[-1]
    Nspm = net.num_throats("spm_resistor")
    if group == "neg":
        spm_ids = np.arange(Nspm)[: len(Ts)]
    else:
        spm_ids = np.arange(Nspm)[-(len(Ts)) :]
    ax_list = []
    x_list = []
    y_list = []
    data_list = []
    for ci, case in enumerate(case_list):
        for ai, amp in enumerate(amp_list):
            if nrows > 1:
                ax = axes[ci][ai]
            else:
                ax = axes[ci]
            data_amalg = data[case][amp][var]["data"].copy()
            ax_list.append(ax)
            cap = data[case][amp]["capacity"]
            if normed:
                mean = data[case][amp][var]["mean"][0]
                data_amalg /= mean
                data_amalg *= 100
                data_amalg -= 100
            filtered_data = data_amalg[:, spm_ids]
            x_data, y_data = np.meshgrid(norm_roll_pos, cap)
            heatmap = filtered_data.astype(float)
            heatmap[heatmap == 0.0] = np.nan
            x_list.append(x_data)
            y_list.append(y_data)
            data_list.append(heatmap)
            im = ax.pcolormesh(x_data, y_data, heatmap, cmap=cm.inferno)
            ax.set_title(case)
            if ai == 0:
                ax.set_ylabel("Capacity [Ah]")
            if (ci == len(case_list) - 1) or nrows == 1:
                ax.set_xlabel("Normalized Roll Position")
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.locator_params(nbins=6)
    return fig


def add_figure_label(ax, index):
    t = ax.text(-0.1, 1.15, abc[index], transform=ax.transAxes, fontsize=14, va="top")
    t.set_bbox(dict(facecolor="white", alpha=1.0, edgecolor="black"))


def stacked_variables(data, case, amp, var_list=[0, 1, 2, 3], ax=None, subi=0):
    net = data[case]["network"]
    spm_vol = net["throat.volume"][net["throat.spm_resistor"]]
    # To do - make this much more robust = replace integers with key
    Q_ohm = data[case][amp][16]["data"]
    Q_irr = data[case][amp][17]["data"]
    Q_rev = data[case][amp][18]["data"]
    Q_ohm_cc = data[case][amp][19]["data"]
    nt, nspm = Q_ohm.shape
    spm_vol_t = np.tile(spm_vol[:, np.newaxis], nt).T
    sum_Q_ohm = np.sum(Q_ohm * spm_vol_t, axis=1)
    sum_Q_irr = np.sum(Q_irr * spm_vol_t, axis=1)
    sum_Q_rev = np.sum(Q_rev * spm_vol_t, axis=1)
    sum_Q_ohm_cc = np.sum(Q_ohm_cc * spm_vol_t, axis=1)
    cmap = cm.inferno
    base = np.zeros(len(sum_Q_ohm))
    cols = cmap(np.linspace(0.1, 0.9, 4))
    labels = []
    for i in [18, 17, 16, 19]:
        text = format_label(i).strip("X-averaged").strip("[W.m-3]")
        labels.append(text.lstrip().rstrip().capitalize())
    for si, source in enumerate([sum_Q_rev, sum_Q_irr, sum_Q_ohm, sum_Q_ohm_cc]):
        ax.fill_between(
            data[case][amp][10]["mean"],
            base,
            base + source,
            color=cols[si],
            label=labels[si],
        )
        base += source
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Total Heat Produced [W]")
    ax.grid()
    add_figure_label(ax, subi)
    plt.legend()
    subi += 1
    return ax


def plot_resistors(net, throats, color, ax):
    conns = net["throat.conns"][throats]
    coords = net["pore.coords"]
    v = coords[conns[:, 1]] - coords[conns[:, 0]]
    z = np.array([0, 0, 1])
    perp = np.cross(v, z)
    zigzag = np.array(
        [0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0]
    )
    segs = len(zigzag)
    p_start = coords[conns[:, 0]]
    x_all = [p_start[:, 0]]
    y_all = [p_start[:, 1]]
    for i in range(segs):
        p_end = p_start + v * (1 / segs) + perp * (2 / segs) * zigzag[i]
        x_all.append(p_end[:, 0])
        y_all.append(p_end[:, 1])
        p_start = p_end
    x_all = np.asarray(x_all)
    y_all = np.asarray(y_all)
    ax.plot(x_all, y_all, color=color)
    return ax


def super_subplot(data, cases_left, cases_right, amp):
    nrows = 3
    ncols = 2
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(int(4 * ncols), int(3 * nrows)),
        sharex=True,
        sharey=False,
    )
    # Top row is current density
    var = 0
    row_num = 0
    ax = axes[row_num][0]
    ncolor = len(cases_left)
    col_array = cmap(np.linspace(0.1, 0.9, ncolor))[::-1]
    for cindex, case in enumerate(cases_left):
        ax = min_mean_max_subplot(
            data,
            case,
            amp,
            var,
            normed=False,
            c=col_array[cindex],
            ax=ax,
            print_amps=False,
        )
        ax.grid()
    ax.set_xlabel("Time [h]")
    ax.set_ylabel(format_label(var))
    ax.legend()
    ax.grid()
    add_figure_label(ax, 0)
    ax = axes[row_num][1]
    for cindex, case in enumerate(cases_right):
        ax = min_mean_max_subplot(
            data,
            case,
            amp,
            var,
            normed=False,
            c=col_array[cindex],
            ax=ax,
            print_amps=False,
        )
        ax.grid()
    ax.set_xlabel("Time [h]")
    ax.set_ylabel(format_label(var))
    ax.legend()
    ax.grid()
    add_figure_label(ax, 1)
    # 2nd row is temperature
    var = 1
    row_num = 1
    ax = axes[row_num][0]
    ncolor = len(cases_left)
    col_array = cmap(np.linspace(0.1, 0.9, ncolor))[::-1]
    for cindex, case in enumerate(cases_left):
        ax = min_mean_max_subplot(
            data,
            case,
            amp,
            var,
            normed=False,
            c=col_array[cindex],
            ax=ax,
            print_amps=False,
        )
        ax.grid()
    ax.set_xlabel("Time [h]")
    ax.set_ylabel(format_label(var))
    ax.legend()
    ax.grid()
    add_figure_label(ax, 2)
    ax = axes[row_num][1]
    for cindex, case in enumerate(cases_right):
        ax = min_mean_max_subplot(
            data,
            case,
            amp,
            var,
            normed=False,
            c=col_array[cindex],
            ax=ax,
            print_amps=False,
        )
        ax.grid()
    ax.set_xlabel("Time [h]")
    ax.set_ylabel(format_label(var))
    ax.legend()
    ax.grid()
    add_figure_label(ax, 3)
    plt.ticklabel_format(axis="y", style="sci")
    ax = axes[2][0]
    stacked_variables(data, cases_left[0], amp, [18, 17, 16, 19], ax, 4)
    ax = axes[2][1]
    stacked_variables(data, cases_right[0], amp, [18, 17, 16, 19], ax, 5)
    plt.tight_layout()


def combined_subplot(
    data, case_list, amp_list, var=0, normed=False, ax=None, legend=False
):
    if ax is None:
        fig, ax = plt.subplots()
    ncolor = len(case_list) * len(amp_list)
    col_array = cmap(np.linspace(0.1, 0.9, ncolor))[::-1]
    print_amps = len(amp_list) > 1
    cindex = 0
    for case in case_list:
        for amp in amp_list:
            ax = min_mean_max_subplot(
                data,
                case,
                amp,
                var,
                normed,
                c=col_array[cindex],
                ax=ax,
                print_amps=print_amps,
            )
            cindex += 1

    ax.set_xlabel("Time [h]")
    ax.set_ylabel(format_label(var))
    if legend:
        ax.legend()


def multi_var_subplot(
    data, case_list, amp_list, var_list, normed=False, landscape=True, nplot=None
):
    if nplot is None:
        nplot = len(var_list)
    if landscape:
        nrows = 1
        ncols = nplot
    else:
        nrows = nplot
        ncols = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(int(6 * ncols), int(4 * nrows)))
    subi = 0
    for vi in range(len(var_list)):
        ax = axes[vi]
        var = var_list[vi]
        legend = True
        combined_subplot(
            data, case_list, amp_list, var=var, normed=normed, ax=ax, legend=legend
        )
        t = ax.text(
            -0.1, 1.15, abc[subi], transform=ax.transAxes, fontsize=14, va="top"
        )
        t.set_bbox(dict(facecolor="white", alpha=1.0, edgecolor="black"))
        ax.grid()
        subi += 1
    plt.tight_layout()
    return fig, axes


def animate_init():
    pass


def animate_data4(data, case, amp, variables=None, filename=None):
    net = data[case]["network"]
    weights = get_weights(net)
    project = net.project
    im_spm_map = np.load(os.path.join(jellybamm.INPUT_DIR, "im_spm_map.npz"))["arr_0"]
    title = filename.split("\\")
    if len(title) == 1:
        title = title[0]
    else:
        title = title[-1]
    plot_left = format_label(variables[0])
    plot_right = format_label(variables[1])
    fig = setup_animation_subplots(plot_left, plot_right)
    mask = np.isnan(im_spm_map)
    if ~np.any(mask):
        mask = im_spm_map == -1
    spm_map_copy = im_spm_map.copy()
    spm_map_copy[np.isnan(spm_map_copy)] = -1
    spm_map_copy = spm_map_copy.astype(int)
    time_var = "Time [h]"
    time = data[case][amp][10]["mean"]
    vars2plot = {}
    vars2plot[plot_left] = data[case][amp][variables[0]]["data"]
    vars2plot[plot_right] = data[case][amp][variables[1]]["data"]
    func_ani = animation.FuncAnimation(
        fig=fig,
        func=update_multi_animation_subplots,
        frames=time.shape[0],
        init_func=animate_init,
        fargs=(
            fig,
            project,
            vars2plot,
            [plot_left, plot_right],
            spm_map_copy,
            mask,
            time_var,
            time,
            weights,
        ),
    )
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=1, metadata=dict(artist="Tom Tranter"), bitrate=-1)
    if ".mp4" not in filename:
        filename = filename + ".mp4"
    func_ani.save(filename, writer=writer, dpi=300)


def setup_animation_subplots(plot_left, plot_right):
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(3, 2, height_ratios=[12, 1, 4], width_ratios=[1, 1])
    plt.subplot(gs[0, 0])
    plt.subplot(gs[1, 0])
    plt.subplot(gs[2, 0])
    plt.subplot(gs[0, 1])
    plt.subplot(gs[1, 1])
    plt.subplot(gs[2, 1])
    return fig


def update_multi_animation_subplots(
    t, fig, project, variables, plot_vars, spm_map, mask, time_var, time, weights
):
    for i, side in enumerate(["left", "right"]):
        data = variables[plot_vars[i]]
        if i == 0:
            global_range = True
        else:
            global_range = True
        fig = update_animation_subplot(
            t,
            fig,
            data,
            plot_vars[i],
            spm_map,
            mask,
            time_var,
            time,
            weights,
            side=side,
            global_range=global_range,
        )


def update_animation_subplot(
    t,
    fig,
    data,
    data_name,
    spm_map,
    mask,
    time_var,
    time,
    weights,
    side="left",
    global_range=True,
):
    print("Updating animation " + side + " frame", t)
    if side == "left":
        ax1 = fig.axes[0]
        ax1c = fig.axes[1]
        ax2 = fig.axes[2]
    else:
        ax1 = fig.axes[3]
        ax1c = fig.axes[4]
        ax2 = fig.axes[5]
    ax1.clear()
    ax1c.clear()
    ax2.clear()
    ax1.set(title=data_name)
    ax2.set(xlabel=time_var)

    arr = np.ones_like(spm_map).astype(float)
    t_data = data[t, :]
    arr[~mask] = t_data[spm_map][~mask]
    arr[mask] = np.nan
    gmin = np.min(data[~np.isnan(data)])
    gmax = np.max(data[~np.isnan(data)])
    if global_range:
        vmin = np.min(data)
        vmax = np.max(data)
    else:
        vmin = np.min(data[t, :])
        vmax = np.max(data[t, :])
    im = ax1.imshow(arr, vmax=vmax, vmin=vmin, cmap=cm.inferno)

    cbar = plt.colorbar(im, cax=ax1c, orientation="horizontal", format="%.2f")
    cbar.ax.locator_params(nbins=6)
    ax2.plot(time, np.max(data, axis=1), "k--")
    ax2.plot(time, np.min(data, axis=1), "k--")
    means = np.zeros(data.shape[0])
    std_devs = np.zeros(data.shape[0])
    if weights is None:
        weights = np.ones_like(data[0, :])
    for _t in range(data.shape[0]):
        (mean, std_dev) = weighted_avg_and_std(data[_t, :], weights)
        means[_t] = mean
        std_devs[_t] = std_dev
    ax2.plot(time, means, "b--")
    ax2.fill_between(time, means - std_devs, means + std_devs)
    ax2.plot([time[t], time[t]], [vmin, vmax], "r")
    grange = gmax - gmin
    ax2.set_ylim(gmin - grange * 0.05, gmax + grange * 0.05)
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    ax2.yaxis.tick_right()
    if t == 0:
        plt.tight_layout()
    return fig


def jellyroll_subplot(
    data, case, amp, var=0, soc_list=[[0.9, 0.7], [0.5, 0.3]], global_range=False, dp=3
):
    soc_arr = np.asarray(soc_list)
    (nrows, ncols) = soc_arr.shape
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12), sharex=True, sharey=True)
    spm_map = np.load(os.path.join(jellybamm.INPUT_DIR, "im_spm_map.npz"))["arr_0"]
    spm_map_copy = spm_map.copy()
    spm_map_copy[np.isnan(spm_map_copy)] = -1
    spm_map_copy = spm_map_copy.astype(int)
    var_data = data[case][amp][var]["data"]
    mask = np.isnan(spm_map)
    if ~np.any(mask):
        mask = spm_map == -1
    arr = np.ones_like(spm_map).astype(float)
    soc = get_SOC_vs_cap(data, case, amp)[:, 1]
    gmax = -1e12
    gmin = 1e12
    arrs = []
    for ir in range(nrows):
        for ic in range(ncols):
            try:
                ax = axes[ir][ic]
            except TypeError:
                ax = axes
            soc_target = soc_arr[ir][ic]
            t = np.argmin((soc - soc_target) ** 2)
            t_data = var_data[t, :]
            arr[~mask] = t_data[spm_map_copy][~mask]
            arr[mask] = np.nan
            vmin = np.min(var_data[t, :])
            vmax = np.max(var_data[t, :])
            if vmin < gmin:
                gmin = vmin
            if vmax > gmax:
                gmax = vmax
            arrs.append(arr.copy())
    out = []
    for ir in range(nrows):
        for ic in range(ncols):
            try:
                ax = axes[ir][ic]
            except TypeError:
                ax = axes
            soc_target = soc_arr[ir][ic]
            t = np.argmin((soc - soc_target) ** 2)
            arr = arrs.pop(0)
            if global_range:
                im = ax.imshow(arr, vmax=gmax, vmin=gmin, cmap=cmap)
            else:
                im = ax.imshow(arr, cmap=cmap)
            ax.set_axis_off()
            if global_range is False:
                plt.colorbar(im, ax=ax, format="%." + str(dp) + "f")
            ax.set_title("SOC: " + str(np.around(soc[t], 2)))
            out.append(arr)
    fig.tight_layout()
    if global_range:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

    return fig, arr


def get_SOC_vs_cap(data, case, amp):
    var_names = get_saved_var_names()
    i_found = None
    for i, vn in enumerate(var_names):
        if "negative" in vn and "lithiation" in vn:
            i_found = i
    lith = data[case][amp][i_found]["mean"]
    cap = data[case][amp]["capacity"]
    soc = lith - lith[-1]
    soc = soc / soc.max()
    return np.vstack((cap, soc)).T
