# python 3 compatibility
from __future__ import division

import numpy as np
import matplotlib, matplotlib.cm
from mpl_toolkits.axes_grid1 import make_axes_locatable, Grid # make_axes_locatable is for colorbars to scale properly in tight_layout
from matplotlib import gridspec # subplot replacement, more versatile
import mpl_toolkits.axes_grid1 as axes_grid1
from matplotlib import animation
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab! For using pyplot remotely.
import matplotlib.pyplot as pl

# for styled eul/lag grid plots:
import seaborn as sns
import matplotlib.colors  # for ListedColormap

pl.ioff()
import egp.basic_types, egp.toolbox
import egp.cosmography
import os, glob

import barcode_plog_analysis as plog

#gridsize = input("Input gridsize: ")

### Base grid functions:

class PowSpec:
    def __init__(self, filename):
        self.filename = filename
        spectrum = np.loadtxt(filename)
        self.k = spectrum[:,0]
        self.P = spectrum[:,1]

def smooth_grid(grid, rel_smoothing_scale):
    grid_field = egp.basic_types.Field(true = grid)
    grid_field.boxlen = 1.
    grid_smooth_field = egp.toolbox.filter_Field(grid_field, egp.toolbox.gaussian_kernel, (rel_smoothing_scale,))
    grid = grid_smooth_field.t
    return grid

def get_grid(filename, slice_index=0, rel_smoothing_scale=None):
    grid = np.memmap(filename, dtype='float64', mode="c")
    size = grid.shape[0]
    gridsize = int(round(size**(1./3)))
    try:
        grid.shape = (gridsize, gridsize, gridsize)
    except ValueError:
        print("In get_grid: somehow shape isn't right. I derived a gridsize of %i, is that right?" % gridsize)
        raise SystemExit
    if rel_smoothing_scale:
        grid = smooth_grid(grid, rel_smoothing_scale)
    return grid


def get_grid_field(filename, boxsize=1, **kwargs):
    grid = get_grid(filename, **kwargs)
    grid_field = egp.basic_types.Field(true=grid, boxsize=boxsize)
    return grid_field


def get_eul_field(filename, boxsize, **kwargs):
    grid = get_grid(filename, **kwargs)
    grid_field = egp.cosmography.DensityField(boxsize=boxsize, input_type='overdensity', true=grid)
    return grid_field



def vminmax_grid_slice(grid_slice, vmin, vmax, v_ratio):
    if not v_ratio:
        if vmin is None:
            vmin = grid_slice.min()
        if vmax is None:
            vmax = grid_slice.max()
    else:
        if (vmin is None) or (vmax is None):
            raise SystemExit("Must give vmin and vmax when using v_ratio!")
        else:
            # adjust vmin2/vmax2 to same ratio as vmin/vmax
            vmin2 = grid_slice.min()
            vmax2 = grid_slice.max()
            if np.abs(vmin) > vmax:
                vfac = vmin / vmax
                vmin2 = vmax2 * vfac
            else:
                vfac = vmax / vmin
                vmax2 = vmin2 * vfac
            vmin = vmin2
            vmax = vmax2
    return vmin, vmax


def determine_norm(norm_in, vmin, vmax, cmap, log_cbar, vmid=None, clip=True,
                   bottom=-1):
    norm = None

    # if a norm is explicitly given, always use it:
    if norm_in is not None:
        norm = norm_in
    # if not, look if we want a linear or logarithmic colorbar:
    else:
        if cmap is None:
            cmap = matplotlib.cm.get_cmap()

        if log_cbar:    # logarithmic colorbar:
            vbase = 1 - bottom

            vmin_log = np.log10(vbase + vmin)
            vmax_log = np.log10(vbase + vmax)
            if vmid is not None:
                vmid_log = np.log10(vbase + vmid)
            else:
                vmid_log = np.log10(vbase + (vmax+vmin)/2)

            if cmap.N % 2 == 0:  # even number of colors:
                bounds = np.hstack((np.linspace(vmin_log, vmid_log,
                                                cmap.N//2 + 1),
                                    np.linspace(vmid_log, vmax_log,
                                                cmap.N//2 + 1)[1:]))
            else:                # odd number of colors:
                dbound_1 = (vmid_log - vmin_log) / (cmap.N/2.)
                dbound_2 = (vmax_log - vmid_log) / (cmap.N/2.)
                bounds = np.hstack((np.arange(vmin_log, vmid_log, dbound_1),
                                    np.arange(vmax_log, vmid_log,
                                              -dbound_2)[::-1]))
        else:           # linear colorbar:
            if vmid is None:
                vmid = (vmax+vmin)/2
            if cmap.N % 2 == 0:  # even number of colors:
                bounds = np.hstack((np.linspace(vmin, vmid, cmap.N//2 + 1),
                                    np.linspace(vmid, vmax, cmap.N//2 + 1)[1:]))
            else:                # odd number of colors:
                dbound_1 = (vmid - vmin) / (cmap.N/2.)
                dbound_2 = (vmax - vmid) / (cmap.N/2.)
                bounds = np.hstack((np.arange(vmin, vmid, dbound_1),
                                    np.arange(vmax, vmid, -dbound_2)[::-1]))

        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N, clip=clip)

    return norm


def determine_ticks(norm, log_cbar, N_ticks=9, fmt="%3.1f", label_add=0.,
                    label_mult=1.):
    ticks = np.hstack((np.linspace(norm.boundaries[0],
                                   norm.boundaries[norm.N//2],
                                   N_ticks//2 + 1),
                       np.linspace(norm.boundaries[norm.N//2],
                                   norm.boundaries[-1],
                                   N_ticks//2 + 1)[1:]))

    if log_cbar:   # logarithmic colorbar
        ticklabels = [fmt % (label_mult * (10**(tick)-2 + label_add)) for tick in ticks]
    else:          # linear colorbar
        ticklabels = [fmt % (label_mult * (tick + label_add)) for tick in ticks]

    return ticks, ticklabels


def set_colorbar_fontsize(cbar, fontsize=None):
    """
    When `fontsize` is `None`, set the colorbar tick and label font sizes to
    the global `matplotlib.rcParams['font.size']`. This is necessary when
    font sizes are changed during code execution, since matplotlib seems unable
    to update colorbar font sizes itself.
    """
    if fontsize is None:
        fontsize = matplotlib.rcParams['font.size']
    cb_texts = (pl.getp(cbar.ax, 'xticklabels') +
                pl.getp(cbar.ax, 'yticklabels') +
                # pl.getp(cbar.ax, 'xticklines') +
                # pl.getp(cbar.ax, 'yticklines') +
                pl.getp(cbar.ax, 'xmajorticklabels') +
                pl.getp(cbar.ax, 'ymajorticklabels') +
                pl.getp(cbar.ax, 'xminorticklabels') +
                pl.getp(cbar.ax, 'yminorticklabels'))
    pl.setp(cb_texts, fontsize=fontsize)


def slab_plot(slab, axes=None, title="", vmin=None,
              vmax=None, rel_smoothing_scale=None, boxsize=None,
              colorbar=True, fix_colorbar=True, v_ratio=False, cmap=None,
              norm=None, log_cbar=False, vmid=None,
              xlabel="Mpc/h", ylabel="Mpc/h", norm_clip=True,
              cut_x=None, cut_y=None, alpha=1., return_objects=False,
              cblabel=None, xlabel_top=False,
              xticks=True, yticks=True, cax=None, return_norm=False,
              remove_bottom_label=False, remove_left_label=False,
              remove_bottom_cblabel=False, cbar_kwargs={},
              tick_fmt="%3.1f", cb_N_ticks=9, cbar_location='right',
              grey_negative_densities=True, determine_ticks_label_add=0.,
              xlabel_top_set_label_position=False,
              **kwargs):
    """
    cut_x, cut_y: tuple in Mpc/h
    """
    if axes is None:
        fig = pl.figure()
        axes = fig.add_subplot(111)

    if xlabel_top:
        if xlabel_top_set_label_position:
            axes.xaxis.set_label_position('top')
        else:
            axes_top = axes.twiny()
            pl.setp(axes_top.get_xticklabels(), visible=False)
            pl.setp(axes.get_xticklabels(), visible=False)

    vmin, vmax = vminmax_grid_slice(slab, vmin, vmax, v_ratio)

    gridsize_x = slab.shape[1]
    if cut_x is not None:
        x_slice = slice(int(np.floor(cut_x[0]/boxsize * gridsize_x)),
                        int(np.ceil(cut_x[1]/boxsize * gridsize_x)))
    else:
        x_slice = slice(0, gridsize_x)

    gridsize_y = slab.shape[0]
    if cut_y is not None:
        y_slice = slice(int(np.floor(cut_y[0]/boxsize * gridsize_y)),
                        int(np.ceil(cut_y[1]/boxsize * gridsize_y)))
    else:
        y_slice = slice(0, gridsize_y)

    extent = None
    if boxsize is not None:
        if cut_x is not None:
            x_extent = (x_slice.start * boxsize / gridsize_x,
                        x_slice.stop * boxsize / gridsize_x)
        else:
            x_extent = (0, boxsize)
        if cut_y is not None:
            y_extent = (y_slice.start * boxsize / gridsize_y,
                        y_slice.stop * boxsize / gridsize_y)
        else:
            y_extent = (0, boxsize)
        extent = [x_extent[0], x_extent[1], y_extent[0], y_extent[1]]

        if xlabel_top_set_label_position:
            axes.set_xlabel(xlabel)
        else:
            if xlabel_top:
                axes_top.set_xlabel(xlabel)
            else:
                axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)

    if grey_negative_densities:
        bottom = -1
    else:
        bottom = slab.min()

    norm = determine_norm(norm, vmin, vmax, cmap, log_cbar, vmid=vmid,
                          clip=norm_clip, bottom=bottom)

    if not log_cbar:
        im = axes.imshow(slab[y_slice, x_slice],
                         extent=extent, cmap=cmap, norm=norm, alpha=alpha,
                         **kwargs)
    else:
        vbase = 1 - bottom
        im = axes.imshow(np.log10(vbase + slab[y_slice, x_slice]),
                         extent=extent, cmap=cmap, norm=norm, alpha=alpha,
                         **kwargs)

    axes.set_title(title)

    if not boxsize:
        axes.set_axis_off()

    if not xticks:
        pl.setp(axes.get_xticklabels(), visible=False)
    # else:
    #     pl.setp(axes.get_xticklabels(), visible=True)
    if not yticks:
        pl.setp(axes.get_yticklabels(), visible=False)
    # else:
    #     pl.setp(axes.get_yticklabels(), visible=True)

    if colorbar:
        if cax is None and fix_colorbar:
            # Don't use pl.gca() instead of axes!
            divider = make_axes_locatable(axes)
            cax = divider.append_axes(cbar_location, "5%", pad="3%")
        cbar = axes.figure.colorbar(im, cax=cax, **cbar_kwargs)
        cbar.set_clim(vmin, vmax)
        ticks, ticklabels = determine_ticks(norm, log_cbar, fmt=tick_fmt,
                                            N_ticks=cb_N_ticks,
                                            label_add=bottom + 1)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
        if cblabel is not None:
            cbar.set_label(cblabel)
        # for some reason, the current rcParams font.size is not read out for
        # colorbars, so we get it manually:
        set_colorbar_fontsize(cbar)


    if remove_bottom_label:
        bottom_ytick = axes.get_yticklabels()[0]
        pl.setp(bottom_ytick, visible=False)

    if remove_bottom_cblabel and colorbar:
        bottom_cbtick = cax.get_yticklabels()[0]
        pl.setp(bottom_cbtick, visible=False)

    if remove_left_label:
        left_xtick = axes.get_xticklabels()[0]
        pl.setp(left_xtick, visible=False)


    # return stuff
    return_bin = (vmin, vmax)
    if return_objects:
        return_bin += (axes, im)
        if colorbar:
            return_bin += (cbar, cax)
    if return_norm:
        return_bin += (norm,)
    return return_bin



def grid_plot_grid(grid, slice_index=0, **kwargs):
    return slab_plot(grid[slice_index], **kwargs)


def grid_plot_field(field, **kwargs):
    return grid_plot_grid(field.t, **kwargs)


def grid_plot(filename, slice_index=0, axes=None, title="", vmin=None,
              vmax=None, rel_smoothing_scale=None, filename2=None, factor=1.,
              boxsize=None, colorbar=True, fix_colorbar=True, v_ratio=False,
              factor_minus_allowed=False, cmap=None, norm=None, log_cbar=False,
              xlabel="Mpc/h", ylabel="Mpc/h", vmid=None, **kwargs):
    """Note: rel_smoothing_scale must be in units of box length (=1)!
    The grid from filename2 is subtracted from that in filename.
    boxsize must be in Mpc/h.
    v_ratio: use argument vmin/vmax to adjust ratio of vmin/vmax of this
             plot, not as the actual vmin/vmax themselves.
    """
    if not filename2:
        grid = get_grid(filename, slice_index, rel_smoothing_scale)
    else:
        grid = get_grid(filename, slice_index)
        grid2 = get_grid(filename2, slice_index)
        grid -= grid2
        if rel_smoothing_scale:
            smooth_grid(grid, rel_smoothing_scale)
    grid = factor*grid
    if not factor_minus_allowed:
        if factor < 0:
            grid -= grid.min()
        # print grid.min()

    return grid_plot_grid(grid, slice_index=slice_index, axes=axes,
                          title=title, vmin=vmin, vmax=vmax, vmid=vmid,
                          rel_smoothing_scale=rel_smoothing_scale,
                          boxsize=boxsize, colorbar=colorbar,
                          fix_colorbar=fix_colorbar, v_ratio=v_ratio,
                          cmap=cmap, norm=norm, log_cbar=log_cbar,
                          xlabel=xlabel, ylabel=ylabel, **kwargs)


N_pal = 256
pal_eul = matplotlib.colors.ListedColormap(sns.color_palette("RdYlBu_r",
                                                             N_pal))
eul_style = {
    'interpolation': 'spline36',
    'cmap': pal_eul,
    'log_cbar': True,
    'xlabel': r"$x_z$ (Mpc/h)",
    'ylabel': r"$x_y$ (Mpc/h)",
}
eul_style_raw = {
    'interpolation': 'nearest',
    'cmap': pal_eul,
    'log_cbar': True,
    'xlabel': r"$x_z$ (Mpc/h)",
    'ylabel': r"$x_y$ (Mpc/h)",
}

pal_rss = matplotlib.colors.ListedColormap(sns.color_palette("RdYlGn_r",
                                                             N_pal))
rss_style = {
    'interpolation': 'spline36',
    'cmap': pal_rss,
    'log_cbar': True,
    'xlabel': r"$\zeta_z$ (Mpc/h)",
    'ylabel': r"$\zeta_y$ (Mpc/h)",
}

rss_style_paper = {
    'interpolation': 'spline36',
    'cmap': pal_rss,
    'log_cbar': True,
    'xlabel': r"$s_z$ (Mpc/h)",
    'ylabel': r"$s_y$ (Mpc/h)",
}

pal_lag = matplotlib.colors.ListedColormap(sns.color_palette("coolwarm",
                                                             N_pal))
lag_style = {
    'interpolation': 'nearest',
    'cmap': pal_lag,
    'log_cbar': False,
    'xlabel': r"$q_z$ (Mpc/h)",
    'ylabel': r"$q_y$ (Mpc/h)",
}
lag_style_smooth = {
    'interpolation': 'spline36',
    'cmap': pal_lag,
    'log_cbar': False,
    'xlabel': r"$q_z$ (Mpc/h)",
    'ylabel': r"$q_y$ (Mpc/h)",
}


def eul_style_grid_plot(grid, boxsize, slice_index=0, axes=None, title="",
                        vmin=None, vmax=None, vmid=0., v_ratio=False,
                        rel_smoothing_scale=None, colorbar=True,
                        fix_colorbar=True, norm=None, **kwargs):
    if type(grid) is str:
        fct = grid_plot
    else:
        fct = grid_plot_grid

    kwargs.update(eul_style)

    vmin, vmax = fct(grid, slice_index=slice_index, axes=axes, title=title, vmin=vmin, vmax=vmax, vmid=vmid, rel_smoothing_scale=rel_smoothing_scale, boxsize=boxsize, colorbar=colorbar, fix_colorbar=fix_colorbar, v_ratio=v_ratio, norm=norm, **kwargs)

    return vmin, vmax


def lag_style_grid_plot(grid, boxsize, slice_index=0, axes=None, title="",
                        vmin=None, vmax=None, vmid=0., v_ratio=False,
                        rel_smoothing_scale=None, colorbar=True,
                        fix_colorbar=True, norm=None, **kwargs):
    if rel_smoothing_scale is None:
        kwargs.update(lag_style)
    else:
        kwargs.update(lag_style_smooth)

    if type(grid) is str:
        fct = grid_plot
    else:
        fct = grid_plot_grid

    vmin, vmax = fct(grid, slice_index=slice_index, axes=axes, title=title, vmin=vmin, vmax=vmax, vmid=vmid, rel_smoothing_scale=rel_smoothing_scale, boxsize=boxsize, colorbar=colorbar, fix_colorbar=fix_colorbar, v_ratio=v_ratio, norm=norm, **kwargs)

    return vmin, vmax


def subplot_axes(rows, cols, textwidth=None, subplot_aspect=5./4, dpi=200):
    """
    `subplot_aspect` is the horizontal divided by the vertical size of a
    subplot.
    """
    if textwidth is None:
        textwidth = 10.  # sensible matplotlib ipython default
    fig = pl.figure(figsize=(textwidth,
                             float(textwidth) / cols * rows / subplot_aspect),
                    dpi=dpi)
    # ax = ImageGrid(fig, 111, # similar to subplot(111)
    #                nrows_ncols = (2, 2),
    #                axes_pad = 0.1,
    #                add_all=True,
    #                label_mode = "L",
    #                )
    ax = []
    for i in range(rows * cols):
        ax.append(fig.add_subplot(rows, cols, i+1))
    return fig, ax


def dump_grid(filename, grid):
    grid.tofile(filename)


# Base power spectrum functions:
# power spectrum units from Martinez & Saar, p. 278

def ps_ksquared_plot(filename, axes=None, title=None, label="",
                     legend=True, xlim=None, xlabel="$k$ (h/Mpc)",
                     ylabel="$k^2 P(k)$ ($h^{-1}$ Mpc)", **kwargs):
    spectrum = PowSpec(filename)
    if not axes:
        fig = pl.figure()
        axes = fig.add_subplot(111)

    axes.plot(spectrum.k, spectrum.P*spectrum.k**2, label=label, **kwargs)

    if axes.get_xscale != 'log':
        axes.set_xscale('log')
    if title:
        axes.set_title(title)
    if legend:
        leg = axes.legend(loc='lower left', fancybox=True, prop={'size': 8})
        leg.get_frame().set_alpha(0.5)
    if xlim:
        axes.set_xlim(xlim)
    return(min(spectrum.k), max(spectrum.k))


def ps_loglog_plot_arrays(k, P, axes=None, title=None, label="", xlim=None,
                   legend=True, return_ylim=False, xlabel="$k$ ($h$ Mpc$^{-1}$)",
                   ylabel="$P(k)$ ($h^{-3}$ Mpc$^3$)", boxsize_gridsize=None,
                   times_k_pow=0, log_y=True, **kwargs):
    if not axes:
        fig = pl.figure()
        axes = fig.add_subplot(111)

    if boxsize_gridsize and xlim is None:
        boxsize = boxsize_gridsize[0]
        gridsize = boxsize_gridsize[1]
        xlim = (2*np.pi/boxsize, np.sqrt(3)*np.pi/boxsize*gridsize)

    vertical = P
    if times_k_pow != 0:
        vertical *= k**times_k_pow

    if log_y:
        axes.loglog(k, vertical, label=label, **kwargs)
    else:
        axes.semilogx(k, vertical, label=label, **kwargs)

    axes.set_xlabel(xlabel)

    if times_k_pow != 0:
        if ylabel == "$P(k)$ ($h^{-3}$ Mpc$^3$)":
            if times_k_pow == 1:
                ylabel = "$k P(k)$ ($h^{{-{0}}}$ Mpc$^{0}$)".format(3-times_k_pow)
            else:
                ylabel = "$k^{0} P(k)$ ($h^{{-{1}}}$ Mpc$^{1}$)".format(times_k_pow, 3-times_k_pow)
    axes.set_ylabel(ylabel)

    if title:
        axes.set_title(title)
    if legend:
        # leg = axes.legend(loc='lower left', fancybox=True, prop={'size': 8})
        # leg.get_frame().set_alpha(0.5)
        axes.legend(loc='best')
    if xlim:
        axes.set_xlim(xlim)
    if return_ylim:
        return (min(k), max(k), min(P), max(P))
    else:
        return(min(k), max(k))


def ps_loglog_plot(filename, **kwargs):
    spectrum = PowSpec(filename)
    return ps_loglog_plot_arrays(spectrum.k, spectrum.P, **kwargs)


#
### Multi panel plots
#

def plot_multi(Nplots):
    xplots = int(np.ceil(np.sqrt(Nplots)))
    yplots = int(np.round(np.sqrt(Nplots)))
    ysize = 12.
    xsize = ysize / yplots * xplots
    fig, ax = pl.subplots(yplots, xplots, figsize=(xsize, ysize))
    fig.tight_layout()
    return fig, ax


def grid_plot_multi(filenames, boxsize=None):
    Nplots = len(filenames)
    fig, ax = plot_multi(Nplots)
    vmin = None
    vmax = None
    for ix, fn in enumerate(filenames):
        title = os.path.basename(fn)
        vmin, vmax = grid_plot(fn, axes=ax.flatten()[ix], vmin=vmin, vmax=vmax,
                               boxsize=boxsize, title=title)
    return fig


def ps_loglog_plot_multi_array(k, P_list, boxsize_gridsize=None,
                               label_appendix="", fig=None, ax=None,
                               fit_y_axis=True, labels=None, ps_kwargs={},
                               **kwargs):
    if not ax:
        fig = pl.figure(figsize=(15, 11))
        ax = fig.add_subplot(111)
    # klim = None
    # if boxsize_gridsize:
    #     boxsize = boxsize_gridsize[0]
    #     gridsize = boxsize_gridsize[1]
    #     klim = (2*np.pi/boxsize, np.sqrt(3)*np.pi/boxsize*gridsize)

    ymin = []
    ymax = []
    for ix, P in enumerate(P_list):
        if labels is None:
            label = ""
        elif len(labels) == len(P_list):
            label = labels[ix]
        else:
            raise SystemExit("In ps_loglog_plot_multi_array: either labels " +
                             "must have same length as P_list, or it " +
                             "must be None!")
        if ix in ps_kwargs.keys():
            these_kwargs = merge_dicts(kwargs, ps_kwargs[ix])
        else:
            these_kwargs = kwargs

        if isinstance(k, list):
            k_i = k[ix]
        else:
            k_i = k
        lim = ps_loglog_plot_arrays(k_i, P, axes=ax, label=label, legend=False,
                                    boxsize_gridsize=boxsize_gridsize,
                                    return_ylim=True, **these_kwargs)
        ymin.append(lim[2])
        ymax.append(lim[3])

    ax.set_ylim((min(ymin), max(ymax)))

    ax.legend(loc='best')
    # leg = ax.legend(loc='lower left', fancybox=True, prop={'size': 8})
    # leg.get_frame().set_alpha(0.5)
    return fig, ax



def ps_loglog_plot_multi(filenames, labels=None, label_appendix="", **kwargs):
    k_list = []
    P_list = []
    make_labels = False
    if labels is None:
        labels = []
        make_labels = True
    elif len(labels) != len(filenames):
        raise SystemExit("In ps_loglog_plot_multi: either labels must " +
                         "have same length as filenames, or it must be " +
                         "None!")

    for fn in filenames:
        spectrum = PowSpec(fn)
        k_list.append(spectrum.k)
        P_list.append(spectrum.P)
        if make_labels:
            labels.append(os.path.basename(fn) + label_appendix)

    return ps_loglog_plot_multi_array(k_list, P_list, labels=labels, **kwargs)

    # if not ax:
    #     fig = pl.figure(figsize=(15, 11))
    #     ax = fig.add_subplot(111)
    # # klim = None
    # # if boxsize_gridsize:
    # #     boxsize = boxsize_gridsize[0]
    # #     gridsize = boxsize_gridsize[1]
    # #     klim = (2*np.pi/boxsize, np.sqrt(3)*np.pi/boxsize*gridsize)

    # ymin = []
    # ymax = []
    # for ix, fn in enumerate(filenames):
    #     if labels is None:
    #         label = os.path.basename(fn) + label_appendix
    #     elif len(labels) == len(filenames):
    #         label = labels[ix]
    #     else:
    #         raise SystemExit("In ps_loglog_plot_multi: either labels must " +
    #                          "have same length as filenames, or it must be " +
    #                          "None!")
    #     if ix in ps_kwargs.keys():
    #         these_kwargs = merge_dicts(kwargs, ps_kwargs[ix])
    #     else:
    #         these_kwargs = kwargs
    #     lim = ps_loglog_plot(fn, axes=ax, label=label, legend=False,# xlim=klim,
    #                          boxsize_gridsize=boxsize_gridsize,
    #                          return_ylim=True, **these_kwargs)
    #     ymin.append(lim[2])
    #     ymax.append(lim[3])

    # ax.set_ylim((min(ymin), max(ymax)))

    # ax.legend(loc='best')
    # # leg = ax.legend(loc='lower left', fancybox=True, prop={'size': 8})
    # # leg.get_frame().set_alpha(0.5)
    # return fig, ax


def grid_plot_multi_glob(fn_glob, boxsize=None):
    filenames = glob.glob(fn_glob)
    filenames = egp.toolbox.natural_sorted(filenames)
    return grid_plot_multi(filenames, boxsize=boxsize)


def ps_loglog_plot_multi_glob(fn_glob, boxsize=None):
    filenames = glob.glob(fn_glob)
    filenames = egp.toolbox.natural_sorted(filenames)
    return ps_loglog_plot_multi(filenames, boxsize=None)


def plot_cross_spec(k, G, ax=None, textwidth=None, dpi=1000,
                    ticks_per_decade=4, scale_ticks_flag=True,
                    legend=True, **kwargs):
    if ax is None:
        fig, ax = subplot_axes(rows=1, cols=1, textwidth=textwidth, dpi=dpi)
        ax = ax[0]
    ax.semilogx(k, G, **kwargs)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(k[1], k[-1])

    # minor x ticks
    activate_minor_gridlines(ax, logx=True)

    if legend:
        leg = ax.legend(loc='best', fancybox=True)#, prop={'size': 8})
        leg.get_frame().set_alpha(0.5)

    # secondary x-axis for spatial scale
    if scale_ticks_flag:
        ax2 = add_scale_ticks(ax, logx=True)  # , add_ticks=[k[1], k[-1]])

    # axis labels
    ax.set_xlabel(r"$k$ [ $h$ $\mathrm{Mpc}$$^{-1}$]")
    ax.set_ylabel(r"$G(k)$")

    if 'label' in kwargs:
        pl.legend(loc='best')
    return ax

#
# STYLE STUFF
#

def activate_minor_gridlines(ax, logx=False, logy=False, axis='both',
                             subs=[2,3,4,5,6,7,8,9]):
    log_locator = pl.LogLocator(subs=subs)
    lin_locator = matplotlib.ticker.AutoMinorLocator()
    if logx:
        ax.get_xaxis().set_minor_locator(log_locator)
    else:
        ax.get_xaxis().set_minor_locator(lin_locator)
    if logy:
        ax.get_yaxis().set_minor_locator(log_locator)
    else:
        ax.get_yaxis().set_minor_locator(lin_locator)
    ax.grid(b=True, which='minor', axis=axis, color='w', linewidth=0.5)


def add_scale_ticks(ax, logx=False, add_ticks=[], ticks_xlim=True,
                    label_fct=lambda k: (2*np.pi/k), label_fmt="%.1f",
                    scale_ticks=[], minor_ticks=True):
    ax2 = ax.twiny()
    if logx:
        ax2.set_xscale('log')
    ax2Ticks = ax.get_xticks()
    print("ax2Ticks: ", ax2Ticks)

    for tick in add_ticks:
        ax2Ticks = np.append(ax2Ticks, tick)

    for tick in scale_ticks:
        ax2Ticks = np.append(ax2Ticks, label_fct(tick))

    xlim = ax.get_xlim()
    if ticks_xlim:
        for xl in xlim:
            if xl not in ax2Ticks:
                ax2Ticks = np.append(ax2Ticks, xl)

    ax2.set_xticks(ax2Ticks)

    if minor_ticks:
        ax2TicksMinor = ax.get_xticks(minor=True)
        ax2.set_xticks(ax2TicksMinor, minor=True)

    ax2.set_xlim(xlim)

    # if minor_ticks:
    #     rotation = 90
    # else:
    #     rotation = 0
    rotation = 0

    ax2Ticklabels = [label_fmt % label_fct(ki) for ki in ax2Ticks]
    ax2.set_xticklabels(ax2Ticklabels, minor=False, rotation=rotation)

    if minor_ticks:
        ax2TickMinorlabels = [label_fmt % label_fct(ki) for ki in ax2TicksMinor]
        ax2.set_xticklabels(ax2TickMinorlabels, minor=True, rotation=rotation)
    ax2.set_xlabel(r'scale [$h^{-1}$ $\mathrm{Mpc}$]')
    return ax2


def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


# Using ColorBrewer Dark2 colors for walls, filaments and clusters, white for
# voids.
# Dark2 = ["#ffffff", "#1b9e77", "#d95f02", "#7570b3"]
Dark2 = ["#ffffff", "#1b9e77", "#d95f02", "#000000"]
pal_nexus = matplotlib.colors.ListedColormap(sns.color_palette(Dark2, 4))
pal_node = matplotlib.colors.ListedColormap(sns.color_palette((Dark2[0],
                                            Dark2[3]), 2))
pal_node_c = matplotlib.colors.ListedColormap(sns.light_palette(Dark2[3]))
pal_fila = matplotlib.colors.ListedColormap(sns.color_palette((Dark2[0],
                                            Dark2[2]), 2))
pal_fila_c = matplotlib.colors.ListedColormap(sns.light_palette(Dark2[2]))
pal_wall = matplotlib.colors.ListedColormap(sns.color_palette((Dark2[0],
                                            Dark2[1]), 2))
pal_wall_c = matplotlib.colors.ListedColormap(sns.light_palette(Dark2[1]))

nexus_style = eul_style.copy()
nexus_style['log_cbar'] = False
nexus_style['cmap'] = pal_nexus

nexus_norm = matplotlib.colors.BoundaryNorm([0, 2, 3, 4, 5], 4)
nexus_part_norm = matplotlib.colors.BoundaryNorm([0, 1, 2], 2)


def plot_nexus_output(grids, density=None, textwidth=16, **kwargs):
    fig, ax = subplot_axes(rows=4, cols=2, textwidth=textwidth)
    grid_plot_grid(grids['all'], axes=ax[0], title="all", norm=nexus_norm,
                   **merge_dicts(nexus_style, kwargs))
    grid_plot_grid(grids['node_clean'], axes=ax[2], title="nodes",
                   norm=nexus_part_norm, **merge_dicts(nexus_style, kwargs,
                                                       {'cmap': pal_node}))
    grid_plot_grid(grids['fila_clean'], axes=ax[4], title="filaments",
                   norm=nexus_part_norm, **merge_dicts(nexus_style, kwargs,
                                                       {'cmap': pal_fila}))
    grid_plot_grid(grids['wall_clean'], axes=ax[6], title="walls",
                   norm=nexus_part_norm, **merge_dicts(nexus_style, kwargs,
                                                       {'cmap': pal_wall}))
    grid_plot_grid(grids['node_maxResponse'], axes=ax[3],
                   title="nodes (max resp.)",
                   **merge_dicts(nexus_style, kwargs, {'cmap': pal_node_c,
                                 'log_cbar': True}))
    grid_plot_grid(grids['fila_maxResponse'], axes=ax[5],
                   title="filaments (max resp.)",
                   **merge_dicts(nexus_style, kwargs, {'cmap': pal_fila_c,
                                 'log_cbar': True}))
    grid_plot_grid(grids['wall_maxResponse'], axes=ax[7],
                   title="walls (max resp.)",
                   **merge_dicts(nexus_style, kwargs, {'cmap': pal_wall_c,
                                 'log_cbar': True}))

    if density is not None:
        grid_plot_grid(density, axes=ax[1], title="density",
                       vmid=0, **merge_dicts(eul_style, kwargs))
    else:
        ax[1].axis('off')


def plot_nexus_output_compact(nexus_all, density, textwidth=16, **kwargs):
    fig, ax = subplot_axes(rows=2, cols=1, textwidth=textwidth)
    _, _, _, cb = grid_plot_grid(nexus_all, axes=ax[0], title="web components",
                                 norm=nexus_norm, return_objects=True,
                                 **merge_dicts(nexus_style, kwargs))
    # cb.ax.get_yaxis().set_ticks([1., 2.5, 3.5, 4.5])
    grid_plot_grid(density, axes=ax[1], title="density",
                   vmid=0, **merge_dicts(eul_style, kwargs))




#
# ## Convenient short-hands:
#
def eul_grid_plot(MCMC_iteration, slice_index=0, axes = None, title = None, vmin = None, vmax = None, rel_smoothing_scale = None, filename2 = None, factor = 1., boxsize = None):
    filename = 'deltaEUL_%(MCMC_iteration)i.dat' % locals()
    return grid_plot(filename, slice_index, axes, title, vmin, vmax, rel_smoothing_scale, filename2, factor, boxsize)

def lag_grid_plot(MCMC_iteration, slice_index=0, axes = None, title = None, vmin = None, vmax = None, rel_smoothing_scale = None, filename2 = None, factor = 1., boxsize = None):
    filename = 'deltaLAG_%(MCMC_iteration)i.dat' % locals()
    return grid_plot(filename, slice_index, axes, title, vmin, vmax, rel_smoothing_scale, filename2, factor, boxsize)

def grid_comparison_row(filename, filename_input, row, rows, slice_index=0, figure = None, title_appendix = None, rel_smoothing_scale = None, negative_sample = False, sync_colorbars = True, boxsize = None):
    """row starts counting at 1!"""
    columns = 3
    if not figure:
        figure = pl.figure()
    
    if not title_appendix:
        title_appendix = ""
    else:
        title_appendix = " " + title_appendix
    
    ax = figure.add_subplot(rows, columns, 2 + (row-1)*columns)
    vmin, vmax = grid_plot(filename_input, slice_index, ax, "input"+title_appendix, rel_smoothing_scale=rel_smoothing_scale, boxsize = boxsize)
    if not sync_colorbars:
        vmin, vmax = None, None
    ax = figure.add_subplot(rows, columns, 1 + (row-1)*columns)
    factor = 1.
    if negative_sample:
        factor = -1.
    grid_plot(filename, slice_index, ax, "sampled"+title_appendix, vmin=vmin, vmax=vmax, rel_smoothing_scale=rel_smoothing_scale, factor = factor, boxsize = boxsize)
    ax = figure.add_subplot(rows, columns, 3 + (row-1)*columns)
    grid_plot(filename, slice_index, ax, "residual"+title_appendix, vmin=vmin, vmax=vmax, filename2 = filename_input, rel_smoothing_scale=rel_smoothing_scale, boxsize = boxsize)

def grid_comparison_row_gridspec(filename, filename_input, slice_index=0, figure = None, gridspec_row = None, title_appendix = None, rel_smoothing_scale = None, negative_sample = False, sync_colorbars = True, boxsize = None):
    """row starts counting at 1!"""
    columns = 3
    if not figure:
        figure = pl.figure()
    if not gridspec_row:
        gridspec_row = gridspec.GridSpec(1, columns)
    
    if not title_appendix:
        title_appendix = ""
    else:
        title_appendix = " " + title_appendix
    
    ax = figure.add_subplot(gridspec_row[0,1]) # middle plot
    vmin, vmax = grid_plot(filename_input, slice_index, ax, "input"+title_appendix, rel_smoothing_scale=rel_smoothing_scale, boxsize = boxsize)
    if not sync_colorbars:
        vmin, vmax = None, None
    ax = figure.add_subplot(gridspec_row[0,0]) # left plot
    factor = 1.
    if negative_sample:
        factor = -1.
    grid_plot(filename, slice_index, ax, "sampled"+title_appendix, vmin=vmin, vmax=vmax, rel_smoothing_scale=rel_smoothing_scale, factor = factor, boxsize = boxsize)
    ax = figure.add_subplot(gridspec_row[0,2]) # right plot
    grid_plot(filename, slice_index, ax, "residual"+title_appendix, vmin=vmin, vmax=vmax, filename2 = filename_input, rel_smoothing_scale=rel_smoothing_scale, boxsize = boxsize)

def grid_comparison_3_axes(filename, filename_input, slice_index=0, axes_list = None, title_appendix = None, rel_smoothing_scale = None, negative_sample = False, sync_colorbars = True, boxsize = None, colorbar = True, fix_colorbar = True):
    """row starts counting at 1!"""
    columns = 3
    if len(axes_list) != columns:
        figure = pl.figure()
        gridspec_row = gridspec.GridSpec(1, columns)
        axes_list = [figure.add_subplot(ss) for ss in gridspec_row]
    
    if not title_appendix:
        title_appendix = ""
    else:
        title_appendix = " " + title_appendix

    if colorbar == "central":
        colorbar = [True, False, False]
    elif colorbar:
        colorbar = [True, True, True]
    else:
        colorbar = [False, False, False]
    
    vmin, vmax = grid_plot(filename_input, slice_index, axes_list[1], "input"+title_appendix, rel_smoothing_scale=rel_smoothing_scale, boxsize = boxsize, colorbar = colorbar[0], fix_colorbar = fix_colorbar)
    if not sync_colorbars:
        vmin, vmax = None, None
    factor = 1.
    if negative_sample:
        factor = -1.
    grid_plot(filename, slice_index, axes_list[0], "sampled"+title_appendix, vmin=vmin, vmax=vmax, rel_smoothing_scale=rel_smoothing_scale, factor = factor, boxsize = boxsize, colorbar = colorbar[1], fix_colorbar = fix_colorbar)
    grid_plot(filename, slice_index, axes_list[2], "residual"+title_appendix, vmin=vmin, vmax=vmax, filename2 = filename_input, rel_smoothing_scale=rel_smoothing_scale, boxsize = boxsize, colorbar = colorbar[2], fix_colorbar = fix_colorbar)

def eul_grid_comparison_row(MCMC_iteration, filename_input = "deltaEULtest.dat", row = 1, rows = 1, slice_index=0, figure = None, rel_smoothing_scale = None, negative_sample = False, sync_colorbars = True, boxsize = None):
    filename = 'deltaEUL_%(MCMC_iteration)i.dat' % locals()
    if rel_smoothing_scale:
        grid_comparison_row(filename, filename_input, row, rows, slice_index, figure, "smoothed Eulerian", rel_smoothing_scale = rel_smoothing_scale, negative_sample = negative_sample, sync_colorbars = sync_colorbars, boxsize = boxsize)
    else:
        grid_comparison_row(filename, filename_input, row, rows, slice_index, figure, "Eulerian", negative_sample = negative_sample, sync_colorbars = sync_colorbars, boxsize = boxsize)

def eul_grid_comparison_row_gridspec(MCMC_iteration, filename_input = "deltaEULtest.dat", slice_index=0, figure = None, gridspec_row = None, rel_smoothing_scale = None, negative_sample = False, sync_colorbars = True, boxsize = None):
    filename = 'deltaEUL_%(MCMC_iteration)i.dat' % locals()
    if rel_smoothing_scale:
        grid_comparison_row_gridspec(filename, filename_input, slice_index, figure, gridspec_row = gridspec_row, title_appendix = "smoothed Eulerian", rel_smoothing_scale = rel_smoothing_scale, negative_sample = negative_sample, sync_colorbars = sync_colorbars, boxsize = boxsize)
    else:
        grid_comparison_row_gridspec(filename, filename_input, slice_index, figure, gridspec_row = gridspec_row, title_appendix = "Eulerian", negative_sample = negative_sample, sync_colorbars = sync_colorbars, boxsize = boxsize)

def eul_grid_comparison_3_axes(MCMC_iteration, filename_input = "deltaEULtest.dat", slice_index=0, axes_list = None, rel_smoothing_scale = None, negative_sample = False, sync_colorbars = True, boxsize = None, colorbar = True, fix_colorbar = True):
    filename = 'deltaEUL_%(MCMC_iteration)i.dat' % locals()
    if rel_smoothing_scale:
        grid_comparison_3_axes(filename, filename_input, slice_index, axes_list = axes_list, title_appendix = "smoothed Eulerian", rel_smoothing_scale = rel_smoothing_scale, negative_sample = negative_sample, sync_colorbars = sync_colorbars, boxsize = boxsize, colorbar = colorbar, fix_colorbar = fix_colorbar)
    else:
        grid_comparison_3_axes(filename, filename_input, slice_index, axes_list = axes_list, title_appendix = "Eulerian", negative_sample = negative_sample, sync_colorbars = sync_colorbars, boxsize = boxsize, colorbar = colorbar, fix_colorbar = fix_colorbar)

def lag_grid_comparison_row(MCMC_iteration, filename_input = "deltaLAGtest.dat", row = 1, rows = 1, slice_index=0, figure = None, rel_smoothing_scale = None, negative_sample = False, sync_colorbars = True, boxsize = None):
    filename = 'deltaLAG_%(MCMC_iteration)i.dat' % locals()
    if rel_smoothing_scale:
        grid_comparison_row(filename, filename_input, row, rows, slice_index, figure, "smoothed Lagrangian", rel_smoothing_scale = rel_smoothing_scale, negative_sample = negative_sample, sync_colorbars = sync_colorbars, boxsize = boxsize)
    else:
        grid_comparison_row(filename, filename_input, row, rows, slice_index, figure, "Lagrangian", negative_sample = negative_sample, sync_colorbars = sync_colorbars, boxsize = boxsize)

def lag_grid_comparison_row_gridspec(MCMC_iteration, filename_input = "deltaLAGtest.dat", slice_index=0, figure = None, gridspec_row = None, rel_smoothing_scale = None, negative_sample = False, sync_colorbars = True, boxsize = None):
    filename = 'deltaLAG_%(MCMC_iteration)i.dat' % locals()
    if rel_smoothing_scale:
        grid_comparison_row_gridspec(filename, filename_input, slice_index, figure, gridspec_row = gridspec_row, title_appendix = "smoothed Lagrangian", rel_smoothing_scale = rel_smoothing_scale, negative_sample = negative_sample, sync_colorbars = sync_colorbars, boxsize = boxsize)
    else:
        grid_comparison_row_gridspec(filename, filename_input, slice_index, figure, gridspec_row = gridspec_row, title_appendix = "Lagrangian", negative_sample = negative_sample, sync_colorbars = sync_colorbars, boxsize = boxsize)

def lag_grid_comparison_3_axes(MCMC_iteration, filename_input = "deltaLAGtest.dat", slice_index=0, axes_list = None, rel_smoothing_scale = None, negative_sample = False, sync_colorbars = True, boxsize = None, colorbar = True, fix_colorbar = True):
    filename = 'deltaLAG_%(MCMC_iteration)i.dat' % locals()
    if rel_smoothing_scale:
        grid_comparison_3_axes(filename, filename_input, slice_index, axes_list = axes_list, title_appendix = "smoothed Lagrangian", rel_smoothing_scale = rel_smoothing_scale, negative_sample = negative_sample, sync_colorbars = sync_colorbars, boxsize = boxsize, colorbar = colorbar, fix_colorbar = fix_colorbar)
    else:
        grid_comparison_3_axes(filename, filename_input, slice_index, axes_list = axes_list, title_appendix = "Lagrangian", negative_sample = negative_sample, sync_colorbars = sync_colorbars, boxsize = boxsize, colorbar = colorbar, fix_colorbar = fix_colorbar)

def lag_ps_ksquared_comparison_plot(MCMC_iteration_list, filename_input = "specLAGtest.dat", axes = None, title = None, shrink_xlim = False):
    xlim_1 = ps_ksquared_plot(filename_input, axes = axes, title = title, label = "input")
    xlims = []
    for MCMC_iteration in MCMC_iteration_list:
        filename = "powSpecit%i.dat" % MCMC_iteration
        xlim_i = ps_ksquared_plot(filename, axes = axes, title = title, label = "iter. %i" % MCMC_iteration)
        xlims.append(xlim_i)
    if shrink_xlim:
        xlower = min( xlim_1[0], min([xlim_i[0] for xlim_i in xlims]) )
        xupper = max( xlim_1[1], min([xlim_i[1] for xlim_i in xlims]) )
        axes.set_xlim(xlower, xupper)

def lag_ps_loglog_comparison_plot(MCMC_iteration_list, filename_input = "specLAGtest.dat", axes = None, title = None, shrink_xlim = False):
    xlim_1 = ps_loglog_plot(filename_input, axes = axes, title = title, label = "input")
    xlims = []
    for MCMC_iteration in MCMC_iteration_list:
        filename = "powSpecit%i.dat" % MCMC_iteration
        xlim_i = ps_loglog_plot(filename, axes = axes, title = title, label = "iter. %i" % MCMC_iteration)
        xlims.append(xlim_i)
    if shrink_xlim:
        xlower = min( xlim_1[0], min([xlim_i[0] for xlim_i in xlims]) )
        xupper = max( xlim_1[1], min([xlim_i[1] for xlim_i in xlims]) )
        axes.set_xlim(xlower, xupper)

def full_comparison(MCMC_iteration, slice_index=0, figure = None, title_appendix = None, rel_smoothing_scale = None, negative_sample = False, sync_colorbars = True, boxsize = None):
    """row starts counting at 1!"""
    columns = 3
    rows = 3
    if not figure:
        figure = pl.figure()

    lag_grid_comparison_row(MCMC_iteration, row = 1, rows = rows, slice_index = slice_index, figure = figure, rel_smoothing_scale = rel_smoothing_scale, negative_sample = negative_sample, sync_colorbars = sync_colorbars, boxsize = boxsize)
    eul_grid_comparison_row(MCMC_iteration, row = 2, rows = rows, slice_index = slice_index, figure = figure, rel_smoothing_scale = rel_smoothing_scale, negative_sample = negative_sample, sync_colorbars = sync_colorbars, boxsize = boxsize)
    ax_ps1 = figure.add_subplot(rows, columns, 1 + 2*columns)
    ax_ps2 = figure.add_subplot(rows, columns, 2 + 2*columns)
    lag_ps_ksquared_comparison_plot([MCMC_iteration], axes = ax_ps1, title = "k^2 vertical axis", shrink_xlim = True)
    lag_ps_loglog_comparison_plot([MCMC_iteration], axes = ax_ps2, title = "log-log axes", shrink_xlim = True)
    
    pl.suptitle("iteration %i" % MCMC_iteration)
    pl.tight_layout()

def full_comparison_gridspec(MCMC_iteration, slice_index=0, figure = None, rel_smoothing_scale = None, negative_sample = False, sync_colorbars = True, boxsize = None):
    """row starts counting at 1!"""
    columns = 3
    rows = 3
    if not figure:
        figure = pl.figure()

    #pl.suptitle("iteration %i" % MCMC_iteration)
    
    row1 = gridspec.GridSpec(1, columns)
    row2 = gridspec.GridSpec(1, columns)
    row3 = gridspec.GridSpec(1, 2)

    ax_ps1 = figure.add_subplot(row3[0,0])
    ax_ps2 = figure.add_subplot(row3[0,1])

    #lag_grid_comparison_row_gridspec(MCMC_iteration, slice_index = slice_index, figure = figure, gridspec_row = row1, rel_smoothing_scale = rel_smoothing_scale, negative_sample = negative_sample, sync_colorbars = sync_colorbars, boxsize = boxsize)
    lag_grid_comparison_row_gridspec(MCMC_iteration, slice_index = slice_index, figure = figure, gridspec_row = row1, rel_smoothing_scale = rel_smoothing_scale, negative_sample = negative_sample, sync_colorbars = sync_colorbars, boxsize = boxsize)
    eul_grid_comparison_row_gridspec(MCMC_iteration, slice_index = slice_index, figure = figure, gridspec_row = row2, rel_smoothing_scale = rel_smoothing_scale, negative_sample = negative_sample, sync_colorbars = sync_colorbars, boxsize = boxsize)
    lag_ps_ksquared_comparison_plot([MCMC_iteration], axes = ax_ps1, title = "k^2 vertical axis", shrink_xlim = True)
    lag_ps_loglog_comparison_plot([MCMC_iteration], axes = ax_ps2, title = "log-log axes", shrink_xlim = True)

    row1.tight_layout(figure, rect=[0, 0.7, 1, 1]) # rect: left, bottom, right, top
    row2.tight_layout(figure, rect=[0, 0.4, 1, 0.7])
    row3.tight_layout(figure, rect=[0, 0, 1, 0.4])

def full_comparison_axes(MCMC_iteration, slice_index=0, figure = None, rel_smoothing_scale = None, negative_sample = False, sync_colorbars = True, boxsize = None):
    """row starts counting at 1!"""
    columns = 3
    rows = 3
    if not figure:
        figure = pl.figure()

    pl.suptitle("iteration %i" % MCMC_iteration)
    
    # ImageGrid rect is [left, bottom, width, height]
    rows12 = axes_grid1.ImageGrid(figure, rect = [0.05, 0.4, 0.9, 0.5], nrows_ncols = (2, 3), cbar_mode = "edge", share_all=True, axes_pad = 0.3)

    ax_ps1 = figure.add_axes([0.05,0.1,0.4,0.2])
    ax_ps2 = figure.add_axes([0.55,0.1,0.4,0.2])

    lag_grid_comparison_3_axes(MCMC_iteration, slice_index = slice_index, axes_list = rows12[:3], rel_smoothing_scale = rel_smoothing_scale, negative_sample = negative_sample, sync_colorbars = sync_colorbars, boxsize = boxsize, colorbar = False)
    eul_grid_comparison_3_axes(MCMC_iteration, slice_index = slice_index, axes_list = rows12[3:], rel_smoothing_scale = rel_smoothing_scale, negative_sample = negative_sample, sync_colorbars = sync_colorbars, boxsize = boxsize, colorbar = False)

    cbar1 = rows12.cbar_axes[0].colorbar(rows12[1].get_images()[0])
    cbar2 = rows12.cbar_axes[1].colorbar(rows12[4].get_images()[0])

    lag_ps_ksquared_comparison_plot([MCMC_iteration], axes = ax_ps1, title = "k^2 vertical axis", shrink_xlim = True)
    lag_ps_loglog_comparison_plot([MCMC_iteration], axes = ax_ps2, title = "log-log axes", shrink_xlim = True)

def full_comparison_axes_savefig(filename, MCMC_iteration, slice_index=0, rel_smoothing_scale = None, negative_sample = False, sync_colorbars = True, boxsize = None):
    figure = pl.figure(figsize = (14,16), dpi=75)
    full_comparison_axes(MCMC_iteration, slice_index = slice_index, figure = figure, rel_smoothing_scale = rel_smoothing_scale, negative_sample = negative_sample, sync_colorbars = sync_colorbars, boxsize = boxsize)
    pl.savefig(filename)
    pl.close()

def movie_full_comparison_update_frame(i, MCMC_iteration_list, slice_index, figure, rel_smoothing_scale, negative_sample, sync_colorbars, boxsize):
    figure.clear()
    full_comparison_axes(MCMC_iteration_list[i], slice_index=slice_index, figure=figure, rel_smoothing_scale=rel_smoothing_scale, negative_sample=negative_sample, sync_colorbars=sync_colorbars, boxsize=boxsize)
    return figure,

def movie_full_comparison(MCMC_iteration_list, slice_index=0, rel_smoothing_scale=None, negative_sample=False, sync_colorbars=True, boxsize=None, filename = None):
    fig = pl.figure(figsize = (14,16), dpi=75)
    anim = animation.FuncAnimation(fig, movie_full_comparison_update_frame, init_func = fig.clear, fargs=(MCMC_iteration_list, slice_index, fig, rel_smoothing_scale, negative_sample, sync_colorbars, boxsize), frames=len(MCMC_iteration_list), interval=500)
    if filename:
        anim.save(filename, bitrate=1200)
    else:
        return anim

#~ def multi_plot(plots):
    #~ """
    #~ The plots parameter should be a list of lists, containing, for
    #~ each item: [function, filename, slice_index, axes, title, kwargs]
    #~ function: one of the "_plot" functions above (not its name, the
              #~ function itself!).
    #~ kwargs:   a dictionary containing extra parameters, e.g.
              #~ rel_smoothing_scale for smooth_grid and vmin and vmax for
              #~ both grid and smooth_grid.
    #~ """
    #~ for plot in plots:
        #~ plot[0](plot[1], slice_index = plot[2], axes = plot[3], **plot[4])

#-----------------------------------------------------------------------
# OLD STUFF:

#def eul_residual_plot(MCMC_iteration, original, slice_index=0, log=False, log_original=False):
    #global gridsize
    #filename = 'deltaEUL_%(MCMC_iteration)i.dat' % locals()
    #grid = np.memmap(filename, dtype='float64', mode='r').reshape(gridsize, gridsize, gridsize)
    #if log:
        #grid = np.log10(grid)
    #grid_original = np.memmap(original, dtype='float64', mode='r').reshape(gridsize, gridsize, gridsize)
    #if log:
        #grid_original = np.log10(grid_original)
    
    #vmin = grid_original[slice_index].min()
    #vmax = grid_original[slice_index].max()
    
    #pl.figure(figsize = (16,4))
    #pl.subplot(1,3,1); pl.imshow(grid[slice_index]); pl.title('sampled field'); pl.clim(vmin, vmax); pl.colorbar()
    #pl.subplot(1,3,2); pl.imshow(grid_original[slice_index]); pl.title('input field'); pl.clim(vmin, vmax); pl.colorbar()
    #pl.subplot(1,3,3); pl.imshow(grid[slice_index]-grid_original[slice_index]); pl.title('residual'); pl.clim(vmin, vmax); pl.colorbar()

#def lag_residual_plot(MCMC_iteration, original, slice_index=0, log=False, log_original=False):
    #global gridsize
    #filename = 'deltaLAG_%(MCMC_iteration)i.dat' % locals()
    #grid = np.memmap(filename, dtype='float64', mode='r').reshape(gridsize, gridsize, gridsize)
    #if log:
        #grid = np.log10(grid)
    #grid_original = np.memmap(original, dtype='float64', mode='r').reshape(gridsize, gridsize, gridsize)
    #if log:
        #grid_original = np.log10(grid_original)
    
    #vmin = grid_original[slice_index].min()
    #vmax = grid_original[slice_index].max()

    #pl.figure(figsize = (16,4))
    #pl.subplot(1,3,1); pl.imshow(grid[slice_index]); pl.title('sampled field'); pl.clim(vmin, vmax); pl.colorbar()
    #pl.subplot(1,3,2); pl.imshow(grid_original[slice_index]); pl.title('input field'); pl.clim(vmin, vmax); pl.colorbar()
    #pl.subplot(1,3,3); pl.imshow(grid[slice_index]-grid_original[slice_index]); pl.title('residual'); pl.clim(vmin, vmax); pl.colorbar()

#def lag_eul_smooth_powerspec_residual_plot(MCMC_iteration, original_lag, original_eul, original_ps_lag, original_ps_eul, patchy_init_spec, patchy_specDeltaIC, patchy_specDeltaZELD, boxlen, smoothing_scale, slice_index=0, log=False, log_original=False):
    #global gridsize

    ## Lagrangian
    #filename = 'deltaLAG_%(MCMC_iteration)i.dat' % locals()
    
    #grid = np.memmap(filename, dtype='float64', mode='r').reshape(gridsize, gridsize, gridsize)
    #grid_field = egp.basic_types.Field(true = grid)
    #grid_field.boxlen = boxlen
    #grid_smooth_field = egp.toolbox.filter_Field(grid_field, egp.toolbox.gaussian_kernel, (smoothing_scale,))
    #grid_smooth = grid_smooth_field.t
    #if log:
        #grid = np.log10(grid)
    #grid_original = np.memmap(original_lag, dtype='float64', mode='r').reshape(gridsize, gridsize, gridsize)
    #grid_original_field = egp.basic_types.Field(true = grid_original)
    #grid_original_field.boxlen = boxlen
    #grid_original_smooth_field = egp.toolbox.filter_Field(grid_original_field, egp.toolbox.gaussian_kernel, (smoothing_scale,))
    #grid_original_smooth = grid_original_smooth_field.t
    #if log:
        #grid_original_smooth = np.log10(grid_original_smooth)
    
    #vmin = grid_original[slice_index].min()
    #vmax = grid_original[slice_index].max()

    #pl.figure(figsize = (16,20))
    #pl.suptitle("MCMC iteration number %i" % MCMC_iteration, fontsize=32)
    
    #pl.subplot(5,3,1); pl.imshow(grid[slice_index]); pl.title('sampled lag. field'); pl.clim(vmin, vmax); pl.colorbar()
    #pl.subplot(5,3,2); pl.imshow(grid_original[slice_index]); pl.title('input lag. field'); pl.clim(vmin, vmax); pl.colorbar()
    #pl.subplot(5,3,3); pl.imshow(grid[slice_index]-grid_original[slice_index]); pl.title('residual'); pl.clim(vmin, vmax); pl.colorbar()
    
    #vmin = grid_original_smooth[slice_index].min()
    #vmax = grid_original_smooth[slice_index].max()

    #pl.subplot(5,3,4); pl.imshow(grid_smooth[slice_index]); pl.title('sampled lag. field, smoothed %3.1f Mpc/h' % smoothing_scale); pl.clim(vmin, vmax); pl.colorbar()
    #pl.subplot(5,3,5); pl.imshow(grid_original_smooth[slice_index]); pl.title('input lag. field, smoothed %3.1f Mpc/h' % smoothing_scale); pl.clim(vmin, vmax); pl.colorbar()
    #pl.subplot(5,3,6); pl.imshow(grid_smooth[slice_index]-grid_original_smooth[slice_index]); pl.title('residual of smoothed images'); pl.clim(vmin, vmax); pl.colorbar()
    
    ## Eulerian
    #filename = 'deltaEUL_%(MCMC_iteration)i.dat' % locals()
    
    #grid = np.memmap(filename, dtype='float64', mode='r').reshape(gridsize, gridsize, gridsize)
    #grid_field = egp.basic_types.Field(true = grid)
    #grid_field.boxlen = boxlen
    #grid_smooth_field = egp.toolbox.filter_Field(grid_field, egp.toolbox.gaussian_kernel, (smoothing_scale,))
    #grid_smooth = grid_smooth_field.t
    #if log:
        #grid = np.log10(grid)
    #grid_original = np.memmap(original_eul, dtype='float64', mode='r').reshape(gridsize, gridsize, gridsize)
    #grid_original_field = egp.basic_types.Field(true = grid_original)
    #grid_original_field.boxlen = boxlen
    #grid_original_smooth_field = egp.toolbox.filter_Field(grid_original_field, egp.toolbox.gaussian_kernel, (smoothing_scale,))
    #grid_original_smooth = grid_original_smooth_field.t
    #if log:
        #grid_original_smooth = np.log10(grid_original_smooth)
    
    #vmin = grid_original[slice_index].min()
    #vmax = grid_original[slice_index].max()

    #pl.subplot(5,3,7); pl.imshow(grid[slice_index]); pl.title('sampled eul. field'); pl.clim(vmin, vmax); pl.colorbar()
    #pl.subplot(5,3,8); pl.imshow(grid_original[slice_index]); pl.title('input eul. field'); pl.clim(vmin, vmax); pl.colorbar()
    #pl.subplot(5,3,9); pl.imshow(grid[slice_index]-grid_original[slice_index]); pl.title('residual'); pl.clim(vmin, vmax); pl.colorbar()
    
    #vmin = grid_original_smooth[slice_index].min()
    #vmax = grid_original_smooth[slice_index].max()

    #pl.subplot(5,3,10); pl.imshow(grid_smooth[slice_index]); pl.title('sampled eul. field, smoothed %3.1f Mpc/h' % smoothing_scale); pl.clim(vmin, vmax); pl.colorbar()
    #pl.subplot(5,3,11); pl.imshow(grid_original_smooth[slice_index]); pl.title('input eul. field, smoothed %3.1f Mpc/h' % smoothing_scale); pl.clim(vmin, vmax); pl.colorbar()
    #pl.subplot(5,3,12); pl.imshow(grid_smooth[slice_index]-grid_original_smooth[slice_index]); pl.title('residual of smoothed images'); pl.clim(vmin, vmax); pl.colorbar()
    
    ## Power spectrum
    #filename = 'powSpecit%(MCMC_iteration)i.dat' % locals()
    
    #barcode_init_spec = np.loadtxt('init_spec.dat')
    #barcode_specLAGtest = np.loadtxt('specLAGtest.dat')
    #barcode_powSpecitN = np.loadtxt(filename)
    #patchy_init_spec = np.loadtxt(patchy_init_spec)
    #patchy_specDeltaIC = np.loadtxt(patchy_specDeltaIC)
    #patchy_specDeltaZELD = np.loadtxt(patchy_specDeltaZELD)

    #ax = pl.subplot(5,3,14)
    #pl.loglog(barcode_init_spec[:,0], barcode_init_spec[:,1], label='barcode init_spec')
    #pl.loglog(patchy_init_spec[:,0], patchy_init_spec[:,1], label='patchy init_spec')
    #leg = ax.legend(loc='lower left', fancybox=True, prop={'size':8})
    #leg.get_frame().set_alpha(0.5)

    #ax = pl.subplot(5,3,15)
    #pl.loglog(barcode_specLAGtest[:,0], barcode_specLAGtest[:,1], label='barcode specLAGtest')
    #pl.loglog(patchy_specDeltaIC[:,0], patchy_specDeltaIC[:,1], label='patchy specDeltaIC')
    #leg = ax.legend(loc='lower left', fancybox=True, prop={'size':8})
    #leg.get_frame().set_alpha(0.5)

    #ax = pl.subplot(5,3,13)
    #pl.loglog(patchy_specDeltaZELD[:,0], patchy_specDeltaZELD[:,1], label='patchy specDeltaZELD')
    #pl.loglog(barcode_powSpecitN[:,0], barcode_powSpecitN[:,1], label='barcode powSpecit%(MCMC_iteration)i' % locals())
    #leg = ax.legend(loc='lower left', fancybox=True, prop={'size':8})
    #leg.get_frame().set_alpha(0.5)

    
