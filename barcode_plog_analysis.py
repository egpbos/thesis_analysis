#!/usr/bin/env python
import numpy as np
import pandas as pd
from matplotlib import pyplot as pl
from mpl_toolkits.axes_grid1 import Grid
import glob


def load_performance_log(filename, skiprows=1):
    raw = np.loadtxt(filename, skiprows=skiprows)
    if len(raw) == 0:
        raise Exception(filename + " has no entries!")
    nice = {}
    nice['acc'] = raw[:, 0]  # accepted
    nice['eps'] = raw[:, 1]  # epsilon
    nice['Neps'] = raw[:, 2]  # N_epsilon
    nice['dH'] = raw[:, 3]
    nice['dK'] = raw[:, 4]
    nice['dE'] = raw[:, 5]
    nice['dprior'] = raw[:, 6]
    nice['dlikeli'] = raw[:, 7]
    nice['psi_prior_i'] = raw[:, 8]
    nice['psi_prior_f'] = raw[:, 9]
    nice['psi_likeli_i'] = raw[:, 10]
    nice['psi_likeli_f'] = raw[:, 11]
    nice['H_kin_i'] = raw[:, 12]
    nice['H_kin_f'] = raw[:, 13]
    return nice


def subdir_plogs(basedir='.'):
    dirlist = []
    plogs = {}
    filelist = glob.glob(basedir + "/*/performance_log.txt")
    if len(filelist) == 0:
        raise Exception("No performance logs in subdirectories of " +
                        basedir + "!")
    dirlist = [fn[len(basedir)+1:-20] for fn in filelist]
    for dn in dirlist:
        try:
            plogs[dn] = load_performance_log(basedir + '/' + dn +
                                             '/performance_log.txt')
        except:
            continue
    return plogs


# used below
def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

# with directory name /dn/:
get_seed = lambda dn: int(dn[:dn.find('_')-1])
get_N = lambda dn: int(dn[find_nth(dn, '_', 2) + 1 : dn.find('N')])
get_L = lambda dn: int(dn[find_nth(dn, '_', 1) + 1 : dn.find('L')])
get_mass = lambda dn: int(dn[find_nth(dn, '_', 3) + 1 : dn.find('m')])
get_ig = lambda dn: dn[find_nth(dn, '_', 4) + 1 : dn.find('ig')]


def get_eps(dn):
    epsstr = dn[find_nth(dn, '_', 5) + 1 : dn.find('e')]
    if epsstr != 'opt':
        return float(epsstr)
    else:
        return epsstr


def get_sigma(dn):
    sigstr = dn[dn.find('_sig') + 4 : dn.find('d')]
    sigfix, sigdel = sigstr.split("+")
    return float(sigfix), float(sigdel)

get_sigfix = lambda dn: get_sigma(dn)[0]
get_sigdel = lambda dn: get_sigma(dn)[1]

# actual usage of above functions
dn_par_fcts_normalization = {'N': get_N, 'L': get_L, 'e': get_eps, 'm': get_mass, 'i': get_ig, 's': get_seed}
dn_par_fcts_variatie_rond_mean = {'N': get_N, 'L': get_L, 'e': get_eps, 'm': get_mass, 'i': get_ig, 's': get_seed, 'sigfix': get_sigfix, 'sigdel': get_sigdel}


def DataFrame_from_subdir_plogs(plogs, dn_par_fcts=dn_par_fcts_normalization):
    dirlist = list(plogs.keys())
    plog_keys = plogs[dirlist[0]].keys()

    df_dict = {}
    for key in dn_par_fcts.keys():
        df_dict[key] = []
    for key in plog_keys:
        df_dict[key] = []
    df_dict['dir'] = []
    df_dict['i_step'] = []

    for dn in dirlist:
        dn_pars = {}
        for key in dn_par_fcts.keys():
            dn_pars[key] = dn_par_fcts[key](dn)

        len_plog = len(plogs[dn][plog_keys[0]])
        for ix_plog in range(len_plog):
            for key in plog_keys:
                df_dict[key].append(plogs[dn][key][ix_plog])
            for key in dn_par_fcts.keys():
                df_dict[key].append(dn_pars[key])
            df_dict['dir'].append(dn)
            df_dict['i_step'].append(ix_plog)

    df = pd.DataFrame(df_dict)
    return df


def df_plog_from_filenames(filenames):
    plogs = {fn[:fn.rfind('/')]:load_performance_log(fn) for fn in filenames}
    return DataFrame_from_subdir_plogs(plogs)


def selecteer_data(y_key, df, equal_filters = {}, x = 'N'):
    filter_str = ""

    # filter for each parameter
    for key in equal_filters.keys():
        df = df[df[key] == equal_filters[key]]
        filter_str += " " + key + "=%i" % equal_filters[key]

    selection = df.groupby(x)[y_key].apply(np.hstack)

    return selection, filter_str


def plot_relatie(y_key, df = None, plogs = None, basedir = '.', equal_filters = {}, x = 'N', y_agg = np.mean, logx = True, logy = True, plot_type = 'line', legend = True, y_erragg = None):
    """y_erragg must be an aggregation function like np.std, or None."""
    if not isinstance(df, pd.DataFrame):
        if not plogs:
            plogs = subdir_plogs(basedir)
        df = DataFrame_from_subdir_plogs(plogs)

    selection, filter_str = selecteer_data(y_key, df, equal_filters = equal_filters, x = x)

    plot_args = {"kind": plot_type,
                 "label": y_key + " (" + y_agg.__name__ + ")" + filter_str,
                 "legend": legend}

    if logx and logy:
        plot_args['loglog'] = True
    else:
        plot_args['logx'] = logx
        plot_args['logy'] = logy

    aggregate = selection.apply(y_agg)
    aggregate.plot(**plot_args)
    # NOG NIET BESCHIKBAAR (TOT PANDAS 0.14):
    # if not y_erragg:
    #     aggregate.plot(**plot_args)
    # else:
    #     errors = selection.apply(y_erragg)
    #     aggregate.plot(yerr = errors, **plot_args)


def NGibbs_from_plog(plog):
    NGibbs = np.cumsum(plog['acc']).astype('int64')
    return NGibbs


def add_NGibbs_ticks(ax, NGibbs, logx=False, add_ticks=[], ticks_xlim=True):
    ix = np.arange(len(NGibbs))

    ax2 = ax.twiny()
    if logx:
        ax2.set_xscale('log')
    ax2Ticks = ax.get_xticks()

    xlim = ax.get_xlim()
    # add ticks at xlims (if not already present)
    if ticks_xlim:
        for xl in xlim:
            if xl not in ax2Ticks:
                ax2Ticks = np.append(ax2Ticks, xl)

    # remove ticks larger than last iteration
    ax2Ticks = ax2Ticks[ax2Ticks < len(NGibbs)]

    # set stuff
    ax2.set_xticks(ax2Ticks)

    ax2TicksMinor = ax.get_xticks(minor=True)

    # add custom added ticks
    for tick in add_ticks:
        ix_NGibbs = NGibbs.searchsorted(tick)
        ax2TicksMinor = np.append(ax2TicksMinor, ix_NGibbs)

    # add minor tick for last step (if not already present)
    if ix[-1] not in ax2TicksMinor:
        ax2TicksMinor = np.append(ax2TicksMinor, ix[-1])

    ax2.set_xticks(ax2TicksMinor, minor=True)

    ax2.set_xlim(xlim)

    ax2Ticklabels = [NGibbs[ix.searchsorted(tick_val)] for tick_val in
                     ax2Ticks]
    ax2.set_xticklabels(ax2Ticklabels, minor=False, rotation=90)

    ax2TickMinorlabels = [NGibbs[ix.searchsorted(tick_val)] for tick_val in
                          ax2TicksMinor]
    ax2.set_xticklabels(ax2TickMinorlabels, minor=True, rotation=90)
    ax2.set_xlabel('N_Gibbs')
    return ax2


### Old performance log functions from barcode_interactive_analysis.py:

def plot_windowed_accrate(accepted, window_size, interval, name, ax = None, acc_opt = 0.65, xlim = (0, 1000), ylim = (0, 1)):
    if not ax:
        fig = pl.figure()
        ax = fig.add_subplot(111)
    windows_shape = (len(accepted) - window_size, window_size)
    windows_strides = (accepted.strides[0], accepted.strides[0])  # no skipping items
    windows = np.lib.stride_tricks.as_strided(accepted, windows_shape, windows_strides)
    accrate_full_windowed = windows.mean(axis=1)
    accrate_full_pre_windowed = np.cumsum(accepted[:window_size]) / np.arange(1, window_size+1)
    accrate_full = np.hstack((accrate_full_pre_windowed, accrate_full_windowed))

    ax.set_title(name)
    accrate = accrate_full[::interval]
    iteration_nr = np.arange(0, len(accrate_full), interval)
    ax.plot(iteration_nr, accrate, '.', label='acc.rate')
    ax.axhline(y=acc_opt, c='brown', ls='--', label='optimal')
    acc200 = np.cumsum(accepted).searchsorted(200)
    ax.axvline(x=acc200, c='magenta', ls=':', label="200 acc'ed")
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_xlabel("iteration attempt #")
    ax.set_ylabel("windowed acceptance rate")

def plot_comparisons(plogs, plog_filenames, order, rows_cols = (3, 6)):
    # theoretical optimal mean and rate 
    dH_mean_opt = 0.41
    acc_opt = 0.65

    interval = 100
    
    xlim = (0, 20000)

    # determine figure size by sizes of individual subplots
    figsize = (rows_cols[1] * 20./6, rows_cols[0] * 12.5/3)

    # total acceptance rate
    fig_acc = pl.figure(figsize = figsize)
    fig_acc.suptitle("Total accumulated acceptance rate after # iteration attempts. Only showing one in each %i iterations." % interval)
    axgrid_acc = Grid(fig_acc, rect=111, nrows_ncols=rows_cols, axes_pad=0.25, label_mode='L')

    for i in range(rows_cols[0]*rows_cols[1]):
        j = order[i]
        accepted = np.cumsum(plogs[j]['acc'])
        accrate_full = accepted / np.arange(1, len(plogs[j]['acc'])+1)
        name = plog_filenames[j][43:plog_filenames[j].rfind("/")]
        axgrid_acc[i].set_title(name)
        accrate = accrate_full[::interval]
        iteration_nr = np.arange(0, len(accrate_full), interval)
        axgrid_acc[i].plot(iteration_nr, accrate, '.', label='acc.rate')
        axgrid_acc[i].axhline(y=acc_opt, c='brown', ls='--', label='optimal')
        acc200 = accepted.searchsorted(200)
        axgrid_acc[i].axvline(x=acc200, c='magenta', ls=':', label="200 acc'ed")
        axgrid_acc[i].axhline(y=accrate_full[acc200:].mean(), c='magenta', ls=':', label="mean, acc'ed > 200")
        axgrid_acc[i].set_ylim(0, 1)
        axgrid_acc[i].set_xlim(xlim)
        axgrid_acc[i].set_xlabel("iteration attempt #")
        axgrid_acc[i].set_ylabel("acceptance rate till then")

    pl.tight_layout()
    pl.subplots_adjust(top=0.9)  # to make sure doesn't overlap with suptitle

    # windowed acceptance rate
    window_size = 500

    fig_accwin = pl.figure(figsize = figsize)
    fig_accwin.suptitle("Windowed acceptance rate (accumulated over %i previous iterations) after # iteration attempts. Only showing one in each %i iterations." % (window_size, interval))
    axgrid_accwin = Grid(fig_accwin, rect=111, nrows_ncols=rows_cols, axes_pad=0.25, label_mode='L')

    for i in range(rows_cols[0]*rows_cols[1]):
        j = order[i]
        accepted = plogs[j]['acc']
        name = plog_filenames[j][43:plog_filenames[j].rfind("/")]
        ax = axgrid_accwin[i]

        plot_windowed_accrate(accepted, window_size, interval, name, ax, acc_opt = acc_opt, xlim = xlim)

    pl.tight_layout()
    pl.subplots_adjust(top=0.9)  # to make sure doesn't overlap with suptitle

    # dH
    fig_dH = pl.figure(figsize = figsize)
    fig_dH.suptitle("dH for each iteration attempt. Only showing one in each %i iterations." % interval)
    axgrid_dH = Grid(fig_dH, rect=111, nrows_ncols=rows_cols, axes_pad=0.25, label_mode='L')

    for i in range(rows_cols[0]*rows_cols[1]):
        j = order[i]

        name = plog_filenames[j][43:plog_filenames[j].rfind("/")]
        axgrid_dH[i].set_title(name)
        axgrid_dH[i].set_yscale('log')

        dH_j_full = plogs[j]['dH']
        dH_j = dH_j_full[::interval]

        pos = np.argwhere(dH_j >= 0)
        neg = np.argwhere(dH_j < 0)
        iteration_nr = np.arange(0, len(dH_j_full), interval)
        axgrid_dH[i].plot(iteration_nr[pos], dH_j[pos], '.b', label='dH > 0')
        axgrid_dH[i].plot(iteration_nr[neg], -dH_j[neg], '.r', label='-dH > 0')
        axgrid_dH[i].axhline(y=dH_mean_opt, c='brown', ls='--', label='optimal')

        accepted = np.cumsum(plogs[j]['acc'])
        acc200 = accepted.searchsorted(200)

        axgrid_dH[i].axvline(x=acc200, c='magenta', ls=':', label="200 acc'ed")
        axgrid_dH[i].axhline(y=plogs[j]['dH'][acc200:].mean(), c='magenta', ls=':', label="mean, acc'ed > 200")
        axgrid_dH[i].set_xlim(xlim)
        axgrid_dH[i].set_xlabel("iteration attempt #")
        axgrid_dH[i].set_ylabel("dH")

    pl.tight_layout()
    pl.subplots_adjust(top=0.9)  # to make sure doesn't overlap with suptitle

    # convergence: likelihood (chi^2 for gaussian likelihood)
    fig_chisq = pl.figure(figsize = figsize)
    fig_chisq.suptitle("Convergence measure: likelihood (is chi-squared for gaussian likelihood) at beginning of iteration. Only showing one in each %i iterations." % interval)
    axgrid_chisq = Grid(fig_chisq, rect=111, nrows_ncols=rows_cols, axes_pad=0.25, label_mode='L')

    for i in range(rows_cols[0]*rows_cols[1]):
        j = order[i]

        likeli_full = plogs[j]['psi_likeli_i']
        likeli = likeli_full[::interval]

        name = plog_filenames[j][43:plog_filenames[j].rfind("/")]
        axgrid_chisq[i].set_title(name)

        iteration_nr = np.arange(0, len(likeli_full), interval)
        axgrid_chisq[i].plot(iteration_nr, likeli, '.', label='likelihood')

        axgrid_chisq[i].set_xlim(xlim)
        axgrid_chisq[i].set_xlabel("iteration attempt #")
        axgrid_chisq[i].set_ylabel("likelihood")

    pl.tight_layout()
    pl.subplots_adjust(top=0.9)  # to make sure doesn't overlap with suptitle

    axgrid_chisq[rows_cols[1]-1].legend()
    axgrid_dH[rows_cols[1]-1].legend()
    axgrid_acc[rows_cols[1]-1].legend()
    axgrid_accwin[rows_cols[1]-1].legend()

    return fig_acc, fig_accwin, fig_dH, fig_chisq
