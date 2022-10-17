#!/usr/bin/python
# Python standard modules
import argparse
import os
import sys
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')
sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')

# Other Python modules
from ast import literal_eval
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
import pandas as pd
import re

# Athena++ modules
import athena_read
import AAT

def main(**kwargs):
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    if not args.file:
        sys.exit('Must specify data files')
    if not args.output:
        sys.exit('Must specify output file')

    theta_min = -np.pi/2.
    theta_max = np.pi/2.
    cm = 'seismic'
    if args.component=='r':
        cbar_label = r'$B_r$'
        col_b = 'Bcc1'
    if args.component=='theta':
        cbar_label = r'$B_\theta$'
        col_b = 'Bcc2'
    if args.component=='phi':
        cbar_label = r'$B_\phi$'
        col_b = 'Bcc3'

    n = len(args.file) # number of data files to read

    # read in butterfly and scale height data
    labels = []
    theta_N = []
    t_lists = [[] for _ in range(n)]
    b_lists = [[] for _ in range(n)]
    t_redundant = [[] for _ in range(n)]
    s_lists = [[] for _ in range(n)]
    for count,f in enumerate(args.file):
        slash_list = [m.start() for m in re.finditer('/', f.name)]
        prob_id = f.name[slash_list[-2]+1:slash_list[-1]]
        l,_,_ = AAT.problem_dictionary(prob_id)
        labels.append(l)

        input_file = f.name[:slash_list[-1]+1] + 'runfiles/athinput.' + prob_id
        data_input = athena_read.athinput(input_file)
        theta_N.append(int(8*data_input['mesh']['nx2']))

        df = pd.read_csv(f, delimiter='\t', usecols=['sim_time', col_b])
        t = df['sim_time'].to_list()
        b = df[col_b].to_list()
        t_lists[count] = t
        for bi in b:
            b_lists[count].append(literal_eval(bi))

        scale_filename = f.name.replace("butterfly", "scale")
        df = pd.read_csv(scale_filename, delimiter='\t', usecols=['sim_time', 'scale_height'])
        t = df['sim_time'].to_list()
        s = df['scale_height'].to_list()
        t_redundant[count] = t
        s_lists[count] = s

    for idx in range(n):
        t_redundant[idx], s_lists[idx] = zip(*sorted(zip(t_redundant[idx], s_lists[idx])))
        t_lists[idx], b_lists[idx] = zip(*sorted(zip(t_lists[idx], b_lists[idx])))

    data_b = np.empty_like(b_lists)
    data_s = np.empty_like(s_lists)
    case = []
    for idx in range(n):
        data_b[idx] = np.asarray(b_lists[idx])
        data_s[idx] = np.asarray(s_lists[idx])
        case.append(np.max(np.abs(data_b[idx])))
    max_extent = max(case)
    #logthresh = 4
    logthresh = abs(int(np.ceil(np.log10(max_extent)))) + 3
    norm = matplotlib.colors.SymLogNorm(linthresh=10**-logthresh, linscale=10**-logthresh, vmin=-max_extent, vmax=max_extent, base=10)

    fig, axs = plt.subplots(nrows=n, ncols=1, constrained_layout=True, figsize=(8,n*2))
    for idx in range(n):
        X = t_lists[idx]
        Y = np.linspace(theta_min,theta_max,theta_N[idx])
        Z = data_b[idx].T

        if n==1:
            pos = axs.pcolorfast(X, Y, Z[:-1,:-1], cmap=cm, norm=norm)
            axs.plot(t_redundant[idx],3.*data_s[idx],'k--')
            axs.plot(t_redundant[idx],-3.*data_s[idx],'k--')
            axs.set_ylabel(r'$\theta - \pi/2$')
            at = AnchoredText(labels[idx], prop=dict(size=15), frameon=True, loc='upper left')
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            axs.add_artist(at)
            axs.ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)
        else:
            pos = axs[idx].pcolorfast(X, Y, Z[:-1,:-1], cmap=cm, norm=norm)
            axs[idx].plot(t_redundant[idx],3.*data_s[idx],'k--')
            axs[idx].plot(t_redundant[idx],-3.*data_s[idx],'k--')
            axs[idx].set_ylabel(r'$\theta - \pi/2$')
            at = AnchoredText(labels[idx], prop=dict(size=15), frameon=True, loc='upper left')
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            axs[idx].add_artist(at)
            for ax in axs:
                ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)

    fig.supxlabel(r'time ($GM/c^3$)', x=0.45, y=-0.03)
    plt.tick_params(axis='both', which='major')

    #generate cbar logarithmic ticks
    maxlog = int(np.ceil(np.log10(max_extent)))
    minlog = maxlog
    logstep = 1
    tick_locations=([-(10**x) for x in range(-logthresh+1, minlog+1, logstep)][::-1]
                    +[0.0]
                    +[(10**x) for x in range(-logthresh+1, maxlog+1, logstep)] )
    cbar = fig.colorbar(pos, ax=axs, extend='both', ticks=tick_locations, format=matplotlib.ticker.LogFormatterMathtext())
    #cbar.set_label(cbar_label, rotation=0)
    cbar.ax.set_title(cbar_label)

    plt.savefig(args.output, bbox_inches='tight')
    plt.close()


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Produces butterfly plots of the specified magnetic field components')
    parser.add_argument('-f', '--file',
                        type=argparse.FileType('r'),
                        nargs='+',
                        default=None,
                        help='list of data files to read, including path')
    parser.add_argument('-c', '--component',
                        type=str,
                        default='phi',
                        choices=['r','theta','phi'],
                        help='magnetic field component to be plotted')
    parser.add_argument('-o', '--output',
                        type=str,
                        default=None,
                        help='name of plot to be created, including path')
    args = parser.parse_args()

    main(**vars(args))
