#!/usr/bin/python
# Python standard modules
import argparse
import sys
import os

dir_path = os.path.dirname(__file__)
lib_path = os.path.join(dir_path, '..', '..', 'dependencies')
sys.path.append(lib_path)

# Other Python modules
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
        cbar_label = r'$\langle B_r \rangle$'
        col_b = 'Bcc1'
    if args.component=='theta':
        cbar_label = r'$\langle B_\theta \rangle$'
        col_b = 'Bcc2'
    if args.component=='phi':
        cbar_label = r'$\langle B_\phi \rangle$'
        col_b = 'Bcc3'

    n = len(args.file) # number of data files to read

    # read in data
    labels = []
    theta_N = []
    t_lists = [[] for _ in range(n)]
    b_lists = [[] for _ in range(n)]
    t_redundant = [[] for _ in range(n)]
    s_lists = [[] for _ in range(n)]
    if args.orbits:
        time_col = 'orbit_time'
    else:
        time_col = 'sim_time'

    for count,f in enumerate(args.file):
        slash_list = [m.start() for m in re.finditer('/', f.name)]
        prob_id = f.name[slash_list[-2]+1:slash_list[-1]]
        l,_,_ = AAT.problem_dictionary(prob_id, args.pres)
        labels.append(l)

        input_file = f.name[:slash_list[-1]+1] + 'runfiles/athinput.' + prob_id
        data_input = athena_read.athinput(input_file)
        theta_N.append(int(8*data_input['mesh']['nx2']))

        # read butterfly data
        df = pd.read_csv(f, delimiter='\t', usecols=[time_col, col_b])
        t = df[time_col].to_list()
        b = df[col_b].to_list()
        if args.orbits:
            t_lists[count] = t
        else:
            t_lists[count] = [ti/1e5 for ti in t] # convert time to units of 10^5 GM/c3
        for bi in b:
            bi = np.fromstring(bi.strip("[]"), sep=', ')
            b_lists[count].append(bi)

        # read scale height data
        scale_filename = f.name.replace("butterfly", "scale")
        df = pd.read_csv(scale_filename, delimiter='\t', usecols=[time_col, 'scale_height'])
        t = df[time_col].to_list()
        s = df['scale_height'].to_list()
        if args.orbits:
            t_redundant[count] = t
        else:
            t_redundant[count] = [ti/1e5 for ti in t] # convert time to units of 10^5 GM/c3
        s_lists[count] = s

    # arrange in time order
    for idx in range(n):
        t_redundant[idx], s_lists[idx] = zip(*sorted(zip(t_redundant[idx], s_lists[idx])))
        t_lists[idx], b_lists[idx] = zip(*sorted(zip(t_lists[idx], b_lists[idx])))

    data_b = np.empty_like(b_lists)
    data_s = np.empty_like(s_lists)

    for idx in range(n):
        data_b[idx] = np.asarray(b_lists[idx])
        data_s[idx] = np.asarray(s_lists[idx])

    if args.pres:
        w = 5
        ylab = r'$\theta$'
    else:
        w = 8
        ylab = r'$\theta - \pi/2$'

    fig, axs = plt.subplots(nrows=n, ncols=1, constrained_layout=True, figsize=(w,n*2))
    if args.orbits:
        x_label = r'time [ISCO orbits]'
    else:
        x_label = r'time [$10^5~GM/c^3$]'
    for idx in range(n):
        X = t_lists[idx]
        Y = np.linspace(theta_min,theta_max,theta_N[idx])
        Z = data_b[idx].T#*1000.
        max_extent = np.max(np.abs(Z))

        if args.log:
            logthresh = abs(int(np.ceil(np.log10(max_extent)))) + 2
            #norm = matplotlib.colors.SymLogNorm(linthresh=10**-logthresh, linscale=10**-logthresh, vmin=-max_extent, vmax=max_extent, base=10)

        if n==1:
            ax = axs
        else:
            ax = axs[idx]

        pos = ax.pcolorfast(X, Y, Z[:-1,:-1], cmap=cm, vmin=-0.5*max_extent, vmax=0.5*max_extent, rasterized=True)
        ax.plot(t_redundant[idx], 3.*data_s[idx], 'k--', rasterized=True)
        ax.plot(t_redundant[idx], -3.*data_s[idx], 'k--', rasterized=True)
        ax.set_ylabel(ylab)
        at = AnchoredText(labels[idx], prop=dict(size=15), frameon=True, loc='upper left')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)

        if args.log:
            maxlog = int(np.ceil(np.log10(max_extent)))
            minlog = maxlog
            logstep = 1
            tick_locations=([-(10**x) for x in range(-logthresh+1, minlog+1, logstep)][::-1]
                            +[0.0]
                            +[(10**x) for x in range(-logthresh+1, maxlog+1, logstep)] )
            if n==1:
                cbar = fig.colorbar(pos, ax=axs, extend='both', ticks=tick_locations, format=matplotlib.ticker.LogFormatterMathtext())
            else:
                cbar = fig.colorbar(pos, ax=axs[idx], extend='both', ticks=tick_locations, format=matplotlib.ticker.LogFormatterMathtext())
        else:
            if n==1:
                cbar = fig.colorbar(pos, ax=axs, extend='both')
            else:
                cbar = fig.colorbar(pos, ax=axs[idx], extend='both')
        cbar.set_label(cbar_label, rotation=0, y=0.54)

    fig.supxlabel(x_label)#, x=0.45, y=-0.03)
    plt.tick_params(axis='both', which='major')

    plt.savefig(args.output, bbox_inches='tight')
    plt.close()


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Produces butterfly plots of the specified magnetic field components')
    parser.add_argument('-f', '--file',
                        type=argparse.FileType('r'),
                        nargs='+',
                        default=None,
                        help='List of data files to read, including path')
    parser.add_argument('-c', '--component',
                        type=str,
                        default='phi',
                        choices=['r','theta','phi'],
                        help='Magnetic field component to be plotted')
    parser.add_argument('-o', '--output',
                        type=str,
                        default=None,
                        help='Name of plot to be created, including path')
    parser.add_argument('--log',
                        action='store_true',
                        help='Use logarithmic colour scale')
    parser.add_argument('--orbits',
                        action='store_true',
                        help='plot against number of ISCO orbits')
    parser.add_argument('--pres',
                        action='store_true',
                        help='Make presentation-quality image')
    args = parser.parse_args()

    main(**vars(args))
