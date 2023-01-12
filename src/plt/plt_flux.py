#!/usr/bin/python
# Python standard modules
import argparse
import sys
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')
sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')

# Other Python modules
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import re

# Athena++ modules
import AAT

def main(**kwargs):
    n = len(args.file) # number of data files to read
    if not args.file:
        sys.exit('Must specify data files')
    if not args.output:
        sys.exit('Must specify output file')

    min_time = []
    colors = []
    labels = []
    for f in args.file:
        slash_list = [m.start() for m in re.finditer('/', f.name)]
        prob_id = f.name[slash_list[-2]+1:slash_list[-1]]
        l,c,t = AAT.problem_dictionary(prob_id, args.pres)
        labels.append(l)
        colors.append(c)
        min_time.append(t)

    t_lists = [[] for _ in range(n)]
    mfu_lists = [[] for _ in range(n)]
    mfl_lists = [[] for _ in range(n)]
    mfu_abs_lists = [[] for _ in range(n)]
    mfl_abs_lists = [[] for _ in range(n)]
    for count,f in enumerate(args.file):
        filename = f.name
        filename = filename.replace("flux", "mass")

        mass_flux = []
        df = pd.read_csv(filename, delimiter='\t', usecols=['sim_time', 'mass_flux'])
        t = df['sim_time'].to_list()
        m = df['mass_flux'].to_list()
        mass_flux = m
        time_m = t

        time_m, mass_flux = zip(*sorted(zip(time_m, mass_flux)))
        time_m = np.array(time_m)
        mass_flux = np.array(mass_flux)
        mass_flux = np.array(mass_flux)
        mass_average = np.average(mass_flux[time_m > min_time[count]])
        rawtoscaled = 1./(np.sqrt(mass_average))
        scale = np.sqrt(4*np.pi)*rawtoscaled

        df = pd.read_csv(f, delimiter='\t')
        t = df['sim_time'].to_list()
        if args.disk:
            mfu = df['mag_flux_u_disk'].to_list()
            mfl = df['mag_flux_l_disk'].to_list()
            mfu_a = df['mag_flux_u_abs_disk'].to_list()
            mfl_a = df['mag_flux_l_abs_disk'].to_list()
        else:
            mfu = df['mag_flux_u'].to_list()
            mfl = df['mag_flux_l'].to_list()
            mfu_a = df['mag_flux_u_abs'].to_list()
            mfl_a = df['mag_flux_l_abs'].to_list()
        t_lists[count] = [ti/1.e5 for ti in t]
        #t_lists[count] = [ti for ti in t]
        mfu_lists[count] = [x*scale for x in mfu]
        mfl_lists[count] = [x*scale for x in mfl]
        mfu_abs_lists[count] = [x*scale for x in mfu_a]
        mfl_abs_lists[count] = [x*scale for x in mfl_a]

        t_lists[count],mfu_lists[count],mfl_lists[count],mfu_abs_lists[count],mfl_abs_lists[count] = zip(*sorted(zip(t_lists[count],mfu_lists[count],mfl_lists[count],mfu_abs_lists[count],mfl_abs_lists[count])))

    mfu_array = [[] for _ in range(n)]
    mfl_array = [[] for _ in range(n)]
    mfu_abs_array = [[] for _ in range(n)]
    mfl_abs_array = [[] for _ in range(n)]
    for ii in range(n):
        mfu_array[ii] = np.array(mfu_lists[ii], dtype=object)
        mfl_array[ii] = np.array(mfl_lists[ii], dtype=object)
        mfu_abs_array[ii] = np.array(mfu_abs_lists[ii], dtype=object)
        mfl_abs_array[ii] = np.array(mfl_abs_lists[ii], dtype=object)
        #t_lists[ii] = t_lists[ii]

    data_u_ratio = np.array(mfu_array, dtype=object)/np.array(mfu_abs_lists, dtype=object)
    data_l_ratio = np.array(mfl_array, dtype=object)/np.array(mfl_abs_lists, dtype=object)
    data_ratio = (np.abs(data_u_ratio) + np.abs(data_l_ratio))/2.
    data_u = mfu_array
    data_l = mfl_array
    data = (np.abs(data_u) + np.abs(data_l))/2.

    lw=1.5
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8,4))
    fig.subplots_adjust(hspace=0.1)
    x_label = r'time [$10^5~GM/c^3$]'
    y1_label = r'$\Phi / \sqrt{\langle\dot{M}\rangle r_g c^2}$'
    y2_label = r'$\Phi /|\Phi|$'
    for i in range(n):
        axs[0].plot(t_lists[i],data[i],label=labels[i],linewidth=lw,color=colors[i])
        axs[1].plot(t_lists[i],data_ratio[i],label=labels[i],linewidth=lw,color=colors[i])

    axs[0].set_ylabel(y1_label)
    axs[1].set_ylabel(y2_label)
    axs[0].set_ylim(top=50,bottom=0)
    axs[1].set_ylim(top=1,bottom=0)
    axs[1].set_xlabel(x_label)

    for ax in axs:
        ax.set_xlim(left=0)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        if args.grid:
            ax.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.3)
            ax.minorticks_on()
            ax.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)

    leg = axs[0].legend(loc='best')
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.savefig(args.output, bbox_inches='tight')
    plt.close()


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot magnetic flux across both hemispheres over time.')
    parser.add_argument('-f', '--file',
                        type=argparse.FileType('r'),
                        nargs='+',
                        default=None,
                        help='list of data files to read, including path')
    parser.add_argument('-o', '--output',
                        type=str,
                        default=None,
                        help='name of plot to be created, including path')
    parser.add_argument('--grid',
                        action='store_true',
                        help='plot grid')
    parser.add_argument('--disk',
                        action='store_true',
                        help='use disk flux only')
    parser.add_argument('--pres',
                        action='store_true',
                        help='make presentation-quality image')
    args = parser.parse_args()

    main(**vars(args))
