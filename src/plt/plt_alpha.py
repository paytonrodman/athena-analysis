#!/usr/bin/python
# Python standard modules
import argparse
import sys
import os

dir_path = os.path.dirname(__file__)
lib_path = os.path.join(dir_path, '..', '..', 'dependencies')
sys.path.append(lib_path)

# Other Python modules
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

    labels = []
    colors = []
    # get pre-defined labels and line colours for each simulation
    for f in args.file:
        slash_list = [m.start() for m in re.finditer('/', f.name)]
        prob_id = f.name[slash_list[-2]+1:slash_list[-1]]
        l,c,_ = AAT.problem_dictionary(prob_id, args.pres)
        labels.append(l)
        colors.append(c)

    t_lists = [[] for _ in range(n)]
    Max_lists = [[] for _ in range(n)]
    Rey_lists = [[] for _ in range(n)]
    a_lists = [[] for _ in range(n)]
    if args.orbits:
        time_col = 'orbit_time'
    else:
        time_col = 'sim_time'

    for count,f in enumerate(args.file):
        df = pd.read_csv(f, delimiter='\t', usecols=[time_col, 'av_Max', 'T_rphi', 'alpha'])
        t = df[time_col].to_list()
        Max_stress = df['av_Max'].to_list()
        T_rphi = df['T_rphi'].to_list()
        alpha = df['alpha'].to_list()

        Rey_stress = [T_rphi[ii] - Max_stress[ii] for ii,val in enumerate(T_rphi)]
        if args.orbits:
            t_lists[count] = t
        else:
            t_lists[count] = [ti/1e5 for ti in t] # convert time to units of 10^5 GM/c3
        Max_lists[count] = Max_stress
        Rey_lists[count] = Rey_stress
        a_lists[count] = alpha

    for ii in range(n):
        t_lists[ii], Max_lists[ii], Rey_lists[ii], a_lists[ii] = zip(*sorted(zip(t_lists[ii], Max_lists[ii], Rey_lists[ii], a_lists[ii])))

    lw = 1.5
    if args.orbits:
        x_label = r'time [ISCO orbits]'
    else:
        x_label = r'time [$10^5~GM/c^3$]'
    if not args.stresses:
        y_label = r'$\langle\alpha_{\rm SS, eff}\rangle$'
    else:
        y_label = r'Stress'

    _, ax1 = plt.subplots(nrows=1, ncols=1, constrained_layout=True, sharex=True)
    if n>1:
        for ii in range(n):
            if not args.stresses:
                ax1.plot(t_lists[ii], a_lists[ii], linewidth=lw, color=colors[ii], label=labels[ii])
            else:
                ax1.plot([],[], color=colors[ii], label=labels[ii])
                ax1.plot(t_lists[ii], Rey_lists[ii], linewidth=0.5, linestyle='dashed', alpha=0.5, color=colors[ii])
                ax1.plot(t_lists[ii], Max_lists[ii], linewidth=0.5, linestyle='dotted', color=colors[ii])
    else:
        if not args.stresses:
            ax1.plot(t_lists[ii], a_lists[ii], linewidth=lw, color=colors[ii], label=r'$\langle\alpha_{\rm SS, eff}\rangle$')
        else:
            ax1.plot(t_lists[ii], Rey_lists[ii], linewidth=0.5, linestyle='dashed', alpha=0.5, color=colors[ii], label='Reynolds stress')
            ax1.plot(t_lists[ii], Max_lists[ii], linewidth=0.5, linestyle='dotted', color=colors[ii], label='Maxwell stress')

    if args.ymax is not None:
        ax1.set_ylim(top=args.ymax)

    ax1.set_xlabel(x_label)#, x=0.5, y=-0.03)
    ax1.set_ylabel(y_label)

    if n==1:
        plt.title(label=labels[0])

    if args.stresses:
        custom_lines = [Line2D([0],[0], color='k', linestyle='dashed', alpha=0.5),
                        Line2D([0],[0], color='k', linestyle='dotted')]
        leg2 = plt.legend(custom_lines, ['Reynolds stress', 'Maxwell stress'], loc=2)
        leg1 = ax1.legend(loc='best')
        plt.gca().add_artist(leg2)
    else:
        leg1 = ax1.legend(loc='best')

    #for line in leg1.get_lines():
    #    line.set_linewidth(1.0)
    plt.savefig(args.output, bbox_inches='tight')
    plt.close()


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot plasma beta over time.')
    parser.add_argument('-f', '--file',
                        type=argparse.FileType('r'),
                        nargs='+',
                        default=None,
                        help='list of data files to read, including path')
    parser.add_argument('-o', '--output',
                        type=str,
                        default=None,
                        help='name of plot to be created, including path')
    parser.add_argument('--ymax',
                        type=float,
                        default=None,
                        help='ymax of plot')
    parser.add_argument('--stresses',
                        action='store_true',
                        help='plot constituent stresses (Maxwell & Reynolds)')
    parser.add_argument('--orbits',
                        action='store_true',
                        help='plot against number of ISCO orbits')
    parser.add_argument('--pres',
                        action='store_true',
                        help='make presentation-quality image')
    args = parser.parse_args()

    main(**vars(args))
