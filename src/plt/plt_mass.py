#!/usr/bin/python
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')
import csv
from ast import literal_eval
import argparse

def main(**kwargs):
    # directory containing data
    problem  = args.prob_id
    root_dir = '/Users/paytonrodman/athena-sim/'
    data_dir = root_dir + problem + '/'
    os.chdir(data_dir)

    time = []
    mass_flux = []
    with open('mass_with_time.csv', newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader, None) # skip header
        for row in csv_reader:
            t = float(row[0])
            #t_orb = float(row[1])
            mf = literal_eval(row[2])

            time.append(t)
            mass_flux.append(mf[0])

    time, mass_flux = zip(*sorted(zip(time, mass_flux)))


    y_var_name = 'mass_flux'
    _, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    lw = 1.5

    if args.logy:
        ax.semilogy(time,mass_flux,linewidth=lw)
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)
    else:
        ax.plot(time,mass_flux,linewidth=lw)
        plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)

    ax.set_xlabel(r'time ($GM/c^3$)')
    ax.set_ylabel(r'$\dot{M}$')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=1)

    #ax.set_ylim(bottom=1e-2,top=0.6)

    if args.grid:
        plt.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.3)
        plt.minorticks_on()
        plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)

    plt.savefig(data_dir + y_var_name + '.png', bbox_inches='tight')
    plt.close()


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot average mass flux in each hemisphere.')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('--grid',
                        action='store_true',
                        help='plot grid')
    parser.add_argument('--logy',
                        action='store_true',
                        help='plot on logy')
    args = parser.parse_args()

    main(**vars(args))
