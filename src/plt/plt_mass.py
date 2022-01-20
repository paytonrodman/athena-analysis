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
    mass_flux_6rg = []
    mass_flux_25rg = []
    mass_flux_50rg = []
    mass_flux_75rg = []
    mass_flux_100rg = []
    with open('mass_with_time.csv', newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader, None) # skip header
        for row in csv_reader:
            t = float(row[0])
            #t_orb = float(row[1])
            mf_all = literal_eval(row[2])

            time.append(t)
            mass_flux_6rg.append(mf_all[0])
            mass_flux_25rg.append(mf_all[1])
            mass_flux_50rg.append(mf_all[2])
            mass_flux_75rg.append(mf_all[3])
            mass_flux_100rg.append(mf_all[4])

    time,mass_flux_6rg,mass_flux_25rg,mass_flux_50rg,mass_flux_75rg,mass_flux_100rg = zip(*sorted(zip(time, mass_flux_6rg,mass_flux_25rg,mass_flux_50rg,mass_flux_75rg,mass_flux_100rg)))


    y_var_name = 'mass_flux'
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(time,mass_flux_6rg,linewidth=1,label='6rg',alpha=1)
    #ax.plot(time,mass_flux_25rg,linewidth=1,label='25rg',alpha=0.8)
    #ax.plot(time,mass_flux_50rg,linewidth=1,label='50rg',alpha=0.8)
    #ax.plot(time,mass_flux_75rg,linewidth=1,label='75rg',alpha=0.8)
    #ax.plot(time,mass_flux_100rg,linewidth=1,label='100rg',alpha=0.8)

    ax.legend()
    ax.set_xlabel(r'time ($GM/c^3$)')
    ax.set_ylabel(r'$\dot{M}$')
    ax.set_xlim(left=0)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.set_ylim(bottom=1e-2,top=0.6)

    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.savefig(data_dir + y_var_name + '.png',dpi=1200)
    plt.close()


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot average mass flux in each hemisphere.')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    args = parser.parse_args()

    main(**vars(args))
