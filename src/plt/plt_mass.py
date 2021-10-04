#!/usr/bin/python
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')
import csv
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
            mf = float(row[1])

            time.append(float(t)*5.)
            mass_flux.append(mf)


    y_var_name = 'mass_flux'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(time,mass_flux)
    ax.set_xlabel('time since start (simulation units)')
    ax.set_ylabel('surface averaged mass flux')
    ax.set_xlim(left=0)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.savefig('/Users/paytonrodman/athena_sim/' + problem + '/' + y_var_name + '.png',dpi=1200)
    plt.close()


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot average mass flux in each hemisphere.')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    args = parser.parse_args()

    main(**vars(args))
