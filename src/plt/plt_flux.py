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
    data_dir = '/Users/paytonrodman/athena_sim/' + problem + '/'
    os.chdir(data_dir)

    time = []
    mag_flux_u = []
    mag_flux_l = []
    with open('flux_with_time.csv', newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader, None) # skip header
        for row in csv_reader:
            t = float(row[0])
            mf_u = float(row[1])
            mf_l = float(row[2])

            time.append(float(t)*5.)
            mag_flux_u.append(mf_u)
            mag_flux_l.append(mf_l)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(time,mag_flux_u,label='upper hem.')
    ax.plot(time,mag_flux_l,label='lower hem.')
    plt.legend(loc='best')
    ax.set_xlabel('time since start (simulation units)')
    ax.set_ylabel(r'average $\Phi_{B}$')
    ax.set_xlim(left=0)
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.savefig('/Users/paytonrodman/athena_sim/' + problem + '/' + 'mag_flux' + '.png',dpi=1200)
    plt.close()

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot various quality factors.')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    args = parser.parse_args()

    main(**vars(args))
