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
    scale_height = []
    with open('scale_with_time.csv', newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader, None) # skip header
        for row in csv_reader:
            t = float(row[0])
            #t_orb = float(row[1])
            sh = float(row[2])

            time.append(t)
            scale_height.append(sh)

    time, scale_height = zip(*sorted(zip(time, scale_height)))

    y_var_name = 'scale_height'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(time,scale_height)
    ax.set_xlabel(r'time ($GM/c^3$)')
    ax.set_ylabel(r'geometric scale height, $h/r$')
    ax.set_xlim(left=0)
    ax.set_ylim(0,0.5)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.savefig(data_dir + y_var_name + '.png',dpi=1200)
    plt.close()


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the scale height of the disk over time.')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    args = parser.parse_args()

    main(**vars(args))
