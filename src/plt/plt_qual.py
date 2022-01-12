#!/usr/bin/python
import numpy as np
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

    filename_csv = args.input + '.csv'

    time = []
    theta_B = []
    Qt_lc,Qt_av,Qt_uc = [],[],[]
    Qp_lc,Qp_av,Qp_uc = [],[],[]
    with open(filename_csv, newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader, None) # skip header
        for row in csv_reader:
            t = float(row[0])
            #t_orb = float(row[1])
            tB_i = float(row[2])
            Qt_all = np.fromstring(row[3].strip("[]"), sep=',')
            Qp_all = np.fromstring(row[4].strip("[]"), sep=',')

            Qt_lc.append(Qt_all[0])
            Qt_av.append(Qt_all[1])
            Qt_uc.append(Qt_all[2])

            Qp_lc.append(Qp_all[0])
            Qp_av.append(Qp_all[1])
            Qp_uc.append(Qp_all[2])

            time.append(t)
            theta_B.append(tB_i)


            # Allow wrapping over 0,pi/2,pi etc to track evolution of theta_B more smoothly
            #if len(tB)>1:
            #    prev_tB = tB[-1]
            #    lower_diff = tB_i - 90.
            #    upper_diff = tB_i + 90.
            #    if (np.abs(prev_tB-lower_diff)) < (np.abs(prev_tB-tB_i)):
            #        theta_B.append(lower_diff)
            #    elif (np.abs(prev_tB-upper_diff)) < (np.abs(prev_tB-tB_i)):
            #        theta_B.append(upper_diff)
            #    else:
            #        theta_B.append(tB_i)
            #else:
            #    theta_B.append(tB_i)
    time,theta_B,Qt_lc,Qt_av,Qt_uc,Qp_lc,Qp_av,Qp_uc = zip(*sorted(zip(time,theta_B,Qt_lc,Qt_av,Qt_uc,Qp_lc,Qp_av,Qp_uc)))

    make_plot(time,theta_B,r'$\theta_B$ (degrees)','theta_B',data_dir)
    make_plot_CI(time,Qt_av,Qt_lc,Qt_uc,r'average $Q_{\theta}$','Q_theta',data_dir)
    make_plot_CI(time,Qp_av,Qp_lc,Qp_uc,r'average $Q_{\phi}$','Q_phi',data_dir)

def make_plot(x,y,ylabel,yname,data_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y)
    ax.set_xlabel(r'time ($GM/c^3$)')
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.savefig(data_dir + yname + '.png',dpi=1200)
    plt.close()

def make_plot_CI(x,y,y_CI_low,y_CI_high,ylabel,yname,data_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y)
    plt.fill_between(x,y_CI_low,y_CI_high,alpha=0.2)
    ax.set_xlabel(r'time ($GM/c^3$)')
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.savefig(data_dir + yname + '.png',dpi=1200)
    plt.close()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot various quality factors.')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-i','--input',
                        default='qual_with_time_0_418_127_128',
                        help='name of the file with quality data to be analysed, including extension, e.g. qual_with_time.csv (default=qual_with_time_0_418_127_128)')
    parser.add_argument('-s', '--slice',
                        action='store_true',
                        help='plot a vertical slice (phi=0) of phi and/or theta Quality factors')

    args = parser.parse_args()

    main(**vars(args))
