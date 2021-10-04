#!/usr/bin/python
import numpy as np
import matplotlib
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
    Bcc1 = []
    Bcc2 = []
    Bcc3 = []
    with open('butterfly_with_time.csv', newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader, None) # skip header
        for row in csv_reader:
            t = float(row[0])
            Bcc1_av = row[1]
            Bcc2_av = row[2]
            Bcc3_av = row[3]

            time.append(float(t)*5.)
            Bcc1.append(literal_eval(Bcc1_av))
            Bcc2.append(literal_eval(Bcc2_av))
            Bcc3.append(literal_eval(Bcc3_av))

    theta_min = -np.pi/2.
    theta_max = np.pi/2.
    t_min = 0
    t_max = time[-1]

    make_plots(Bcc1,problem,r'$B_r$','butterfly_Bcc1',t_min,t_max,theta_min,theta_max)
    make_plots(Bcc2,problem,r'$B_\theta$','butterfly_Bcc2',t_min,t_max,theta_min,theta_max)
    make_plots(Bcc3,problem,r'$B_\phi$','butterfly_Bcc3',t_min,t_max,theta_min,theta_max)


def make_plots(data,problem,xlabel,save_name,x_min,x_max,y_min,y_max):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    max_extent = np.max(np.abs(np.asarray(data).T))
    pos = ax.imshow(np.asarray(data).T, extent=[x_min,x_max,y_min,y_max], cmap='seismic', norm=matplotlib.colors.Normalize(vmin=-max_extent, vmax=max_extent),interpolation='bicubic')
    ax.axis('auto')
    ax.set_xlabel('time since start (simulation units)',fontsize=14)
    ax.set_ylabel(r'$\theta - \pi/2$',fontsize=14)
    cbar1 = plt.colorbar(pos,extend='both')
    cbar1.ax.set_ylabel(xlabel,fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(data_dir + save_name + '.png',dpi=1200)
    plt.close()

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Produces butterfly plots of the specified magnetic field components')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    args = parser.parse_args()

    main(**vars(args))
