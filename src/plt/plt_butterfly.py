#!/usr/bin/python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')
sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')
import csv
import AAT
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
            #t_orb = float(row[1])
            Bcc1_av = row[2]
            Bcc2_av = row[3]
            Bcc3_av = row[4]

            time.append(t)
            Bcc1.append(literal_eval(Bcc1_av))
            Bcc2.append(literal_eval(Bcc2_av))
            Bcc3.append(literal_eval(Bcc3_av))

    time, Bcc1, Bcc2, Bcc3 = zip(*sorted(zip(time, Bcc1, Bcc2, Bcc3)))

    theta_min = -np.pi/2.
    theta_max = np.pi/2.
    t_min = 0
    t_max = time[-1]
    logthresh = 4
    logstep = 1

    make_butterfly_plots(Bcc1,data_dir,r'$B_r$','butterfly_Bcc1',t_min,t_max,theta_min,theta_max,None,logthresh,logstep)
    make_butterfly_plots(Bcc2,data_dir,r'$B_\theta$','butterfly_Bcc2',t_min,t_max,theta_min,theta_max,None,logthresh,logstep)
    make_butterfly_plots(Bcc3,data_dir,r'$B_\phi$','butterfly_Bcc3',t_min,t_max,theta_min,theta_max,None,logthresh,logstep)

    #make_cyclic_plots(Bcc3)


def make_cyclic_plots(data,theta_min,theta_max):
    theta = np.linspace(theta_min, theta_max, num=np.shape(data)[1], endpoint=False)
    th_low = AAT.find_nearest(theta,-1.)
    th_upp = AAT.find_nearest(theta,1.)

    Bcc3_reduc = Bcc[:,th_low:th_upp]
    print(np.shape(Bcc3_reduc))


def make_butterfly_plots(data,data_dir,xlabel,save_name,x_min,x_max,y_min,y_max,max_extent,logthresh,logstep):
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(111)
    if max_extent is None:
        max_extent = np.max(np.abs(np.asarray(data).T))
    norm = matplotlib.colors.SymLogNorm(linthresh=10**-logthresh, linscale=10**-logthresh, vmin=-max_extent, vmax=max_extent, base=10)
    #norm = matplotlib.colors.SymLogNorm(linthresh=0.01*max_extent, linscale=0.01*max_extent, vmin=-max_extent, vmax=max_extent, base=10)
    pos = ax.imshow(np.asarray(data).T, extent=[x_min,x_max,y_min,y_max], cmap='seismic', norm=norm)
    ax.axis('auto')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)
    ax.set_xlabel(r'time ($GM/c^3$)')
    ax.set_ylabel(r'$\theta - \pi/2$')
    plt.tick_params(axis='both', which='major')

    #generate logarithmic ticks
    maxlog = int(np.ceil(np.log10(max_extent)))
    minlog = maxlog
    tick_locations=([-(10**x) for x in range(-logthresh+1, minlog+1, logstep)][::-1]
                    +[0.0]
                    +[(10**x) for x in range(-logthresh+1, maxlog+1, logstep)] )
    cbar = fig.colorbar(pos, ax=ax, extend='both', ticks=tick_locations, format=matplotlib.ticker.LogFormatterMathtext())
    cbar.ax.set_ylabel(xlabel, rotation=0)

    #plt.tight_layout()
    plt.savefig(data_dir + save_name + '.png',dpi=1200)
    plt.close()


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Produces butterfly plots of the specified magnetic field components')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    args = parser.parse_args()

    main(**vars(args))
