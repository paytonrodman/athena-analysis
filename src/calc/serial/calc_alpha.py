#!/usr/bin/env python3
#
# calc_alpha.py
#
# A program to calculate the azimuthally-averaged field of an Athena++ disk,
# to determine the effective alpha (stress)
#
# Usage: python calc_alpha.py [options]
#
import numpy as np
import os
import sys
#sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')
import athena_read
import glob
import re
import csv
import AAT
import argparse
import matplotlib
import matplotlib.pyplot as plt

def main(**kwargs):
    problem  = args.prob_id
    root_dir = "/Users/paytonrodman/athena-sim/"
    #root_dir = '/home/per29/rds/rds-accretion-zyNhkonJSR8/'
    prob_dir = root_dir + problem + '/'
    data_dir = prob_dir + 'data/'
    runfile_dir = prob_dir + 'runfiles/'
    os.chdir(data_dir)

    files = glob.glob('./*.athdf')
    times = np.empty(0)
    for f in files:
        time_sec = re.findall(r'\b\d+\b', f)
        if args.update:
            if float(time_sec[0]) not in times and float(time_sec[0]) not in csv_time:
                times = np.append(times, float(time_sec[0]))
        else:
            if float(time_sec[0]) not in times:
                times = np.append(times, float(time_sec[0]))
    if len(times)==0:
        sys.exit('No new timesteps to analyse in the given directory. Exiting.')

    #times = [0.,5000.,10000.,15000.,20000.,25000.,30000.]
    #times = [15000.,17500.,20000.,22500.,25000.,27500.,30000.,32500.,35000.]
    #times = [0.,2500.,5000.,7500.]

    data_input = athena_read.athinput(runfile_dir + 'athinput.' + problem)
    scale_height = data_input['problem']['h_r']
    data_init = athena_read.athdf(problem + '.cons.00000.athdf', quantities=['x1v','x2v'])
    x1v = data_init['x1v']
    x2v = data_init['x2v']
    th_u_id = AAT.find_nearest(x2v, np.pi/2. + scale_height)
    th_l_id = AAT.find_nearest(x2v, np.pi/2. - scale_height)

    av_stress = []
    labels = []
    for t in sorted(times):
        print(t)
        str_t = str(int(t)).zfill(5)
        data_cons = athena_read.athdf(problem + '.cons.' + str_t + '.athdf', quantities=['x1v','x2v','x3v','Bcc1','Bcc3'])
        data_prim = athena_read.athdf(problem + '.prim.' + str_t + '.athdf', quantities=['press'])

        #unpack data
        x1v = data_cons['x1v']
        x2v = data_cons['x2v']
        Bcc1 = data_cons['Bcc1']
        Bcc3 = data_cons['Bcc3']
        press = data_prim['press']

        stress = np.average((-2.*Bcc1*Bcc3),axis=0)
        alpha = stress/np.average(press,axis=0)

        if args.plot_2D:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            max_extent = np.max(np.abs(np.asarray(alpha)))
            if max_extent==0:
                max_extent = 1e-5
            max_extent = 1
            #norm = matplotlib.colors.SymLogNorm(linthresh=0.01, linscale=0.01, vmin=-maxextent, vmax=max_extent, base=10)
            #norm = matplotlib.colors.Normalize(vmin=-max_extent, vmax=max_extent)
            n = matplotlib.colors.LogNorm(vmin=1e-5, vmax=max_extent)
            pos = ax.imshow(alpha,
                            extent = [np.min(x1v),np.max(x1v),np.min(x2v),np.max(x2v)],
                            aspect = 'auto',
                            cmap = 'viridis',
                            norm = n)
            ax.set_xlabel(r'$r$',fontsize=14)
            ax.set_ylabel(r'$\theta$',fontsize=14)
            cbar1 = plt.colorbar(pos,extend='both')
            cbar1.ax.set_ylabel(r'$\alpha_{\rm eff}$',fontsize=14)
            plt.savefig(prob_dir + 'alpha/' + problem + '.' + str_t +'.png', bbox_inches='tight',dpi=1200)
            plt.close()

        if args.plot_1D:
            alpha_sh = np.average(alpha[th_l_id:th_u_id,:],axis=0)
            #av_stress.append(alpha_sh)
            #labels.append(int(t))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x1v,alpha_sh)
            ax.set_xlabel(r'radius ($r_g$)')
            ax.set_ylabel(r'average $\alpha$ (within 1$H/r$)')
            ax.set_xlim(0, 300)
            ax.set_ylim(0, 0.16)
            title_str = 'time = ' + str(data_cons['Time'])
            ax.set_title(title_str)
            ax.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
            ax.minorticks_on()
            ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.tight_layout()
            plt.savefig(prob_dir + 'alpha/average/' + problem + '.average' + '.' + str_t +'.png', bbox_inches='tight',dpi=1200)
            plt.close()

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate average magnetic field values within the midplane at a given radius (r)')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='specify whether the results being analysed are from a restart')
    parser.add_argument('-p2d', '--plot_2D',
                        action="store_true",
                        help='specify whether to plot the image from each timestep')
    parser.add_argument('-p1d', '--plot_1D',
                        action="store_true",
                        help='specify whether to plot the radial profile from each timestep')
    args = parser.parse_args()

    main(**vars(args))
