#!/usr/bin/env python3
#
# calc_alpha.py
#
# A program to calculate the azimuthally-averaged field of an Athena++ disk using MPI,
# to determine the effective alpha (stress)
#
# To run:
# mpirun -n [n] python calc_alpha.py [options]
# for [n] cores.
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
import argparse
from mpi4py import MPI
import matplotlib
import matplotlib.pyplot as plt

def main(**kwargs):
    # get number of processors and processor rank
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    problem  = args.prob_id
    root_dir = "/Users/paytonrodman/athena-sim/"
    #root_dir = '/home/per29/rds/rds-accretion-zyNhkonJSR8/'
    prob_dir = root_dir + problem + '/'
    data_dir = prob_dir + 'data/'
    os.chdir(data_dir)

    # check if data file already exists
    csv_time = np.empty(0)
    if args.update:
        with open(prob_dir + 'alpha_with_time.csv', 'r', newline='') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            next(csv_reader, None) # skip header
            for row in csv_reader:
                csv_time = np.append(csv_time, float(row[0]))

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

    times = [0,5000,10000,15000,20000,25000,30000]
    #times = [25000]

    # distribute files to cores
    count = len(times) // size  # number of files for each process to analyze
    remainder = len(times) % size  # extra files if times is not a multiple of size
    if rank < remainder:  # processes with rank < remainder analyze one extra catchment
        start = rank * (count + 1)  # index of first file to analyze
        stop = start + count + 1  # index of last file to analyze
    else:
        start = rank * count + remainder
        stop = start + count
    local_times = times[start:stop] # get the times to be analyzed by each rank

    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_cons = athena_read.athdf(problem + '.cons.' + str_t + '.athdf', quantities=['x1v','x2v','x3v','Bcc1','Bcc3'])
        data_prim = athena_read.athdf(problem + '.prim.' + str_t + '.athdf', quantities=['press'])

        #unpack data
        x1v = data_cons['x1v']
        x2v = data_cons['x2v']
        Bcc1 = data_cons['Bcc1']
        Bcc3 = data_cons['Bcc3']
        press = data_prim['press']

        stress = np.average((Bcc1*Bcc3),axis=2)
        alpha = stress/np.average(press,axis=2)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        max_extent = np.max(np.abs(np.asarray(alpha).T))
        if max_extent==0:
            max_extent = 1e-5
        #max_extent = 0.4
        #norm = matplotlib.colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=-max_extent, vmax=max_extent)
        #norm = matplotlib.colors.Normalize(vmin=-max_extent, vmax=max_extent)
        pos = ax.imshow(alpha.T,
                        extent = [np.min(x1v),np.max(x1v),np.min(x2v),np.max(x2v)],
                        aspect = 'auto',
                        cmap = 'RdBu',
                        norm = matplotlib.colors.SymLogNorm(linthresh=0.01, linscale=0.01, vmin=-max_extent, vmax=max_extent, base=10))
        ax.set_xlabel(r'$r$',fontsize=14)
        ax.set_ylabel(r'$\theta$',fontsize=14)
        cbar1 = plt.colorbar(pos,extend='both')
        cbar1.ax.set_ylabel(r'$\alpha_{\rm eff}$',fontsize=14)

        plt.savefig(prob_dir + problem + '.' + str_t +'.png', bbox_inches='tight',dpi=1200)
        plt.close()


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate average magnetic field values within the midplane at a given radius (r)')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='specify whether the results being analysed are from a restart')
    args = parser.parse_args()

    main(**vars(args))
