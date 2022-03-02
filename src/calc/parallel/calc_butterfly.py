#!/usr/bin/env python3
#
# calc_butterfly.py
#
# A program to calculate the azimuthally-averaged field of an Athena++ disk using MPI,
# for use in butterfly plots
#
# To run:
# mpirun -n [n] python calc_butterfly.py [options]
# for [n] cores.
#
import numpy as np
import os
import sys
#sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')
import athena_read
import AAT
import glob
import re
import csv
import argparse
from math import sqrt
from mpi4py import MPI

def main(**kwargs):
    # get number of processors and processor rank
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    root_dir = "/Users/paytonrodman/athena-sim/"
    #root_dir = '/home/per29/rds/rds-accretion-zyNhkonJSR8/'
    prob_dir = root_dir + args.prob_id + '/'
    data_dir = prob_dir + 'data/'
    filename_output = 'butterfly_with_time.csv'
    os.chdir(data_dir)

    # check if data file already exists
    csv_times = np.empty(0)
    if args.update:
        with open(prob_dir + filename_output, 'r', newline='') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            next(csv_reader, None) # skip header
            for row in csv_reader:
                csv_times = np.append(csv_times, float(row[0]))

    # compile a list of unique times associated with data files
    files = glob.glob('./' + args.prob_id + '.cons.*.athdf')
    file_times = np.empty(0)
    for f in files:
        current_time = re.findall(r'\b\d+\b', f)
        if args.update:
            if float(current_time[0]) not in file_times and float(current_time[0]) not in csv_times:
                file_times = np.append(file_times, float(current_time[0]))
        else:
            if float(current_time[0]) not in times:
                file_times = np.append(file_times, float(current_time[0]))
    if len(file_times)==0:
        sys.exit('No new timesteps to analyse in the given directory. Exiting.')

    # distribute files to cores
    files_per_process = len(times) // size
    remainder = len(times) % size
    if rank < remainder:  # processes with rank < remainder analyze one extra catchment
        start = rank * (files_per_process + 1)
        stop = start + files_per_process + 1
    else:
        start = rank * files_per_process + remainder
        stop = start + files_per_process

    local_times = times[start:stop] # get the times to be analyzed by each rank

    data_init = athena_read.athdf(args.prob_id + '.cons.00000.athdf', quantities=['x1v'])
    x1v_init = data_init['x1v']
    if kwargs['r'] is not None:
        r_id = AAT.find_nearest(x1v_init, kwargs['r'])
    else:
        r_id = AAT.find_nearest(x1v_init, 25.) # approx. middle of high res region

    if rank==0:
        if not args.update:
            with open(prob_dir + filename_output, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["sim_time", "orbit_time", "Bcc1", "Bcc2", "Bcc3"])
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_cons = athena_read.athdf(args.prob_id + '.cons.' + str_t + '.athdf', quantities=['Bcc1','Bcc2','Bcc3'])

        #unpack data
        Bcc1 = data_cons['Bcc1']
        Bcc2 = data_cons['Bcc2']
        Bcc3 = data_cons['Bcc3']

        Bcc1_theta = np.average(Bcc1[:,:,r_id],axis=0).tolist()
        Bcc2_theta = np.average(Bcc2[:,:,r_id],axis=0).tolist()
        Bcc3_theta = np.average(Bcc3[:,:,r_id],axis=0).tolist()

        r_ISCO = 6. # location of ISCO in PW potential
        T_period = 2.*np.pi*sqrt(r_ISCO)*(r_ISCO - 2.)
        sim_t = data_cons['Time']
        orbit_t = sim_t/T_period

        with open(prob_dir + filename_output, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([sim_t,orbit_t,Bcc1_theta,Bcc2_theta,Bcc3_theta])


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate average magnetic field values within the midplane at a given radius (r)')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='specify whether the results being analysed are from a restart')
    parser.add_argument('--r',
                        type=float,
                        default=None,
                        help=('value of r where averages are calculated (default: 25rg)'))
    args = parser.parse_args()

    main(**vars(args))
