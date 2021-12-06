#!/usr/bin/env python3
#
# calc_scale.py
#
# A program to calculate the geometric scale height of an Athena++ disk using MPI.
#
# To run:
# mpirun -n [n] python calc_scale.py [options]
# for [n] cores.
#
import sys
import numpy as np
import os
sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
#sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')
import athena_read
import AAT
import glob
import re
import csv
import argparse
import gc # garbage collector
from mpi4py import MPI

def main(**kwargs):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    problem  = args.prob_id
    #root_dir = "/Users/paytonrodman/athena-sim/"
    root_dir = '/home/per29/rds/rds-accretion-zyNhkonJSR8/'
    prob_dir = root_dir + problem + '/'
    data_dir = prob_dir + 'data/'
    runfile_dir = prob_dir + 'runfiles/'
    os.chdir(data_dir)

    csv_time = np.empty(0)
    # check if data file already exists
    if args.update:
        with open(prob_dir + 'scale_with_time.csv', 'r', newline='') as f:
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

    count = len(times) // size  # number of files for each process to analyze
    remainder = len(times) % size  # extra files if times is not a multiple of size
    if rank < remainder:  # processes with rank < remainder analyze one extra catchment
        start = rank * (count + 1)  # index of first file to analyze
        stop = start + count + 1  # index of last file to analyze
    else:
        start = rank * count + remainder
        stop = start + count
    local_times = times[start:stop] # get the times to be analyzed by each rank

    data_input = athena_read.athinput(runfile_dir + 'athinput.' + problem)
    mass = np.copy(data_input['problem']['mass'])
    x1min = np.copy(data_input['mesh']['x1min'])
    del data_input
    # get mesh data for all files (static)
    data_init = athena_read.athdf(problem + '.cons.00000.athdf')
    x1v = np.copy(data_init['x1v']) # r
    x2v = np.copy(data_init['x2v']) # theta
    x3v = np.copy(data_init['x3v']) # phi
    x1f = np.copy(data_init['x1f']) # r
    x2f = np.copy(data_init['x2f']) # theta
    x3f = np.copy(data_init['x3f']) # phi
    del data_init
    gc.collect()

    dx1f,dx2f,dx3f = AAT.calculate_delta(x1f,x2f,x3f)
    phi,theta,r = np.meshgrid(x3v,x2v,x1v, sparse=False, indexing='ij')
    dphi,dtheta,dr = np.meshgrid(dx3f,dx2f,dx1f, sparse=False, indexing='ij')
    dOmega = np.sin(theta)*dtheta*dphi

    if not args.update:
        if rank==0:
            with open(prob_dir + 'scale_with_time.csv', 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["sim_time", "orbit_time", "scale_height"])
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        filename_cons = problem + '.cons.' + str_t + '.athdf'
        data_cons = athena_read.athdf(filename_cons)

        #unpack data
        dens = np.copy(data_cons['dens'])
        del data_cons
        gc.collect()

        # Calculations
        polar_ang = np.sum(theta*dens*dOmega,axis=(0,1))/np.sum(dens*dOmega,axis=(0,1))
        h_up = (theta-polar_ang)**2. * dens*dOmega
        h_down = dens*dOmega
        scale_h = np.sqrt(np.sum(h_up,axis=(0,1))/np.sum(h_down,axis=(0,1)))
        scale_h_av = np.average(scale_h,weights=dx1f)

        v_Kep0 = np.sqrt(mass/x1min)
        Omega0 = v_Kep0/x1min
        T0 = 2.*np.pi/Omega0
        orbit_t = t/T0
        sim_t = t

        with open(prob_dir + 'scale_with_time.csv', 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            row = [sim_t,orbit_t,scale_h_av]
            writer.writerow(row)


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the average geometric scale height over the disk')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='specify whether the results being analysed are from a restart')
    args = parser.parse_args()

    main(**vars(args))
