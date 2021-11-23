#!/usr/bin/python3.6
#
# calc_scale.py
#
# A program to calculate the geometric scale height of an Athena++ disk using MPI.
#
# To run:
# mpirun -n [n] python calc_scale.py
# for [n] cores.
#
import sys
sys.path.insert(0,'/home/per29/.local/lib/python3.6/site-packages/')
from mpi4py import MPI
import numpy as np
import os
#sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')
import athena_read
import AAT
import glob
import re
import csv
import argparse

def main(**kwargs):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print('My rank is ',rank)

    problem  = args.prob_id
    root_dir = "/Users/paytonrodman/athena-sim/"
    #root_dir = '/home/per29/rds/rds-accretion-zyNhkonJSR8/'
    prob_dir = root_dir + problem + '/'
    data_dir = prob_dir + 'data/'
    os.chdir(data_dir)

    csv_time = []
    # check if data file already exists
    if args.update:
        with open(prob_dir + 'scale_with_time.csv', 'r', newline='') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            next(csv_reader, None) # skip header
            for row in csv_reader:
                csv_time.append(float(row[0]))
    else: # create empty file
        with open(prob_dir + 'scale_with_time.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["time", "scale_height"])

    files = glob.glob('./*.athdf')
    times = []
    for f in files:
        time_sec = re.findall(r'\b\d+\b', f)
        if args.update:
            if float(time_sec[0]) not in times and float(time_sec[0]) not in csv_time:
                times.append(float(time_sec[0]))
        else:
            if float(time_sec[0]) not in times:
                times.append(float(time_sec[0]))
    if len(times)==0:
        sys.exit('No new timesteps to analyse in the given directory. Exiting.')
    times = sorted(times)

    # get mesh data for all files (static)
    data_init = athena_read.athdf(problem + '.cons.00000.athdf')
    x1v = data_init['x1v'] # r
    x2v = data_init['x2v'] # theta
    x3v = data_init['x3v'] # phi
    x1f = data_init['x1f'] # r
    x2f = data_init['x2f'] # theta
    x3f = data_init['x3f'] # phi
    dx1f,dx2f,dx3f = AAT.calculate_delta(x1f,x2f,x3f)
    phi,theta,r = np.meshgrid(x3v,x2v,x1v, sparse=False, indexing='ij')
    dphi,dtheta,dr = np.meshgrid(dx3f,dx2f,dx1f, sparse=False, indexing='ij')
    dOmega = np.sin(theta)*dtheta*dphi

    scale_height = []
    scale_time = []
    for t_select in range(comm.rank, len(times), size):
        t = times[t_select]
        print('file number: ',t)
        str_t = str(int(t)).zfill(5)

        filename_cons = problem + '.cons.' + str_t + '.athdf'
        data_cons = athena_read.athdf(filename_cons)

        #unpack data
        dens = data_cons['dens']
        # Calculations
        polar_ang = np.sum(theta*dens*dOmega,axis=(0,1))/np.sum(dens*dOmega,axis=(0,1))
        h_up = (theta-polar_ang)**2. * dens*dOmega
        h_down = dens*dOmega
        scale_h = np.sqrt(np.sum(h_up,axis=(0,1))/np.sum(h_down,axis=(0,1)))
        scale_h_av = np.average(scale_h,weights=dx1f)

        # rank 0 gathers data
        sbuf = comm.gather(scale_h_av,root=0)
        tbuf = comm.gather(t,root=0)

        # rank 0 has to write into the HDF file
        if comm.rank == 0:
            # Append each data point
            with open(prob_dir + 'scale_with_time.csv', 'a', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(zip(tbuf,sbuf))

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
