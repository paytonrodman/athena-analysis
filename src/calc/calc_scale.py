#!/usr/bin/env python3
#
# calc_scale.py
#
# A program to calculate the geometric scale height of an Athena++ disk using MPI.
#
# Usage: mpirun -n [nprocs] calc_scale.py [options]
#
# Python standard modules
import argparse
import sys
import os
sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
#sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')

# Other Python modules
import numpy as np
from mpi4py import MPI
import csv

# Athena++ modules
import athena_read
import AAT

def main(**kwargs):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    os.chdir(args.data)

    file_times = AAT.add_time_to_list(args.update, args.output)
    local_times = AAT.distribute_files_to_cores(file_times, size, rank)

    # get mesh data for all files (static)
    data_init = athena_read.athdf(args.problem_id + '.cons.00000.athdf',
                                    quantities=['x1v','x2v','x3v','x1f','x2f','x3f'])
    x1v = data_init['x1v']
    x2v = data_init['x2v']
    x3v = data_init['x3v']
    x1f = data_init['x1f']
    x2f = data_init['x2f']
    x3f = data_init['x3f']

    dx1f,dx2f,dx3f = AAT.calculate_delta(x1f,x2f,x3f)
    _,theta,_ = np.meshgrid(x3v,x2v,x1v, sparse=False, indexing='ij')
    dphi,dtheta,_ = np.meshgrid(dx3f,dx2f,dx1f, sparse=False, indexing='ij')
    dOmega = np.sin(theta)*dtheta*dphi # solid angle

    if rank==0:
        if not args.update:
            with open(args.output, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["sim_time", "orbit_time", "scale_height"])
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_cons = athena_read.athdf(args.problem_id + '.cons.' + str_t + '.athdf',
                                        quantities=['dens'])
        dens = data_cons['dens']

        # calculate the geometric scale height (see Hogg & Reynolds 2018 for details)
        polar_ang = np.sum(theta*dens*dOmega,axis=(0,1))/np.sum(dens*dOmega,axis=(0,1))
        h_up = (theta-polar_ang)**2. * dens*dOmega
        h_down = dens*dOmega
        scale_h = np.sqrt(np.sum(h_up,axis=(0,1))/np.sum(h_down,axis=(0,1)))
        scale_h_av = np.average(scale_h,weights=dx1f)

        sim_t = data_cons['Time']
        orbit_t = AAT.calculate_orbit_time(sim_t)

        with open(args.output, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            row = [sim_t,orbit_t,scale_h_av]
            writer.writerow(row)


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates the average geometric scale height over the disk')
    parser.add_argument('problem_id',
                        help='root name for data files, e.g. high_res')
    parser.add_argument('data',
                        help='location of data folder, possibly including path')
    parser.add_argument('output',
                        help='name of output to be (over)written, possibly including path')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='append new results to an existing data file')
    args = parser.parse_args()

    main(**vars(args))
