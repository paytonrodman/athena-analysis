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
import os
import sys
sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
#sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')
import csv
import argparse
from mpi4py import MPI
import numpy as np
import athena_read
import AAT

def main(**kwargs):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #root_dir = "/Users/paytonrodman/athena-sim/"
    root_dir = '/home/per29/rds/rds-accretion-zyNhkonJSR8/'
    prob_dir = root_dir + kwargs['prob_id'] + '/'
    data_dir = prob_dir + 'data/'
    filename_output = 'scale_with_time.csv'
    os.chdir(data_dir)

    file_times = AAT.add_time_to_list(kwargs['update'], prob_dir, filename_output, kwargs['prob_id'])
    local_times = AAT.distribute_files_to_cores(file_times, size, rank)

    # get mesh data for all files (static)
    data_init = athena_read.athdf(kwargs['prob_id'] + '.cons.00000.athdf',
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
        if not kwargs['update']:
            with open(prob_dir + filename_output, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["sim_time", "orbit_time", "scale_height"])
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_cons = athena_read.athdf(kwargs['prob_id'] + '.cons.' + str_t + '.athdf',
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

        with open(prob_dir + filename_output, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            row = [sim_t,orbit_t,scale_h_av]
            writer.writerow(row)


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates the average geometric scale height over the disk')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='specify whether the results being analysed are from a restart')
    args = parser.parse_args()

    main(**vars(args))
