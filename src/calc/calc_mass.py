#!/usr/bin/env python3
#
# calc_mass.py
#
# A program to calculate the mass accretion rate of an Athena++ disk using MPI.
#
# Usage: mpirun -n [nprocs] calc_mass.py [options]
#
# Output: a .csv file containing
#         (1) simulation file number
#         (2) simulation time in GM/c^3
#         (3) simulation time in N_(ISCO orbits)
#         (4) average mass flux through ISCO
#
# Python standard modules
import argparse
import sys
import os

dir_path = os.path.dirname(__file__)
lib_path = os.path.join(dir_path, '..', '..', 'dependencies')
sys.path.append(lib_path)

# Other Python modules
import numpy as np
from mpi4py import MPI
import csv

# Athena++ modules (require sys.path.append above)
import athena_read
import AAT

def main(**kwargs):
    # get number of processors and processor rank
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # check that the output file has type .csv
    root, ext = os.path.splitext(args.output)
    if ext != '.csv':
        ext = '.csv'
    output_file = root + ext

    os.chdir(args.data)

    # make list of files/times to analyse, distribute to cores
    file_times = AAT.add_time_to_list(args.update, output_file)
    local_times = AAT.distribute_files_to_cores(file_times, size, rank)

    init_data = athena_read.athdf(args.problem_id + '.cons.00000.athdf', quantities=['x1v'])
    x1v_init = init_data['x1v'] # r
    r_id = AAT.find_nearest(x1v_init, 6.)

    if rank==0:
        if not args.update: # create output file with header
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['sim_time', 'orbit_time', 'mass_flux'])
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_cons = athena_read.athdf(args.problem_id + '.cons.' + str_t + '.athdf',
                            quantities=['dens','mom1','mom2','mom3'])

        #unpack data
        x2v = data_cons['x2v']
        x3v = data_cons['x3v']
        x1f = data_cons['x1f']
        x2f = data_cons['x2f']
        x3f = data_cons['x3f']
        dens = data_cons['dens']
        mom1 = data_cons['mom1']
        mom2 = data_cons['mom2']
        mom3 = data_cons['mom3']

        _,dx2f,dx3f = AAT.calculate_delta(x1f,x2f,x3f)
        v1,_,_ = AAT.calculate_velocity(mom1,mom2,mom3,dens)

        mf_i = []
        for j in range(len(x2v)):
            for k in range(len(x3v)):
                dS = (x1f[r_id])**2. * np.sin(x2f[j]) * dx2f[j] * dx3f[k] # r^2 sin(theta) dtheta dphi
                mf_i.append(-dens[k,j,r_id] * v1[k,j,r_id] * dS)
        mf_local = np.sum(mf_i)

        sim_t = data_cons['Time']
        orbit_t = AAT.calculate_orbit_time(sim_t)

        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            row = [sim_t,orbit_t,mf_local]
            writer.writerow(row)


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate instantaneous mass flux across inner radial boundary')
    parser.add_argument('problem_id',
                        type=str,
                        help='root name for data files, e.g. b200')
    parser.add_argument('data',
                        type=str,
                        help='location of data folder, possibly including path')
    parser.add_argument('output',
                        type=str,
                        help='name of csv output to be (over)written, possibly including path')
    parser.add_argument('-u', '--update',
                        action='store_true',
                        help='append new results to an existing data file')
    args = parser.parse_args()

    main(**vars(args))
