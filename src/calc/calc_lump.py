#!/usr/bin/env python3
#
# calc_lump.py
#
# A program that returns a slice of density in the disk midplane, for visualising
# the m=1 mode.
#
# Usage: mpirun -n [nprocs] calc_lump.py [options]
#
# Output: a .csv file containing
#         (1) simulation file number
#         (2) simulation time in GM/c^3
#         (3) simulation time in N_(ISCO orbits)
#         (4) line density, averaged over 3H
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
import pandas as pd
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

    # retrieve lists of scale height with time
    if rank==0:
        df = pd.read_csv(args.scale, delimiter='\t', usecols=['sim_time', 'scale_height'])
        scale_time_list = df['sim_time'].to_list()
        scale_height_list = df['scale_height'].to_list()
    else:
        scale_time_list = None
        scale_height_list = None
    scale_height_list = comm.bcast(scale_height_list, root=0)
    scale_time_list = comm.bcast(scale_time_list, root=0)

    if rank==0:
        if not args.update:  # create output file with header
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['file_time', 'sim_time', 'orbit_time', 'line_density'])

    comm.barrier()
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_cons = athena_read.athdf(args.problem_id + '.cons.' + str_t + '.athdf',
                                        quantities=['dens'])

        # find corresponding entry in scale height data
        scale_index = AAT.find_nearest(scale_time_list,data_cons['Time'])
        scale_height = scale_height_list[scale_index]

        #unpack data
        x2v = data_cons['x2v']
        x3v = data_cons['x3v']
        dens = data_cons['dens']

        # define bounds of region to average over
        th_u = AAT.find_nearest(x2v, np.pi/2. + (3.*scale_height))
        th_l = AAT.find_nearest(x2v, np.pi/2. - (3.*scale_height))
        ph_u = AAT.find_nearest(x3v, np.pi)
        ph_l = AAT.find_nearest(x3v, 0)

        # take averages on each side
        dens_u = np.average(dens[ph_u, th_l:th_u, :], axis=0)
        dens_l = np.average(dens[ph_l, th_l:th_u, :], axis=0)
        dens_all = list(np.concatenate((np.flip(dens_l,0),dens_u), axis=0))

        sim_t = data_cons['Time']
        orbit_t = AAT.calculate_orbit_time(sim_t)

        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            row = [int(t),sim_t,orbit_t,dens_all]
            writer.writerow(row)


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate plasma beta from raw simulation data.')
    parser.add_argument('problem_id',
                        type=str,
                        help='root name for data files, e.g. b200')
    parser.add_argument('data',
                        type=str,
                        help='location of data folder, possibly including path')
    parser.add_argument('scale',
                        type=str,
                        help='location of scale height file, possibly including path')
    parser.add_argument('output',
                        type=str,
                        help='name of csv output to be (over)written, possibly including path')
    parser.add_argument('-u', '--update',
                        action='store_true',
                        help='append new results to an existing data file')
    args = parser.parse_args()

    main(**vars(args))
