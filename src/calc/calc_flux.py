#!/usr/bin/env python3
#
# calc_flux.py
#
# A program to calculate the magnetic flux threading the ISCO of an Athena++ disk using MPI
#
# Usage: mpirun -n [nprocs] calc_flux.py [options]
#
# Output: a .csv file containing
#         (1) simulation file number
#         (2) simulation time in GM/c^3
#         (3) simulation time in N_(ISCO orbits)
#         (4) upper-hemisphere-averaged magnetic flux
#         (5) lower-hemisphere-averaged magnetic flux
#         (6) upper-hemisphere-averaged absolute magnetic flux
#         (7) lower-hemisphere-averaged absolute magnetic flux
#         (8) upper-hemisphere-averaged magnetic flux, from the disk only
#         (9) lower-hemisphere-averaged magnetic flux, from the disk only
#         (10) upper-hemisphere-averaged absolute magnetic flux, from the disk only
#         (11) lower-hemisphere-averaged absolute magnetic flux, from the disk only
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
        if not args.update: # create output file with header
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['file_time', 'sim_time', 'orbit_time',
                                    'mag_flux_u', 'mag_flux_l',
                                    'mag_flux_u_abs', 'mag_flux_l_abs',
                                    'mag_flux_u_disk', 'mag_flux_l_disk',
                                    'mag_flux_u_abs_disk', 'mag_flux_l_abs_disk'])
    comm.barrier()
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_cons = athena_read.athdf(args.problem_id + '.cons.' + str_t + '.athdf',
                                        quantities=['dens','mom1','Bcc1'])

        # find corresponding entry in scale height data
        scale_index = AAT.find_nearest(scale_time_list,data_cons['Time'])
        scale_height = scale_height_list[scale_index]

        #unpack data
        x1v = data_cons['x1v']
        x2v = data_cons['x2v']
        x3v = data_cons['x3v']
        x1f = data_cons['x1f']
        x2f = data_cons['x2f']
        x3f = data_cons['x3f']
        Bcc1 = data_cons['Bcc1']

        # define bounds of region to average over
        r_id = AAT.find_nearest(x1v, 6.)
        th_id = AAT.find_nearest(x2v, np.pi/2.)
        th_l = AAT.find_nearest(x2v, np.pi/2. - (3.*scale_height))
        th_u = AAT.find_nearest(x2v, np.pi/2. + (3.*scale_height))

        _,dx2f,dx3f = AAT.calculate_delta(x1f,x2f,x3f)

        mf_l = []
        mf_u = []
        mf_l_d = []
        mf_u_d = []
        # calculate flux at each point on the surface x1f[r_id]
        for j in range(th_id):
            for k in range(len(x3v)):
                dS = (x1f[r_id]**2.) * np.sin(x2f[j]) * dx2f[j] * dx3f[k]
                mf_u.append(Bcc1[k,j,r_id]*dS)
                if j<th_l:
                    mf_u_d.append(Bcc1[k,j,r_id]*dS)
        for j in range(th_id,len(x2v)):
            for k in range(len(x3v)):
                dS = (x1f[r_id]**2.) * np.sin(x2f[j]) * dx2f[j] * dx3f[k]
                mf_l.append(Bcc1[k,j,r_id]*dS)
                if j>th_u:
                    mf_l_d.append(Bcc1[k,j,r_id]*dS)

        # integrate fluxes over surface
        mag_flux_u = np.sqrt(4.*np.pi) * np.sum(mf_u)
        mag_flux_u_abs = np.sqrt(4.*np.pi) * np.sum(np.abs(mf_u))
        mag_flux_l = np.sqrt(4.*np.pi) * np.sum(mf_l)
        mag_flux_l_abs = np.sqrt(4.*np.pi) * np.sum(np.abs(mf_l))

        mag_flux_u_disk = np.sqrt(4.*np.pi) * np.sum(mf_u_d)
        mag_flux_u_abs_disk = np.sqrt(4.*np.pi) * np.sum(np.abs(mf_u_d))
        mag_flux_l_disk = np.sqrt(4.*np.pi) * np.sum(mf_l_d)
        mag_flux_l_abs_disk = np.sqrt(4.*np.pi) * np.sum(np.abs(mf_l_d))

        sim_t = data_cons['Time']
        orbit_t = AAT.calculate_orbit_time(sim_t)

        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            row = [int(t), sim_t, orbit_t,
                    mag_flux_u, mag_flux_l,
                    mag_flux_u_abs, mag_flux_l_abs,
                    mag_flux_u_disk, mag_flux_l_disk,
                    mag_flux_u_abs_disk, mag_flux_l_abs_disk]
            writer.writerow(row)


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate various quality factors from raw simulation data.')
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
