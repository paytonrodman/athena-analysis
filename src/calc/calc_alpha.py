#!/usr/bin/env python3
#
# calc_alpha.py
#
# A program to calculate the Shakura-Sunyaev alpha of an Athena++ disk using MPI.
#
# Usage: mpirun -n [nprocs] calc_alpha.py [options]
#
# IN PROGRESS
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
    # get number of processors and processor rank
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    os.chdir(args.data)

    file_times = AAT.add_time_to_list(args.update, args.output)
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
        if not args.update:
            with open(args.output, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["sim_time", "orbit_time", "T_rphi", "alpha"])
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_prim = athena_read.athdf(args.problem_id + ".prim." + str_t + ".athdf",
                                        quantities=['press'])
        data_cons = athena_read.athdf(args.problem_id + ".cons." + str_t + ".athdf",
                                        quantities=['dens',
                                                    'mom1','mom3',
                                                    'Bcc1','Bcc3'])

        # find corresponding entry in scale height data
        scale_index = AAT.find_nearest(scale_time_list,data_cons['Time'])
        scale_height = scale_height_list[scale_index]

        #unpack data
        x1v = data_cons['x1v']
        x2v = data_cons['x2v']
        x3v = data_cons['x3v']
        dens = data_cons['dens']
        mom1 = data_cons['mom1']
        mom3 = data_cons['mom3']
        Bcc1 = data_cons['Bcc1']
        Bcc3 = data_cons['Bcc3']
        press = data_prim['press']

        # define bounds of region to average over
        th_u = AAT.find_nearest(x2v, np.pi/2. + (3.*scale_height))
        th_l = AAT.find_nearest(x2v, np.pi/2. - (3.*scale_height))

        # calculate rotational velocity
        r,_,_ = np.meshgrid(x3v,x2v,x1v, sparse=False, indexing='ij')
        Omega_kep = np.sqrt(GM/(x1v**3.))
        Omega_kep = np.broadcast_to(Omega_kep, (np.shape(dens)[0], np.shape(dens)[1], np.shape(dens)[2]))
        dmom3 = mom3 - r*Omega_kep

        # restrict range (arrays ordered by [phi,theta,r]!)
        press = press[:, th_l:th_u, :]
        dens = dens[:, th_l:th_u, :]
        mom1 = mom1[:, th_l:th_u, :]
        Bcc1 = Bcc1[:, th_l:th_u, :]
        Bcc3 = Bcc3[:, th_l:th_u, :]
        dmom3 = dmom3[:, th_l:th_u, :]

        # calculate Shakura-Sunyaev alpha from stresses
        Reynolds_stress = dens*mom1*dmom3
        Maxwell_stress = -Bcc1*Bcc3
        T_rphi = Reynolds_stress + Maxwell_stress
        T_rphi = np.average(T_rphi, axis=(1)) # average over vertical height, theta
        alpha_SS = T_rphi/np.average(press, axis=(1))

        sim_t = data_cons['Time']
        orbit_t = AAT.calculate_orbit_time(sim_t)

        with open(args.output, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            row = [sim_t, orbit_t, T_rphi, alpha_SS]
            writer.writerow(row)


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate plasma beta from raw simulation data.')
    parser.add_argument('problem_id',
                        help='root name for data files, e.g. high_res')
    parser.add_argument('data',
                        help='location of data folder, possibly including path')
    parser.add_argument('input',
                        help='location of athinput file, possibly including path')
    parser.add_argument('scale',
                        help='location of scale height file, possibly including path')
    parser.add_argument('output',
                        help='name of output to be (over)written, possibly including path')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='append new results to an existing data file')
    args = parser.parse_args()

    main(**vars(args))
