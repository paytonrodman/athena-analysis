#!/usr/bin/env python3
#
# calc_alpha.py
#
# A program to calculate the Shakura-Sunyaev alpha of an Athena++ disk using MPI
#
# Usage: mpirun -n [nprocs] calc_alpha.py [options]
#
# Output: a .csv file containing
#         (1) simulation file time
#         (2) simulation time in GM/c^3
#         (3) simulation time in N_(ISCO orbits)
#         (4) disk-averaged Maxwell stress
#         (5) disk-averaged stress tensor T_{r,phi}
#         (6) disk-averaged alpha
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
import pandas as pd
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
    output = args.output
    root, ext = os.path.splitext(output)
    if ext != '.csv':
        ext = '.csv'
    output = root + ext

    os.chdir(args.data)

    # make list of files/times to analyse, distribute to cores
    file_times = AAT.add_time_to_list(args.update, output)
    local_times = AAT.distribute_files_to_cores(file_times, size, rank)

    # find bounds of high resolution region
    data_input = athena_read.athinput(args.input)
    if 'refinement3' in data_input:
        x1_high_max = data_input['refinement3']['x1max']
    elif 'refinement2' in data_input:
        x1_high_max = data_input['refinement2']['x1max']
    elif 'refinement1' in data_input:
        x1_high_max = data_input['refinement1']['x1max']
    else:
        x1_high_max = data_input['mesh']['x1max']

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
            with open(output, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['file_time', 'sim_time', 'orbit_time', 'av_Max', 'T_rphi', 'alpha'])
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_prim = athena_read.athdf(args.problem_id + '.prim.' + str_t + '.athdf',
                                        quantities=['press'])
        data_cons = athena_read.athdf(args.problem_id + '.cons.' + str_t + '.athdf',
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

        sim_t = data_cons['Time']

        # manage memory
        del data_cons
        del data_prim

        # define bounds of region to average over
        r_u = AAT.find_nearest(x1v, x1_high_max)
        th_u = AAT.find_nearest(x2v, np.pi/2. + (3.*scale_height))
        th_l = AAT.find_nearest(x2v, np.pi/2. - (3.*scale_height))

        # calculate rotational velocity
        r,_,_ = np.meshgrid(x3v,x2v,x1v, sparse=False, indexing='ij')
        GM = 1. #code units
        Omega_kep = np.sqrt(GM/(x1v**3.))
        Omega_kep = np.broadcast_to(Omega_kep, (np.shape(dens)[0], np.shape(dens)[1], np.shape(dens)[2]))
        dmom3 = mom3 - r*Omega_kep

        # restrict range (arrays ordered by [phi,theta,r]!)
        press = press[:, th_l:th_u, :r_u]
        dens = dens[:, th_l:th_u, :r_u]
        mom1 = mom1[:, th_l:th_u, :r_u]
        Bcc1 = Bcc1[:, th_l:th_u, :r_u]
        Bcc3 = Bcc3[:, th_l:th_u, :r_u]
        dmom3 = dmom3[:, th_l:th_u, :r_u]

        # calculate Shakura-Sunyaev alpha from stresses
        Reynolds_stress = dens*mom1*dmom3
        Maxwell_stress = -Bcc1*Bcc3
        T_rphi = Reynolds_stress + Maxwell_stress
        alpha_SS = np.average(T_rphi)/np.average(press)

        orbit_t = AAT.calculate_orbit_time(sim_t)

        with open(output, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            row = [int(t), sim_t, orbit_t, np.average(Maxwell_stress), np.average(T_rphi), alpha_SS]
            writer.writerow(row)


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate Shakura-Sunyaev alpha from raw simulation data.')
    parser.add_argument('problem_id',
                        type=str,
                        help='root name for data files, e.g. b200')
    parser.add_argument('data',
                        type=str,
                        help='location of data folder, possibly including path')
    parser.add_argument('input',
                        type=str,
                        help='location of athinput file, possibly including path')
    parser.add_argument('scale',
                        type=str,
                        help='location of scale height csv file (produced by calc_scale.py), possibly including path')
    parser.add_argument('output',
                        type=str,
                        help='name of csv output to be (over)written, possibly including path')
    parser.add_argument('-u', '--update',
                        action='store_true',
                        help='append new results to an existing data file')
    args = parser.parse_args()

    main(**vars(args))
