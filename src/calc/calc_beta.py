#!/usr/bin/env python3
#
# calc_beta.py
#
# A program to calculate the plasma beta of an Athena++ disk using MPI.
#
# Usage: mpirun -n [nprocs] calc_beta.py [options]
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

    os.chdir(args.data)

    # make list of files/times to analyse, distribute to cores
    file_times = AAT.add_time_to_list(args.update, args.output)
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

    # assign one core the job of creating output file with header
    if rank==0:
        if not args.update: # create output file with header
            with open(args.output, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['file_time', 'sim_time', 'orbit_time', 'plasma_beta'])

    comm.barrier()
    # cores loop through their assigned list of times
    for t in local_times:
        str_t = str(int(t)).zfill(5)

        # read in conservative and primitive data
        data_prim = athena_read.athdf(args.problem_id + '.prim.' + str_t + '.athdf',
                                        quantities=['press'])
        data_cons = athena_read.athdf(args.problem_id + '.cons.' + str_t + '.athdf',
                                        quantities=['dens','Bcc1','Bcc2','Bcc3'])

        # find corresponding entry in scale height data
        scale_index = AAT.find_nearest(scale_time_list,data_cons['Time'])
        scale_height = scale_height_list[scale_index]

        #unpack data
        x1f = data_cons['x1f']
        x2f = data_cons['x2f']
        x3f = data_cons['x3f']
        x1v = data_cons['x1v']
        x2v = data_cons['x2v']
        x3v = data_cons['x3v']
        dens = data_cons['dens']
        Bcc1 = data_cons['Bcc1']
        Bcc2 = data_cons['Bcc2']
        Bcc3 = data_cons['Bcc3']
        press = data_prim['press']

        # define bounds of region to average over
        r_u = AAT.find_nearest(x1v, x1_high_max)
        th_u = AAT.find_nearest(x2v, np.pi/2. + (3.*scale_height))
        th_l = AAT.find_nearest(x2v, np.pi/2. - (3.*scale_height))

        # create 3D meshes
        r,theta,_ = np.meshgrid(x3v,x2v,x1v, sparse=False, indexing='ij')
        dx1f,dx2f,dx3f = AAT.calculate_delta(x1f,x2f,x3f)
        dphi,dtheta,dr = np.meshgrid(dx3f,dx2f,dx1f, sparse=False, indexing='ij')

        # restrict range (arrays ordered by [phi,theta,r]!)
        dphi = dphi[:, th_l:th_u, :r_u]
        dtheta = dtheta[:, th_l:th_u, :r_u]
        dr = dr[:, th_l:th_u, :r_u]
        r = r[:, th_l:th_u, :r_u]
        theta = theta[:, th_l:th_u, :r_u]
        pressure = press[:, th_l:th_u, :r_u]
        density = dens[:, th_l:th_u, :r_u]

        volume = (r**2.)*np.sin(theta)*dr*dtheta*dphi

        # initialise weights for averaging
        sum_p = 0.
        numWeight_p = 0.
        sum_b = 0.
        numWeight_b = 0.

        # Find volume centred total magnetic field
        bcc_all = np.square(Bcc1[:, th_l:th_u, :r_u]) + np.square(Bcc2[:, th_l:th_u, :r_u]) + np.square(Bcc3[:, th_l:th_u, :r_u])
        # Density- and volume weighted pressure/magnetic field
        numWeight_p = np.sum(pressure*density*volume) # value * weight
        sum_p       = np.sum(density*volume) # weight
        numWeight_b = np.sum(bcc_all*density*volume) # value * weight
        sum_b       = np.sum(density*volume) # weight

        pres_av = numWeight_p/sum_p
        bcc_av = numWeight_b/sum_b
        if bcc_av>0: # if B is zero, beta is undefined
            current_beta = 2. * pres_av / bcc_av
        else:
            current_beta = np.nan

        # determine orbital time at ISCO
        sim_t = data_cons['Time']
        orbit_t = AAT.calculate_orbit_time(sim_t)

        # append result to output file
        with open(args.output, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            row = [int(t),sim_t,orbit_t,current_beta]
            writer.writerow(row)


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate plasma beta from raw simulation data.')
    parser.add_argument('problem_id',
                        help='root name for data files, e.g. high_res')
    parser.add_argument('data',
                        help='location of data folder, possibly including path')
    parser.add_argument('input',
                        help='location of .athinput file, possibly including path')
    parser.add_argument('scale',
                        help='location of scale height file, possibly including path')
    parser.add_argument('output',
                        help='name of output to be (over)written, possibly including path')
    parser.add_argument('-u', '--update',
                        action='store_true',
                        help='append new results to an existing data file')
    args = parser.parse_args()

    main(**vars(args))
