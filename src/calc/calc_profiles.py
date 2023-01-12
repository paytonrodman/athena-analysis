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
sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
#sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')

# Other Python modules
import csv
import numpy as np
from mpi4py import MPI
import pandas as pd

# Athena++ modules
import athena_read
import AAT

def main(**kwargs):
    # get number of processors and processor rank
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    os.chdir(args.data)

    file_times = AAT.add_time_to_list(args.update, args.output1)
    if args.problem_id=='high_res':
        file_times = file_times[file_times>39341] # t > 2e5
    elif args.problem_id=='high_beta':
        file_times = file_times[file_times>4000] # t > 2e4
    local_times = AAT.distribute_files_to_cores(file_times, size, rank)

    data_input = athena_read.athinput(args.input)
    scale_height = data_input['problem']['h_r']

    data_init = athena_read.athdf(args.problem_id + '.cons.00000.athdf', quantities=['x2v'])
    x2v = data_init['x2v']
    th_u = AAT.find_nearest(x2v, np.pi/2. + (2.*scale_height))
    th_l = AAT.find_nearest(x2v, np.pi/2. - (2.*scale_height))

    #dens_all = []
    #mom1_all = []
    #mom2_all = []
    #mom3_all = []
    #temp_all = []
    #Bcc1_all = []
    #Bcc2_all = []
    #Bcc3_all = []
    #rot_all = []
    if rank==0:
        if not args.update:
            with open(args.output1, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["sim_time", "orbit_time", "dens", "mom1", "mom2", "mom3",
                                    "temp", "Bcc1", "Bcc2", "Bcc3", "rotational_speed"])
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_prim = athena_read.athdf(args.problem_id + ".prim." + str_t + ".athdf",
                                        quantities=['press'])
        data_cons = athena_read.athdf(args.problem_id + ".cons." + str_t + ".athdf",
                                        quantities=['x1v','x2v','x3v',
                                                    'dens','mom1','mom2','mom3',
                                                    'Bcc1','Bcc2','Bcc3'])

        #unpack data
        x1v = data_cons['x1v']
        x2v = data_cons['x2v']
        x3v = data_cons['x3v']
        density = data_cons['dens']
        mom1 = data_cons['mom1']
        mom2 = data_cons['mom2']
        mom3 = data_cons['mom3']
        Bcc1 = data_cons['Bcc1']
        Bcc2 = data_cons['Bcc2']
        Bcc3 = data_cons['Bcc3']
        press = data_prim['press']

        temp = press/density
        v3 = mom3/density
        GM = 1.

        r,_,_ = np.meshgrid(x3v,x2v,x1v, sparse=False, indexing='ij')
        orbital_rotation = v3**2./r**2.
        Omega_kep = np.sqrt(GM/(x1v**3.))
        Omega_kep = np.broadcast_to(Omega_kep, (np.shape(density)[0], np.shape(density)[1], np.shape(density)[2]))

        dens = density[:, th_l:th_u, :]
        mom1 = mom1[:, th_l:th_u, :]
        mom2 = mom2[:, th_l:th_u, :]
        mom3 = mom3[:, th_l:th_u, :]
        temp = temp[:, th_l:th_u, :]
        Bcc1 = Bcc1[:, th_l:th_u, :]
        Bcc2 = Bcc2[:, th_l:th_u, :]
        Bcc3 = Bcc3[:, th_l:th_u, :]

        #orbital_rotation = np.average(orbital_rotation, axis=(0,1))
        #Omega_kep = np.average(Omega_kep, axis=(0,1))

        # weight rotation by density
        numWeight_orb = np.sum(orbital_rotation*density) #value * weight
        sum_orb       = np.sum(density) # weight
        weighted_rotation = numWeight_orb/sum_orb
        ratio = weighted_rotation/Omega_kep

        # average over theta and phi
        dens_profile = np.average(dens, axis=(0,1))
        mom1_profile = np.average(mom1, axis=(0,1))
        mom2_profile = np.average(mom2, axis=(0,1))
        mom3_profile = np.average(mom3, axis=(0,1))
        temp_profile = np.average(temp, axis=(0,1))
        Bcc1_profile = np.average(Bcc1, axis=(0,1))
        Bcc2_profile = np.average(Bcc2, axis=(0,1))
        Bcc3_profile = np.average(Bcc3, axis=(0,1))
        ratio_profile = np.average(ratio, axis=(0,1))

        sim_t = data_cons['Time']
        orbit_t = AAT.calculate_orbit_time(sim_t)

        with open(args.output1, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([sim_t,orbit_t,dens_profile,mom1_profile,mom2_profile,mom3_profile,
                                temp_profile,Bcc1_profile,Bcc2_profile,Bcc3_profile,ratio_profile])

        #dens_all.append(dens_profile)
        #mom1_all.append(mom1_profile)
        #mom2_all.append(mom2_profile)
        #mom3_all.append(mom3_profile)
        #temp_all.append(temp_profile)
        #Bcc1_all.append(Bcc1_profile)
        #Bcc2_all.append(Bcc2_profile)
        #Bcc3_all.append(Bcc3_profile)
        #rot_all.append(ratio)

    comm.barrier()
    if rank == 0:
        df = pd.read_csv(args.output1, delimiter='\t', usecols=['sim_time', 'dens', 'mom1', 'mom2', 'mom3',
                                                                'temp', 'Bcc1', 'Bcc2', 'Bcc3',
                                                                'rotational_speed'])
        time = df['sim_time'].to_list()
        d = df['dens'].to_list()
        m1 = df['mom1'].to_list()
        m2 = df['mom2'].to_list()
        m3 = df['mom3'].to_list()
        t = df['temp'].to_list()
        b1 = df['Bcc1'].to_list()
        b2 = df['Bcc2'].to_list()
        b3 = df['Bcc3'].to_list()
        r = df['rotational_speed'].to_list()

        len_r = len(np.fromstring(d[0].strip("[]"), sep=' '))

        d_arr = np.empty([len(time), len_r])
        m1_arr = np.empty([len(time), len_r])
        m2_arr = np.empty([len(time), len_r])
        m3_arr = np.empty([len(time), len_r])
        t_arr = np.empty([len(time), len_r])
        b1_arr = np.empty([len(time), len_r])
        b2_arr = np.empty([len(time), len_r])
        b3_arr = np.empty([len(time), len_r])
        r_arr = np.empty([len(time), len_r])

        for ii in range(len(time)):
            d_arr[ii] = np.fromstring(d[ii].strip("[]"), sep=' ')
            m1_arr[ii] = np.fromstring(m1[ii].strip("[]"), sep=' ')
            m2_arr[ii] = np.fromstring(m2[ii].strip("[]"), sep=' ')
            m3_arr[ii] = np.fromstring(m3[ii].strip("[]"), sep=' ')
            t_arr[ii] = np.fromstring(t[ii].strip("[]"), sep=' ')
            b1_arr[ii] = np.fromstring(b1[ii].strip("[]"), sep=' ')
            b2_arr[ii] = np.fromstring(b2[ii].strip("[]"), sep=' ')
            b3_arr[ii] = np.fromstring(b3[ii].strip("[]"), sep=' ')
            r_arr[ii] = np.fromstring(r[ii].strip("[]"), sep=' ')

        # average over time
        dens_av = np.mean(d_arr, axis=0)
        mom1_av = np.mean(m1_arr, axis=0)
        mom2_av = np.mean(m2_arr, axis=0)
        mom3_av = np.mean(m3_arr, axis=0)
        temp_av = np.mean(t_arr, axis=0)
        Bcc1_av = np.mean(b1_arr, axis=0)
        Bcc2_av = np.mean(b2_arr, axis=0)
        Bcc3_av = np.mean(b3_arr, axis=0)
        rot_av = np.mean(r_arr, axis=0)

        np.save(args.output + 'dens_profile.npy', dens_av)
        np.save(args.output + 'mom1_profile.npy', mom1_av)
        np.save(args.output + 'mom2_profile.npy', mom2_av)
        np.save(args.output + 'mom3_profile.npy', mom3_av)
        np.save(args.output + 'temp_profile.npy', temp_av)
        np.save(args.output + 'Bcc1_profile.npy', Bcc1_av)
        np.save(args.output + 'Bcc2_profile.npy', Bcc2_av)
        np.save(args.output + 'Bcc3_profile.npy', Bcc3_av)
        np.save(args.output + 'rot_profile.npy', rot_av)


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate plasma beta from raw simulation data.')
    parser.add_argument('problem_id',
                        help='root name for data files, e.g. high_res')
    parser.add_argument('data',
                        help='location of data folder, including path')
    parser.add_argument('input',
                        help='location of athinput file, including path')
    parser.add_argument('output1',
                        help='output file for intermediate results, including path')
    parser.add_argument('output',
                        help='location of output folder, including path')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='append new results to an existing data file')
    args = parser.parse_args()

    main(**vars(args))
