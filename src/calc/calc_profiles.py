#!/usr/bin/env python3
#
# calc_profiles.py
#
# A program to calculate radial profiles of various parameters at late times in the disk.
#
# Usage: mpirun -n [nprocs] calc_profiles.py [options]
#
# Output: profiles.csv, Bcc[1,2,3]_profile.npy, mom[1,2,3]_profile.npy, dens_profile.npy, rot_profile.npy, temp_profile.npy
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
    file_times.sort()
    _,_,t_min = AAT.problem_dictionary(args.problem_id, False) # get minimum time
    file_times_restricted = []
    if rank==0:
        for f in file_times:
            str_f = str(int(f)).zfill(5)
            # read in some small slice of the file to check the time
            data_check = athena_read.athdf(args.problem_id + '.cons.' + str_f + '.athdf',
                                           x1_min=5,x1_max=6,x2_min=0,x2_max=0.1,x3_min=0,x3_max=0.1)
            sim_t = data_check['Time']
            if sim_t >= t_min:
                file_times_restricted.append(f)

    local_times = AAT.distribute_files_to_cores(file_times_restricted, size, rank)

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
            with open(args.output1, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['sim_time', 'orbit_time', 'dens', 'mom1', 'mom2', 'mom3',
                                    'temp', 'Bcc1', 'Bcc2', 'Bcc3',
                                    'reynolds_stress', 'maxwell_stress', 'alpha_SS'])

    comm.barrier()
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_prim = athena_read.athdf(args.problem_id + '.prim.' + str_t + '.athdf',
                                        quantities=['press'])
        data_cons = athena_read.athdf(args.problem_id + '.cons.' + str_t + '.athdf',
                                        quantities=['dens','mom1','mom2','mom3',
                                                    'Bcc1','Bcc2','Bcc3'])

        # find corresponding entry in scale height data
        scale_index = AAT.find_nearest(scale_time_list, data_cons['Time'])
        scale_height = scale_height_list[scale_index]

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

        # define bounds of region to average over
        th_u = AAT.find_nearest(x2v, np.pi/2. + (2.*scale_height))
        th_l = AAT.find_nearest(x2v, np.pi/2. - (2.*scale_height))

        # calculate rotational velocity
        r,_,_ = np.meshgrid(x3v,x2v,x1v, sparse=False, indexing='ij')
        #orbital_rotation = v3**2./r**2.
        Omega_kep = np.sqrt(GM/(x1v**3.))
        Omega_kep = np.broadcast_to(Omega_kep, (np.shape(density)[0], np.shape(density)[1], np.shape(density)[2]))
        dmom3 = mom3 - r*Omega_kep

        # restrict range (arrays ordered by [phi,theta,r]!)
        dens = density[:, th_l:th_u, :]
        mom1 = mom1[:, th_l:th_u, :]
        mom2 = mom2[:, th_l:th_u, :]
        mom3 = mom3[:, th_l:th_u, :]
        temp = temp[:, th_l:th_u, :]
        Bcc1 = Bcc1[:, th_l:th_u, :]
        Bcc2 = Bcc2[:, th_l:th_u, :]
        Bcc3 = Bcc3[:, th_l:th_u, :]
        press = press[:, th_l:th_u, :]
        dmom3 = dmom3[:, th_l:th_u, :]

        # calculate Shakura-Sunyaev alpha from stresses
        stress_Rey = dens*mom1*dmom3
        stress_Max = -Bcc1*Bcc3
        T_rphi = stress_Rey + stress_Max
        alpha_SS = T_rphi/press

        #orbital_rotation = np.average(orbital_rotation, axis=(0,1))
        #Omega_kep = np.average(Omega_kep, axis=(0,1))

        # weight rotation by density
        #numWeight_orb = np.sum(orbital_rotation*density) #value * weight
        #sum_orb       = np.sum(density) # weight
        #weighted_rotation = numWeight_orb/sum_orb
        #orbit_v_ratio = weighted_rotation/Omega_kep

        # average over theta and phi
        dens_profile = np.average(dens, axis=(0,1))
        mom1_profile = np.average(mom1, axis=(0,1))
        mom2_profile = np.average(mom2, axis=(0,1))
        mom3_profile = np.average(mom3, axis=(0,1))
        temp_profile = np.average(temp, axis=(0,1))
        Bcc1_profile = np.average(Bcc1, axis=(0,1))
        Bcc2_profile = np.average(Bcc2, axis=(0,1))
        Bcc3_profile = np.average(Bcc3, axis=(0,1))
        #orbit_v_ratio_profile = np.average(orbit_v_ratio, axis=(0,1))
        stress_Rey_profile = np.average(stress_Rey, axis=(0,1))
        stress_Max_profile = np.average(stress_Max, axis=(0,1))
        alpha_SS_profile = np.average(alpha_SS, axis=(0,1))

        sim_t = data_cons['Time']
        orbit_t = AAT.calculate_orbit_time(sim_t)

        with open(args.output1, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([sim_t,orbit_t,dens_profile,mom1_profile,mom2_profile,mom3_profile,
                                temp_profile,Bcc1_profile,Bcc2_profile,Bcc3_profile,
                                stress_Rey_profile, stress_Max_profile, alpha_SS_profile])

    # now take average over time
    comm.barrier()
    if rank == 0:
        df = pd.read_csv(args.output1, delimiter='\t', usecols=['sim_time', 'dens', 'mom1', 'mom2', 'mom3',
                                                                'temp', 'Bcc1', 'Bcc2', 'Bcc3', 'rotational_speed',
                                                                'reynolds_stress', 'maxwell_stress', 'alpha_SS'])
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
        alp = df['alpha_SS'].to_list()

        len_r = len(np.fromstring(d[0].strip('[]'), sep=' '))

        d_arr = np.empty([len(time), len_r])
        m1_arr = np.empty([len(time), len_r])
        m2_arr = np.empty([len(time), len_r])
        m3_arr = np.empty([len(time), len_r])
        t_arr = np.empty([len(time), len_r])
        b1_arr = np.empty([len(time), len_r])
        b2_arr = np.empty([len(time), len_r])
        b3_arr = np.empty([len(time), len_r])
        r_arr = np.empty([len(time), len_r])
        alp_arr = np.empty([len(time), len_r])

        for ii in range(len(time)):
            d_arr[ii] = np.fromstring(d[ii].strip('[]'), sep=' ')
            m1_arr[ii] = np.fromstring(m1[ii].strip('[]'), sep=' ')
            m2_arr[ii] = np.fromstring(m2[ii].strip('[]'), sep=' ')
            m3_arr[ii] = np.fromstring(m3[ii].strip('[]'), sep=' ')
            t_arr[ii] = np.fromstring(t[ii].strip('[]'), sep=' ')
            b1_arr[ii] = np.fromstring(b1[ii].strip('[]'), sep=' ')
            b2_arr[ii] = np.fromstring(b2[ii].strip('[]'), sep=' ')
            b3_arr[ii] = np.fromstring(b3[ii].strip('[]'), sep=' ')
            r_arr[ii] = np.fromstring(r[ii].strip('[]'), sep=' ')
            alp_arr[ii] = np.fromstring(alp[ii].strip('[]'), sep=' ')

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
        alp_av = np.mean(alp_arr, axis=0)

        np.save(args.output + 'dens_profile.npy', dens_av)
        np.save(args.output + 'mom1_profile.npy', mom1_av)
        np.save(args.output + 'mom2_profile.npy', mom2_av)
        np.save(args.output + 'mom3_profile.npy', mom3_av)
        np.save(args.output + 'temp_profile.npy', temp_av)
        np.save(args.output + 'Bcc1_profile.npy', Bcc1_av)
        np.save(args.output + 'Bcc2_profile.npy', Bcc2_av)
        np.save(args.output + 'Bcc3_profile.npy', Bcc3_av)
        np.save(args.output + 'rot_profile.npy', rot_av)
        np.save(args.output + 'alpha_profile.npy', alp_av)


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate plasma beta from raw simulation data.')
    parser.add_argument('problem_id',
                        help='root name for data files, e.g. high_res')
    parser.add_argument('data',
                        help='location of data folder, including path')
    parser.add_argument('scale',
                        help='location of scale height file, possibly including path')
    parser.add_argument('output1',
                        help='output file for intermediate results, including path')
    parser.add_argument('output',
                        help='location of output folder, including path')
    parser.add_argument('-u', '--update',
                        action='store_true',
                        help='append new results to an existing data file')
    args = parser.parse_args()

    main(**vars(args))
