#!/usr/bin/env python3
#
# calc_profiles.py
#
# A program to calculate radial profiles of various parameters at late times in the disk using MPI.
#
# Usage: mpirun -n [nprocs] calc_profiles.py [options]
#
# Output: Bcc[1,2,3]_profile.npy, mom[1,2,3]_profile.npy, dens_profile.npy, alpha_profile.npy, temp_profile.npy
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

# Athena++ modules (require sys.path.append above)
import athena_read
import AAT

def main(**kwargs):
    # get number of processors and processor rank
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    os.chdir(args.data)

    # create list of quantities to read from data
    if rank==0: # master node only
        if args.filetime_short and args.problem_id not in ['b200','b5']:
            sys.exit('--filetime_short argument can only be used with long simulations. Exiting.')
        if 'all' in args.profile_data:
            args.profile_data = ['dens','mom','temp','Bcc','alpha']
            quantities_cons = ['dens','mom1','mom2','mom3','Bcc1','Bcc2','Bcc3']
            quantities_prim = ['press']
        else:
            quantities_cons = []
            quantities_prim = []
            if 'dens' in args.profile_data:
                quantities_cons.append('dens')
            if 'mom' in args.profile_data:
                quantities_cons.append('mom1')
                quantities_cons.append('mom2')
                quantities_cons.append('mom3')
            if 'temp' in args.profile_data:
                quantities_cons.append('dens')
                quantities_prim.append('press')
            if 'Bcc' in args.profile_data:
                quantities_cons.append('Bcc1')
                quantities_cons.append('Bcc2')
                quantities_cons.append('Bcc3')
            if 'alpha' in args.profile_data:
                quantities_cons.append('x1v')
                quantities_cons.append('dens')
                quantities_cons.append('mom1')
                quantities_cons.append('mom3')
                quantities_cons.append('Bcc1')
                quantities_cons.append('Bcc3')
                quantities_prim.append('press')

        # remove duplicates
        quantities_cons = (list(set(quantities_cons)))
        quantities_prim = (list(set(quantities_prim)))
    else:
        quantities_cons = None
        quantities_prim = None
    comm.barrier()
    quantities_cons = comm.bcast(quantities_cons, root=0)
    quantities_prim = comm.bcast(quantities_prim, root=0)

    # retrieve lists of file names and simulation times
    if rank==0:
        df = pd.read_csv(args.filetime, delimiter='\t', usecols=['file_time', 'sim_time'])
        filetime_list = df['file_time'].to_list()
        simtime_list = df['sim_time'].to_list()

        _,_,t_min = AAT.problem_dictionary(args.problem_id, False) # get minimum required time for this simulation

        # restrict file list to times above minimum (and below maximum, if applicable)
        if args.filetime_short is not None:
            df = pd.read_csv(args.filetime_short, delimiter='\t', usecols=['file_time', 'sim_time'])
            t_max = np.max(df['sim_time'].to_list())
            file_times_restricted = [f for ii,f in enumerate(filetime_list) if simtime_list[ii]>t_min and simtime_list[ii]<t_max]
        else:
            file_times_restricted = [f for ii,f in enumerate(filetime_list) if simtime_list[ii]>t_min]

        if len(file_times_restricted)==0:
            sys.exit('No file times meet requirements. Exiting.')
    else:
        file_times_restricted = None
    comm.barrier()
    file_times_restricted = comm.bcast(file_times_restricted, root=0)

    local_times = AAT.distribute_files_to_cores(file_times_restricted, size, rank)

    # retrieve lists of scale height with time and broadcast
    if rank==0:
        df = pd.read_csv(args.scale, delimiter='\t', usecols=['sim_time', 'scale_height'])
        scale_time_list = df['sim_time'].to_list()
        scale_height_list = df['scale_height'].to_list()
    else:
        scale_time_list = None
        scale_height_list = None
    comm.barrier()
    scale_height_list = comm.bcast(scale_height_list, root=0)
    scale_time_list = comm.bcast(scale_time_list, root=0)

    if len(local_times)>0: # skip for nodes that have no assigned times
        for num_elem,t in enumerate(local_times):
            str_t = str(int(t)).zfill(5)

            data_cons = athena_read.athdf(args.problem_id + '.cons.' + str_t + '.athdf',
                                            quantities=quantities_cons)
            if len(quantities_prim)>0:
                data_prim = athena_read.athdf(args.problem_id + '.prim.' + str_t + '.athdf',
                                                quantities=quantities_prim)


            # find corresponding entry in scale height data
            scale_index = AAT.find_nearest(scale_time_list, data_cons['Time'])
            scale_height = scale_height_list[scale_index]

            # define bounds of region to average over
            x2v = data_cons['x2v']
            th_u = AAT.find_nearest(x2v, np.pi/2. + (3.*scale_height))
            th_l = AAT.find_nearest(x2v, np.pi/2. - (3.*scale_height))

            #unpack data
            if 'dens' in args.profile_data:
                dens = data_cons['dens'][:, th_l:th_u, :]
            if 'mom' in args.profile_data:
                mom1 = data_cons['mom1'][:, th_l:th_u, :]
                mom2 = data_cons['mom2'][:, th_l:th_u, :]
                mom3 = data_cons['mom3'][:, th_l:th_u, :]
            if 'temp' in args.profile_data:
                dens = data_cons['dens'][:, th_l:th_u, :]
                press = data_prim['press'][:, th_l:th_u, :]
                temp = press/dens
            if 'Bcc' in args.profile_data:
                Bcc1 = data_cons['Bcc1'][:, th_l:th_u, :]
                Bcc2 = data_cons['Bcc2'][:, th_l:th_u, :]
                Bcc3 = data_cons['Bcc3'][:, th_l:th_u, :]
            if 'alpha' in args.profile_data:
                GM = 1.
                x1v = data_cons['x1v']

                dens = data_cons['dens'][:, th_l:th_u, :]
                mom3 = data_cons['mom3'][:, th_l:th_u, :]
                mom1 = data_cons['mom1'][:, th_l:th_u, :]
                Bcc1 = data_cons['Bcc1'][:, th_l:th_u, :]
                Bcc3 = data_cons['Bcc3'][:, th_l:th_u, :]
                press = data_prim['press'][:, th_l:th_u, :]

                # calculate rotational velocity for Reynolds stress
                Omega_kep = np.sqrt(GM/(x1v**3.))
                Omega_kep = np.broadcast_to(Omega_kep, (np.shape(dens)[0], np.shape(dens)[1], np.shape(dens)[2]))
                dmom3 = mom3 - Omega_kep

                # calculate Shakura-Sunyaev alpha from stresses
                stress_Rey = dens*mom1*dmom3
                stress_Max = -Bcc1*Bcc3
                T_rphi = stress_Rey + stress_Max
                alpha = T_rphi/press


            # average over theta and phi
            if 'dens' in args.profile_data:
                dens_profile = np.average(dens, axis=(0,1))
            if 'mom' in args.profile_data:
                mom1_profile = np.average(mom1, axis=(0,1))
                mom2_profile = np.average(mom2, axis=(0,1))
                mom3_profile = np.average(mom3, axis=(0,1))
            if 'temp' in args.profile_data:
                temp_profile = np.average(temp, axis=(0,1))
            if 'Bcc' in args.profile_data:
                Bcc1_profile = np.average(Bcc1, axis=(0,1))
                Bcc2_profile = np.average(Bcc2, axis=(0,1))
                Bcc3_profile = np.average(Bcc3, axis=(0,1))
            if 'alpha' in args.profile_data:
                alpha_profile = np.average(alpha, axis=(0,1))

            # add current value to running average
            if num_elem==0:
                if 'dens' in args.profile_data:
                    dens_av_local = [dens_profile]
                if 'mom' in args.profile_data:
                    mom1_av_local = [mom1_profile]
                    mom2_av_local = [mom2_profile]
                    mom3_av_local = [mom3_profile]
                if 'temp' in args.profile_data:
                    temp_av_local = [temp_profile]
                if 'Bcc' in args.profile_data:
                    Bcc1_av_local = [Bcc1_profile]
                    Bcc2_av_local = [Bcc2_profile]
                    Bcc3_av_local = [Bcc3_profile]
                if 'alpha' in args.profile_data:
                    alpha_av_local = [alpha_profile]
            else:
                if 'dens' in args.profile_data:
                    dens_av_local = addToAverage(dens_av_local, num_elem, dens_profile)
                if 'mom' in args.profile_data:
                    mom1_av_local = addToAverage(mom1_av_local, num_elem, mom1_profile)
                    mom2_av_local = addToAverage(mom2_av_local, num_elem, mom2_profile)
                    mom3_av_local = addToAverage(mom3_av_local, num_elem, mom3_profile)
                if 'temp' in args.profile_data:
                    temp_av_local = addToAverage(temp_av_local, num_elem, temp_profile)
                if 'Bcc' in args.profile_data:
                    Bcc1_av_local = addToAverage(Bcc1_av_local, num_elem, Bcc1_profile)
                    Bcc2_av_local = addToAverage(Bcc2_av_local, num_elem, Bcc2_profile)
                    Bcc3_av_local = addToAverage(Bcc3_av_local, num_elem, Bcc3_profile)
                if 'alpha' in args.profile_data:
                    alpha_av_local = addToAverage(alpha_av_local, num_elem, alpha_profile)

        N = num_elem+1

        # send local averages to master
        if 'dens' in args.profile_data:
            weighted_dens = dens_av_local*N
            all_weighted_dens = comm.gather(weighted_dens, root=0)
        if 'mom' in args.profile_data:
            weighted_mom1 = mom1_av_local*N
            weighted_mom2 = mom2_av_local*N
            weighted_mom3 = mom3_av_local*N
            all_weighted_mom1 = comm.gather(weighted_mom1, root=0)
            all_weighted_mom2 = comm.gather(weighted_mom2, root=0)
            all_weighted_mom3 = comm.gather(weighted_mom3, root=0)
        if 'temp' in args.profile_data:
            weighted_temp = temp_av_local*N
            all_weighted_temp = comm.gather(weighted_temp, root=0)
        if 'Bcc' in args.profile_data:
            weighted_Bcc1 = Bcc1_av_local*N
            weighted_Bcc2 = Bcc2_av_local*N
            weighted_Bcc3 = Bcc3_av_local*N
            all_weighted_Bcc1 = comm.gather(weighted_Bcc1, root=0)
            all_weighted_Bcc2 = comm.gather(weighted_Bcc2, root=0)
            all_weighted_Bcc3 = comm.gather(weighted_Bcc3, root=0)
        if 'alpha' in args.profile_data:
            weighted_alpha = alpha_av_local*N
            all_weighted_alpha = comm.gather(weighted_alpha, root=0)

        all_elem = comm.gather(N, root=0)

    # master performs average over all locals
    if rank == 0:
        if 'dens' in args.profile_data:
            dens_av = np.sum(all_weighted_dens, axis=0)/np.sum(all_elem, axis=0)
        if 'mom' in args.profile_data:
            mom1_av = np.sum(all_weighted_mom1, axis=0)/np.sum(all_elem, axis=0)
            mom2_av = np.sum(all_weighted_mom2, axis=0)/np.sum(all_elem, axis=0)
            mom3_av = np.sum(all_weighted_mom3, axis=0)/np.sum(all_elem, axis=0)
        if 'temp' in args.profile_data:
            temp_av = np.sum(all_weighted_temp, axis=0)/np.sum(all_elem, axis=0)
        if 'Bcc' in args.profile_data:
            Bcc1_av = np.sum(all_weighted_Bcc1, axis=0)/np.sum(all_elem, axis=0)
            Bcc2_av = np.sum(all_weighted_Bcc2, axis=0)/np.sum(all_elem, axis=0)
            Bcc3_av = np.sum(all_weighted_Bcc3, axis=0)/np.sum(all_elem, axis=0)
        if 'alpha' in args.profile_data:
            alpha_av = np.sum(all_weighted_alpha, axis=0)/np.sum(all_elem, axis=0)


        if args.filetime_short is not None:
            if 'dens' in args.profile_data:
                np.save(args.output + 'dens_profile_short.npy', dens_av)
            if 'mom' in args.profile_data:
                np.save(args.output + 'mom1_profile_short.npy', mom1_av)
                np.save(args.output + 'mom2_profile_short.npy', mom2_av)
                np.save(args.output + 'mom3_profile_short.npy', mom3_av)
            if 'temp' in args.profile_data:
                np.save(args.output + 'temp_profile_short.npy', temp_av)
            if 'Bcc' in args.profile_data:
                np.save(args.output + 'Bcc1_profile_short.npy', Bcc1_av)
                np.save(args.output + 'Bcc2_profile_short.npy', Bcc2_av)
                np.save(args.output + 'Bcc3_profile_short.npy', Bcc3_av)
            if 'alpha' in args.profile_data:
                np.save(args.output + 'alpha_profile_short.npy', alpha_av)
        else:
            if 'dens' in args.profile_data:
                np.save(args.output + 'dens_profile.npy', dens_av)
            if 'mom' in args.profile_data:
                np.save(args.output + 'mom1_profile.npy', mom1_av)
                np.save(args.output + 'mom2_profile.npy', mom2_av)
                np.save(args.output + 'mom3_profile.npy', mom3_av)
            if 'temp' in args.profile_data:
                np.save(args.output + 'temp_profile.npy', temp_av)
            if 'Bcc' in args.profile_data:
                np.save(args.output + 'Bcc1_profile.npy', Bcc1_av)
                np.save(args.output + 'Bcc2_profile.npy', Bcc2_av)
                np.save(args.output + 'Bcc3_profile.npy', Bcc3_av)
            if 'alpha' in args.profile_data:
                np.save(args.output + 'alpha_profile.npy', alpha_av)


def addToAverage(average, size, value):
    return (size * average + value) / (size + 1);


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate plasma beta from raw simulation data.')
    parser.add_argument('problem_id',
                        type=str,
                        help='root name for data files, e.g. b200')
    parser.add_argument('data',
                        type=str,
                        help='location of data folder, including path')
    parser.add_argument('scale',
                        type=str,
                        help='location of scale height file, possibly including path')
    parser.add_argument('filetime',
                        type=str,
                        help='location of time file (generated by create_timelist.py), possibly including path')
    parser.add_argument('output',
                        type=str,
                        help='location of output folder, including path')
    parser.add_argument('profile_data',
                        choices=['dens','mom','temp','Bcc','alpha','all'],
                        nargs='*',
                        help='possible profiles to calculate')
    parser.add_argument('-s', '--filetime_short',
                        help='location of short time file, possibly including path')
    args = parser.parse_args()

    main(**vars(args))
