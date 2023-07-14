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
import gc # only necessary for very big files

# Athena++ modules (require sys.path.append above)
import athena_read
import AAT

def main(**kwargs):
    if args.filetime_short and args.problem_id not in ['high_res','high_beta']:
        sys.exit('--filetime_short argument can only be used with high_res or high_beta. Exiting.')

    # get number of processors and processor rank
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    os.chdir(args.data)

    # retrieve lists of file times and file names
    if rank==0:
        df = pd.read_csv(args.filetime, delimiter='\t', usecols=['file_time', 'sim_time'])
        filetime_list = df['file_time'].to_list()
        simtime_list = df['sim_time'].to_list()

        _,_,t_min = AAT.problem_dictionary(args.problem_id, False) # get minimum required time

        # if applicable, use maximum time of the higher resolution simulation
        if args.filetime_short is not None:
            df = pd.read_csv(args.filetime_short, delimiter='\t', usecols=['file_time', 'sim_time'])
            simtime_list = df['sim_time'].to_list()
            t_max = np.max(simtime_list)
            file_times_restricted = [f for ii,f in enumerate(filetime_list) if simtime_list[ii]>t_min and simtime_list[ii]<t_max]
        else:
            file_times_restricted = [f for ii,f in enumerate(filetime_list) if simtime_list[ii]>t_min]
    else:
        file_times_restricted = None
    file_times_restricted = comm.bcast(file_times_restricted, root=0)
    comm.barrier()

    if not file_times_restricted: # if list is empty
        sys.exit('No file times meet requirements. Exiting.')

    local_times = AAT.distribute_files_to_cores(file_times_restricted, size, rank)

    # retrieve lists of scale height with time and broadcast
    if rank==0:
        df = pd.read_csv(args.scale, delimiter='\t', usecols=['sim_time', 'scale_height'])
        scale_time_list = df['sim_time'].to_list()
        scale_height_list = df['scale_height'].to_list()
    else:
        scale_time_list = None
        scale_height_list = None
    scale_height_list = comm.bcast(scale_height_list, root=0)
    scale_time_list = comm.bcast(scale_time_list, root=0)
    comm.barrier()

    if len(local_times)>0: # skip for nodes that have no assigned times
        for num_elem,t in enumerate(local_times):
            print("Time: ", t)
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
            density = data_cons['dens']
            mom1 = data_cons['mom1']
            mom2 = data_cons['mom2']
            mom3 = data_cons['mom3']
            Bcc1 = data_cons['Bcc1']
            Bcc2 = data_cons['Bcc2']
            Bcc3 = data_cons['Bcc3']
            press = data_prim['press']
            temp = press/density
            GM = 1.

            # clear references to large objects and free memory
            del data_cons, data_prim
            gc.collect()

            # calculate rotational velocity for Reynolds stress
            Omega_kep = np.sqrt(GM/(x1v**3.))
            Omega_kep = np.broadcast_to(Omega_kep, (np.shape(density)[0], np.shape(density)[1], np.shape(density)[2]))
            dmom3 = mom3 - Omega_kep

            del Omega_kep
            gc.collect()

            # define bounds of region to average over
            th_u = AAT.find_nearest(x2v, np.pi/2. + (3.*scale_height))
            th_l = AAT.find_nearest(x2v, np.pi/2. - (3.*scale_height))

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
            alpha = T_rphi/press

            # average over theta and phi
            dens_profile = np.average(dens, axis=(0,1))
            mom1_profile = np.average(mom1, axis=(0,1))
            mom2_profile = np.average(mom2, axis=(0,1))
            mom3_profile = np.average(mom3, axis=(0,1))
            temp_profile = np.average(temp, axis=(0,1))
            Bcc1_profile = np.average(Bcc1, axis=(0,1))
            Bcc2_profile = np.average(Bcc2, axis=(0,1))
            Bcc3_profile = np.average(Bcc3, axis=(0,1))
            alpha_profile = np.average(alpha, axis=(0,1))

            del dens, mom1, mom2, mom3, temp, Bcc1, Bcc2, Bcc3, alpha
            gc.collect()

            if num_elem==0:
                dens_av_local = [dens_profile]
                mom1_av_local = [mom1_profile]
                mom2_av_local = [mom2_profile]
                mom3_av_local = [mom3_profile]
                temp_av_local = [temp_profile]
                Bcc1_av_local = [Bcc1_profile]
                Bcc2_av_local = [Bcc2_profile]
                Bcc3_av_local = [Bcc3_profile]
                alpha_av_local = [alpha_profile]
            else:
                dens_av_local = addToAverage(dens_av_local, num_elem, dens_profile)
                mom1_av_local = addToAverage(mom1_av_local, num_elem, mom1_profile)
                mom2_av_local = addToAverage(mom2_av_local, num_elem, mom2_profile)
                mom3_av_local = addToAverage(mom3_av_local, num_elem, mom3_profile)
                temp_av_local = addToAverage(temp_av_local, num_elem, temp_profile)
                Bcc1_av_local = addToAverage(Bcc1_av_local, num_elem, Bcc1_profile)
                Bcc2_av_local = addToAverage(Bcc2_av_local, num_elem, Bcc2_profile)
                Bcc3_av_local = addToAverage(Bcc3_av_local, num_elem, Bcc3_profile)
                alpha_av_local = addToAverage(alpha_av_local, num_elem, alpha_profile)

            del dens_profile
            del mom1_profile, mom2_profile, mom3_profile
            del temp_profile
            del Bcc1_profile, Bcc2_profile, Bcc3_profile
            del alpha_profile
            gc.collect()


        N = num_elem+1

        weighted_dens = dens_av_local*N
        weighted_mom1 = mom1_av_local*N
        weighted_mom2 = mom2_av_local*N
        weighted_mom3 = mom3_av_local*N
        weighted_temp = temp_av_local*N
        weighted_Bcc1 = Bcc1_av_local*N
        weighted_Bcc2 = Bcc2_av_local*N
        weighted_Bcc3 = Bcc3_av_local*N
        weighted_alpha = alpha_av_local*N

        all_weighted_dens = comm.gather(weighted_dens, root=0)
        all_weighted_mom1 = comm.gather(weighted_mom1, root=0)
        all_weighted_mom2 = comm.gather(weighted_mom2, root=0)
        all_weighted_mom3 = comm.gather(weighted_mom3, root=0)
        all_weighted_temp = comm.gather(weighted_temp, root=0)
        all_weighted_Bcc1 = comm.gather(weighted_Bcc1, root=0)
        all_weighted_Bcc2 = comm.gather(weighted_Bcc2, root=0)
        all_weighted_Bcc3 = comm.gather(weighted_Bcc3, root=0)
        all_weighted_alpha = comm.gather(weighted_alpha, root=0)

        all_elem = comm.gather(N, root=0)

    if rank == 0:
        dens_av = np.sum(all_weighted_dens, axis=0)/np.sum(all_elem, axis=0)
        mom1_av = np.sum(all_weighted_mom1, axis=0)/np.sum(all_elem, axis=0)
        mom2_av = np.sum(all_weighted_mom2, axis=0)/np.sum(all_elem, axis=0)
        mom3_av = np.sum(all_weighted_mom3, axis=0)/np.sum(all_elem, axis=0)
        temp_av = np.sum(all_weighted_temp, axis=0)/np.sum(all_elem, axis=0)
        Bcc1_av = np.sum(all_weighted_Bcc1, axis=0)/np.sum(all_elem, axis=0)
        Bcc2_av = np.sum(all_weighted_Bcc2, axis=0)/np.sum(all_elem, axis=0)
        Bcc3_av = np.sum(all_weighted_Bcc3, axis=0)/np.sum(all_elem, axis=0)
        alpha_av = np.sum(all_weighted_alpha, axis=0)/np.sum(all_elem, axis=0)

        if filetime_short is not None:
            np.save(args.output + 'dens_profile_short.npy', dens_av)
            np.save(args.output + 'mom1_profile_short.npy', mom1_av)
            np.save(args.output + 'mom2_profile_short.npy', mom2_av)
            np.save(args.output + 'mom3_profile_short.npy', mom3_av)
            np.save(args.output + 'temp_profile_short.npy', temp_av)
            np.save(args.output + 'Bcc1_profile_short.npy', Bcc1_av)
            np.save(args.output + 'Bcc2_profile_short.npy', Bcc2_av)
            np.save(args.output + 'Bcc3_profile_short.npy', Bcc3_av)
            np.save(args.output + 'alpha_profile_short.npy', alpha_av)
        else:
            np.save(args.output + 'dens_profile.npy', dens_av)
            np.save(args.output + 'mom1_profile.npy', mom1_av)
            np.save(args.output + 'mom2_profile.npy', mom2_av)
            np.save(args.output + 'mom3_profile.npy', mom3_av)
            np.save(args.output + 'temp_profile.npy', temp_av)
            np.save(args.output + 'Bcc1_profile.npy', Bcc1_av)
            np.save(args.output + 'Bcc2_profile.npy', Bcc2_av)
            np.save(args.output + 'Bcc3_profile.npy', Bcc3_av)
            np.save(args.output + 'alpha_profile.npy', alpha_av)


def addToAverage(average, size, value):
    return (size * average + value) / (size + 1);


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate plasma beta from raw simulation data.')
    parser.add_argument('problem_id',
                        help='root name for data files, e.g. high_res')
    parser.add_argument('data',
                        help='location of data folder, including path')
    parser.add_argument('scale',
                        help='location of scale height file, possibly including path')
    parser.add_argument('filetime',
                        help='location of time file, possibly including path')
    parser.add_argument('output',
                        help='location of output folder, including path')
    parser.add_argument('-s', '--filetime_short',
                        help='location of short time file, possibly including path')
    args = parser.parse_args()

    main(**vars(args))
