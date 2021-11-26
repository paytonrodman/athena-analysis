#!/usr/bin/env python3
#
# calc_scale.py
#
# A program to calculate the geometric scale height of an Athena++ disk using MPI.
#
# To run:
# mpirun -n [n] python calc_scale.py [options]
# for [n] cores.
#

### TO DO
### Need to read in input file to get mass and omegaKep. Apply change to all other files too.
import sys
sys.path.insert(0,'/home/per29/.local/lib/python3.6/site-packages/')
from mpi4py import MPI
import numpy as np
import os
#sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')
import athena_read
import AAT
import glob
import re
import csv
import argparse

def main(**kwargs):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    problem  = args.prob_id
    root_dir = "/Users/paytonrodman/athena-sim/"
    #root_dir = '/home/per29/rds/rds-accretion-zyNhkonJSR8/'
    prob_dir = root_dir + problem + '/'
    data_dir = prob_dir + 'data/'
    runfile_dir = prob_dir + 'runfiles/'
    os.chdir(data_dir)

    csv_time = np.empty(0)
    # check if data file already exists
    if args.update:
        with open(prob_dir + 'mass_with_time.csv', 'r', newline='') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            next(csv_reader, None) # skip header
            for row in csv_reader:
                csv_time = np.append(csv_time, float(row[0]))

    files = glob.glob('./*.athdf')

    times = np.empty(0)
    for f in files:
        time_sec = re.findall(r'\b\d+\b', f)
        if args.update:
            if float(time_sec[0]) not in times and float(time_sec[0]) not in csv_time:
                times = np.append(times, float(time_sec[0]))
        else:
            if float(time_sec[0]) not in times:
                times = np.append(times, float(time_sec[0]))
    if len(times)==0:
        sys.exit('No new timesteps to analyse in the given directory. Exiting.')

    count = len(times) // size  # number of files for each process to analyze
    remainder = len(times) % size  # extra files if times is not a multiple of size
    if rank < remainder:  # processes with rank < remainder analyze one extra catchment
        start = rank * (count + 1)  # index of first file to analyze
        stop = start + count + 1  # index of last file to analyze
    else:
        start = rank * count + remainder
        stop = start + count
    local_times = times[start:stop] # get the times to be analyzed by each rank

    data_input = athena_read.athinput(runfile_dir + 'athinput.' + problem)
    mass = data_input['problem']['mass']
    x1min = data_input['mesh']['x1min']
    # get mesh data for all files (static)
    data_init = athena_read.athdf(problem + '.cons.00000.athdf')
    x1v = data_init['x1v'] # r
    x2v = data_init['x2v'] # theta
    x3v = data_init['x3v'] # phi
    x1f = data_init['x1f'] # r
    x2f = data_init['x2f'] # theta
    x3f = data_init['x3f'] # phi
    dx1f,dx2f,dx3f = AAT.calculate_delta(x1f,x2f,x3f)
    phi,theta,r = np.meshgrid(x3v,x2v,x1v, sparse=False, indexing='ij')
    dphi,dtheta,dr = np.meshgrid(dx3f,dx2f,dx1f, sparse=False, indexing='ij')
    dOmega = np.sin(theta)*dtheta*dphi

    local_scale_h = []
    local_orbit_time = []
    local_sim_time = []
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        filename_cons = problem + '.cons.' + str_t + '.athdf'
        data_cons = athena_read.athdf(filename_cons)

        #unpack data
        dens = data_cons['dens']
        # Calculations
        polar_ang = np.sum(theta*dens*dOmega,axis=(0,1))/np.sum(dens*dOmega,axis=(0,1))
        h_up = (theta-polar_ang)**2. * dens*dOmega
        h_down = dens*dOmega
        scale_h = np.sqrt(np.sum(h_up,axis=(0,1))/np.sum(h_down,axis=(0,1)))
        scale_h_av = np.average(scale_h,weights=dx1f)
        local_scale_h.append(scale_h_av)

        v_Kep0 = np.sqrt(mass/x1min)
        Omega0 = v_Kep0/x1min
        T0 = 2.*np.pi/Omega0
        local_orbit_time.append(t/T0)
        local_sim_time.append(t)

    if rank > 0:
        comm.Send(np.asarray(local_scale_h), dest=0, tag=14)  # send results to process 0
        comm.Send(np.asarray(local_orbit_time), dest=0, tag=20)
        comm.Send(np.asarray(local_sim_time), dest=0, tag=24)
    else:
        final_sc_h = np.copy(local_scale_h)  # initialize final results with results from process 0
        final_orb_t = np.copy(local_orbit_time)
        final_sim_t = np.copy(local_sim_time)

        for i in range(1, size):  # determine the size of the array to be received from each process
            if i < remainder:
                rank_size = count + 1
            else:
                rank_size = count

            tmp_sh = np.empty((rank_size-1, final_sc_h.shape[0]), dtype=np.float)  # create empty array to receive results
            comm.Recv(tmp_mf, source=i, tag=14)  # receive results from the process
            final_sc_h = np.vstack((final_sc_h, tmp_sh))  # add the received results to the final results

            tmp_orb_t = np.empty((rank_size-1, final_orb_t.shape[0]), dtype=np.int)
            comm.Recv(tmp_orb_t, source=i, tag=20)
            final_orb_t = np.vstack((final_orb_t, tmp_orb_t))

            tmp_sim_t = np.empty((rank_size-1, final_sim_t.shape[0]), dtype=np.int)
            comm.Recv(tmp_sim_t, source=i, tag=24)
            final_sim_t = np.vstack((final_sim_t, tmp_sim_t))

        sh_out = final_sc_h.flatten()
        orb_t_out = final_orb_t.flatten()
        sim_t_out = final_sim_t.flatten()
        print("flatten sh: ", sh_out)
        print("flatten orb t: ", orb_t_out)
        print("flatten sim t: ", sim_t_out)


    if rank == 0:
        sim_t_out,orb_t_out,sh_out = (list(t) for t in zip(*sorted(zip(sim_t_out,orb_t_out,sh_out))))
        os.chdir(prob_dir)
        if args.update:
            with open('mass_with_time.csv', 'a', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(zip(sim_t_out,orb_t_out,sh_out))
        else:
            with open('mass_with_time.csv', 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["sim_time", "orbit_time", "scale_height"])
                writer.writerows(zip(sim_t_out,orb_t_out,sh_out))

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the average geometric scale height over the disk')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='specify whether the results being analysed are from a restart')
    args = parser.parse_args()

    main(**vars(args))
