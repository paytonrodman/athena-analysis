#!/usr/bin/env python3
#
# calc_mass.py
#
# A program to calculate the mass accretion rate of an Athena++ disk using MPI.
#
# To run:
# mpirun -n [n] python calc_mass.py [options]
# for [n] cores.
import numpy as np
import os
import sys
sys.path.append('/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
#sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')
import athena_read
import AAT
import glob
import re
import csv
import argparse
from mpi4py import MPI


def main(**kwargs):
    # get number of processors and processor rank
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    problem  = args.prob_id
    #root_dir = "/Users/paytonrodman/athena-sim/"
    root_dir = '/home/per29/rds/rds-accretion-zyNhkonJSR8/'
    prob_dir = root_dir + problem + '/'
    data_dir = prob_dir + 'data/'
    runfile_dir = prob_dir + 'runfiles/'
    os.chdir(data_dir)

    # check if data file already exists
    csv_time = np.empty(0)
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

    data_input = athena_read.athinput(runfile_dir + 'athinput.' + problem)
    mass = data_input['problem']['mass']
    x1min = data_input['mesh']['x1min']

    #times = np.asarray([0,10,20,30,40,50])
    # distribute files to cores
    count = len(times) // size  # number of files for each process to analyze
    remainder = len(times) % size  # extra files if times is not a multiple of size
    if rank < remainder:  # processes with rank < remainder analyze one extra catchment
        start = rank * (count + 1)  # index of first file to analyze
        stop = start + count + 1  # index of last file to analyze
    else:
        start = rank * count + remainder
        stop = start + count
    local_times = times[start:stop] # get the times to be analyzed by each rank

    local_mf_total = []
    local_orbit_time = []
    local_sim_time = []
    #print("times are: ", local_times)
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_cons = athena_read.athdf(problem + '.cons.' + str_t + '.athdf')

        #unpack data
        x2v = data_cons['x2v'] # theta
        x3v = data_cons['x3v'] # phi
        x1f = data_cons['x1f'] # r
        x2f = data_cons['x2f'] # theta
        x3f = data_cons['x3f'] # phi
        dens = data_cons['dens']
        mom1 = data_cons['mom1']
        mom2 = data_cons['mom2']
        mom3 = data_cons['mom3']
        # Calculations
        dx1f,dx2f,dx3f = AAT.calculate_delta(x1f,x2f,x3f)
        v1,v2,v3 = AAT.calculate_velocity(mom1,mom2,mom3,dens)

        mf = []
        for j in range(len(x2v)):
            for k in range(len(x3v)):
                dOmega = np.sin(x2f[j]) * dx2f[j] * dx3f[k]
                mf_i = -dens[k,j,0] * v1[k,j,0] * (x1f[0])**2. * dOmega
                mf.append(mf_i)
        local_mf_total.append(np.sum(mf))

        v_Kep0 = np.sqrt(mass/x1min)
        Omega0 = v_Kep0/x1min
        T0 = 2.*np.pi/Omega0
        local_orbit_time.append(t/T0)
        local_sim_time.append(float(t))

    send_mf = np.array(local_mf_total)
    send_orbit = np.array(local_orbit_time)
    send_sim = np.array(local_sim_time)
    # Collect local array sizes using the high-level mpi4py gather
    sendcounts = np.array(comm.gather(len(send_mf), 0))

    if rank == 0:
        recv_mf = np.empty(sum(sendcounts), dtype=float)
        recv_orbit = np.empty(sum(sendcounts), dtype=float)
        recv_sim = np.empty(sum(sendcounts), dtype=float)
    else:
        recv_mf = None
        recv_orbit = None
        recv_sim = None

    comm.Gatherv(sendbuf=send_mf, recvbuf=(recv_mf, sendcounts), root=0)
    comm.Gatherv(sendbuf=send_orbit, recvbuf=(recv_orbit, sendcounts), root=0)
    comm.Gatherv(sendbuf=send_sim, recvbuf=(recv_sim, sendcounts), root=0)
    if rank == 0:
        print("Gathered array mf: {}".format(recv_mf))
        print("Gathered array orb: {}".format(recv_orbit))
        print("Gathered array sim: {}".format(recv_sim))
        mf_out = recv_mf.flatten()
        orb_t_out = recv_orbit.flatten()
        sim_t_out = recv_sim.flatten()

    if rank == 0:
        sim_t_out,orb_t_out,mf_out = (list(t) for t in zip(*sorted(zip(sim_t_out,orb_t_out,mf_out))))
        os.chdir(prob_dir)
        if args.update:
            with open('mass_with_time.csv', 'a', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(zip(sim_t_out,orb_t_out,mf_out))
        else:
            with open('mass_with_time.csv', 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["sim_time", "orbit_time", "mass_flux"])
                writer.writerows(zip(sim_t_out,orb_t_out,mf_out))

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate instantaneous mass flux across inner radial boundary')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='specify whether the results being analysed are from a restart')
    args = parser.parse_args()

    main(**vars(args))
