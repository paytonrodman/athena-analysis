#!/usr/bin/env python3
#
# calc_flux.py
#
# A program to calculate the magnetic flux threading the ISCO of an Athena++ disk using MPI
#
# To run:
# mpirun -n [n] python calc_flux.py [options]
# for [n] cores.
#
import numpy as np
import os
import sys
sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
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
        with open(prob_dir + 'flux_with_time.csv', 'r', newline='') as f:
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

    data_init = athena_read.athdf(problem + '.cons.00000.athdf')
    x2v_init = data_init['x2v']
    th_id = AAT.find_nearest(x2v_init, np.pi/2.)
    data_input = athena_read.athinput(runfile_dir + 'athinput.' + problem)
    mass = data_input['problem']['mass']
    x1min = data_input['mesh']['x1min']

    local_mag_flux_u = []
    local_mag_flux_l = []
    local_orbit_time = []
    local_sim_time = []
    for t in local_times:
        #print('file number: ', t)
        str_t = str(int(t)).zfill(5)
        data_cons = athena_read.athdf(problem + '.cons.' + str_t + '.athdf')

        #unpack data
        x2v = data_cons['x2v'] # theta
        x3v = data_cons['x3v'] # phi
        x1f = data_cons['x1f'] # r
        x2f = data_cons['x2f'] # theta
        x3f = data_cons['x3f'] # phi
        Bcc1 = data_cons['Bcc1']

        # Calculations
        dx1f,dx2f,dx3f = AAT.calculate_delta(x1f,x2f,x3f)

        mf_l = []
        mf_u = []
        for j in range(th_id):
            for k in range(len(x3v)):
                dS = (x1f[0]**2.)*np.sin(x2f[j])*dx2f[j]*dx3f[k] # r^2 * sin(theta) * dtheta * dphi
                mf_i = Bcc1[k,j,0]*dS
                mf_u.append(mf_i)
        for j in range(th_id,len(x2v)):
            for k in range(len(x3v)):
                dS = (x1f[0]**2.)*np.sin(x2f[j])*dx2f[j]*dx3f[k] # r^2 * sin(theta) * dtheta * dphi
                mf_i = Bcc1[k,j,0]*dS
                mf_l.append(mf_i)

        local_mag_flux_u.append(np.sum(mf_u))
        local_mag_flux_l.append(np.sum(mf_l))

        v_Kep0 = np.sqrt(mass/x1min)
        Omega0 = v_Kep0/x1min
        T0 = 2.*np.pi/Omega0
        local_orbit_time.append(t/T0)
        local_sim_time.append(float(t))

    send_mfu = np.array(local_mag_flux_u)
    send_mfl = np.array(local_mag_flux_l)
    send_orbit = np.array(local_orbit_time)
    send_sim = np.array(local_sim_time)
    # Collect local array sizes using the high-level mpi4py gather
    sendcounts = np.array(comm.gather(len(send_mfu), 0))

    if rank == 0:
        recv_mfu = np.empty(sum(sendcounts), dtype=float)
        recv_mfl = np.empty(sum(sendcounts), dtype=float)
        recv_orbit = np.empty(sum(sendcounts), dtype=float)
        recv_sim = np.empty(sum(sendcounts), dtype=float)
    else:
        recv_mfu = None
        recv_mfl = None
        recv_orbit = None
        recv_sim = None

    comm.Gatherv(sendbuf=send_mfu, recvbuf=(recv_mfu, sendcounts), root=0)
    comm.Gatherv(sendbuf=send_mfl, recvbuf=(recv_mfl, sendcounts), root=0)
    comm.Gatherv(sendbuf=send_orbit, recvbuf=(recv_orbit, sendcounts), root=0)
    comm.Gatherv(sendbuf=send_sim, recvbuf=(recv_sim, sendcounts), root=0)
    if rank == 0:
        mfu_out = recv_mfu.flatten()
        mfl_out = recv_mfl.flatten()
        orb_t_out = recv_orbit.flatten()
        sim_t_out = recv_sim.flatten()

    if rank == 0:
        sim_t_out,orb_t_out,mfu_out,mfl_out = (list(t) for t in zip(*sorted(zip(sim_t_out,orb_t_out,mfu_out,mfl_out))))
        os.chdir(prob_dir)
        if args.update:
            with open('flux_with_time.csv', 'a', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(zip(sim_t_out,orb_t_out,mfu_out,mfl_out))
        else:
            with open('flux_with_time.csv', 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["sim_time", "orbit_time", "mag_flux_u", "mag_flux_l"])
                writer.writerows(zip(sim_t_out,orb_t_out,mfu_out,mfl_out))

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate various quality factors from raw simulation data.')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='specify whether the results being analysed are from a restart')
    args = parser.parse_args()

    main(**vars(args))
