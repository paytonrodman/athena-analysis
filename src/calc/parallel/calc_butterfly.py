#!/usr/bin/env python3
#
import numpy as np
import os
import sys
#sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')
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
    root_dir = "/Users/paytonrodman/athena-sim/"
    #root_dir = '/home/per29/rds/rds-accretion-zyNhkonJSR8/'
    prob_dir = root_dir + problem + '/'
    data_dir = prob_dir + 'data/'
    runfile_dir = prob_dir + 'runfiles/'
    os.chdir(data_dir)

    # check if data file already exists
    csv_time = np.empty(0)
    if args.update:
        with open(prob_dir + 'butterfly_with_time.csv', 'r', newline='') as f:
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
    x1v_init = data_init['x1v']
    data_input = athena_read.athinput(runfile_dir + 'athinput.' + problem)
    mass = data_input['problem']['mass']
    x1min = data_input['mesh']['x1min']

    if kwargs['r'] is not None:
        r_id = AAT.find_nearest(x1v_init, kwargs['r'])
    else:
        r_id = AAT.find_nearest(x1v_init, 25.) # approx. middle of high res region

    local_Bcc1_theta = []
    local_Bcc2_theta = []
    local_Bcc3_theta = []
    local_Bpol = []
    local_orbit_time = []
    local_sim_time = []
    for t in local_times:
        #print('file number: ,' t)
        str_t = str(int(t)).zfill(5)
        data_cons = athena_read.athdf(problem + '.cons.' + str_t + '.athdf')

        #unpack data
        Bcc1 = data_cons['Bcc1']
        Bcc2 = data_cons['Bcc2']
        Bcc3 = data_cons['Bcc3']
        Bpol_i = Bcc1 + Bcc2

        local_Bcc1_theta.append(np.average(Bcc1[r_id,:,:],axis=1).tolist())
        local_Bcc2_theta.append(np.average(Bcc2[r_id,:,:],axis=1).tolist())
        local_Bcc3_theta.append(np.average(Bcc3[r_id,:,:],axis=1).tolist())
        local_Bpol.append(np.average(Bpol_i[r_id,:,:],axis=1).tolist())

        v_Kep0 = np.sqrt(mass/x1min)
        Omega0 = v_Kep0/x1min
        T0 = 2.*np.pi/Omega0
        local_orbit_time.append(t/T0)
        local_sim_time.append(float(t))

    send_b1 = np.array(local_Bcc1_theta)
    send_b2 = np.array(local_Bcc2_theta)
    send_b3 = np.array(local_Bcc3_theta)
    send_bpol = np.array(local_Bpol)
    send_orbit = np.array(local_orbit_time)
    send_sim = np.array(local_sim_time)
    # Collect local array sizes using the high-level mpi4py gather
    sendcounts = np.array(comm.gather(len(send_b1), 0))

    if rank == 0:
        recv_b1 = np.empty(sum(sendcounts), dtype=float)
        recv_b2 = np.empty(sum(sendcounts), dtype=float)
        recv_b3 = np.empty(sum(sendcounts), dtype=float)
        recv_bpol = np.empty(sum(sendcounts), dtype=float)
        recv_orbit = np.empty(sum(sendcounts), dtype=float)
        recv_sim = np.empty(sum(sendcounts), dtype=float)
    else:
        recv_b1 = None
        recv_b2 = None
        recv_b3 = None
        recv_bpol = None
        recv_orbit = None
        recv_sim = None

    comm.Gatherv(sendbuf=send_b1, recvbuf=(recv_b1, sendcounts), root=0)
    comm.Gatherv(sendbuf=send_b2, recvbuf=(recv_b2, sendcounts), root=0)
    comm.Gatherv(sendbuf=send_b3, recvbuf=(recv_b3, sendcounts), root=0)
    comm.Gatherv(sendbuf=send_bpol, recvbuf=(recv_bpol, sendcounts), root=0)
    comm.Gatherv(sendbuf=send_orbit, recvbuf=(recv_orbit, sendcounts), root=0)
    comm.Gatherv(sendbuf=send_sim, recvbuf=(recv_sim, sendcounts), root=0)
    if rank == 0:
        b1_out = recv_b1.flatten()
        b2_out = recv_b2.flatten()
        b3_out = recv_b3.flatten()
        bpol_out = recv_bpol.flatten()
        orb_t_out = recv_orbit.flatten()
        sim_t_out = recv_sim.flatten()

    if rank == 0:
        sim_t_out,orb_t_out,b1_out,b2_out,b3_out,bpol_out = (list(t) for t in zip(*sorted(zip(sim_t_out,orb_t_out,b1_out,b2_out,b3_out,bpol_out))))
        os.chdir(prob_dir)
        if args.update:
            with open('butterfly_with_time.csv', 'a', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(zip(sim_t_out,orb_t_out,b1_out,b2_out,b3_out,bpol_out))
        else:
            with open('butterfly_with_time.csv', 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["sim_time", "orbit_time", "Bcc1", "Bcc2", "Bcc3", "Bpol"])
                writer.writerows(zip(sim_t_out,orb_t_out,b1_out,b2_out,b3_out,bpol_out))

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate average magnetic field values within the midplane at a given radius (r)')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='specify whether the results being analysed are from a restart')
    parser.add_argument('--r',
                        type=float,
                        default=None,
                        help=('value of r where averages are calculated (default: 25rg)'))
    args = parser.parse_args()

    main(**vars(args))
