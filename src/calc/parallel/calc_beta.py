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
        with open(prob_dir + 'beta_with_time.csv', 'r', newline='') as f:
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

    data_input = athena_read.athinput(runfile_dir + 'athinput.' + problem)
    scale_height = data_input['problem']['h_r']
    mass = data_input['problem']['mass']
    x1min = data_input['mesh']['x1min']
    data_init = athena_read.athdf(problem + '.cons.00000.athdf')
    x2v = data_init['x2v']
    th_u = AAT.find_nearest(x2v, np.pi/2. + (3.*scale_height))
    th_l = AAT.find_nearest(x2v, np.pi/2. - (3.*scale_height))

    local_beta = []
    local_orbit_time = []
    local_sim_time = []
    for t in local_times:
        #print('file number: ', t)
        str_t = str(int(t)).zfill(5)
        data_prim = athena_read.athdf(problem + ".prim." + str_t + ".athdf")
        data_cons = athena_read.athdf(problem + ".cons." + str_t + ".athdf")

        #unpack data
        dens = data_cons['dens']
        Bcc1 = data_cons['Bcc1']
        Bcc2 = data_cons['Bcc2']
        Bcc3 = data_cons['Bcc3']
        press = data_prim['press']

        current_beta = calculate_beta(th_u,th_l,dens,press,Bcc1,Bcc2,Bcc3)
        local_beta.append(current_beta)

        v_Kep0 = np.sqrt(mass/x1min)
        Omega0 = v_Kep0/x1min
        T0 = 2.*np.pi/Omega0
        local_orbit_time.append(t/T0)
        local_sim_time.append(float(t))

    send_beta = np.array(local_beta)
    send_orbit = np.array(local_orbit_time)
    send_sim = np.array(local_sim_time)
    # Collect local array sizes using the high-level mpi4py gather
    sendcounts = np.array(comm.gather(len(send_beta), 0))

    if rank == 0:
        recv_beta = np.empty(sum(sendcounts), dtype=float)
        recv_orbit = np.empty(sum(sendcounts), dtype=float)
        recv_sim = np.empty(sum(sendcounts), dtype=float)
    else:
        recv_beta = None
        recv_orbit = None
        recv_sim = None

    comm.Gatherv(sendbuf=send_beta, recvbuf=(recv_beta, sendcounts), root=0)
    comm.Gatherv(sendbuf=send_orbit, recvbuf=(recv_orbit, sendcounts), root=0)
    comm.Gatherv(sendbuf=send_sim, recvbuf=(recv_sim, sendcounts), root=0)
    if rank == 0:
        print("Gathered array beta: {}".format(recv_beta))
        print("Gathered array orb: {}".format(recv_orbit))
        print("Gathered array sim: {}".format(recv_sim))
        beta_out = recv_beta.flatten()
        orb_t_out = recv_orbit.flatten()
        sim_t_out = recv_sim.flatten()

    if rank == 0:
        sim_t_out,orb_t_out,beta_out = (list(t) for t in zip(*sorted(zip(sim_t_out,orb_t_out,beta_out))))
        os.chdir(prob_dir)
        if args.update:
            with open('beta_with_time.csv', 'a', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(zip(sim_t_out,orb_t_out,beta_out))
        else:
            with open('beta_with_time.csv', 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["sim_time", "orbit_time", "plasma_beta"])
                writer.writerows(zip(sim_t_out,orb_t_out,beta_out))

def calculate_beta(th_u,th_l,dens,press,Bcc1,Bcc2,Bcc3):
    """Calculate the mean plasma beta within a specified region.

    Args:
        th_u: the upper boundary in x2.
        th_l: the lower boundary in x2.
        dens: the number density.
        press: the gas pressure.
        Bcc1: the cell-centred magnetic field in the x1 direction.
        Bcc2: the cell-centred magnetic field in the x2 direction.
        Bcc3: the cell-centred magnetic field in the x3 direction.
    Returns:
        the mean plasma beta.

    """
    # Density-weighted mean gas pressure
    sum_p = 0.
    numWeight_p = 0.
    sum_b = 0.
    numWeight_b = 0.

    pressure = press[:,th_l:th_u,:]
    density = dens[:,th_l:th_u,:]
    # Find volume centred total magnetic field
    bcc_all = np.sqrt(np.square(Bcc1[:,th_l:th_u,:]) +
                      np.square(Bcc2[:,th_l:th_u,:]) +
                      np.square(Bcc3[:,th_l:th_u,:]))

    numWeight_p = np.sum(pressure*density)
    sum_p       = np.sum(density)
    numWeight_b = np.sum(bcc_all*density)
    sum_b       = np.sum(density)

    pres_av = numWeight_p/sum_p
    bcc_av = numWeight_b/sum_b
    if bcc_av>0:
        current_beta = 2. * pres_av / (bcc_av**2.)
    else:
        current_beta = np.nan

    return current_beta

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate plasma beta from raw simulation data.')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='specify whether the results being analysed are from a restart (reduces computational time by only appending to files, rather than rewriting)')
    args = parser.parse_args()

    main(**vars(args))
