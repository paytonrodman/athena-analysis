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
import scipy.stats
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

    data_input = athena_read.athinput(runfile_dir + 'athinput.' + problem)
    x1min = data_input['mesh']['x1min'] # bounds of simulation
    x1max = data_input['mesh']['x1max']
    x2min = data_input['mesh']['x2min']
    x2max = data_input['mesh']['x2max']
    x1_high_min = data_input['refinement3']['x1min'] # bounds of high resolution region
    x1_high_max = data_input['refinement3']['x1max']
    x2_high_min = data_input['refinement3']['x2min']
    x2_high_max = data_input['refinement3']['x2max']
    mass = data_input['problem']['mass']

    init_data = athena_read.athdf(problem + '.cons.00000.athdf')
    x1v_init = init_data['x1v'] # r
    x2v_init = init_data['x2v'] # theta

    if kwargs['r_lower'] is not None:
        if not x1min <= kwargs['r_lower'] < x1max:
            sys.exit('Error: Lower r value must be between %d and %d' % x1min,x1max)
        rl = AAT.find_nearest(x1v_init, kwargs['r_lower'])
    else:
        rl = AAT.find_nearest(x1v_init, x1_high_min)
    if kwargs['r_upper'] is not None:
        if not x1min <= kwargs['r_upper'] < x1max:
            sys.exit('Error: Upper r value must be between %d and %d' % x1min,x1max)
        ru = AAT.find_nearest(x1v_init, kwargs['r_upper'])
    else:
        ru = AAT.find_nearest(x1v_init, x1_high_max)
    if kwargs['theta_lower'] is not None:
        if not x2min <= kwargs['theta_lower'] < x2max:
            sys.exit('Error: Lower theta value must be between %d and %d' % x2min,x2max)
        tl = AAT.find_nearest(x2v_init, kwargs['theta_lower'])
    else:
        tl = AAT.find_nearest(x2v_init, x2_high_min)
    if kwargs['theta_upper'] is not None:
        if not x2min <= kwargs['theta_upper'] < x2max:
            sys.exit('Error: Upper theta value must be between %d and %d' % x2min,x2max)
        tu = AAT.find_nearest(x2v_init, kwargs['theta_upper'])
    else:
        tu = AAT.find_nearest(x2v_init, x2_high_max)

    if rl==ru:
        ru += 1
    if tl==tu:
        tu += 1

    filename_output = 'qual_with_time_' + str(rl) + '_' + str(ru) + '_' + str(tl) + '_' + str(tu) + '.csv'
    # check if data file already exists
    csv_time = np.empty(0)
    if args.update:
        with open(prob_dir + filename_output, 'r', newline='') as f:
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
    if rank < remainder:  # processes with rank < remainder analyze one extra file
        start = rank * (count + 1)  # index of first file to analyze
        stop = start + count + 1  # index of last file to analyze
    else:
        start = rank * count + remainder
        stop = start + count
    local_times = times[start:stop] # get the times to be analyzed by each rank

    local_theta_B = []
    local_Q_theta_low, local_Q_theta_av, local_Q_theta_high = [], [], []
    local_Q_phi_low, local_Q_phi_av, local_Q_phi_high = [], [], []
    local_orbit_time = []
    local_sim_time = []
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_prim = athena_read.athdf(problem + '.prim.' + str_t + '.athdf')
        data_cons = athena_read.athdf(problem + '.cons.' + str_t + '.athdf')

        #constants
        gamma = 5./3.
        GM = 1.

        #unpack data
        x1v = data_cons['x1v'] # r
        x2v = data_cons['x2v'] # theta
        x3v = data_cons['x3v'] # phi
        x1f = data_cons['x1f'] # r
        x2f = data_cons['x2f'] # theta
        x3f = data_cons['x3f'] # phi
        dens = data_cons['dens']
        mom1 = data_cons['mom1']
        mom2 = data_cons['mom2']
        mom3 = data_cons['mom3']
        Bcc1 = data_cons['Bcc1']
        Bcc2 = data_cons['Bcc2']
        Bcc3 = data_cons['Bcc3']
        press = data_prim['press']

        # Calculations
        dx1f,dx2f,dx3f = AAT.calculate_delta(x1f,x2f,x3f)
        v1,v2,v3 = AAT.calculate_velocity(mom1,mom2,mom3,dens)
        Omega_kep = np.sqrt(GM/(x1v**3.)) #Keplerian angular velocity in midplane

        tB = magnetic_angle(Bcc1,Bcc2)
        tB_av = np.average(tB[rl:ru,tl:tu,:])

        Qt,Qp = quality_factors(x1v,x2v,x3v,dx1f,dx2f,dx3f,dens,press,v2,Bcc1,Bcc2,Bcc3,Omega_kep,gamma)
        Qt = Qt[rl:ru,tl:tu,:]
        Qp = Qp[rl:ru,tl:tu,:]
        Qt_av,Qt_lc,Qt_uc = mean_confidence_interval(Qt.flatten(), confidence=0.95)
        Qp_av,Qp_lc,Qp_uc = mean_confidence_interval(Qp.flatten(), confidence=0.95)

        local_theta_B.append(tB_av)

        local_Q_theta_low.append(Qt_lc)
        local_Q_theta_av.append(Qt_av)
        local_Q_theta_high.append(Qt_uc)

        local_Q_phi_low.append(Qp_lc)
        local_Q_phi_av.append(Qp_av)
        local_Q_phi_high.append(Qp_uc)

        v_Kep0 = np.sqrt(mass/x1min)
        Omega0 = v_Kep0/x1min
        T0 = 2.*np.pi/Omega0
        local_orbit_time.append(t/T0)
        local_sim_time.append(float(t))

    send_thB = np.array(local_theta_B)
    send_Qtl = np.array(local_Q_theta_low)
    send_Qta = np.array(local_Q_theta_av)
    send_Qth = np.array(local_Q_theta_high)
    send_Qpl = np.array(local_Q_phi_low)
    send_Qpa = np.array(local_Q_phi_av)
    send_Qph = np.array(local_Q_phi_high)
    send_orbit = np.array(local_orbit_time)
    send_sim = np.array(local_sim_time)
    # Collect local array sizes using the high-level mpi4py gather
    sendcounts = np.array(comm.gather(len(send_thB), 0))

    if rank == 0:
        recv_thB = np.empty(sum(sendcounts), dtype=float)
        recv_Qtl = np.empty(sum(sendcounts), dtype=float)
        recv_Qta = np.empty(sum(sendcounts), dtype=float)
        recv_Qth = np.empty(sum(sendcounts), dtype=float)
        recv_Qpl = np.empty(sum(sendcounts), dtype=float)
        recv_Qpa = np.empty(sum(sendcounts), dtype=float)
        recv_Qph = np.empty(sum(sendcounts), dtype=float)
        recv_orbit = np.empty(sum(sendcounts), dtype=float)
        recv_sim = np.empty(sum(sendcounts), dtype=float)
    else:
        recv_thB = None
        recv_Qtl = None
        recv_Qta = None
        recv_Qth = None
        recv_Qpl = None
        recv_Qpa = None
        recv_Qph = None
        recv_orbit = None
        recv_sim = None

    comm.Gatherv(sendbuf=send_thB, recvbuf=(recv_thB, sendcounts), root=0)
    comm.Gatherv(sendbuf=send_Qtl, recvbuf=(recv_Qtl, sendcounts), root=0)
    comm.Gatherv(sendbuf=send_Qta, recvbuf=(recv_Qta, sendcounts), root=0)
    comm.Gatherv(sendbuf=send_Qth, recvbuf=(recv_Qth, sendcounts), root=0)
    comm.Gatherv(sendbuf=send_Qpl, recvbuf=(recv_Qpl, sendcounts), root=0)
    comm.Gatherv(sendbuf=send_Qpa, recvbuf=(recv_Qpa, sendcounts), root=0)
    comm.Gatherv(sendbuf=send_Qph, recvbuf=(recv_Qph, sendcounts), root=0)
    comm.Gatherv(sendbuf=send_orbit, recvbuf=(recv_orbit, sendcounts), root=0)
    comm.Gatherv(sendbuf=send_sim, recvbuf=(recv_sim, sendcounts), root=0)
    if rank == 0:
        thB_out = recv_thB.flatten()
        Qtl_out = recv_Qtl.flatten()
        Qta_out = recv_Qta.flatten()
        Qth_out = recv_Qth.flatten()
        Qpl_out = recv_Qpl.flatten()
        Qpa_out = recv_Qpa.flatten()
        Qph_out = recv_Qph.flatten()
        orb_t_out = recv_orbit.flatten()
        sim_t_out = recv_sim.flatten()


    if rank == 0:
        sim_t_out,orb_t_out,thB_out,Qtl_out,Qta_out,Qth_out,Qpl_out,Qpa_out,Qph_out = (list(t) for t in zip(*sorted(zip(sim_t_out,orb_t_out,thB_out,Qtl_out,Qta_out,Qth_out,Qpl_out,Qpa_out,Qph_out))))
        os.chdir(prob_dir)
        if args.update:
            with open(filename_output, 'a', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(zip(sim_t_out,orb_t_out,thB_out,Qtl_out,Qta_out,Qth_out,Qpl_out,Qpa_out,Qph_out))
        else:
            with open(filename_output, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["sim_time", "orbit_time", "theta_B", "Q_theta_low", "Q_theta_av", "Q_theta_high", "Q_phi_low", "Q_phi_av", "Q_phi_high"])
                writer.writerows(zip(sim_t_out,orb_t_out,thB_out,Qtl_out,Qta_out,Qth_out,Qpl_out,Qpa_out,Qph_out))


def mean_confidence_interval(data, confidence=0.95):
    """Calculate the 95% confidence interval.

    Args:
        data: the data to perfom calculations on.
        confidence: the desired confidence interval (default:0.95).
    Returns:
        the 95% confidence interval.

    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def magnetic_angle(Bcc1,Bcc2):
    """Calculate the magnetic angle, as per Hogg & Reynolds (2018) and others.

    Args:
        Bcc1: the cell-centred magnetic field in the x1 direction.
        Bcc2: the cell-centred magnetic field in the x2 direction.
    Returns:
        the magnetic angle.

    """
    theta_B = (-np.arctan(Bcc1/Bcc2)) * (180./np.pi)
    return theta_B

def quality_factors(x1v,x2v,x3v,dx1f,dx2f,dx3f,dens,press,v2,Bcc1,Bcc2,Bcc3,Omega_kep,gamma):
    """Calculate the quality factors in x2 and x3.

    Args:
        x1v,x2v,x3v: the volume-centred coordinates for x1, x2, and x3 directions.
        dx1f,dx2f,dx3f: the length of each cell in the x1, x2, and x3 directions.
        dens: the number density.
        press: the gas pressure.
        v2: the gas velocity in the x2 direction.
        Bcc1: the cell-centred magnetic field in the x1 direction.
        Bcc2: the cell-centred magnetic field in the x2 direction.
        Bcc3: the cell-centred magnetic field in the x3 direction.
        Omega_kep: the equatorial Keplerian angular velocity.
        gamma: the ratio of specific heats.
    Returns:
        the quality factors in x2 and x3.

    """
    vA_theta,vA_phi = Alfven_vel(dens,press,Bcc1,Bcc2,Bcc3,gamma)
    lambda_MRI_theta = 2.*np.pi*np.sqrt(16./15.)*np.abs(vA_theta)/Omega_kep
    lambda_MRI_phi = 2.*np.pi*np.sqrt(16./15.)*np.abs(vA_phi)/Omega_kep

    phi,theta,r = np.meshgrid(x3v,x2v,x1v, sparse=False, indexing='ij')
    dphi,dtheta,dr = np.meshgrid(dx3f,dx2f,dx1f, sparse=False, indexing='ij')

    Q_theta = lambda_MRI_theta/np.sqrt(r*dphi)
    Q_phi = lambda_MRI_phi/np.sqrt(r*np.abs(np.sin(phi))*dtheta)
    return Q_theta,Q_phi

def Alfven_vel(dens,press,Bcc1,Bcc2,Bcc3,gamma):
    """Calculate the Alfven velocity.

    Args:
        dens: the number density.
        press: the gas pressure.
        Bcc1: the cell-centred magnetic field in the x1 direction.
        Bcc2: the cell-centred magnetic field in the x2 direction.
        Bcc3: the cell-centred magnetic field in the x3 direction.
        gamma: the ratio of specific heats.
    Returns:
        the Alfven velocity.

    """
    w = dens + (gamma/(gamma - 1.))*press
    B2 = Bcc1**2. + Bcc2**2. + Bcc3**2.
    vA_theta = Bcc2/(np.sqrt(w+B2)) #Alfven velocity of theta component of B
    vA_phi = Bcc3/(np.sqrt(w+B2)) #Alfven velocity of phi component of B
    return vA_theta,vA_phi


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate various quality factors from raw simulation data.')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='specify whether the results being analysed are from a restart')
    parser.add_argument('-rl', '--r_lower',
                        type=float,
                        default=None,
                        help='value of lower r bound of region being analysed, must be between x1min and x1max (default=5)')
    parser.add_argument('-ru', '--r_upper',
                        type=float,
                        default=None,
                        help='value of upper r bound of region being analysed, must be between x1min and x1max (default=100)')
    parser.add_argument('-tl', '--theta_lower',
                        type=float,
                        default=None,
                        help='value of lower theta bound of region being analysed, must be between x2min and x2max (default=0.982)')
    parser.add_argument('-tu', '--theta_upper',
                        type=float,
                        default=None,
                        help='value of upper theta bound of region being analysed, must be between x2min and x2max (default=2.159)')
    args = parser.parse_args()

    main(**vars(args))
