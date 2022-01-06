#!/usr/bin/python
#
# Calculate magnetic plasma beta
#
# Usage: python calc_beta.py [problem_id] [-u]
#
import numpy as np
import os
import sys
sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')
#sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
import athena_read
import AAT
import glob
import re
import csv
import argparse

def main(**kwargs):
    problem  = args.prob_id
    root_dir = "/Users/paytonrodman/athena-sim/"
    #root_dir = '/home/per29/rds/rds-accretion-zyNhkonJSR8/'
    prob_dir = root_dir + problem + '/'
    data_dir = prob_dir + 'data/'
    runfile_dir = prob_dir + 'runfiles/'
    os.chdir(data_dir)

    data_input = athena_read.athinput(runfile_dir + 'athinput.' + problem)
    mass = data_input['problem']['mass']
    x1min = data_input['mesh']['x1min']

    csv_time = []
    # check if data file already exists
    if args.update:
        with open(prob_dir + 'beta_with_time.csv', 'r', newline='') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            next(csv_reader, None) # skip header
            for row in csv_reader:
                csv_time.append(float(row[0]))

    files = glob.glob('./*.athdf')
    times = []
    for f in files:
        time_sec = re.findall(r'\b\d+\b', f)
        if args.update:
            if float(time_sec[0]) not in times and float(time_sec[0]) not in csv_time:
                times.append(float(time_sec[0]))
        else:
            if float(time_sec[0]) not in times:
                times.append(float(time_sec[0]))
    if len(times)==0:
        sys.exit('No new timesteps to analyse in the given directory. Exiting.')

    data_input = athena_read.athinput(runfile_dir + 'athinput.' + problem)
    scale_height = data_input['problem']['h_r']
    data_init = athena_read.athdf(problem + '.cons.00000.athdf')
    x1v = data_init['x1v']
    x2v = data_init['x2v']
    th_u = AAT.find_nearest(x2v, np.pi/2. + (2.*scale_height))
    th_l = AAT.find_nearest(x2v, np.pi/2. - (2.*scale_height))
    r_u = AAT.find_nearest(x1v, 100.)

    orbit_time = []
    sim_time = []
    beta_list = []
    for t in sorted(times):
        str_t = str(int(t)).zfill(5)

        data_prim = athena_read.athdf(problem + ".prim." + str_t + ".athdf")
        data_cons = athena_read.athdf(problem + ".cons." + str_t + ".athdf")

        #unpack data
        dens = data_cons['dens']
        Bcc1 = data_cons['Bcc1']
        Bcc2 = data_cons['Bcc2']
        Bcc3 = data_cons['Bcc3']
        press = data_prim['press']

        current_beta = calculate_beta(r_u,th_u,th_l,dens,press,Bcc1,Bcc2,Bcc3)
        beta_list.append(current_beta)

        v_Kep0 = np.sqrt(mass/x1min)
        Omega0 = v_Kep0/x1min
        T0 = 2.*np.pi/Omega0
        orbit_time.append(t/T0)
        sim_time.append(t)

    sim_time,orbit_time,beta_list = (list(t) for t in zip(*sorted(zip(sim_time,orbit_time,beta_list))))
    os.chdir(prob_dir)
    if args.update:
        with open('beta_with_time.csv', 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(sim_time,orbit_time,beta_list))
    else:
        with open('beta_with_time.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["sim_time", "orbit_time", "plasma_beta"])
            writer.writerows(zip(sim_time,orbit_time,beta_list))

def calculate_beta(r_u,th_u,th_l,x1v,x2v,x3v,x1f,x2f,x3f,dens,press,Bcc1,Bcc2,Bcc3):
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
    r,theta,_ = np.meshgrid(x3v,x2v,x1v, sparse=False, indexing='ij')
    dx1f,dx2f,dx3f = AAT.calculate_delta(x1f,x2f,x3f)
    dphi,dtheta,dr = np.meshgrid(dx3f,dx2f,dx1f, sparse=False, indexing='ij')

    # Density-weighted mean gas pressure
    sum_p = 0.
    numWeight_p = 0.
    sum_b = 0.
    numWeight_b = 0.

    dphi = dphi[:r_u,th_l:th_u,:]
    dtheta = dtheta[:r_u,th_l:th_u,:]
    dr = dr[:r_u,th_l:th_u,:]
    r = r[:r_u,th_l:th_u,:]
    theta = theta[:r_u,th_l:th_u,:]
    pressure = press[:r_u,th_l:th_u,:]
    density = dens[:r_u,th_l:th_u,:]

    volume = (r**2.)*np.sin(theta)*dr*dtheta*dphi
    # Find volume centred total magnetic field
    bcc_all = np.square(Bcc1[:r_u,th_l:th_u,:]) +
              np.square(Bcc2[:r_u,th_l:th_u,:]) +
              np.square(Bcc3[:r_u,th_l:th_u,:])

    numWeight_p = np.sum(pressure*density*volume)
    sum_p       = np.sum(density*volume)
    numWeight_b = np.sum(bcc_all*density*volume)
    sum_b       = np.sum(density*volume)

    pres_av = numWeight_p/sum_p
    bcc_av = numWeight_b/sum_b
    if bcc_av>0:
        current_beta = 2. * pres_av / bcc_av
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
