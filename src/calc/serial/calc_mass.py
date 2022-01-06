#!/usr/bin/python
#
# Calculate mass flux crossing inner boundary
#
# Usage: python calc_mass.py [problem_id] [-u]
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
    init_data = athena_read.athdf(problem + '.cons.00000.athdf')
    x1v_init = init_data['x1v'] # r
    r_id_0 = AAT.find_nearest(x1v_init, 6.) # find index of ISCO
    r_id_1 = AAT.find_nearest(x1v_init, 25.)
    r_id_2 = AAT.find_nearest(x1v_init, 50.)
    r_id_3 = AAT.find_nearest(x1v_init, 75.)
    r_id_4 = AAT.find_nearest(x1v_init, 100.)

    csv_time = []
    # check if data file already exists
    if args.update:
        with open(prob_dir + 'mass_with_time.csv', 'r', newline='') as f:
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

    orbit_time = []
    sim_time = []
    mf0_total,mf1_total,mf2_total,mf3_total,mf4_total = [],[],[],[],[]
    for t in sorted(times):
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
        _,dx2f,dx3f = AAT.calculate_delta(x1f,x2f,x3f)
        v1,_,v3 = AAT.calculate_velocity(mom1,mom2,mom3,dens)

        mf0,mf1,mf2,mf3,mf4 = [],[],[],[],[]
        for j in range(len(x2v)):
            for k in range(len(x3v)):
                dOmega = np.sin(x2f[j]) * dx2f[j] * dx3f[k]
                mf0_i = -dens[k,j,r_id_0] * v1[k,j,r_id_0] * (x1f[r_id_0])**2. * dOmega
                mf1_i = -dens[k,j,r_id_1] * v1[k,j,r_id_1] * (x1f[r_id_1])**2. * dOmega
                mf2_i = -dens[k,j,r_id_2] * v1[k,j,r_id_2] * (x1f[r_id_2])**2. * dOmega
                mf3_i = -dens[k,j,r_id_3] * v1[k,j,r_id_3] * (x1f[r_id_3])**2. * dOmega
                mf4_i = -dens[k,j,r_id_4] * v1[k,j,r_id_4] * (x1f[r_id_4])**2. * dOmega
                mf0.append(mf0_i)
                mf1.append(mf1_i)
                mf2.append(mf2_i)
                mf3.append(mf3_i)
                mf4.append(mf4_i)

        mf0_total.append(np.sum(mf0))
        mf1_total.append(np.sum(mf1))
        mf2_total.append(np.sum(mf2))
        mf3_total.append(np.sum(mf3))
        mf4_total.append(np.sum(mf4))

        v_Kep0 = np.sqrt(mass/x1min)
        Omega0 = v_Kep0/x1min
        T0 = 2.*np.pi/Omega0
        orbit_time.append(t/T0)
        sim_time.append(t)


    sim_time,orbit_time,mf0_total,mf1_total,mf2_total,mf3_total,mf4_total = (list(t) for t in zip(*sorted(zip(sim_time,orbit_time,mf0_total,mf1_total,mf2_total,mf3_total,mf4_total))))
    os.chdir(prob_dir)
    if args.update:
        with open('mass_with_time.csv', 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(sim_time,orbit_time,mf0_total,mf1_total,mf2_total,mf3_total,mf4_total))
    else:
        with open('mass_with_time.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["sim_time", "orbit_time", "mass_flux_6rg", "mass_flux_25rg", "mass_flux_50rg", "mass_flux_75rg", "mass_flux_100rg"])
            writer.writerows(zip(sim_time,orbit_time,mf0_total,mf1_total,mf2_total,mf3_total,mf4_total))

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
