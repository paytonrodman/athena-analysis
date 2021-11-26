#!/usr/bin/python
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

def main(**kwargs):
    problem  = args.prob_id
    root_dir = "/Users/paytonrodman/athena-sim/"
    #root_dir = '~/rds/rds-accretion-zyNhkonJSR8/'
    prob_dir = root_dir + problem + '/'
    data_dir = prob_dir + 'data/'
    runfile_dir = prob_dir + 'runfiles/'
    os.chdir(data_dir)

    csv_time = []
    # check if data file already exists
    if args.update:
        with open(prob_dir + 'butterfly_with_time.csv', 'r', newline='') as f:
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

    data_init = athena_read.athdf(problem + '.cons.00000.athdf')
    x1v_init = data_init['x1v']
    x2v_init = data_init['x2v']
    data_input = athena_read.athinput(runfile_dir + 'athinput.' + problem)
    mass = data_input['problem']['mass']
    x1min = data_init['x1v']['x1min']

    if kwargs['r'] is not None:
        r_id = AAT.find_nearest(x1v_init, kwargs['r'])
    else:
        r_id = AAT.find_nearest(x1v_init, 25.) # approx. middle of high res region

    Bcc1_theta = []
    Bcc2_theta = []
    Bcc3_theta = []
    orbit_time = []
    sim_time = []
    for t in sorted(times):
        #print('file number: ,' t)
        str_t = str(int(t)).zfill(5)

        data_cons = athena_read.athdf(problem + '.cons.' + str_t + '.athdf')

        #unpack data
        x2f = data_cons['x2f'] # theta
        Bcc1 = data_cons['Bcc1']
        Bcc2 = data_cons['Bcc2']
        Bcc3 = data_cons['Bcc3']

        Bcc1_theta.append(np.average(Bcc1[r_id,:,:],axis=1).tolist())
        Bcc2_theta.append(np.average(Bcc2[r_id,:,:],axis=1).tolist())
        Bcc3_theta.append(np.average(Bcc3[r_id,:,:],axis=1).tolist())

        v_Kep0 = np.sqrt(mass/x1min)
        Omega0 = v_Kep0/x1min
        T0 = 2.*np.pi/Omega0
        orbit_time.append(t/T0)

        sim_time.append(t)

    sim_time,orbit_time,Bcc1_theta,Bcc2_theta,Bcc3_theta = (list(t) for t in zip(*sorted(zip(sim_time,orbit_time,Bcc1_theta,Bcc2_theta,Bcc3_theta))))
    os.chdir(prob_dir)
    if args.update:
        with open('butterfly_with_time.csv', 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(sim_time,orbit_time,Bcc1_theta,Bcc2_theta,Bcc3_theta))
    else:
        with open('butterfly_with_time.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["sim_time", "orbit_time", "Bcc1", "Bcc2", "Bcc3"])
            writer.writerows(zip(sim_time,orbit_time,Bcc1_theta,Bcc2_theta,Bcc3_theta))

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
