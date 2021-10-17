#!/usr/bin/python
import numpy as np
import os
import sys
sys.path.insert(0, '../../dependencies')
import athena_read
import AAT
import glob
import re
import csv
import argparse

def main(**kwargs):
    problem  = args.prob_id
    root_dir = "/Users/paytonrodman/athena-sim/"
    data_dir = root_dir + problem + "/data/"
    os.chdir(data_dir)

    csv_time = []
    # check if data file already exists
    if args.update:
        with open('../beta_with_time.csv', 'r', newline='') as f:
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

    data_input = athena_read.athinput("../athinput." + problem)
    scale_height = data_input['problem']['h_r']
    data_init = athena_read.athdf(problem + ".cons.00000.athdf")
    x2v = data_init['x2v']
    th_u = AAT.find_nearest(x2v, np.pi/2. + (3.*scale_height))
    th_l = AAT.find_nearest(x2v, np.pi/2. - (3.*scale_height))

    beta_list = []
    for t in sorted(times):
        print("file number: ", t)
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
        beta_list.append(current_beta)

    times, beta_list = (list(t) for t in zip(*sorted(zip(times, beta_list))))
    os.chdir("../")
    if args.update:
        with open('beta_with_time.csv', 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(times,beta_list))
    else:
        with open('beta_with_time.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["Time", "plasma_beta"])
            writer.writerows(zip(times,beta_list))

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
