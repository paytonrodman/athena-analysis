#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')
import athena_read
import glob
import re
import csv
import argparse

def main(**kwargs):
    problem  = args.prob_id
    data_dir = "/Users/paytonrodman/athena_sim/" + problem + "/data/"
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

    filename_input = "../athinput." + problem
    data_input = athena_read.athinput(filename_input)
    nx1 = data_input['mesh']['nx1']
    nx2 = data_input['mesh']['nx2']
    nx3 = data_input['mesh']['nx3']
    x1min = data_input['mesh']['x1min']
    x1max = data_input['mesh']['x1max']
    x2min = data_input['mesh']['x2min']
    x2max = data_input['mesh']['x2max']
    x3min = data_input['mesh']['x3min']
    x3max = data_input['mesh']['x3max']

    th_id = nx2/2. # midplane

    beta_list = []
    prim_var_names = ['press']
    for t in times:
        print("file number: ", t)
        str_t = str(int(t)).zfill(5)

        filename_cons = problem + ".cons." + str_t + ".athdf"
        filename_prim = problem + ".prim." + str_t + ".athdf"

        data_prim = athena_read.athdf(filename_prim)
        data_cons = athena_read.athdf(filename_cons)

        #constants
        gamma = 5./3.
        GM = 1.
        beta_0 = 200.

        #unpack data
        x1v = data_cons['x1v'] # r
        x3v = data_cons['x3v'] # phi
        dens = data_cons['dens']
        Bcc1 = data_cons['Bcc1']
        Bcc2 = data_cons['Bcc2']
        Bcc3 = data_cons['Bcc3']
        press = data_prim['press']

        current_beta = calculate_beta(beta_0,th_id,x1v,x3v,press,dens,Bcc1,Bcc2,Bcc3)
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

def calculate_beta(beta_0,th_id,x1v,x3v,press,dens,Bcc1,Bcc2,Bcc3):
    # Density-weighted mean gas pressure
    sum_p = 0.
    numWeight_p = 0.
    sum_b = 0.
    numWeight_b = 0.
    for k,phi in enumerate(x3v):
        #for j,theta in enumerate(x2v):
        for i,r in enumerate(x1v):
            j = int(th_id)
            pressure = press[k,j,i]
            density = dens[k,j,i]
            # Find volume centred total magnetic field
            bcc_all = np.sqrt(np.square(Bcc1[k,j,i]) +
                              np.square(Bcc2[k,j,i]) +
                              np.square(Bcc3[k,j,i]))
            numWeight_p += pressure*density
            sum_p       += density
            numWeight_b += bcc_all*density
            sum_b       += density

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