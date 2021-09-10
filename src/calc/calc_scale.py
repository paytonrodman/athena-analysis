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
    root_dir = "/Users/paytonrodman/athena_sim/"
    data_dir = root_dir + problem + "/data/"
    os.chdir(data_dir)

    csv_time = []
    # check if data file already exists
    if args.update:
        with open('../scale_with_time.csv', 'r', newline='') as f:
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

    scale_height = []
    for t in sorted(times):
        print("file number: ", t)
        str_t = str(int(t)).zfill(5)

        filename_cons = problem + ".cons." + str_t + ".athdf"
        data_cons = athena_read.athdf(filename_cons)

        #unpack data
        x2v = data_cons['x2v'] # theta
        x1f = data_cons['x1f'] # r
        x2f = data_cons['x2f'] # theta
        x3f = data_cons['x3f'] # phi
        dens = data_cons['dens']
        # Calculations
        dx1f,dx2f,dx3f = AAT.calculate_delta(x1f,x2f,x3f)

        up = x2v*dens*np.sin(x2v)*dx2f*dx3f
        down = dens*np.sin(x2v)*dx2f*dx3f
        polar_ang = np.sum(up,axis=(1,2))/np.sum(down,axis=(1,2))

        h_up = (x2v-polar_ang)**2. * dens*np.sin(x2v)*dx2f*dx3f
        h_down = dens*np.sin(x2v)*dx2f*dx3f
        scale_h = np.sqrt(np.sum(h_up,axis=(1,2))/np.sum(h_down,axis=(1,2)))
        scale_h_av = np.average(scale_h,weights=dx1f)

        scale_height.append(scale_h_av)



    times,scale_height = (list(t) for t in zip(*sorted(zip(times,scale_height))))
    os.chdir("../")
    if args.update:
        with open('scale_with_time.csv', 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(times,scale_height))
    else:
        with open('scale_with_time.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["Time", "scale_height"])
            writer.writerows(zip(times,scale_height))


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the average geometric scale height over the disk')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='specify whether the results being analysed are from a restart')
    args = parser.parse_args()

    main(**vars(args))
