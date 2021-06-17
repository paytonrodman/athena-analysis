#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
#sys.path.insert(0, '/home/per29/athena-public-version-master/vis/python')
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')
import athena_read
import math
import glob
import re
import argparse

def main(**kwargs):
    problem  = args.prob_id
    time = args.time

    os.chdir("/Users/paytonrodman/athena_sim/" + problem + "/data/")

    number_str = str(time)
    zero_fill_num = number_str.zfill(5)

    filename_cons = problem + ".cons." + zero_fill_num + ".athdf"
    filename_prim = problem + ".prim." + zero_fill_num + ".athdf"
    data_p = athena_read.athdf(filename_prim)
    data_c = athena_read.athdf(filename_cons)
    input = athena_read.athinput("../athinput." + problem)

    #constants
    gamma = input['hydro']['gamma']
    GM = input['problem']['mass']
    beta_0 = input['problem']['beta_0']

    #unpack data
    x1f = data_c['x1f'] # r
    x2f = data_c['x2f'] # theta
    x3f = data_c['x3f'] # phi

    if kwargs['r_max'] is not None:
        ru = find_nearest(x1f, kwargs['r_max'])
    else:
        ru = len(x1f) - 1
    if kwargs['r_min'] is not None:
        rl = find_nearest(x1f, kwargs['r_min'])
    else:
        rl = 0
    if kwargs['th_max'] is not None:
        tu = find_nearest(x2f, kwargs['th_max'])
    else:
        tu = len(x2f) - 1
    if kwargs['th_min'] is not None:
        tl = find_nearest(x2f, kwargs['th_min'])
    else:
        tl = 0
    if kwargs['ph_max'] is not None:
        pu = find_nearest(x3f, kwargs['ph_max'])
    else:
        pu = len(x3f) - 1
    if kwargs['ph_min'] is not None:
        pl = find_nearest(x3f, kwargs['ph_min'])
    else:
        pl = 0

    x1f = x1f[rl:ru+1]
    x2f = x2f[tl:tu+1]
    x3f = x3f[pl:pu+1]
    x1v = data_c['x1v'][rl:ru] # r
    x2v = data_c['x2v'][tl:tu] # theta
    x3v = data_c['x3v'][pl:pu] # phi
    dens = data_c['dens'][pl:pu,tl:tu,rl:ru]
    Etot = data_c['Etot'][pl:pu,tl:tu,rl:ru]
    mom1 = data_c['mom1'][pl:pu,tl:tu,rl:ru]
    mom2 = data_c['mom2'][pl:pu,tl:tu,rl:ru]
    mom3 = data_c['mom3'][pl:pu,tl:tu,rl:ru]
    Bcc1 = data_c['Bcc1'][pl:pu,tl:tu,rl:ru]
    Bcc2 = data_c['Bcc2'][pl:pu,tl:tu,rl:ru]
    Bcc3 = data_c['Bcc3'][pl:pu,tl:tu,rl:ru]
    press = data_p['press'][pl:pu,tl:tu,rl:ru]

    delta_theta = np.diff(x3f)
    delta_phi = np.diff(x2f)
    vol = calc_volume(x1f,x2f,x3f)
    v1,v2,v3 = calc_velocity(mom1,mom2,mom3,vol,dens)

    current_beta,beta_fact = calculate_beta(x1v,x2v,x3v,
                                            rl,ru,tl,tu,pl,pu,
                                            dens,press,Bcc1,Bcc2,Bcc3,beta_0)
    print(f"current beta: {current_beta}")
    print(f"required multiplicative factor for beta_0={beta_0}: {beta_fact}")

def calculate_beta(x1v,x2v,x3v,rl,ru,tl,tu,pl,pu,dens,press,Bcc1,Bcc2,Bcc3,beta_0):
    sum_p = 0.
    numWeight_p = 0.
    sum_b = 0.
    numWeight_b = 0.
    for k,phi in enumerate(x3v):
        for j,theta in enumerate(x2v):
            for i,r in enumerate(x1v):
                density = dens[k,j,i]
                pressure = press[k,j,i]
                bcc_all = np.sqrt(np.square(Bcc1[k,j,i]) +
                                  np.square(Bcc2[k,j,i]) +
                                  np.square(Bcc3[k,j,i]))

                numWeight_p += pressure*density
                sum_p       += density
                numWeight_b += bcc_all*density
                sum_b       += density

    pres_av = numWeight_p/sum_p
    bcc_av = numWeight_b/sum_b
    # Normalise B to give preferred beta
    current_beta = 2. * pres_av / (bcc_av**2.)
    beta_fact = np.sqrt(current_beta / beta_0)
    return current_beta,beta_fact

def find_nearest(array, value):
    array = np.asarray(array);
    idx = (np.abs(array - value)).argmin();
    return idx;

def calc_volume(x1f,x2f,x3f):
    vol = np.empty((len(x1f)-1,len(x2f)-1,len(x3f)-1))
    dx1f = np.diff(x1f)
    dx2f = np.diff(x2f)
    dx3f = np.diff(x3f)
    for idx1,x1_len in enumerate(dx1f):
        for idx2,x2_len in enumerate(dx2f):
            for idx3,x3_len in enumerate(dx3f):
                vol[idx1,idx2,idx3] = x1_len*x2_len*x3_len
    return vol

def calc_velocity(mom1,mom2,mom3,vol,dens):
    v1 = mom1*vol.T/dens
    v2 = mom2*vol.T/dens
    v3 = mom3*vol.T/dens
    return v1,v2,v3



# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Determine multiplicative factor needed to produce a given plasma beta value, based on raw simulation data.')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('--time',
                        type=int,
                        default=0,
                        help=('time slice to analyse; use --time=<val> if <val> has negative sign (default: 0)'))
    parser.add_argument('--r_min',
                        type=float,
                        default=None,
                        help=('minimum r value of region to be analysed (default: minimum of domain)'))
    parser.add_argument('--r_max',
                        type=float,
                        default=None,
                        help=('maximum r value of region to be analysed (default: maximum of domain)'))
    parser.add_argument('--th_min',
                        type=float,
                        default=None,
                        help=('minimum theta value of region to be analysed (default: minimum of domain)'))
    parser.add_argument('--th_max',
                        type=float,
                        default=None,
                        help=('maximum theta value of region to be analysed (default: maximum of domain)'))
    parser.add_argument('--ph_min',
                        type=float,
                        default=None,
                        help=('minimum phi value of region to be analysed (default: minimum of domain)'))
    parser.add_argument('--ph_max',
                        type=float,
                        default=None,
                        help=('maximum phi value of region to be analysed (default: maximum of domain)'))
    args = parser.parse_args()

    main(**vars(args))
