#!/usr/bin/python
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')
import csv
import numpy as np
import scipy.stats
from ast import literal_eval
import argparse

def main(**kwargs):
    # directory containing data
    problem  = args.prob_id
    root_dir = '/Users/paytonrodman/athena-sim/'
    data_dir = root_dir + problem + '/'
    os.chdir(data_dir)
    time = []
    alpha = []
    C = []
    with open('dyn_with_time.csv', newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader, None) # skip header
        for row in csv_reader:
            t = float(row[0])
            #t_orb = float(row[1])
            a = row[2]
            c = row[3]
            time.append(t)
            alpha.append(literal_eval(a))
            C.append(literal_eval(c))

    time, alpha, C = zip(*sorted(zip(time, alpha, C)))

    print(np.shape(alpha))

    for num in range(0,3):
        x = alpha[:,num]
        C = C_all[num]
        if num==0:
            xlab2 = r'Offset ($C$)'
            ylab2 = r'$\alpha_d$'
            f3 = '_r'
        elif num==1:
            xlab2 = r'Offset ($C$)'
            ylab2 = r'$\alpha_d$'
            f3 = '_th'
        elif num==2:
            xlab2 = r'Offset ($C$)'
            ylab2 = r'$\alpha_d$'
            f3 = '_ph'

        fig, ax = plt.subplots()
        x = C
        y = alpha
        #idx = [randint(0, np.size(x)-1) for p in range(0, 1000)]
        #x = x[idx]
        #y = y[idx]
        plt.plot(x,y,'k.', markersize=1)
        ax.set(xlabel=xlab2, ylabel=ylab2)
        title_str = "t=" + str(sim_t)
        ax.set_title(title_str)
        plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
        plt.tight_layout()
        plt.savefig(prob_dir+'/'+f1+f2+f3+'_fit.png',dpi=300)
        plt.close()
        #plt.show()







# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate average magnetic field values within the midplane at a given radius (r)')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='specify whether the results being analysed are from a restart')
    parser.add_argument('-a', '--average',
                        type=str,
                        default='azimuthal',
                        help='specify averaging method (azimuthal,gaussian)')
    parser.add_argument('-H', '--hemisphere',
                        type=str,
                        default='upper',
                        help='specify which hemisphere to average in (upper, lower)')
    parser.add_argument('--r',
                        type=float,
                        default=None,
                        help=('value of r where averages are calculated (default: 25rg)'))
    args = parser.parse_args()

    main(**vars(args))
