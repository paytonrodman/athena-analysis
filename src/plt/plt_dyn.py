#!/usr/bin/python
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')
import csv
import numpy as np
import argparse

def main(**kwargs):
    # directory containing data
    problem  = args.prob_id
    root_dir = '/Users/paytonrodman/athena-sim/'
    prob_dir = root_dir + problem
    data_dir = prob_dir + '/dyn/average/'
    os.chdir(data_dir)

    av = args.average
    hem = args.hemisphere

    time = []
    time_orb = []
    alpha = []
    C = []
    with open(av[:3]+'_'+hem+'_dyn_with_time.csv', newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader, None) # skip header
        for row in csv_reader:
            t = float(row[0])
            t_orb = float(row[1])
            a = row[2]
            c = row[3]
            time.append(t)
            time_orb.append(t_orb)
            alpha.append(np.fromstring(a.strip("[]"), sep=', '))
            C.append(np.fromstring(c.strip("[]"), sep=', '))


    time, time_orb, alpha, C = zip(*sorted(zip(time, time_orb, alpha, C)))

    time = np.asarray(time)
    time_orb = np.asarray(time_orb)
    alpha = np.asarray(alpha)
    C = np.asarray(C)

    if args.path:
        time_mask = (time_orb > 350.) & (time_orb < 400.)
    else:
        time_mask = (time_orb > 50.)
    alpha = alpha[time_mask]
    C = C[time_mask]

    for num in range(0,3):
        x = alpha[:,num]
        y = C[:,num]
        if num==0:
            xlab = r'Offset ($C$)'
            ylab = r'$\alpha_d$'
            f = 'r'
        elif num==1:
            xlab = r'Offset ($C$)'
            ylab = r'$\alpha_d$'
            f = 'th'
        elif num==2:
            xlab = r'Offset ($C$)'
            ylab = r'$\alpha_d$'
            f = 'ph'

        #N = np.size(x)
        #base = plt.cm.get_cmap('viridis')
        #color_list = base(np.linspace(0, 1, N))
        #cmap_name = base.name + str(N)
        #cm = plt.cm.get_cmap('jet', N)
        #c = np.random.randint(N, size=np.size(x))

        #idx = [randint(0, np.size(x)-1) for p in range(0, 1000)]
        #x = x[idx]
        #y = y[idx]
        #plt.scatter(x,y,c=c, cmap=cm, s=50)
        plt.plot(x,y,'k.',markersize=5)
        if args.path:
            plt.plot(x,y,'r')
        plt.ylabel(ylab)
        plt.xlabel(xlab)
        plt.grid(True)
        #title_str = "t=" + str(sim_t)
        #plt.title(title_str)
        plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
        plt.tight_layout()
        plt.savefig(data_dir+'/'+av[:3]+'_'+hem+'_'+f+'_fit.png',dpi=300)
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
    parser.add_argument('-p', '--path',
                        action="store_true",
                        help='specify whether to plot the path over time')
    args = parser.parse_args()

    main(**vars(args))
