#!/usr/bin/python
# Python standard modules
import argparse
import sys
import os

dir_path = os.path.dirname(__file__)
lib_path = os.path.join(dir_path, '..', '..', 'dependencies')
sys.path.append(lib_path)

# Other Python modules
import matplotlib.pyplot as plt
import numpy as np
import re

# Athena++ modules
import athena_read
import AAT

def main(**kwargs):
    n = len(args.prob_id) # number of data files to read
    if not args.prob_id:
        sys.exit('Must specify problem IDs')
    if not args.output:
        sys.exit('Must specify output file')

    dens = [[] for _ in range(n)]
    mom1 = [[] for _ in range(n)]
    temp = [[] for _ in range(n)]
    alpha = [[] for _ in range(n)]
    if args.short:
        dens_short = [[] for _ in range(n)]
        mom1_short = [[] for _ in range(n)]
        temp_short = [[] for _ in range(n)]
        alpha_short = [[] for _ in range(n)]


    x1v = [[] for _ in range(n)]
    labels = []
    colors = []
    for count,f in enumerate(args.prob_id):
        # get pre-defined labels and line colours for each simulation
        slash_list = [m.start() for m in re.finditer('/', f)]
        prob_id = f[slash_list[-2]+1:slash_list[-1]]
        l,c,_ = AAT.problem_dictionary(prob_id, args.pres)
        labels.append(l)
        colors.append(c)

        mesh_file = f + 'data/'+ prob_id + '.cons.00000.athdf'
        data_mesh = athena_read.athdf(mesh_file, quantities=['x1v'])
        x1v[count] = data_mesh['x1v']

        dens_file = f + 'dens_profile.npy'
        mom1_file = f + 'mom1_profile.npy'
        temp_file = f + 'temp_profile.npy'
        alpha_file = f + 'alpha_profile.npy'

        dens_i = np.load(dens_file, mmap_mode='r')
        mom1_i = np.load(mom1_file, mmap_mode='r')
        temp_i = np.load(temp_file, mmap_mode='r')
        alpha_i = np.load(alpha_file, mmap_mode='r')

        dens[count] = dens_i[0]
        mom1[count] = mom1_i[0]
        temp[count] = temp_i[0]
        alpha[count] = alpha_i[0]

        if args.short:
            if prob_id in ['high_res','high_beta']:
                dens_file_short = f + 'dens_profile_short.npy'
                mom1_file_short = f + 'mom1_profile_short.npy'
                temp_file_short = f + 'temp_profile_short.npy'
                alpha_file_short = f + 'alpha_profile_short.npy'

                dens_short_i = np.load(dens_file_short)
                mom1_short_i = np.load(mom1_file_short)
                temp_short_i = np.load(temp_file_short)
                alpha_short_i = np.load(alpha_file_short)

                dens_short[count] = dens_short_i[0]
                mom1_short[count] = mom1_short_i[0]
                temp_short[count] = temp_short_i[0]
                alpha_short[count] = alpha_short_i[0]
            else:
                dens_short[count] = np.nan*np.ones_like(dens[count])
                mom1_short[count] = np.nan*np.ones_like(mom1[count])
                temp_short[count] = np.nan*np.ones_like(temp[count])
                alpha_short[count] = np.nan*np.ones_like(alpha[count])


    ylabels = [r'$\langle\langle \rho \rangle\rangle^*$',
               r'$\langle\langle v_r \rangle\rangle^*$',
               r'$\langle\langle \frac{P}{\rho c^2} \rangle\rangle^*$',
               r'$\langle\langle \alpha_{\rm SS} \rangle\rangle^*$']

    lw = 1.5
    n_plots = len(ylabels)

    fig, axs = plt.subplots(nrows=n_plots, ncols=1, sharex=True, figsize=(4,2*n_plots))
    fig.subplots_adjust(hspace=0.0)
    for ii in range(n):
        if args.logr:
            axs[0].semilogx(x1v[ii], dens[ii], linewidth=lw, color=colors[ii], label=labels[ii], rasterized=True)
            axs[1].semilogx(x1v[ii], mom1[ii], linewidth=lw, color=colors[ii], label=labels[ii], rasterized=True)
            axs[2].semilogx(x1v[ii], temp[ii], linewidth=lw, color=colors[ii], label=labels[ii], rasterized=True)
            axs[3].loglog(x1v[ii], alpha[ii], linewidth=lw, color=colors[ii], label=labels[ii], rasterized=True)
            if args.short:
                axs[0].semilogx(x1v[ii], dens_short[ii], linewidth=lw, color=colors[ii], linestyle='--', rasterized=True)
                axs[1].semilogx(x1v[ii], mom1_short[ii], linewidth=lw, color=colors[ii], linestyle='--', rasterized=True)
                axs[2].semilogx(x1v[ii], temp_short[ii], linewidth=lw, color=colors[ii], linestyle='--', rasterized=True)
                axs[3].loglog(x1v[ii], alpha_short[ii], linewidth=lw, color=colors[ii], linestyle='--', rasterized=True)
        else:
            axs[0].plot(x1v[ii], dens[ii], linewidth=lw, color=colors[ii], label=labels[ii], rasterized=True)
            axs[1].plot(x1v[ii], mom1[ii], linewidth=lw, color=colors[ii], label=labels[ii], rasterized=True)
            axs[2].plot(x1v[ii], temp[ii], linewidth=lw, color=colors[ii], label=labels[ii], rasterized=True)
            axs[3].semilogy(x1v[ii], alpha[ii], linewidth=lw, color=colors[ii], label=labels[ii], rasterized=True)
            if args.short:
                axs[0].plot(x1v[ii], dens_short[ii], linewidth=lw, color=colors[ii], linestyle='--', rasterized=True)
                axs[1].plot(x1v[ii], mom1_short[ii], linewidth=lw, color=colors[ii], linestyle='--', rasterized=True)
                axs[2].plot(x1v[ii], temp_short[ii], linewidth=lw, color=colors[ii], linestyle='--', rasterized=True)
                axs[3].semilogy(x1v[ii], alpha_short[ii], linewidth=lw, color=colors[ii], linestyle='--', rasterized=True)
    for ii in range(n_plots):
        axs[ii].set_ylabel(ylabels[ii])
        if not args.logr:
            axs[ii].set_xlim(left=0)
            axs[ii].ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)

    axs[-1].set_xlabel(r'radius ($r_g$)')

    leg = axs[1].legend(loc='best')
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.savefig(args.output, bbox_inches='tight')
    plt.close()


def directory_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot magnetic field curvature, kappa, over time for different disk regions.')
    parser.add_argument('-p', '--prob_id',
                        type=directory_path,
                        nargs='+',
                        default=None,
                        help='list of directory paths for each problem ID')
    parser.add_argument('-o', '--output',
                        type=str,
                        default=None,
                        help='name of plot to be created, including path')
    parser.add_argument('--logr',
                        action='store_true',
                        help='plot r in log')
    parser.add_argument('--short',
                        action='store_true',
                        help='add lines for same time as strong-field case')
    parser.add_argument('--pres',
                        action='store_true',
                        help='make presentation-quality image')
    args = parser.parse_args()

    main(**vars(args))
