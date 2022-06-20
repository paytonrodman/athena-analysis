#!/usr/bin/env python3
#
# plt_qual_map.py
#
# A program to plot a 2D map of the azimuthally-averaged quality factors generated by calc_qual.py
# Based on plot_spherical.py
#
# Usage: python plt_qual_map.py [options]
#
# Python standard modules
import argparse
import sys
#sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')

# Other Python modules
import numpy as np

# Athena++ modules
import athena_read

def main(**kwargs):
    # Load Python plotting modules
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    # Set resolution of plot in dots per square inch
    if args.dpi is not None:
        resolution = args.dpi

    data_mesh = athena_read.athdf(args.data_file_init, quantities=['x1f','x2f','x3f'])
    # Extract basic coordinate information
    x1f = data_mesh['x1f']
    x2f = data_mesh['x2f']
    x3f = data_mesh['x3f']

    # Set radial extent
    if args.r_max is not None:
        r_max = args.r_max
    else:
        r_max = x1f[-1]

    # Create scalar grid
    if args.midplane:
        r_grid, phi_grid = np.meshgrid(x1f, x3f)
        x_grid = r_grid * np.cos(phi_grid)
        y_grid = r_grid * np.sin(phi_grid)
    else:
        r_grid, theta_grid = np.meshgrid(x1f, x2f)
        x_grid = r_grid * np.sin(theta_grid)
        y_grid = r_grid * np.cos(theta_grid)

    Q_theta = np.load(args.Qtheta_file)
    Q_phi = np.load(args.Qphi_file)
    tB = np.load(args.tB_file)

    if args.midplane:
        vals_Q = Q_phi
        vals_T = np.mean(tB, axis=1)
    else:
        vals_Q = Q_theta
        vals_T = np.mean(tB, axis=0)

    for id in ["Q","T"]:
        if id=="Q":
            vals = vals_Q
            if args.midplane:
                output = args.output_location + 'Qphi.png'
            else:
                output = args.output_location + 'Qtheta.png'
        else:
            vals = vals_T
            output = args.output_location + 'tB.png'
        fig = plt.figure()
        fig.add_subplot(111)
        if args.vmin is not None:
            vmin = args.vmin
        else:
            if args.logc:
                vmin = np.nanmin(vals[np.isfinite(vals)])
            else:
                vmin = 0
        if args.vmax is not None:
            vmax = args.vmax
        else:
            vmax = np.nanmax(vals[np.isfinite(vals)])

        if args.logc:
            norm = colors.LogNorm(vmin, vmax)
        else:
            norm = colors.Normalize(vmin, vmax)
        im = plt.pcolormesh(x_grid, y_grid, vals, cmap="magma", norm=norm, shading='auto')
        cbar = plt.colorbar(im, extend='both')
        plt.gca().set_aspect('equal')
        if not args.midplane:
            plt.xlim((0, r_max))
        else:
            plt.xlim((-r_max, r_max))
        plt.ylim((-r_max, r_max))
        if args.midplane:
            cbar.set_label(r'$Q_{\phi}$')
            plt.xlabel(r'$x$')
            plt.ylabel(r'$y$')
        else:
            cbar.set_label(r'$Q_{\theta}$')
            plt.xlabel(r'$x$')
            plt.ylabel(r'$z$')
        plt.title('Temporally and spatially averaged quality factor')
        plt.savefig(output, bbox_inches='tight', dpi=resolution)


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate various quality factors from raw simulation data.')
    parser.add_argument('Qtheta_file',
                        help='name of data file containing averaged Q_theta values, possibly including path')
    parser.add_argument('Qphi_file',
                        help='name of data file containing averaged Q_phi values, possibly including path')
    parser.add_argument('tB_file',
                        help='name of data file containing averaged tB values, possibly including path')
    parser.add_argument('data_file_init',
                        help='name of data file containing initial mesh data, possibly including path')
    parser.add_argument('output_location',
                        help='folder to save outputs to, possibly including path')
    parser.add_argument('-a', '--average',
                        action='store_true',
                        help='flag indicating phi-averaging should be done')
    parser.add_argument('-r', '--r_max',
                        type=float,
                        default=None,
                        help='maximum radial extent of plot')
    parser.add_argument('-m',
                        '--midplane',
                        action='store_true',
                        help=('flag indicating plot should be midplane (r,phi) rather '
                              'than (r,theta)'))
    parser.add_argument('--logc',
                        action='store_true',
                        help='flag indicating data should be colormapped logarithmically')
    parser.add_argument('--dpi',
                        type=int,
                        default=200,
                        help='resolution of saved image (dots per square inch)')
    parser.add_argument('--vmin',
                        type=float,
                        default=None,
                        help='minimum value for colormap')
    parser.add_argument('--vmax',
                        type=float,
                        default=None,
                        help='maximum value for colormap')
    args = parser.parse_args()

    main(**vars(args))
