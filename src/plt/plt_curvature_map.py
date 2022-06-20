#!/usr/bin/env python3
#
# plt_curvature_map.py
#
# A program to plot a snapshot of azimuthally-averaged kappa values in the r-theta plane.
#
# Usage: python plt_curvature_map.py [options]
#
# Python standard modules
import argparse
import warnings
import sys
sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')

# Other Python modules
import numpy as np

# Athena++ modules
import athena_read

# Main function
def main(**kwargs):
    if kwargs['stream'] is None:
        sys.exit('Must specify stream')

    # Load function for transforming coordinates
    #from scipy.ndimage import map_coordinates

    # Load Python plotting modules
    if kwargs['output_file'] != 'show':
        import matplotlib
        matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    # Determine if vector quantities should be read
    quantities = [kwargs['quantity']]
    quantities.append(kwargs['stream'] + '1')
    quantities.append(kwargs['stream'] + '2')

    # Read mesh and input data
    data_input = athena_read.athinput(kwargs['input_file'])
    # Read data
    data = athena_read.athdf(kwargs['data_file'], quantities=quantities)

    # Extract boundaries of each SMR level
    if 'refinement3' not in data_input:
        sys.exit('Simulation must have 3 levels of refinement in mesh. Exiting.')
    lower_boundaries = np.array([data_input['refinement3']['x2min'],
                                data_input['refinement2']['x2min'],
                                data_input['refinement1']['x2min']])
    upper_boundaries = np.array([data_input['refinement3']['x2max'],
                                data_input['refinement2']['x2max'],
                                data_input['refinement1']['x2max']])
    # Renormalise theta values to measure from positive x-axis
    lower_boundaries = -lower_boundaries + np.pi/2
    upper_boundaries = -upper_boundaries + np.pi/2

    # Set resolution of plot in dots per square inch
    if kwargs['dpi'] is not None:
        resolution = kwargs['dpi']

    # Extract basic coordinate information
    phi = data['x3v']
    r_face = data['x1f']
    theta_face = data['x2f']
    nx3 = len(phi)

    # Create scalar grid
    theta_face_extended = np.concatenate((theta_face, 2.0*np.pi - theta_face[-2::-1]))
    r_grid, theta_grid = np.meshgrid(r_face, theta_face_extended)
    x_grid = r_grid * np.sin(theta_grid)
    y_grid = r_grid * np.cos(theta_grid)

    # Perform slicing/averaging of vector data
    vals_r_right = data[kwargs['stream'] + '1'][0, :, :].T
    vals_r_left = data[kwargs['stream'] + '1'][int(nx3/2), :, :].T
    vals_theta_right = data[kwargs['stream'] + '2'][0, :, :].T
    vals_theta_left = -data[kwargs['stream'] + '2'][int(nx3/2), :, :].T
    # Join vector data through boundaries
    vals_r = np.hstack((vals_r_left[:, :1], vals_r_right[:, :-1],
                        vals_r_left[:, -2::-1], vals_r_right[:, :1]))
    vals_theta = np.hstack((vals_theta_left[:, :1], vals_theta_right[:, :-1],
                            vals_theta_left[:, -2::-1], vals_theta_right[:, :1]))

    # Transform vector data to Cartesian components
    r_vals = r_grid[:-1, :-1]
    sin_theta = np.sin(theta_grid[:-1, :-1])
    cos_theta = np.cos(theta_grid[:-1, :-1])
    dx_dr = sin_theta
    dz_dr = cos_theta
    dx_dtheta = (r_vals * cos_theta) / r_vals
    dz_dtheta = (-r_vals * sin_theta) / r_vals

    vals_x = dx_dr.T * vals_r + dx_dtheta.T * vals_theta
    vals_z = dz_dr.T * vals_r + dz_dtheta.T * vals_theta

    # curvature kappa = ||b dot nabla b|| where b = B / ||B|| and ||x|| is the modulus of x
    # and nabla is the differential operator, i d/dx + j d/dy + k d/dz
    bx = vals_x.T / np.sqrt(vals_x.T**2. + vals_z.T**2.)
    bz = vals_z.T / np.sqrt(vals_x.T**2. + vals_z.T**2.)

    dbx = np.array(np.gradient(bx))
    dx = np.array(np.gradient(x_grid))
    dbz = np.array(np.gradient(bz))
    dz = np.array(np.gradient(y_grid))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            category=RuntimeWarning)
        divbx = np.true_divide(dbx[1,:,:], dx[1,:-1,:-1])
        divbz = np.true_divide(dbz[0,:,:], dz[0,:-1,:-1])

    bdivb = divbx*bx + divbz*bz
    bdivb = np.abs(bdivb)

    fig = plt.figure()
    ax = fig.add_subplot(111, polar='True')
    vmin = np.nanmin(bdivb[np.isfinite(bdivb)])
    vmax = np.nanmax(bdivb[np.isfinite(bdivb)])
    if vmin == 0:
        finite = bdivb[np.isfinite(bdivb)]
        vmin = np.nanmin(finite[finite>0])
    #norm=colors.SymLogNorm(linthresh=0.0001, linscale=0.0001, vmin=np.nanmin(bdivb[np.isfinite(bdivb)]), vmax=np.nanmax(bdivb[np.isfinite(bdivb)]), base=10)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(theta_grid[:-1, :-1], r_grid[:-1, :-1], bdivb, shading='auto', norm=norm)
    ax.set_theta_zero_location("N")
    plt.axis('off')
    cbar = plt.colorbar(im, extend='both')
    cbar.set_label(r'$|\kappa|$')

    if kwargs['time']:
        plt.title('t={:.2e} GM/c3'.format(data['Time']))
    if kwargs['output_file'] == 'show':
        plt.show()
    else:
        plt.savefig(kwargs['output_file'], bbox_inches='tight', dpi=resolution)


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file',
                        help='name of data file, possibly including path')
    parser.add_argument('input_file',
                        help='name of athinput file, possibly including path')
    parser.add_argument('quantity',
                        help='name of quantity to be plotted')
    parser.add_argument('output_file',
                        help=('name of output to be (over)written, possibly including '
                              'path; use "show" to show interactive plot instead'))
    parser.add_argument('-s', '--stream',
                        default=None,
                        help='name of vector quantity to use to make stream plot')
    parser.add_argument('--time',
                        action='store_true',
                        help=('flag indicating title should be set to time'))
    parser.add_argument('--dpi',
                        type=int,
                        default=None,
                        help='resolution of saved image (dots per square inch)')
    args = parser.parse_args()
    main(**vars(args))
