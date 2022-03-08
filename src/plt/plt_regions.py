#! /usr/bin/env python

"""
Script for plotting vertical (r,theta) slices of data in
spherical coordinates.

Run "plot_spherical.py -h" to see description of inputs.

See documentation on athena_read.athdf() for important notes about reading files
with mesh refinement.

Users are encouraged to make their own versions of this script for improved
results by adjusting figure size, spacings, tick locations, axes labels, etc.
The script must also be modified to plot any functions of the quantities in the
file, including combinations of multiple quantities.

Requires scipy if making a stream plot.
"""

# Python standard modules
import argparse
import warnings
import sys

# Other Python modules
import numpy as np

sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')
# Athena++ modules
import athena_read


# Main function
def main(**kwargs):

    # Load function for transforming coordinates
    if kwargs['stream'] is not None:
        from scipy.ndimage import map_coordinates

    # Load Python plotting modules
    if kwargs['output_file'] != 'show':
        import matplotlib
        matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    # Determine if vector quantities should be read
    quantities = [kwargs['quantity']]
    if kwargs['stream'] is not None:
        quantities.append(kwargs['stream'] + '1')
        quantities.append(kwargs['stream'] + '2')

    # Define grid compression in theta-direction
    h = 1.0

    # Read mesh and input data
    data_input = athena_read.athinput(kwargs['input_file'])
    # Read data
    data = athena_read.athdf(kwargs['data_file'], quantities=quantities)
    x1max = data_input['mesh']['x1max']

    # Extract boundaries of each SMR level
    if 'refinement3' in data_input:
        lower_boundaries = np.array([data_input['refinement3']['x2min'],
                                    data_input['refinement2']['x2min'],
                                    data_input['refinement1']['x2min']])
        upper_boundaries = np.array([data_input['refinement3']['x2max'],
                                    data_input['refinement2']['x2max'],
                                    data_input['refinement1']['x2max']])
    elif 'refinement2' in data_input:
        lower_boundaries = np.array([data_input['refinement2']['x2min'],
                                    data_input['refinement1']['x2min']])
        upper_boundaries = np.array([data_input['refinement2']['x2max'],
                                    data_input['refinement1']['x2max']])
    elif 'refinement1' in data_input:
        lower_boundaries = np.array([data_input['refinement1']['x2min']])
        upper_boundaries = np.array([data_input['refinement1']['x2max']])
    else:
        lower_boundaries = np.array([data_input['mesh']['x2min']])
        upper_boundaries = np.array([data_input['mesh']['x2max']])
    # Renormalise theta values to measure from positive x-axis
    lower_boundaries = -lower_boundaries + np.pi/2
    upper_boundaries = -upper_boundaries + np.pi/2

    # Set resolution of plot in dots per square inch
    if kwargs['dpi'] is not None:
        resolution = kwargs['dpi']

    # Extract basic coordinate information
    coordinates = data['Coordinates']
    r = data['x1v']
    theta = data['x2v']
    phi = data['x3v']
    r_face = data['x1f']
    theta_face = data['x2f']
    nx1 = len(r)
    nx2 = len(theta)
    nx3 = len(phi)

    # Set radial extent
    if kwargs['r_max'] is not None:
        r_max = kwargs['r_max']
    else:
        r_max = r_face[-1]

    # Create scalar grid
    theta_face_extended = np.concatenate((theta_face, 2.0*np.pi - theta_face[-2::-1]))
    r_grid, theta_grid = np.meshgrid(r_face, theta_face_extended)
    x_grid = r_grid * np.sin(theta_grid)
    y_grid = r_grid * np.cos(theta_grid)

    # Create streamline grid
    if kwargs['stream'] is not None:
        x_stream = np.linspace(-r_max, r_max, kwargs['stream_samples'])
        z_stream = np.linspace(-r_max, r_max, kwargs['stream_samples'])
        x_grid_stream, z_grid_stream = np.meshgrid(x_stream, z_stream)
        r_grid_stream_coord = (x_grid_stream.T**2 + z_grid_stream.T**2) ** 0.5
        theta_grid_stream_coord = np.pi - \
            np.arctan2(x_grid_stream.T, -z_grid_stream.T)
        theta_grid_stream_pix = ((theta_grid_stream_coord + theta[0])
                                 / (2.0*np.pi + 2.0 * theta[0])) * (2 * nx2 + 1)
        r_grid_stream_pix = np.empty_like(r_grid_stream_coord)
        for (i, j), r_val in np.ndenumerate(r_grid_stream_coord):
            index = sum(r < r_val) - 1
            if index < 0:
                r_grid_stream_pix[i, j] = -1
            elif index < nx1 - 1:
                r_grid_stream_pix[i, j] = index + \
                    (r_val - r[index]) / (r[index + 1] - r[index])
            else:
                r_grid_stream_pix[i, j] = nx1

    # Perform slicing/averaging of scalar data
    if kwargs['average']:
        vals_right = np.mean(data[kwargs['quantity']], axis=0)
        vals_left = vals_right
    else:
        vals_right = 0.5 * (data[kwargs['quantity']][-1, :, :] + data[kwargs['quantity']][0, :, :])
        vals_left = 0.5 * (data[kwargs['quantity']][int((nx3/2)-1), :, :]
                           + data[kwargs['quantity']][int(nx3/2), :, :])

    # Join scalar data through boundaries
    vals = np.vstack((vals_right, vals_left[::-1, :]))

    # Perform slicing/averaging of vector data
    if kwargs['stream'] is not None:
        if kwargs['stream_average']:
            vals_r_right = np.mean(data[kwargs['stream'] + '1'], axis=0).T
            vals_r_left = vals_r_right
            vals_theta_right = np.mean(data[kwargs['stream'] + '2'], axis=0).T
            vals_theta_left = -vals_theta_right
        else:
            vals_r_right = data[kwargs['stream'] + '1'][0, :, :].T
            vals_r_left = data[kwargs['stream'] + '1'][int(nx3/2), :, :].T
            vals_theta_right = data[kwargs['stream'] + '2'][0, :, :].T
            vals_theta_left = -data[kwargs['stream'] + '2'][int(nx3/2), :, :].T

        # Join vector data through boundaries
        vals_r = np.hstack((vals_r_left[:, :1], vals_r_right, vals_r_left[:, ::-1],
                            vals_r_right[:, :1]))
        vals_r = map_coordinates(vals_r, (r_grid_stream_pix, theta_grid_stream_pix),
                                 order=1, cval=np.nan)
        vals_theta = np.hstack((vals_theta_left[:, :1], vals_theta_right,
                                vals_theta_left[:, ::-1], vals_theta_right[:, :1]))
        vals_theta = map_coordinates(vals_theta,
                                     (r_grid_stream_pix, theta_grid_stream_pix),
                                     order=1, cval=np.nan)

        # Transform vector data to Cartesian components
        r_vals = r_grid_stream_coord
        sin_theta = np.sin(theta_grid_stream_coord)
        cos_theta = np.cos(theta_grid_stream_coord)
        dx_dr = sin_theta
        dz_dr = cos_theta
        dx_dtheta = r_vals * cos_theta
        dz_dtheta = -r_vals * sin_theta
        if not (coordinates == 'schwarzschild' or coordinates == 'kerr-schild'):
            dx_dtheta /= r_vals
            dz_dtheta /= r_vals
        vals_x = dx_dr * vals_r + dx_dtheta * vals_theta
        vals_z = dz_dr * vals_r + dz_dtheta * vals_theta

    # Determine colormapping properties
    cmap = plt.get_cmap(kwargs['colormap'])
    vmin = kwargs['vmin']
    vmax = kwargs['vmax']
    if kwargs['logc']:
        norm = colors.LogNorm(vmin, vmax)
    else:
        norm = colors.Normalize(vmin, vmax)

    # Make plot
    plt.figure()
    im = plt.pcolormesh(x_grid, y_grid, vals, cmap=cmap, norm=norm)
    for index in range(len(upper_boundaries)):
        y_limit = x1max*np.tan(upper_boundaries[index])
        plt.plot([-x1max,x1max], [-y_limit,y_limit],'white',linewidth=1)
        y_limit = x1max*np.tan(lower_boundaries[index])
        plt.plot([-x1max,x1max], [-y_limit,y_limit],'white',linewidth=1)

    if kwargs['stream'] is not None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                'invalid value encountered in greater_equal',
                RuntimeWarning,
                'numpy')
            plt.streamplot(x_stream, z_stream, vals_x.T, vals_z.T,
                           density=kwargs['stream_density'], color='k', linewidth=0.5,
                           arrowsize=0.5)

    plt.gca().set_aspect('equal')
    plt.xlim((-r_max, r_max))
    plt.ylim((-r_max, r_max))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$z$')
    plt.axis('off')
    if kwargs['time']:
        #plt.title('t='+str(int(data['Time'])))
        plt.title('t={:.2e} GM/c3'.format(data['Time']))
    #plt.colorbar(im)
    if kwargs['output_file'] == 'show':
        plt.show()
    else:
        plt.savefig(kwargs['output_file'], bbox_inches='tight', dpi=resolution)


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file',
                        help='name of input file, possibly including path')
    parser.add_argument('input_file',
                        help='name of athinput file, possibly including path')
    parser.add_argument('quantity',
                        help='name of quantity to be plotted')
    parser.add_argument('output_file',
                        help=('name of output to be (over)written, possibly including '
                              'path; use "show" to show interactive plot instead'))
    parser.add_argument('-a', '--average',
                        action='store_true',
                        help='flag indicating phi-averaging should be done')
    parser.add_argument('-r', '--r_max',
                        type=float,
                        default=None,
                        help='maximum radial extent of plot')
    parser.add_argument('-c',
                        '--colormap',
                        default=None,
                        help=('name of Matplotlib colormap to use instead of default'))
    parser.add_argument('--vmin',
                        type=float,
                        default=None,
                        help=('data value to correspond to colormap minimum; use '
                              '--vmin=<val> if <val> has negative sign'))
    parser.add_argument('--vmax',
                        type=float,
                        default=None,
                        help=('data value to correspond to colormap maximum; use '
                              '--vmax=<val> if <val> has negative sign'))
    parser.add_argument('--logc',
                        action='store_true',
                        help='flag indicating data should be colormapped logarithmically')
    parser.add_argument('-s', '--stream',
                        default=None,
                        help='name of vector quantity to use to make stream plot')
    parser.add_argument('--stream_average',
                        action='store_true',
                        help='flag indicating phi-averaging on stream plot data')
    parser.add_argument('--stream_density',
                        type=float,
                        default=1.0,
                        help='density of stream lines')
    parser.add_argument('--stream_samples',
                        type=int,
                        default=100,
                        help='linear size of stream line sampling grid')
    parser.add_argument('--time',
                        action='store_true',
                        help=('flag indicating title should be set to time'))
    parser.add_argument('--dpi',
                        type=int,
                        default=None,
                        help='resolution of saved image (dots per square inch)')
    args = parser.parse_args()
    main(**vars(args))
