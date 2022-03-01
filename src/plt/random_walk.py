#!/usr/bin/env python3
#
# random_walk.py
#
# A program to generate a n-dimensional random walk plot
#
# To run:
# python random_walk.py [options]
#
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main(**kwargs):
    dims = args.dim
    step_n = args.step_num
    step_set = [-10,0,10]
    origin = np.zeros((1,dims))

    step_shape = (step_n,dims)
    steps = np.random.choice(a=step_set, size=step_shape)
    path = np.concatenate([origin, steps]).cumsum(0)
    start = path[:1]
    stop = path[-1:]

    yhat = savitzky_golay(path.flatten(), 501, 3) # window size 51, polynomial order 3
    zero_crossings = np.where(np.diff(np.signbit(np.asarray(yhat))))[0]
    print(zero_crossings)



    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #plt.plot(path)
    #plt.plot(zero_crossings,np.zeros_like(zero_crossings), 'k.')
    #plt.show()

    if args.plotpath:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(path)
        plt.plot(path*-1.)
        plt.plot(yhat, color='black')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        ax.set_xlabel(r'time ($GM/c^3$)',fontsize=14)
        ax.set_ylabel(r'random walk path',fontsize=14)
        plt.show()



def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')



# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A program to generate a n-dimensional random walk plot')
    parser.add_argument('-d', '--dim',
                        type=int,
                        default=1,
                        help='specify dimensions of random walk (default: 1)')
    parser.add_argument('-n', '--step_num',
                        type=int,
                        default=1000,
                        help='specify number of steps to take (default: 1000)')
    parser.add_argument('--plotpath',
                        action="store_true",
                        help='specify whether to plot the path of the random walk')
    args = parser.parse_args()

    main(**vars(args))
