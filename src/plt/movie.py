#!/usr/bin/env python3
#
# movie.py
#
# A program to combine data stills into a movie
#
# To run:
# python movie.py [options]
#
import os
import ffmpeg
import argparse

def main(**kwargs):
    #root_dir = '/Users/paytonrodman/athena-sim/'
    root_dir = '/home/per29/rds/rds-accretion-zyNhkonJSR8/'
    os.chdir(root_dir + args.prob_id + '/img/')

    (
        ffmpeg
        .input('*.png', pattern_type='glob', framerate=1)
        .output('movie_' + args.plot_id + '.mp4')
        .run()
    )

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create movie from stills')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('plot_id',
                        help='base name of the sort of plot being animated, e.g. side_stream, side_plain, or above')
    args = parser.parse_args()

    main(**vars(args))
