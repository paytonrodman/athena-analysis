#!/usr/bin/env python3
#
# movie.py
#
# A program to combine data stills into a movie
#
# To run:
# python cv_movie.py [options]
#
import os
import glob
import cv2
import argparse

def main(**kwargs):
    root_dir = '/Users/paytonrodman/athena-sim/'
    #root_dir = '/home/per29/rds/rds-accretion-zyNhkonJSR8/'
    os.chdir(root_dir + args.prob_id + '/' + args.image_folder + '/')

    img_array = []
    for filename in sorted(glob.glob('*.png')):
        img = cv2.imread(filename)
        height,width,_ = img.shape
        size = (width,height)
        img_array.append(img)

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    out = cv2.VideoWriter(args.output, fourcc, args.fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    cv2.destroyAllWindows()

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create movie from stills')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('image_folder',
                        help='name of the folder under prob_id, where stills are located, e.g. img')
    parser.add_argument('codec',
                        help='codec used to create movie, e.g. MJPG')
    parser.add_argument('output',
                        help='output filename and extension, e.g. high_res.avi')
    parser.add_argument('fps',
                        type=int,
			            default=5,
                        help='frames per second (default=5)')
    args = parser.parse_args()

    main(**vars(args))
