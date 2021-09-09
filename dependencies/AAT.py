import numpy as np

def calculate_velocity(mom1,mom2,mom3,dens):
    """Calculates the 3-vector velocity from 3-vector momentum density and scalar density."""
    v1 = mom1/dens
    v2 = mom2/dens
    v3 = mom3/dens
    return v1,v2,v3

def calculate_delta(x1,x2,x3):
    """For 3 vector arrays x1,x2,x3, return the difference between each adjacent entry."""
    dx1 = np.diff(x1)
    dx2 = np.diff(x2)
    dx3 = np.diff(x3)
    return dx1,dx2,dx3

def find_nearest(array, value):
    """For a given array, find the index of the entry closest to 'value'."""
    array = np.asarray(array);
    idx = (np.abs(array - value)).argmin();
    return idx;
