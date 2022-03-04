def calculate_velocity(mom1,mom2,mom3,dens):
    """Calculates the 3-vector velocity from 3-vector momentum density and scalar density."""
    v1 = mom1/dens
    v2 = mom2/dens
    v3 = mom3/dens
    return v1,v2,v3

def calculate_delta(x1,x2,x3):
    """For 3 vector arrays x1,x2,x3, return the difference between each adjacent entry."""
    import numpy as np
    dx1 = np.diff(x1)
    dx2 = np.diff(x2)
    dx3 = np.diff(x3)
    return dx1,dx2,dx3

def find_nearest(array, value):
    """For a given array, find the index of the entry closest to 'value'."""
    import numpy as np
    array = np.asarray(array);
    idx = (np.abs(array - value)).argmin();
    return idx;

def add_time_to_list(update_flag, directory, output_filename, problem_id):
    """Compile a unique list of files that have not been analysed before."""
    import glob
    import csv
    import re
    import numpy as np

    # check if data file already exists
    csv_times = np.empty(0)
    if update_flag:
        with open(prob_dir + filename_output, 'r', newline='') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            next(csv_reader, None) # skip header
            for row in csv_reader:
                csv_times = np.append(csv_times, float(row[0]))

    # compile a list of unique times associated with data files
    files = glob.glob('./' + problem_id + '.cons.*.athdf')
    file_times = np.empty(0)
    for f in files:
        current_time = re.findall(r'\b\d+\b', f)
        if update_flag:
            if float(current_time[0]) not in file_times and float(current_time[0]) not in csv_times:
                file_times = np.append(file_times, float(current_time[0]))
        else:
            if float(current_time[0]) not in file_times:
                file_times = np.append(file_times, float(current_time[0]))
    if len(file_times)==0:
        sys.exit('No new timesteps to analyse in the given directory. Exiting.')

    return file_times

def distribute_files_to_cores(file_list, n_process, rank):
    """Distribute files to cores in a balanced way."""

    files_per_process = len(file_list) // n_process
    remainder = len(file_list) % n_process
    if rank < remainder:  # processes with rank < remainder analyze one extra catchment
        start = rank * (files_per_process + 1)
        stop = start + files_per_process + 1
    else:
        start = rank * files_per_process + remainder
        stop = start + files_per_process

    local_times = file_list[start:stop] # get the times to be analyzed by each rank
    return local_times

def calculate_orbit_time(simulation_time):
    """Calculate the orbital time at the ISCO."""

    import numpy as np

    r_ISCO = 6. # location of ISCO in PW potential
    T_period = 2.*np.pi*np.sqrt(r_ISCO)*(r_ISCO - 2.)
    orbit_time = simulation_time/T_period
    return orbit_time
