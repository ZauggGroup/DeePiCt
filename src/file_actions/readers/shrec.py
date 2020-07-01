import csv
import numpy as np

particle_dict = {'3qm1': {'class': 1, 'radius': 5},
                 '1s3x': {'class': 2, 'radius': 6},
                 '3h84': {'class': 3, 'radius': 7},
                 '3gl1': {'class': 4, 'radius': 6},
                 '2cg9': {'class': 5, 'radius': 10},
                 '3d2f': {'class': 6, 'radius': 8},
                 '1u6g': {'class': 7, 'radius': 10},
                 '3cf3': {'class': 8, 'radius': 12},
                 '1bxn': {'class': 9, 'radius': 10},
                 '1qvr': {'class': 10, 'radius': 11},
                 '4b4t': {'class': 11, 'radius': 15},
                 '4d8q': {'class': 12, 'radius': 12},
                 }


def read_shrec_motl(path_to_motl: str):
    """
    Output: array whose first entries are the rows of the motif list
    Usage example:
    motl = read_motl_from_csv(path_to_csv_motl)
    list_of_max=[row[0] for row in motl]
    """
    motl = []

    with open(path_to_motl, 'r') as csvfile:
        motlreader = csv.reader(csvfile, delimiter='|')
        for row in motlreader:
            line = row[0].split(" ")
            motl_line = [val for val in line if val != '']
            motl_line = [particle_dict[motl_line[0]]['class']] + \
                        [float(val) for val in motl_line[1:]]
            motl += [motl_line]
    return np.array(motl)
