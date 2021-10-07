import os

import numpy as np
import pandas as pd

from file_actions.readers.em import read_em


def load_motl(path_to_dataset: str) -> np.array:
    _, data_file_extension = os.path.splitext(path_to_dataset)
    assert data_file_extension in [".em", ".csv"], "file in non valid format."
    if data_file_extension == ".em":
        em_header, motl = read_em(path_to_emfile=path_to_dataset)
    else: # data_file_extension == ".csv":
        motl = read_motl_from_csv(path_to_csv_motl=path_to_dataset)
    return motl


def load_motl_as_df(path_to_motl):
    _, data_file_extension = os.path.splitext(path_to_motl)
    column_names = ['score', 'x_', 'y_', 'peak', 'tilt_x', 'tilt_y', 'tilt_z',
                    'x', 'y', 'z', 'empty_1', 'empty_2', 'empty_3', 'x-shift',
                    'y-shift', 'z-shift', 'phi', 'psi', 'theta', 'class']
    assert data_file_extension in [".em", ".csv"], "file in non valid format."
    if data_file_extension == ".em":
        header, motl = read_em(path_to_emfile=path_to_motl)
        motl_df = pd.DataFrame(motl, columns=column_names)
    else:
        motl_df = pd.read_csv(path_to_motl, header=None)
        motl_df.columns = column_names
    return motl_df


def read_motl_from_csv(path_to_csv_motl: str):
    """
    Output: array whose first entries are the rows of the motif list
    Usage example:
    motl = read_motl_from_csv(path_to_csv_motl)
    list_of_max=[row[0] for row in motl]
    """
    if os.stat(path_to_csv_motl).st_size == 0:
        motl_df = pd.DataFrame({"x": [], "y": [], "z": [], "score": []})
    else:
        motl_df = pd.read_csv(path_to_csv_motl)
        if motl_df.shape[1] == 20:
            motl_df = pd.read_csv(path_to_csv_motl,
                                  names=['score', 'x_', 'y_', 'peak', 'tilt_x', 'tilt_y', 'tilt_z',
                                         'x', 'y', 'z', 'empty_1', 'empty_2', 'empty_3', 'x-shift',
                                         'y-shift', 'z-shift', 'phi', 'psi', 'theta', 'class'])
        elif motl_df.shape[1] == 3:
            motl_df = pd.read_csv(path_to_csv_motl, names=["x", "y", "z"])
            motl_df["score"] = np.nan
    return motl_df
