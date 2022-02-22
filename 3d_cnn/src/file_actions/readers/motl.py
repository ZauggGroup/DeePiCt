import os

import numpy as np
import pandas as pd

from file_actions.readers.em import read_em


def load_motl(path_to_dataset: str) -> np.array:
    _, data_file_extension = os.path.splitext(path_to_dataset)
    assert data_file_extension in [".em", ".csv"], "file in non valid format."
    if data_file_extension == ".em":
        em_header, motl = read_em(path_to_emfile=path_to_dataset)
    else:  # data_file_extension == ".csv":
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


def read_txt_list(txt_path):
    """
    Coordinate lists reader, assuming the txt file each line is composed as:
    x\ty\tz
    i.e. the coordinates are separated by tabs.
    """
    txt_list_df = pd.read_csv(txt_path, sep="\t", names=["x", "y", "z"])
    return txt_list_df


def generate_empty_motl():
    names = ['score', 'x_', 'y_', 'peak', 'tilt_x', 'tilt_y', 'tilt_z',
             'x', 'y', 'z', 'empty_1', 'empty_2', 'empty_3', 'x-shift',
             'y-shift', 'z-shift', 'phi', 'psi', 'theta', 'class']
    empty_dict = {}
    for name in names:
        empty_dict[name] = []
    return pd.DataFrame(empty_dict)


def read_csv_list(csv_path):
    if pd.read_csv(csv_path).shape[1] == 20:
        csv_list_df = pd.read_csv(csv_path, names=['score', 'x_', 'y_', 'peak', 'tilt_x', 'tilt_y', 'tilt_z',
                                                   'x', 'y', 'z', 'empty_1', 'empty_2', 'empty_3', 'x-shift',
                                                   'y-shift', 'z-shift', 'phi', 'psi', 'theta', 'class'])
    elif pd.read_csv(csv_path).shape[1] == 3:
        csv_list_df = pd.read_csv(csv_path, names=['x', 'y', 'z'])
    elif pd.read_csv(csv_path).shape[1] == 4:
        csv_list_df = pd.read_csv(csv_path, names=['score', 'x', 'y', 'z'])
    else:
        csv_list_df = generate_empty_motl()
    return csv_list_df


def read_motl(motl_path):
    assert len(motl_path) > 0, "not a valid path"
    motl_ext = os.path.basename(motl_path).split(".")[-1]
    if motl_ext == "csv":
        motl_df = read_csv_list(motl_path)
    elif motl_ext == "em":
        motl_df = read_em_list(motl_path)
    elif motl_ext == "txt":
        motl_df = read_txt_list(motl_path)
    else:
        motl_df = read_txt_list(motl_path)
    return motl_df


def read_em_list(em_path):
    header, value = read_em(path_to_emfile=em_path)
    names = ['score', 'x_', 'y_', 'peak', 'tilt_x', 'tilt_y', 'tilt_z',
             'x', 'y', 'z', 'empty_1', 'empty_2', 'empty_3', 'x-shift',
             'y-shift', 'z-shift', 'phi', 'psi', 'theta', 'class']
    em_dict = {}
    for column, name in enumerate(names):
        em_dict[name] = value[:, column]
    em_list_df = pd.DataFrame(data=em_dict)
    em_list_df[["x_", "y_", "x", "y", "z"]] = em_list_df[["x_", "y_", "x", "y", "z"]].astype('int64')
    return em_list_df
