import os

import numpy as np

from file_actions.readers.em import read_em
from file_actions.readers.hdf import _load_hdf_dataset
from file_actions.readers.mrc import read_mrc


def load_tomogram(path_to_dataset: str) -> np.array:
    """
    Verified that they open according to same coordinate system
    """
    _, data_file_extension = os.path.splitext(path_to_dataset)
    print("file in {} format".format(data_file_extension))
    assert data_file_extension in [".em", ".hdf", ".mrc", ".rec"], \
        "file in non valid format."
    if data_file_extension == ".em":
        em_header, dataset = read_em(path_to_emfile=path_to_dataset)
    elif data_file_extension == ".hdf":
        dataset = _load_hdf_dataset(hdf_file_path=path_to_dataset)
    elif data_file_extension in [".mrc", ".rec"]:
        dataset = read_mrc(path_to_mrc=path_to_dataset)
    return dataset
