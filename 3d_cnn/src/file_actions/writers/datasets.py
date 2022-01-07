import os

import numpy as np

from file_actions.writers.h5 import write_dataset_hdf
from file_actions.writers.mrc import write_mrc_dataset


def write_dataset(output_path: str, array: np.array):
    _, file_extension = os.path.splitext(output_path)
    assert file_extension in ['.hdf', '.mrc', '.rec'], "not a valid extension"

    if file_extension == '.hdf':
        write_dataset_hdf(output_path=output_path, tomo_data=array)
    elif file_extension in ['.mrc', '.rec']:
        write_mrc_dataset(mrc_path=output_path, array=array)
    else:
        print("not a valid extension")
    return
