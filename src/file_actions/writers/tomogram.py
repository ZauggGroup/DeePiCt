import os
import numpy as np

from file_actions.writers.h5 import write_dataset_hdf
from file_actions.writers.mrc import write_mrc_dataset


def write_tomogram(output_path: str, tomo_data: np.array) -> None:
    ext = os.path.splitext(output_path)[-1].lower()
    if ext == ".mrc":
        write_mrc_dataset(mrc_path=output_path, array=tomo_data)
    elif ext == ".hdf":
        write_dataset_hdf(output_path=output_path, tomo_data=tomo_data)
    return
