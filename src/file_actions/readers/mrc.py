import mrcfile
import numpy as np


def read_mrc(path_to_mrc: str, dtype=None) -> np.array:
    with mrcfile.open(path_to_mrc) as mrc:
        tomo = mrc.data
        if dtype is not None:
            tomo = tomo.astype(dtype=dtype)
    return tomo
