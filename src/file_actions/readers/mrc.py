import mrcfile
import numpy as np


def read_mrc(path_to_mrc: str) -> np.array:
    with mrcfile.open(path_to_mrc) as mrc:
        tomo = mrc.data
    return tomo
