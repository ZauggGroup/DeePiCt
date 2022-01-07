from os.path import join

import h5py

from performance.math_utils import radians2degrees
from tomogram_utils.transformations import rotate_ref
from file_actions.readers.tomograms import load_tomogram
from constants import h5_internal_paths


def create_template_catalogue(output_path: str, reference_file: str,
                              angles_file: str, in_degrees=False):
    reference = load_tomogram(reference_file)
    angles = load_tomogram(angles_file)
    if in_degrees:
        zxz_angles_in_degrees = angles
    else:
        zxz_angles_in_degrees = radians2degrees(angles)
    with h5py.File(output_path, 'w') as f:
        for index, angle in enumerate(list(zxz_angles_in_degrees)):
            rotation_name = str(index)
            rotated_reference = \
                rotate_ref(ref=reference,
                           zxz_angles_in_degrees=angle,
                           mode="nearest")
            internal_path = join(h5_internal_paths.RAW_SUBTOMOGRAMS,
                                 rotation_name)
            f[internal_path] = rotated_reference[:]
    return
