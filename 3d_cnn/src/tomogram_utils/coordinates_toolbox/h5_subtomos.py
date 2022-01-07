from os.path import join
from typing import List

import h5py
import numpy as np
from functools import reduce

from file_actions.readers.tomograms import load_tomogram
from tomogram_utils.coordinates_toolbox.subtomos import \
    get_subtomo_corners_within_dataset

from constants import h5_internal_paths
from tensors.actions import crop_window
from tomogram_utils.coordinates_toolbox.utils import invert_tom_coordinate_system


def get_first_raw_subtomo_shape_from_h5file(f):
    subtomo_name = list(f[h5_internal_paths.RAW_SUBTOMOGRAMS])[0]
    subtomo_path = join(h5_internal_paths.RAW_SUBTOMOGRAMS, subtomo_name)
    subtomo_shape = f[subtomo_path].shape
    return subtomo_shape


def compute_best_cross_correlation_angle(array: np.array, mask: np.array,
                                         h5file: h5py.File,
                                         ref_start_corners: tuple or List[int],
                                         ref_side_lengths: tuple or List[
                                             int]) -> tuple:
    angles = list(h5file[h5_internal_paths.RAW_SUBTOMOGRAMS])
    angles = sorted([int(angle) for angle in angles])
    correlations = list()
    for angle in angles:
        rotation_name = str(angle)
        internal_path = join(h5_internal_paths.RAW_SUBTOMOGRAMS,
                             rotation_name)
        template = h5file[internal_path][:]
        template = crop_window(input_array=template,
                               shape_to_crop=ref_side_lengths,
                               window_corner=ref_start_corners)

        mask_adj = crop_window(input_array=mask,
                               shape_to_crop=ref_side_lengths,
                               window_corner=ref_start_corners)

        template_mass = reduce((lambda x, y: x * y), template.shape)
        assert template_mass != 0, "the template volume is zero"
        correlations.append(
            np.sum(array * template * mask_adj) / template_mass)
    best_angle_index = np.argmax(correlations)
    best_cross_correlation = correlations[best_angle_index]
    return best_cross_correlation, best_angle_index


def compute_list_best_cross_correlation_angles(
        list_of_peak_coordinates: list,
        catalogue_path: str,
        path_to_mask: str,
        path_to_dataset: str,
        reference_rotation_angles_file: str,
        in_tom_format=True) -> tuple:
    dataset = load_tomogram(path_to_dataset=path_to_dataset)
    mask = load_tomogram(path_to_dataset=path_to_mask)
    dataset_shape = dataset.shape
    with h5py.File(catalogue_path, 'r') as h5file:
        subtomo_shape = get_first_raw_subtomo_shape_from_h5file(h5file)
        subtomo_center = tuple([sh // 2 for sh in subtomo_shape])
        list_best_angle_indices = list()
        list_best_cross_correlations = list()
        if in_tom_format:
            list_of_peak_coordinates_in_python_system = list(
                map(invert_tom_coordinate_system, list_of_peak_coordinates))
        else:
            list_of_peak_coordinates_in_python_system = list_of_peak_coordinates
        for point in list_of_peak_coordinates_in_python_system:
            point = [int(entry) for entry in point]
            start_corners, end_corners, side_lengths = \
                get_subtomo_corners_within_dataset(dataset_shape=dataset_shape,
                                                   subtomo_shape=subtomo_shape,
                                                   center=point)
            if tuple(side_lengths) == subtomo_shape:
                ref_start_corners = (0, 0, 0)
            else:
                ref_start_corners, _, _ = get_subtomo_corners_within_dataset(
                    dataset_shape=subtomo_shape,
                    subtomo_shape=side_lengths,
                    center=subtomo_center)
            array = crop_window(input_array=dataset, shape_to_crop=side_lengths,
                                window_corner=start_corners)
            best_cross_correlation, best_angle_index = \
                compute_best_cross_correlation_angle(
                    array=array, mask=mask,
                    h5file=h5file,
                    ref_start_corners=ref_start_corners,
                    ref_side_lengths=side_lengths)

            list_best_cross_correlations.append(best_cross_correlation)
            list_best_angle_indices.append(best_angle_index)
    angles_reference = load_tomogram(
        path_to_dataset=reference_rotation_angles_file)

    list_best_angles = list()
    for best_angle_index in list_best_angle_indices:
        angle = angles_reference[best_angle_index]
        list_best_angles.append(angle)

    return list_best_cross_correlations, list_best_angles
