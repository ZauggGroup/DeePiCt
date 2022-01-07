import functools
from os.path import join

import h5py
import numpy as np
import torch
import torch.nn as nn

from tomogram_utils.coordinates_toolbox.subtomos import get_coord_from_name, \
    get_subtomo_corners, read_subtomo_names, get_subtomo_corner_and_side_lengths
from tomogram_utils.coordinates_toolbox.subtomos import \
    get_subtomo_corner_side_lengths_and_padding
from tomogram_utils.coordinates_toolbox.utils import shift_coordinates_by_vector
from constants import h5_internal_paths
from tomogram_utils.peak_toolbox.utils import extract_peaks


def _get_numb_peaks(subtomo_shape: tuple, min_peak_distance: int) -> int:
    numb_peaks = [shape / 2 / min_peak_distance for shape in
                  subtomo_shape]
    numb_peaks = functools.reduce(lambda x, y: x * y, numb_peaks)
    return int(numb_peaks)


def _extract_data_subtomo(h5file: h5py.File,  # before h5py._hl.files.File
                          subtomo_h5_internal_path: str,
                          subtomo_side_lengths: list,
                          overlap: int,
                          class_number=0) -> np.array:
    return h5file[subtomo_h5_internal_path][class_number,
           overlap:subtomo_side_lengths[0] + overlap,
           overlap:subtomo_side_lengths[1] + overlap,
           overlap:subtomo_side_lengths[2] + overlap]


def _get_peaks_per_subtomo(h5file: h5py.File, subtomo_name: str,
                           subtomo_shape: tuple, output_shape: tuple,
                           subtomos_internal_path: str, numb_peaks: int,
                           min_peak_distance: int,
                           overlap: int) -> tuple:  # todo verify this

    subtomo_corner, subtomo_side_lengths = \
        get_subtomo_corner_and_side_lengths(subtomo_name,
                                            subtomo_shape,
                                            output_shape)
    print(subtomo_side_lengths)
    print("Subtomogram corner", subtomo_corner)
    # subtomo_corner -= overlap * np.array([1, 1, 1])
    print("subtomo corner after shift", subtomo_corner)
    subtomo_h5_internal_path = join(subtomos_internal_path,
                                    subtomo_name)

    data_subtomo = _extract_data_subtomo(
        h5file=h5file,
        subtomo_h5_internal_path=subtomo_h5_internal_path,
        subtomo_side_lengths=subtomo_side_lengths,
        overlap=0)  # ToDo check!
    long_edge_if_true = [True for length, dim in
                         zip(subtomo_side_lengths, subtomo_shape) if
                         dim - 2 * overlap]
    shape_minus_overlap = tuple(
        [dim - 2 * overlap for dim in subtomo_side_lengths])
    mask_out_overlap = np.ones(shape_minus_overlap)
    padding = [(overlap, overlap) for _ in range(3)]
    mask_out_overlap = np.pad(mask_out_overlap, padding, "constant")
    data_subtomo = mask_out_overlap * data_subtomo

    subtomo_list_of_maxima, subtomo_list_of_maxima_coords = \
        extract_peaks(dataset=data_subtomo, numb_peaks=numb_peaks,
                      radius=min_peak_distance)
    shifted_subtomo_maxima_coords = \
        shift_coordinates_by_vector(subtomo_list_of_maxima_coords,
                                    subtomo_corner)
    return subtomo_list_of_maxima, shifted_subtomo_maxima_coords


def get_peaks_per_subtomo_with_overlap(h5file: h5py.File, subtomo_name: str,
                                       subtomo_shape: tuple,
                                       output_shape: tuple,
                                       subtomos_internal_path: str,
                                       numb_peaks: int,
                                       min_peak_distance: int,
                                       overlap: int) -> tuple:
    subtomo_corner, subtomo_side_lengths, zero_padding = \
        get_subtomo_corner_side_lengths_and_padding(subtomo_name,
                                                    subtomo_shape,
                                                    output_shape,
                                                    overlap // 2)

    subtomo_h5_internal_path = join(subtomos_internal_path,
                                    subtomo_name)

    data_subtomo = _extract_data_subtomo(
        h5file=h5file,
        subtomo_h5_internal_path=subtomo_h5_internal_path,
        subtomo_side_lengths=subtomo_side_lengths,
        overlap=0)
    shape_minus_overlap = tuple([dim - pad[0] - pad[1] for pad, dim in
                                 zip(zero_padding, data_subtomo.shape)])
    mask_out_overlap = np.ones(shape_minus_overlap)
    mask_out_overlap = np.pad(mask_out_overlap, zero_padding, "constant")
    data_subtomo = mask_out_overlap * data_subtomo

    subtomo_list_of_maxima, subtomo_list_of_maxima_coords = \
        extract_peaks(dataset=data_subtomo, numb_peaks=numb_peaks,
                      radius=min_peak_distance)
    shifted_subtomo_maxima_coords = \
        shift_coordinates_by_vector(subtomo_list_of_maxima_coords,
                                    subtomo_corner)
    return subtomo_list_of_maxima, shifted_subtomo_maxima_coords


def get_peaks_per_subtomo_with_overlap_multiclass(
        h5file: h5py.File, subtomo_name: str,
        subtomo_shape: tuple,
        output_shape: tuple,
        subtomos_internal_path: str,
        numb_peaks: int,
        class_number: int,
        min_peak_distance: int,
        overlap: int,
        final_activation: nn.Module = None,
        threshold: float = -np.inf) -> tuple:
    subtomo_corner, subtomo_side_lengths, zero_padding = \
        get_subtomo_corner_side_lengths_and_padding(subtomo_name,
                                                    subtomo_shape,
                                                    output_shape,
                                                    overlap // 2)

    subtomo_h5_internal_path = join(subtomos_internal_path,
                                    subtomo_name)
    # print(subtomo_corner, subtomo_side_lengths, zero_padding)
    assert np.min(subtomo_side_lengths) > 0
    data_subtomo = _extract_data_subtomo(
        h5file=h5file,
        subtomo_h5_internal_path=subtomo_h5_internal_path,
        subtomo_side_lengths=subtomo_side_lengths,
        overlap=0, class_number=class_number)
    if final_activation is None:
        print("No final activation for peak extraction.")
    else:
        data_subtomo = np.array(
            final_activation(torch.from_numpy(data_subtomo).double()))

    shape_minus_overlap = tuple([dim - pad[0] - pad[1] for pad, dim in
                                 zip(zero_padding, data_subtomo.shape)])
    mask_out_overlap = np.ones(shape_minus_overlap)
    mask_out_overlap = np.pad(mask_out_overlap, zero_padding, "constant")
    data_subtomo = mask_out_overlap * data_subtomo
    # print("Extracting peaks of ", subtomo_name)
    subtomo_list_of_maxima, subtomo_list_of_maxima_coords = \
        extract_peaks(dataset=data_subtomo, numb_peaks=numb_peaks,
                      radius=min_peak_distance, threshold=threshold)
    shifted_subtomo_maxima_coords = \
        shift_coordinates_by_vector(subtomo_list_of_maxima_coords,
                                    subtomo_corner)
    return subtomo_list_of_maxima, shifted_subtomo_maxima_coords


def _extract_data_subtomo_old(h5file: h5py._hl.files.File,
                              subtomo_h5_internal_path: str,
                              subtomo_side_lengths: list) -> np.array:
    return h5file[subtomo_h5_internal_path][0, :subtomo_side_lengths[0],
           :subtomo_side_lengths[1], :subtomo_side_lengths[2]]


def _get_peaks_per_subtomo_old(h5file: h5py._hl.files.File, subtomo_name: str,
                               subtomo_shape: tuple, output_shape: tuple,
                               subtomos_internal_path: str, numb_peaks: int,
                               min_peak_distance: int) -> tuple:
    subtomo_corner, subtomo_side_lengths = \
        get_subtomo_corner_and_side_lengths(subtomo_name,
                                            subtomo_shape,
                                            output_shape)
    # subtomo_corner -= 12*np.array([1,1,1])
    print("Subtomogram corner", subtomo_corner)
    print("subtomogram side_lengths = ", subtomo_side_lengths)

    subtomo_h5_internal_path = join(subtomos_internal_path,
                                    subtomo_name)

    data_subtomo = _extract_data_subtomo_old(h5file, subtomo_h5_internal_path,
                                             subtomo_side_lengths)

    subtomo_list_of_maxima, subtomo_list_of_maxima_coords = \
        extract_peaks(dataset=data_subtomo, numb_peaks=numb_peaks,
                      radius=min_peak_distance)

    shifted_subtomo_maxima_coords = \
        shift_coordinates_by_vector(subtomo_list_of_maxima_coords,
                                    subtomo_corner)
    return subtomo_list_of_maxima, shifted_subtomo_maxima_coords


def get_peaks_from_subtomograms(subtomo_file_path: str, label_name: str,
                                subtomo_shape: tuple,
                                output_shape: tuple,
                                min_peak_distance: int) -> tuple:
    list_of_maxima = []
    list_of_maxima_coords = []
    numb_peaks = _get_numb_peaks(subtomo_shape, min_peak_distance)
    print("Number of peaks per subtomogram will be", numb_peaks)
    with h5py.File(subtomo_file_path, 'r') as h5file:
        subtomos_internal_path = join(
            h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS, label_name)
        for subtomo_name in list(h5file[subtomos_internal_path]):
            subtomo_list_of_maxima, shifted_subtomo_maxima_coords = \
                _get_peaks_per_subtomo_old(h5file, subtomo_name, subtomo_shape,
                                           output_shape,
                                           subtomos_internal_path, numb_peaks,
                                           min_peak_distance)

            list_of_maxima += subtomo_list_of_maxima
            list_of_maxima_coords += shifted_subtomo_maxima_coords
    return list_of_maxima, list_of_maxima_coords


def get_peaks_from_subtomograms_with_overlap(subtomo_file_path: str,
                                             label_name: str,
                                             subtomo_shape: tuple,
                                             output_shape: tuple,
                                             min_peak_distance: int,
                                             overlap: int) -> tuple:
    list_of_maxima = []
    list_of_maxima_coords = []
    internal_subtomo_shape = subtomo_shape - 2 * overlap * np.array([1, 1, 1])
    print(internal_subtomo_shape)
    numb_peaks = _get_numb_peaks(tuple(internal_subtomo_shape),
                                 min_peak_distance)
    print("Number of peaks per subtomogram will be", numb_peaks)
    with h5py.File(subtomo_file_path, 'r') as h5file:
        subtomos_internal_path = join(
            h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS, label_name)
        for subtomo_name in list(h5file[subtomos_internal_path]):
            print(subtomo_name)
            subtomo_list_of_maxima, shifted_subtomo_maxima_coords = \
                _get_peaks_per_subtomo(h5file, subtomo_name,
                                       tuple(internal_subtomo_shape),
                                       output_shape,
                                       subtomos_internal_path, numb_peaks,
                                       min_peak_distance,
                                       overlap)
            list_of_maxima += subtomo_list_of_maxima
            list_of_maxima_coords += shifted_subtomo_maxima_coords
    return list_of_maxima, list_of_maxima_coords


def _filter_coordinates_per_subtomo(coordinates, subtomo_corners_init,
                                    subtomo_corners_end, shift_z):
    # print("subtomo_corners_init", subtomo_corners_init)
    # print("subtomo_corners_end", subtomo_corners_end)
    subtomo_corners_init_xyz = list(reversed(subtomo_corners_init))
    subtomo_corners_end_xyz = list(reversed(subtomo_corners_end))
    subtomo_corners_init_xyz = np.array(subtomo_corners_init_xyz) + np.array(
        [0, 0, shift_z])
    subtomo_corners_end_xyz = np.array(subtomo_corners_end_xyz) + np.array(
        [0, 0, shift_z])  # hasta aca bien!

    # print("subtomo", subtomo_corners_init_xyz, subtomo_corners_end_xyz)
    selected_coordinates = []
    discarded_coordinates = []
    for point in coordinates:
        is_in_subtomo = all(p >= c_init and p <= c_end for p, c_init, c_end in
                            zip(point, subtomo_corners_init_xyz,
                                subtomo_corners_end_xyz))
        if is_in_subtomo:
            selected_coordinates += [point]
        else:
            discarded_coordinates += [point]
            print("discarded!")
    return selected_coordinates, discarded_coordinates


def filter_test_coordinates(coordinates: list, subtomo_file_path: str,
                            split: int, dataset_shape: tuple,
                            subtomo_shape: tuple, shift: int):
    subtomo_names = read_subtomo_names(subtomo_file_path)
    test_coordinates = []
    train_coordinates = []
    for subtomo_name in subtomo_names[split:]:
        subtomo_center = get_coord_from_name(subtomo_name)
        subtomo_corners_init, subtomo_corners_end, _ = get_subtomo_corners(
            output_shape=dataset_shape, subtomo_shape=subtomo_shape,
            subtomo_center=subtomo_center)
        filtered_coordinates, discarded_coordinates = \
            _filter_coordinates_per_subtomo(coordinates,
                                            subtomo_corners_init,
                                            subtomo_corners_end,
                                            shift)
        test_coordinates += filtered_coordinates
        train_coordinates += discarded_coordinates
    return test_coordinates, train_coordinates


def _get_subtomo_coorners(subtomo_corners_init: list, subtomo_corners_end: list,
                          shift_z: int):
    subtomo_corners_init_xyz = list(reversed(subtomo_corners_init))
    subtomo_corners_end_xyz = list(reversed(subtomo_corners_end))
    subtomo_corners_init_xyz = np.array(subtomo_corners_init_xyz) + np.array(
        [0, 0, shift_z])
    subtomo_corners_end_xyz = np.array(subtomo_corners_end_xyz) + np.array(
        [0, 0, shift_z])
    return subtomo_corners_init_xyz, subtomo_corners_end_xyz


def _get_subtomo_corners_from_split(subtomo_names: list, split: int,
                                    subtomo_shape: tuple, dataset_shape: tuple,
                                    shift: int):
    subtomos_corners = []
    for subtomo_name in subtomo_names[split:]:
        subtomo_center = get_coord_from_name(subtomo_name)
        subtomo_corners_init, subtomo_corners_end, _ = get_subtomo_corners(
            output_shape=dataset_shape, subtomo_shape=subtomo_shape,
            subtomo_center=subtomo_center)
        corners = _get_subtomo_coorners(subtomo_corners_init,
                                        subtomo_corners_end, shift)
        subtomos_corners += [corners]
    return subtomos_corners


def _check_point_in_subtomo(corners: tuple, point: list):
    subtomo_corners_init_xyz, subtomo_corners_end_xyz = corners
    is_in_subtomo = all(
        p >= c_init and p <= c_end for p, c_init, c_end in
        zip(point, subtomo_corners_init_xyz,
            subtomo_corners_end_xyz))
    return is_in_subtomo


def select_coordinates_in_subtomos(coordinates: list, subtomo_file_path: str,
                                   split: int,
                                   data_order, dataset_shape: tuple,
                                   subtomo_shape: tuple,
                                   shift: int):
    subtomo_names = read_subtomo_names(subtomo_file_path)
    if isinstance(data_order, str):
        # print("Keeping data order from the h5 file.")
        subtomo_names_reordered = subtomo_names
    else:
        subtomo_names_reordered = [subtomo_names[i] for i in data_order]

    subtomos_corners = _get_subtomo_corners_from_split(subtomo_names_reordered,
                                                       split,
                                                       subtomo_shape,
                                                       dataset_shape, shift)
    selected_coordinates = []
    discarded_coordinates = []
    for point in coordinates:
        flag = "undetected"
        for corners in subtomos_corners:
            is_in_subtomo = _check_point_in_subtomo(corners, point)
            if is_in_subtomo and flag == "undetected":
                selected_coordinates += [point]
                flag = "detected"
        if flag == "undetected":
            discarded_coordinates += [point]
    return selected_coordinates, discarded_coordinates


def select_coordinates_and_values_in_subtomos(coordinates: list,
                                              values: list,
                                              subtomo_file_path: str,
                                              split: int,
                                              data_order,
                                              dataset_shape: tuple,
                                              subtomo_shape: tuple,
                                              shift: int):
    subtomo_names = read_subtomo_names(subtomo_file_path)
    if isinstance(data_order, str):
        print("Keeping data order from the h5 file.")
        subtomo_names_reordered = subtomo_names
    else:
        subtomo_names_reordered = [subtomo_names[i] for i in data_order]
    subtomos_corners = _get_subtomo_corners_from_split(subtomo_names_reordered,
                                                       split,
                                                       subtomo_shape,
                                                       dataset_shape,
                                                       shift)
    selected_coordinates = []
    selected_coordinates_values = []
    discarded_coordinates = []
    discarded_coordinates_values = []
    for value, point in zip(values, coordinates):
        flag = "undetected"
        for corners in subtomos_corners:
            is_in_subtomo = _check_point_in_subtomo(corners, point)
            if is_in_subtomo and flag == "undetected":
                selected_coordinates += [point]
                selected_coordinates_values += [value]
                flag = "detected"
        if flag == "undetected":
            discarded_coordinates += [point]
            discarded_coordinates_values += [value]
    return selected_coordinates, discarded_coordinates, \
           selected_coordinates_values, discarded_coordinates_values
