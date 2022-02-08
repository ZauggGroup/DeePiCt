import csv
import datetime
import os
import re
import time
from functools import reduce
from os import makedirs
from os.path import join

import h5py
import numpy as np
import pandas as pd
import torch.nn as nn

from constants import h5_internal_paths
from constants.dataset_tables import ModelsTableHeader
from tomogram_utils.coordinates_toolbox.subtomos import \
    get_subtomo_corner_and_side_lengths
from tomogram_utils.coordinates_toolbox.utils import \
    arrange_coordinates_list_by_score, filtering_duplicate_coords_with_values, \
    to_tom_coordinate_system, shift_coordinates_by_vector
from tomogram_utils.peak_toolbox.subtomos import \
    get_peaks_per_subtomo_with_overlap, \
    get_peaks_per_subtomo_with_overlap_multiclass
from tomogram_utils.peak_toolbox.utils import read_motl_data


def motl_writer(path_to_output_folder: str, list_of_peak_scores: list,
                list_of_peak_coords: list, in_tom_format=False,
                order_by_score=True, list_of_angles: list or bool = False,
                motl_name: None or str = None):
    """
    Already modified to match em_motl format
    Format of MOTL:
       The following parameters are stored in the matrix MOTIVELIST of dimension
       (NPARTICLES, 20)):
       column
          1         : Score Coefficient from localisation algorithm
          2         : x-coordinate in full tomogram
          3         : y-coordinate in full tomogram
          4         : peak number
          5         : running index of tilt series (optional)
          8         : x-coordinate in full tomogram
          9         : y-coordinate in full tomogram
          10        : z-coordinate in full tomogram
          14        : x-shift in subvolume (AFTER rotation of reference)
          15        : y-shift in subvolume
          16        : z-shift in subvolume
          17        : Phi
          18        : Psi
          19        : Theta
          20        : class number
    For more information check tom package documentation (e.g. tom_chooser.m).
    """
    numb_peaks = len(list_of_peak_scores)
    joint_list = list(zip(list_of_peak_scores, list_of_peak_coords))

    if order_by_score:
        print("saving coordinates ordered by decreasing score value")
        joint_list = sorted(joint_list, key=lambda pair: pair[0], reverse=1)
    else:
        print("saving coordinates without sorting by score value")

    if motl_name is None:
        motl_file_name = join(path_to_output_folder,
                              'motl_' + str(numb_peaks) + '.csv')
    else:
        motl_file_name = join(path_to_output_folder, motl_name)
    with open(motl_file_name, 'w', newline='') as csvfile:
        motlwriter = csv.writer(csvfile, delimiter=' ', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        for index, tuple_val_point in enumerate(joint_list):
            val, point = tuple_val_point
            if in_tom_format:
                x, y, z = point
            else:
                x, y, z = to_tom_coordinate_system(point)
            coordinate_in_tom_format = str(x) + ',' + str(y) + ',' + str(z)
            if not list_of_angles:
                angle_str = '0,0,0'
            else:
                phi, psi, theta = list_of_angles[index]
                angle_str = str(phi) + ',' + str(psi) + ',' + str(theta)
            xy_columns = ',' + str(x) + ',' + str(y) + ','
            class_str = '1'  # by default, maybe useful to list in arguments
            tail = ',0,0,0,0,0,0,' + angle_str + ',' + class_str

            row = str(val) + xy_columns + str(
                index) + ',0,0,0,' + coordinate_in_tom_format + tail
            motlwriter.writerow([row])
    print("The motive list has been writen in", motl_file_name)
    return motl_file_name


def build_tom_motive_list(list_of_peak_coordinates: list,
                          list_of_peak_scores=None,
                          list_of_angles_in_degrees=None,
                          list_of_classes=None,
                          in_tom_format=True) -> pd.DataFrame:
    """
    This function builds a motive list of particles, according to the tom format
    standards:
        The following parameters are stored in the data frame motive_list of
    dimension (NPARTICLES, 20):
       column
          1         : Score Coefficient from localisation algorithm
          2         : x-coordinate in full tomogram
          3         : y-coordinate in full tomogram
          4         : peak number
          5         : running index of tilt series (optional)
          8         : x-coordinate in full tomogram
          9         : y-coordinate in full tomogram
          10        : z-coordinate in full tomogram
          14        : x-shift in subvolume (AFTER rotation of reference)
          15        : y-shift in subvolume
          16        : z-shift in subvolume
          17        : Phi
          18        : Psi
          19        : Theta
          20        : class number
    For more information check tom package documentation (e.g. tom_chooser.m).

    :param list_of_peak_coordinates: list of points in a dataset where a
    particle has been identified. The points in this list hold the format
    np.array([px, py, pz]) where px, py, pz are the indices of the dataset in
    the tom coordinate system.
    :param list_of_peak_scores: list of scores (e.g. cross correlation)
    associated to each point in list_of_peak_coordinates.
    :param list_of_angles_in_degrees: list of Euler angles in degrees,
    np.array([phi, psi, theta]), according to the convention in the
    tom_rotatec.h function: where the rotation is the composition
    rot_z(psi)*rot_x(theta)*rot_z(phi) (again, where xyz are the tom coordinate
    system). For more information on the Euler convention see help from
    datasets.transformations.rotate_ref.
    :param list_of_classes: list of integers of length n_particles representing
    the particle class associated to each particle coordinate.
    :return:
    """
    empty_cell_value = 0  # float('nan')
    if in_tom_format:
        xs, ys, zs = list(np.array(list_of_peak_coordinates, int).transpose())
    else:
        zs, ys, xs = list(np.array(list_of_peak_coordinates, int).transpose())

    n_particles = len(list_of_peak_coordinates)
    tom_indices = list(range(1, 1 + n_particles))
    create_const_list = lambda x: [x for _ in range(n_particles)]

    if list_of_peak_scores is None:
        list_of_peak_scores = create_const_list(empty_cell_value)
    if list_of_angles_in_degrees is None:
        phis, psis, thetas = create_const_list(empty_cell_value), \
                             create_const_list(empty_cell_value), \
                             create_const_list(empty_cell_value)
    else:
        phis, psis, thetas = list_of_angles_in_degrees

    if list_of_classes is None:
        list_of_classes = create_const_list(1)
    motive_list_df = pd.DataFrame({})
    motive_list_df['score'] = list_of_peak_scores
    motive_list_df['x_'] = xs
    motive_list_df['y_'] = ys
    motive_list_df['peak'] = tom_indices
    motive_list_df['tilt_x'] = empty_cell_value
    motive_list_df['tilt_y'] = empty_cell_value
    motive_list_df['tilt_z'] = empty_cell_value
    motive_list_df['x'] = xs
    motive_list_df['y'] = ys
    motive_list_df['z'] = zs
    motive_list_df['empty_1'] = empty_cell_value
    motive_list_df['empty_2'] = empty_cell_value
    motive_list_df['empty_3'] = empty_cell_value
    motive_list_df['x-shift'] = empty_cell_value
    motive_list_df['y-shift'] = empty_cell_value
    motive_list_df['z-shift'] = empty_cell_value
    motive_list_df['phi'] = phis
    motive_list_df['psi'] = psis
    motive_list_df['theta'] = thetas
    motive_list_df['class'] = list_of_classes
    return motive_list_df


def new_motl_writer(path_to_output_folder: str, list_of_peak_coordinates: list,
                    list_of_peak_scores=None, list_of_angles_in_degrees=None,
                    list_of_classes: int or list = 1, in_tom_format=False,
                    order_by_score=True, motl_name=None) -> str:
    n_particles = len(list_of_peak_coordinates)
    if order_by_score:
        assert list_of_peak_scores is not None
        print("saving coordinates ordered by decreasing score value")
        list_of_peak_scores, list_of_peak_coordinates = \
            arrange_coordinates_list_by_score(list_of_peak_scores,
                                              list_of_peak_coordinates)
    else:
        print("Building the motive list without sorting by score.")

    if not in_tom_format:
        list_of_peak_coordinates = list(
            map(to_tom_coordinate_system, list_of_peak_coordinates))
    else:
        print("Coordinates already in tom format.")

    if motl_name is None:
        motl_file_name = join(path_to_output_folder,
                              'motl_' + str(n_particles) + '.csv')
    else:
        motl_file_name = join(path_to_output_folder, motl_name)

    motive_list_df = build_tom_motive_list(
        list_of_peak_coordinates=list_of_peak_coordinates,
        list_of_peak_scores=list_of_peak_scores,
        list_of_angles_in_degrees=list_of_angles_in_degrees,
        list_of_classes=list_of_classes)

    motive_list_df.to_csv(motl_file_name, index=False, header=False)
    print("Motive list saved in", motl_file_name)
    return motl_file_name


def _write_table_header(directory_path: str, param: str,
                        table_writer: csv.writer):
    now = datetime.datetime.now()
    table_writer.writerow([str(now)])
    table_writer.writerow(["From jobs in " + directory_path])
    table_writer.writerow(["CONTENTS"])
    table_writer.writerow(["_job_name"])
    table_writer.writerow(["_K"])
    table_writer.writerow(["_" + param])
    table_writer.writerow(["_classes"])
    table_writer.writerow(["_auPRC"])
    return


def write_jobs_table(directory_path: str, table_name: str, param: str,
                     star_files: list, jobs_statistics_dict: dict):
    table_file_path = join(directory_path, table_name)
    with open(table_file_path, 'w') as csvfile:
        table_writer = csv.writer(csvfile, delimiter=' ',
                                  quotechar='|', quoting=csv.QUOTE_MINIMAL)
        _write_table_header(directory_path, param, table_writer)
        for job_parameters in star_files:
            job_path, k, param_value, classes = job_parameters
            classes = set(classes)
            job_name = re.findall(r"(job\d\d\d)", job_path)[0]
            _, _, _, au_prc, _ = jobs_statistics_dict[job_name]
            row = [job_name, k, param_value, classes, au_prc]
            table_writer.writerow(row)
    return


def write_global_motl_from_overlapping_subtomograms(subtomograms_path: str,
                                                    motive_list_output_dir: str,
                                                    overlap: int,
                                                    label_name: str,
                                                    output_shape: tuple,
                                                    subtomo_shape: tuple,
                                                    numb_peaks: int,
                                                    min_peak_distance: int,
                                                    number_peaks_uniquify: int,
                                                    z_shift: int) -> str:
    with h5py.File(subtomograms_path, 'r') as h5file:
        subtomos_internal_path = join(
            h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS, label_name)
        print(list(h5file[subtomos_internal_path]))
        list_of_maxima = []
        list_of_maxima_coords = []
        overlap_shift = overlap * np.array([1, 1, 1])
        z_shift_vector = [z_shift, 0, 0]
        for subtomo_name in list(h5file[subtomos_internal_path]):
            subtomo_list_of_maxima, subtomo_maxima_coords = \
                get_peaks_per_subtomo_with_overlap(
                    h5file=h5file,
                    subtomo_name=subtomo_name,
                    subtomo_shape=subtomo_shape,
                    output_shape=output_shape,
                    subtomos_internal_path=subtomos_internal_path,
                    numb_peaks=numb_peaks,
                    min_peak_distance=min_peak_distance,
                    overlap=overlap)
            print("Peaks in ", subtomo_name, " computed")
            subtomo_corner, _ = get_subtomo_corner_and_side_lengths(
                subtomo_name,
                subtomo_shape,
                output_shape)

            subtomo_maxima_coords = shift_coordinates_by_vector(
                coordinates=subtomo_maxima_coords, shift_vector=-overlap_shift)
            subtomo_maxima_coords = shift_coordinates_by_vector(
                coordinates=subtomo_maxima_coords, shift_vector=z_shift_vector)

            list_of_maxima += subtomo_list_of_maxima
            list_of_maxima_coords += subtomo_maxima_coords

        motl_file_name = unique_coordinates_motl_writer(
            path_to_output_folder=motive_list_output_dir,
            list_of_peak_scores=list_of_maxima,
            list_of_peak_coords=list_of_maxima_coords,
            number_peaks_to_uniquify=number_peaks_uniquify,
            minimum_peaks_distance=min_peak_distance)
    return motl_file_name


def write_global_motl_from_overlapping_subtomograms_multiclass(
        subtomograms_path: str,
        motive_list_output_dir: str,
        overlap: int,
        label_name: str,
        output_shape: tuple,
        subtomo_shape: tuple,
        numb_peaks: int,
        min_peak_distance: int,
        class_number: int,
        number_peaks_uniquify: int,
        z_shift: int,
        final_activation: nn.Module = None,
        threshold: float = -np.inf) -> str:
    with h5py.File(subtomograms_path, 'r') as h5file:
        subtomos_internal_path = join(
            h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS, label_name)
        list_of_maxima = []
        list_of_maxima_coords = []
        overlap_shift = overlap * np.array([1, 1, 1])
        z_shift_vector = [z_shift, 0, 0]
        print("shift_vector [z_shift, y_shift, x_shift] =", z_shift_vector)
        start = time.time()
        subtomo_names_list = list(h5file[subtomos_internal_path])
        total_subtomos = len(subtomo_names_list)
        for subtomo_indx, subtomo_name in enumerate(subtomo_names_list):
            print("(", subtomo_indx + 1, "/",
                  total_subtomos, ") ", subtomo_name)
            subtomo_list_of_maxima, subtomo_maxima_coords = \
                get_peaks_per_subtomo_with_overlap_multiclass(
                    h5file=h5file,
                    subtomo_name=subtomo_name,
                    subtomo_shape=subtomo_shape,
                    output_shape=output_shape,
                    subtomos_internal_path=subtomos_internal_path,
                    numb_peaks=numb_peaks,
                    class_number=class_number,
                    min_peak_distance=min_peak_distance,
                    overlap=overlap,
                    final_activation=final_activation,
                    threshold=threshold)
            print(len(subtomo_maxima_coords), "peaks")
            subtomo_corner, _ = get_subtomo_corner_and_side_lengths(
                subtomo_name,
                subtomo_shape,
                output_shape)

            subtomo_maxima_coords = shift_coordinates_by_vector(
                coordinates=subtomo_maxima_coords, shift_vector=-overlap_shift)
            subtomo_maxima_coords = shift_coordinates_by_vector(
                coordinates=subtomo_maxima_coords, shift_vector=z_shift_vector)

            list_of_maxima += subtomo_list_of_maxima
            list_of_maxima_coords += subtomo_maxima_coords
        end = time.time()
        print("The total number of peaks is:", len(list_of_maxima_coords),
              "and the elapsed time to compute (ms) =", end - start)

        # To arrange by score is necesary, otherwise most are trash
        # from the last tomogram:
    values = []
    coordinates = []
    peaks_by_value = sorted(list(zip(list_of_maxima, list_of_maxima_coords)),
                            key=lambda x: x[0], reverse=True)
    for val, zyx_coord in peaks_by_value[:number_peaks_uniquify]:
        values += [val]
        coordinates += [zyx_coord]
    start = time.time()
    print("Sorted list with peaks:", len(values))
    motl_file_name = unique_coordinates_motl_writer(
        path_to_output_folder=motive_list_output_dir,
        list_of_peak_scores=list_of_maxima,
        list_of_peak_coords=list_of_maxima_coords,
        number_peaks_to_uniquify=number_peaks_uniquify,
        minimum_peaks_distance=min_peak_distance,
        class_number=class_number)
    end = time.time()
    print("Elapsed time for uniquifying motl (ms):", end - start)
    return motl_file_name


def unique_coordinates_motl_writer(path_to_output_folder: str,
                                   list_of_peak_scores: list,
                                   list_of_peak_coords: list,
                                   number_peaks_to_uniquify: int,
                                   minimum_peaks_distance: int,
                                   class_number=0,
                                   in_tom_format=False,
                                   motl_name=None,
                                   uniquify_by_score=False
                                   ) -> str:
    """
    Motl writer for given coordinates and score values. The format of resuting
    motl follows the TOM package one: 20 columns and N rows, where N is the
    number of coordinates to be stored.
    This function uniquifies the coordinates for a given minimum distance
    between peaks.

    :param path_to_output_folder: Destination folder
    :param list_of_peak_scores: list of scores associated to each coordinate
    :param list_of_peak_coords: list of coordinates
    :param number_peaks_to_uniquify: number of coordinates to filter for
    unique coordinates. Useful when N is huge.
    :param minimum_peaks_distance:  minimum distance between peaks to be
    considered as different particles.
    :param class_number: parameter that can be used for the motl name
    :param in_tom_format: True if coordinates are in x, y, z format according to
    TOM reader.
    :param motl_name: default is None, otherwise it can be an optional string
    with the name of output motl file.
    :param uniquify_by_score: if True, when filtering to uniquify we would keep
    the repeated coordinate holding the highest score value. By default is
    set to False.
    :return: motl file path
    """

    # Arrange by score value (necessary step to not get the trash from low
    # scores when analyzing peaks from cnn):
    values = []
    coordinates = []
    # To arrange by score:
    # Todo read angles from motl to copy them into new motl
    for val, zyx_coord in sorted(
            list(zip(list_of_peak_scores, list_of_peak_coords)),
            key=lambda x: x[0], reverse=1):
        values += [val]
        coordinates += [zyx_coord]

    start = time.time()
    values, coordinates = filtering_duplicate_coords_with_values(
        motl_coords=coordinates[:number_peaks_to_uniquify],
        motl_values=values[:number_peaks_to_uniquify],
        min_peak_distance=minimum_peaks_distance,
        preference_by_score=uniquify_by_score)
    end = time.time()
    print("elapsed time for filtering coordinates", end - start, "sec")
    numb_peaks = len(values)

    if motl_name is None:
        motl_name = 'motl_' + str(numb_peaks) + '_class_' + str(
            class_number) + '.csv'
    else:
        print("motif list name given as ", motl_name)

    motl_file_name = new_motl_writer(
        path_to_output_folder=path_to_output_folder,
        list_of_peak_coordinates=coordinates,
        list_of_peak_scores=values,
        list_of_angles_in_degrees=None,
        list_of_classes=1,
        in_tom_format=in_tom_format,
        order_by_score=False,
        motl_name=motl_name)
    return motl_file_name


def filter_duplicate_values_by_score(list_of_peak_scores: list,
                                     list_of_peak_coords: list,
                                     number_peaks_to_uniquify: int,
                                     minimum_peaks_distance: int,
                                     uniquify_by_score=False):
    values = []
    coordinates = []
    for val, zyx_coord in sorted(
            list(zip(list_of_peak_scores, list_of_peak_coords)),
            key=lambda x: x[0], reverse=1):
        values += [val]
        coordinates += [zyx_coord]

    start = time.time()
    values, coordinates = filtering_duplicate_coords_with_values(
        motl_coords=coordinates[:number_peaks_to_uniquify],
        motl_values=values[:number_peaks_to_uniquify],
        min_peak_distance=minimum_peaks_distance,
        preference_by_score=uniquify_by_score)
    end = time.time()
    print("elapsed time for filtering coordinates", end - start, "sec")
    return values, coordinates


def unite_motls(path_to_motl1: str, path_to_motl2: str,
                path_to_output_motl_dir: str,
                output_motl_name: str):
    makedirs(name=path_to_output_motl_dir, exist_ok=True)
    values1, coordinates1, angles1 = read_motl_data(path_to_motl=path_to_motl1)
    values2, coordinates2, angles2 = read_motl_data(path_to_motl=path_to_motl2)
    values = list(values1) + list(values2)
    coordinates = list(coordinates1) + list(coordinates2)
    angles = list(angles1) + list(angles2)
    motl_writer(path_to_output_folder=path_to_output_motl_dir,
                list_of_peak_coords=coordinates,
                list_of_peak_scores=values,
                list_of_angles=angles,
                in_tom_format=True,
                motl_name=output_motl_name)


def record_model_descriptor(model_name: str, model_path, model_dir: str, log_dir: str, depth: int,
                             initial_features: int, n_epochs: int, training_paths_list: list, split: float,
                             output_classes: int, segmentation_names: list, box_size: int, partition_name: str,
                             processing_tomo: str, retrain: str or bool, path_to_old_model: str,
                             models_notebook_path: str, encoder_dropout: float = np.nan,
                             decoder_dropout: float = np.nan, batch_norm: bool = False, overlap: int = 12,
                             cv_fold: int or None = None, cv_test_tomos: list or None = None):
    """
    :param model_name:
    :param label_name:
    :param model_dir:
    :param log_dir:
    :param depth:
    :param initial_features:
    :param n_epochs:
    :param training_paths_list:
    :param split:
    :param output_classes:
    :param segmentation_names:
    :param retrain:
    :param path_to_old_model:
    :param models_notebook_path:
    :param encoder_dropout:
    :param decoder_dropout:
    :param batch_norm:
    :return:
    """
    training_paths = reduce(lambda x, y: x + ", " + y, training_paths_list)
    segmentation_names = reduce(lambda x, y: x + ", " + y, segmentation_names)
    print(training_paths, segmentation_names)
    if cv_test_tomos is not None:
        cv_testing_paths = reduce(lambda x, y: x + ", " + y, cv_test_tomos)
    else:
        cv_testing_paths = ""

    now = datetime.datetime.now()
    date = str(now.day) + "/" + str(now.month) + "/" + str(now.year)

    pass


def write_on_models_notebook(model_name: str, label_name: str, model_dir: str, log_dir: str, depth: int,
                             initial_features: int, n_epochs: int, training_paths_list: list, split: float,
                             output_classes: int, segmentation_names: list, box_size: int, partition_name: str,
                             processing_tomo: str, retrain: str or bool, path_to_old_model: str,
                             models_notebook_path: str, encoder_dropout: float = np.nan,
                             decoder_dropout: float = np.nan, batch_norm: bool = False, overlap: int = 12,
                             cv_fold: int or None = None, cv_test_tomos: list or None = None):
    """
    :param model_name:
    :param label_name:
    :param model_dir:
    :param log_dir:
    :param depth:
    :param initial_features:
    :param n_epochs:
    :param training_paths_list:
    :param split:
    :param output_classes:
    :param segmentation_names:
    :param retrain:
    :param path_to_old_model:
    :param models_notebook_path:
    :param encoder_dropout:
    :param decoder_dropout:
    :param batch_norm:
    :return:
    """
    model_path = os.path.join(model_dir, model_name + ".pkl")
    ModelsHeader = ModelsTableHeader()
    training_paths = reduce(lambda x, y: x + ", " + y, training_paths_list)
    segmentation_names = reduce(lambda x, y: x + ", " + y, segmentation_names)
    print(training_paths, segmentation_names)
    if cv_test_tomos is not None:
        cv_testing_paths = reduce(lambda x, y: x + ", " + y, cv_test_tomos)
    else:
        cv_testing_paths = ""

    now = datetime.datetime.now()
    date = str(now.day) + "/" + str(now.month) + "/" + str(now.year)
    mini_notebook_df = pd.DataFrame({ModelsHeader.model_name: model_name},
                                    index=[0])
    mini_notebook_df[ModelsHeader.model_name] = model_name
    mini_notebook_df[ModelsHeader.label_name] = label_name
    mini_notebook_df[ModelsHeader.model_path] = model_path
    mini_notebook_df[ModelsHeader.log_path] = log_dir
    mini_notebook_df[ModelsHeader.depth] = depth
    mini_notebook_df[ModelsHeader.initial_features] = initial_features
    mini_notebook_df[ModelsHeader.epochs] = n_epochs
    mini_notebook_df[ModelsHeader.training_set] = training_paths
    mini_notebook_df[ModelsHeader.testing_set] = cv_testing_paths
    mini_notebook_df[ModelsHeader.fold] = cv_fold
    mini_notebook_df[ModelsHeader.train_split] = split
    mini_notebook_df[ModelsHeader.output_classes] = output_classes
    mini_notebook_df[ModelsHeader.segmentation_names] = segmentation_names
    mini_notebook_df[ModelsHeader.retrain] = str(retrain)
    mini_notebook_df[ModelsHeader.old_model] = path_to_old_model
    mini_notebook_df[ModelsHeader.training_date] = date
    mini_notebook_df[ModelsHeader.batch_norm] = str(batch_norm)
    mini_notebook_df[ModelsHeader.encoder_dropout] = encoder_dropout
    mini_notebook_df[ModelsHeader.decoder_dropout] = decoder_dropout
    mini_notebook_df[ModelsHeader.box_size] = box_size
    mini_notebook_df[ModelsHeader.overlap] = overlap
    mini_notebook_df[ModelsHeader.processing_tomo] = processing_tomo
    mini_notebook_df[ModelsHeader.partition_name] = partition_name

    models_notebook_dir = os.path.dirname(models_notebook_path)
    makedirs(models_notebook_dir, exist_ok=True)
    if os.path.isfile(models_notebook_path):
        models_notebook_df = pd.read_csv(models_notebook_path,
                                         dtype={ModelsHeader.model_name: str})
        if model_name in models_notebook_df[ModelsHeader.model_name].values:
            print("Substituting model row in models table")
            index = models_notebook_df.index[models_notebook_df[ModelsHeader.model_name] == model_name].tolist()
            assert len(index) == 1
            index = index[0]
            models_notebook_df.iloc[index, :] = mini_notebook_df.iloc[0, :]
        else:
            models_notebook_df = models_notebook_df.append(mini_notebook_df, sort="False")

        models_notebook_df.to_csv(path_or_buf=models_notebook_path, index=False)
    else:
        mini_notebook_df.to_csv(path_or_buf=models_notebook_path, index=False)
    return


def write_statistics(statistics_file: str, statistics_label: str,
                     tomo_name: str, stat_measure: float):
    dict_stats = {'tomo_name': [tomo_name],
                  statistics_label: [stat_measure]}
    mini_stats_df = pd.DataFrame(dict_stats)
    if os.path.isfile(statistics_file):
        print("The statistics file exists")
        stats_df = pd.read_csv(statistics_file)
        stats_df['tomo_name'] = stats_df['tomo_name'].astype(str)
        if statistics_label in stats_df.keys():
            print("Model's statistics label exists")
            if tomo_name in stats_df['tomo_name'].values:
                row = stats_df['tomo_name'] == tomo_name
                stats_df.loc[row, statistics_label] = [stat_measure]
            else:
                stats_df = stats_df.append(mini_stats_df, sort=False)
        else:
            print("Tomo name does not exist")
            stats_df = pd.merge(stats_df, mini_stats_df, on='tomo_name',
                                how='outer')
        stats_df.to_csv(path_or_buf=statistics_file, index=False)

    else:
        print("The statistics file does not exist, we will create it.")
        path = os.path.dirname(statistics_file)
        if path == '':
            print("Writing statistics file in current directory.")
        else:
            makedirs(path, exist_ok=True)
        mini_stats_df.to_csv(path_or_buf=statistics_file, index=False)
    return
