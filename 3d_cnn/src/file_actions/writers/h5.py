import os
import os.path
import random
from os import makedirs
from os.path import join

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from constants import h5_internal_paths
from file_actions.readers.em import read_em
from file_actions.readers.motl import read_motl_from_csv
from file_actions.readers.tomograms import load_tomogram
from file_actions.writers.mrc import write_mrc_dataset
from networks.unet import UNet3D
from pytorch_cnn.classes.io import get_device
from tensors.actions import crop_window_around_point
from tomogram_utils.coordinates_toolbox import subtomos
from tomogram_utils.coordinates_toolbox.utils import \
    extract_coordinates_from_em_motl
from tomogram_utils.peak_toolbox.utils import paste_sphere_in_dataset
from tomogram_utils.peak_toolbox.utils import read_motl_coordinates_and_values


def write_dataset_hdf(output_path: str, tomo_data: np.array):
    with h5py.File(output_path, 'w') as f:
        f[h5_internal_paths.HDF_INTERNAL_PATH] = tomo_data
    print("The hdf file has been writen in ", output_path)


def write_dataset_from_subtomograms(output_path, subtomo_path, output_shape,
                                    subtomo_shape,
                                    subtomos_internal_path):
    tomo_data = np.zeros(output_shape)
    with h5py.File(subtomo_path, 'r') as f:
        for subtomo_name in list(f[subtomos_internal_path]):
            subtomo_center = subtomos.get_coord_from_name(subtomo_name)
            init_points, end_points, lengths = subtomos.get_subtomo_corners(
                output_shape,
                subtomo_shape,
                subtomo_center)
            print(init_points, end_points, lengths)
            subtomo_h5_internal_path = join(subtomos_internal_path,
                                            subtomo_name)
            data_slices = [slice(i, e) for i, e in zip(init_points, end_points)]
            data_slices = tuple(data_slices)
            subtomo_slices = tuple([slice(0, l) for l in lengths])
            tomo_data[data_slices] = f[subtomo_h5_internal_path][subtomo_slices]
    write_dataset_hdf(output_path, tomo_data)
    del tomo_data


def assemble_tomo_from_subtomos(output_path: str, partition_file_path: str,
                                output_shape: tuple, subtomo_shape: tuple or list,
                                subtomos_internal_path: str,
                                class_number: int, overlap: int,
                                final_activation: None or 'sigmoid' = None,
                                reconstruction_type: str = "prediction"):
    print("Assembling data from", partition_file_path, ":")
    tomo_data = -10 * np.ones(output_shape)  # such that sigmoid(-10) ~ 0
    inner_subtomo_shape = tuple([subtomo_dim - 2 * overlap for
                                 subtomo_dim in subtomo_shape])
    with h5py.File(partition_file_path, 'r') as f:
        subtomo_names = list(f[subtomos_internal_path])
        total_subtomos = len(subtomo_names)
        output_shape_overlap = tuple([sh + overlap for sh in output_shape])
        for index, subtomo_name in zip(tqdm(range(total_subtomos)),
                                       subtomo_names):
            subtomo_center = subtomos.get_coord_from_name(subtomo_name)
            start_corner, end_corner, lengths = \
                subtomos.get_subtomo_corners(output_shape=output_shape_overlap,
                                             subtomo_shape=inner_subtomo_shape,
                                             subtomo_center=subtomo_center)

            volume_slices = [slice(overlap, overlap + l) for l in lengths]
            if np.min(lengths) > 0:
                overlap_shift = overlap * np.array([1, 1, 1])
                start_corner -= overlap_shift
                end_corner -= overlap_shift
                subtomo_h5_internal_path = join(subtomos_internal_path,
                                                subtomo_name)
                if reconstruction_type == "prediction":
                    channels, *rest = f[subtomo_h5_internal_path][:].shape
                    assert class_number < channels
                    # noinspection PyTypeChecker
                    channel_slices = [class_number] + volume_slices
                    channel_slices = tuple(channel_slices)
                    subtomo_data = f[subtomo_h5_internal_path][:]
                    internal_subtomo_data = subtomo_data[channel_slices]
                else:
                    volume_slices = tuple(volume_slices)
                    internal_subtomo_data = f[subtomo_h5_internal_path][
                        volume_slices]
                tomo_slices = tuple(
                    [slice(s, e) for s, e in zip(start_corner, end_corner)])
                tomo_data[tomo_slices] = internal_subtomo_data
    if final_activation is not None:
        sigmoid = nn.Sigmoid()
        tomo_data = sigmoid(torch.from_numpy(tomo_data).float())
        tomo_data = tomo_data.float().numpy()
    ext = os.path.splitext(output_path)[-1].lower()
    if ext == ".mrc":
        write_mrc_dataset(mrc_path=output_path, array=tomo_data)
    elif ext == ".hdf":
        write_dataset_hdf(output_path, tomo_data)
    return


def write_clustering_labels_subtomos(output_path: str, subtomo_path: str,
                                     output_shape: tuple, subtomo_shape: tuple,
                                     subtomos_internal_path: str,
                                     label_name: str, class_number: int,
                                     overlap: int):
    output_shape_with_overlap = output_shape  # [dim + overlap_thickness for
    # dim in
    # output_shape]
    print("The output shape is", output_shape_with_overlap)
    tomo_data = np.zeros(output_shape_with_overlap)

    internal_subtomo_shape = tuple([subtomo_dim - 2 * overlap for
                                    subtomo_dim in subtomo_shape])

    label_subtomos_internal_path = join(subtomos_internal_path, label_name)

    print("To reconstruct: ", label_subtomos_internal_path)
    with h5py.File(subtomo_path, 'r') as f:
        subtomos_set = f[label_subtomos_internal_path]
        print(list(subtomos_set))
        for subtomo_name in list(subtomos_set):
            subtomo_center = subtomos.get_coord_from_name(subtomo_name)
            start_corner, end_corner, lengths = subtomos.get_subtomo_corners(
                output_shape,
                internal_subtomo_shape,
                subtomo_center)
            overlap_shift = overlap * np.array([1, 1, 1])
            start_corner -= overlap_shift
            end_corner -= overlap_shift
            if len(subtomos_set[subtomo_name].shape) > 3:
                channels = subtomos_set[subtomo_name].shape[0]
                internal_subtomo_data = np.zeros(lengths)
                if channels > 1:
                    assert class_number < channels
                    # leave out the background class
                    channel_data = subtomos_set[subtomo_name][
                                   class_number, overlap:lengths[0] + overlap,
                                   overlap:lengths[1] + overlap,
                                   overlap:lengths[2] + overlap]
                    print("channel ", 0, ", min, max = ", np.min(channel_data),
                          np.max(channel_data))
                    internal_subtomo_data += channel_data
            else:
                print("subtomos_set[subtomo_name].shape = ",
                      subtomos_set[subtomo_name].shape)

                internal_subtomo_data = subtomos_set[subtomo_name][
                                        overlap:lengths[0] + overlap,
                                        overlap:lengths[1] + overlap,
                                        overlap:lengths[2] + overlap]
            slices = [slice(i, e) for i, e in zip(start_corner, end_corner)]
            tomo_data[slices] = internal_subtomo_data
            print("internal_subtomo_data = ", internal_subtomo_data.shape)

    write_dataset_hdf(output_path, tomo_data)
    print("right before deleting the maximum is", np.max(tomo_data))
    del tomo_data


def write_dataset_from_subtomos_with_overlap_dice_multiclass(
        output_path,
        subtomo_path,
        output_shape,
        subtomo_shape,
        subtomos_internal_path,
        class_number,
        overlap) -> None:
    """
    Works for single class and multi class.... check and substitute the other
    functions
    :param output_path:
    :param subtomo_path:
    :param output_shape:
    :param subtomo_shape:
    :param subtomos_internal_path:
    :param class_number:
    :param overlap:
    :return:
    """
    output_shape_with_overlap = output_shape
    print("The output shape is", output_shape_with_overlap)
    tomo_data = np.zeros(output_shape_with_overlap)

    internal_subtomo_shape = tuple([subtomo_dim - 2 * overlap for
                                    subtomo_dim in subtomo_shape])
    with h5py.File(subtomo_path, 'r') as f:
        for subtomo_name in list(f[subtomos_internal_path]):
            subtomo_center = subtomos.get_coord_from_name(subtomo_name)
            start_corner, end_corner, lengths = subtomos.get_subtomo_corners(
                output_shape,
                internal_subtomo_shape,
                subtomo_center)
            volume_slices = [slice(overlap, l + overlap) for l in lengths]
            overlap_shift = overlap * np.array([1, 1, 1])
            start_corner -= overlap_shift
            end_corner -= overlap_shift
            subtomo_h5_internal_path = join(subtomos_internal_path,
                                            subtomo_name)
            channels = f[subtomo_h5_internal_path][:].shape[0]
            internal_subtomo_data = np.zeros(lengths)
            if channels > 1:
                if isinstance(class_number, list):
                    for channel in class_number:
                        channel_data = f[subtomo_h5_internal_path][
                            channel,
                            volume_slices[0],
                            volume_slices[1],
                            volume_slices[2]]
                        print("channel ", channel, ", min, max = ",
                              np.min(channel_data),
                              np.max(channel_data))
                        internal_subtomo_data += channel_data
                elif isinstance(class_number, int):
                    channel_data = f[subtomo_h5_internal_path][class_number,
                                                               volume_slices[0],
                                                               volume_slices[1],
                                                               volume_slices[2]]
                    print("channel ", class_number, ", min, max = ",
                          np.min(channel_data),
                          np.max(channel_data))
                    internal_subtomo_data += channel_data
                else:
                    print("class_number = ", class_number)
                    print("class_number should be an int or a list of ints")
            else:
                internal_subtomo_data = f[subtomo_h5_internal_path][
                    0, volume_slices[0], volume_slices[1], volume_slices[2]]
            slices = [slice(i, e) for i, e in zip(start_corner, end_corner)]
            tomo_data[slices] = internal_subtomo_data
            print("internal_subtomo_data = ", internal_subtomo_data.shape)
    write_dataset_hdf(output_path, tomo_data)
    print("right before deleting the maximum is", np.max(tomo_data))
    return


def write_dataset_from_subtomos_with_overlap_multiclass_exponentiating(
        output_path,
        subtomo_path,
        output_shape,
        subtomo_shape,
        subtomos_internal_path,
        overlap):
    output_shape_with_overlap = output_shape  # [dim + overlap_thickness for
    # dim in
    # output_shape]
    print("The actual output shape is", output_shape_with_overlap)
    tomo_data = np.zeros(output_shape_with_overlap)

    internal_subtomo_shape = tuple([subtomo_dim - 2 * overlap for
                                    subtomo_dim in subtomo_shape])
    with h5py.File(subtomo_path, 'r') as f:
        for subtomo_name in list(f[subtomos_internal_path]):
            subtomo_center = subtomos.get_coord_from_name(subtomo_name)
            start_corner, end_corner, lengths = subtomos.get_subtomo_corners(
                output_shape,
                internal_subtomo_shape,
                subtomo_center)
            volume_slices = [slice(overlap, overlap + l) for l in lengths]
            overlap_shift = overlap * np.array([1, 1, 1])
            start_corner -= overlap_shift
            end_corner -= overlap_shift
            subtomo_h5_internal_path = join(subtomos_internal_path,
                                            subtomo_name)
            channels = f[subtomo_h5_internal_path][:].shape[0]
            internal_subtomo_data = np.zeros(lengths)
            for n in range(channels - 1):  # leave out the background class
                channel_data = f[subtomo_h5_internal_path][n + 1,
                                                           volume_slices[0],
                                                           volume_slices[1],
                                                           volume_slices[2]]
                print("channel ", n, ", min, max = ", np.min(channel_data),
                      np.max(channel_data))
                internal_subtomo_data += np.exp(channel_data)
            slices = [slice(i, e) for i, e in zip(start_corner, end_corner)]
            tomo_data[slices] = internal_subtomo_data
            print("internal_subtomo_data = ", internal_subtomo_data.shape)
    write_dataset_hdf(output_path, tomo_data)
    print("right before deleting the maximum is", np.max(tomo_data))
    del tomo_data


def write_subtomograms_from_dataset(output_path, padded_dataset,
                                    window_centers, crop_shape):
    for window_center in window_centers:
        with h5py.File(output_path, 'a') as f:
            subtomo_name = "subtomo_{0}".format(str(window_center))
            subtomo_h5_internal_path = join(h5_internal_paths.RAW_SUBTOMOGRAMS,
                                            subtomo_name)
            subtomo_data = crop_window_around_point(input_array=padded_dataset,
                                                    crop_shape=crop_shape,
                                                    window_center=window_center)
            f[subtomo_h5_internal_path] = subtomo_data
    print("Partition written to", output_path)


def write_joint_raw_and_labels_subtomograms(output_path: str,
                                            padded_raw_dataset: np.array,
                                            padded_labels_dataset: np.array,
                                            label_name: str,
                                            window_centers: list,
                                            crop_shape: tuple):
    with h5py.File(output_path, 'w') as f:
        for window_center in window_centers:
            print("window_center", window_center)
            subtomo_name = "subtomo_{0}".format(str(window_center))
            subtomo_raw_h5_internal_path = join(
                h5_internal_paths.RAW_SUBTOMOGRAMS,
                subtomo_name)
            subtomo_raw_data = crop_window_around_point(
                input_array=padded_raw_dataset,
                crop_shape=crop_shape,
                window_center=window_center)

            subtomo_label_h5_internal_path = join(
                h5_internal_paths.LABELED_SUBTOMOGRAMS, label_name)
            subtomo_label_h5_internal_path = join(
                subtomo_label_h5_internal_path,
                subtomo_name)

            subtomo_label_data = crop_window_around_point(
                input_array=padded_labels_dataset,
                crop_shape=crop_shape,
                window_center=window_center)
            if np.max(subtomo_label_data) > 0.5:
                f[subtomo_raw_h5_internal_path] = subtomo_raw_data
                f[subtomo_label_h5_internal_path] = subtomo_label_data
            else:
                print("subtomo ", subtomo_name, "discarded")
    return


def generate_classification_training_set(path_to_output_h5: str,
                                         path_to_dataset: str,
                                         motl_path: str, label: str,
                                         subtomo_size: int or tuple or list):
    assert isinstance(subtomo_size, (int, tuple, list))
    if isinstance(subtomo_size, int):
        crop_shape = (subtomo_size, subtomo_size, subtomo_size)
    else:
        crop_shape = subtomo_size

    _, coordinates = read_motl_coordinates_and_values(motl_path)
    dataset = load_tomogram(path_to_dataset)

    if os.path.isfile(path_to_output_h5):
        mode = 'a'
    else:
        mode = 'w'

    makedirs(os.path.dirname(path_to_output_h5), exist_ok=True)
    with h5py.File(path_to_output_h5, mode) as f:
        internal_path = h5_internal_paths.LABELED_SUBTOMOGRAMS
        internal_path = join(internal_path, label)
        for point in coordinates:
            x, y, z = [int(entry) for entry in point]
            subtomo_name = "subtomo_" + str(point)
            subtomo = crop_window_around_point(input_array=dataset,
                                               crop_shape=crop_shape,
                                               window_center=(z, y, x))
            subtomo_path = join(internal_path, subtomo_name)
            f[subtomo_path] = subtomo[:]
    return path_to_output_h5


def generate_classification_training_set_per_tomo(dataset_table: str,
                                                  tomo_name: str,
                                                  semantic_classes: list,
                                                  path_to_output_h5: str,
                                                  box_side: int or tuple):
    df = pd.read_csv(dataset_table)
    df['tomo_name'] = df['tomo_name'].astype(str)
    tomo_df = df[df['tomo_name'] == tomo_name]

    for semantic_class in semantic_classes:
        motl_label = "path_to_motl_clean_" + semantic_class
        motl_path = tomo_df.iloc[0][motl_label]
        path_to_dataset = tomo_df.iloc[0]['eman2_filetered_tomo']
        generate_classification_training_set(path_to_output_h5, path_to_dataset,
                                             motl_path, semantic_class,
                                             box_side)
    return


def write_raw_subtomograms_intersecting_mask(output_path: str,
                                             padded_raw_dataset: np.array,
                                             padded_mask_dataset: np.array,
                                             window_centers: list,
                                             crop_shape: tuple):
    with h5py.File(output_path, 'w') as f:
        n = len(window_centers)
        for index, window_center in zip(tqdm(range(n)), window_centers):
            subtomo_name = "subtomo_{0}".format(str(window_center))
            subtomo_raw_h5_internal_path = join(
                h5_internal_paths.RAW_SUBTOMOGRAMS,
                subtomo_name)
            subtomo_raw_data = crop_window_around_point(
                input_array=padded_raw_dataset,
                crop_shape=crop_shape,
                window_center=window_center)

            subtomo_label_data = crop_window_around_point(
                input_array=padded_mask_dataset,
                crop_shape=crop_shape,
                window_center=window_center)
            if np.max(subtomo_label_data) > 0:
                f[subtomo_raw_h5_internal_path] = subtomo_raw_data
    return


def write_joint_raw_and_labels_subtomograms_dice_multiclass(
        output_path: str,
        padded_raw_dataset: np.array,
        padded_labels_list: list,  # list of padded labeled data sets
        segmentation_names: list,
        window_centers: list,
        crop_shape: tuple):
    with h5py.File(output_path, 'w') as f:
        for window_center in window_centers:
            print("window_center", window_center)
            subtomo_name = "subtomo_{0}".format(str(window_center))
            subtomo_raw_h5_internal_path = join(
                h5_internal_paths.RAW_SUBTOMOGRAMS,
                subtomo_name)
            subtomo_raw_data = crop_window_around_point(
                input_array=padded_raw_dataset,
                crop_shape=crop_shape,
                window_center=window_center)

            subtomo_label_data_list = []
            subtomo_label_h5_internal_path_list = []
            segmentation_max = 0
            for label_name, padded_label in zip(segmentation_names,
                                                padded_labels_list):
                subtomo_label_data = crop_window_around_point(
                    input_array=padded_label,
                    crop_shape=crop_shape,
                    window_center=window_center)
                subtomo_label_data_list += [subtomo_label_data]
                print("subtomo_max = ", np.max(subtomo_label_data))
                subtomo_label_h5_internal_path = join(
                    h5_internal_paths.LABELED_SUBTOMOGRAMS, label_name)
                subtomo_label_h5_internal_path = join(
                    subtomo_label_h5_internal_path,
                    subtomo_name)
                subtomo_label_h5_internal_path_list += [
                    subtomo_label_h5_internal_path]
                segmentation_max = np.max(
                    [segmentation_max, np.max(subtomo_label_data)])
            if segmentation_max > 0.5:
                f[subtomo_raw_h5_internal_path] = subtomo_raw_data
                for subtomo_label_h5_internal_path, subtomo_label_data in zip(
                        subtomo_label_h5_internal_path_list,
                        subtomo_label_data_list):
                    f[subtomo_label_h5_internal_path] = subtomo_label_data
            else:
                print("subtomo ", subtomo_name, "discarded")
    return


def write_segmented_data(data_path: str, output_segmentation: np.array,
                         label_name: str) -> np.array:
    with h5py.File(data_path, 'a') as f:
        for subtomo_indx, subtomo_name in enumerate(
                list(f[h5_internal_paths.RAW_SUBTOMOGRAMS])):
            segmented_subtomo_path = join(
                h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS,
                label_name)
            subtomo_h5_internal_path = join(segmented_subtomo_path,
                                            subtomo_name)
            f[subtomo_h5_internal_path] = \
                output_segmentation[subtomo_indx, :, :, :, :]


def _check_segmentation_existence(data_file: h5py.File, label_name: str):
    if 'predictions' in list(data_file['volumes']):
        predictions = list(
            data_file[h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS])
        if label_name in predictions:
            flag = 'segmentation_exists'
        else:
            flag = 'segmentation_nonexistent'
    else:
        flag = 'segmentation_nonexistent'
    return flag


def _get_subtomos_names_to_segment(file: h5py.File, label_name: str, flag: str):
    if 'segmentation_nonexistent' == flag:
        print("The segmentation", label_name, " does not exist yet.")
        subtomo_names = list(file[h5_internal_paths.RAW_SUBTOMOGRAMS])
        total_subtomos = len(subtomo_names)
        predicted_subtomos_names = []
    else:
        print("The segmentation", label_name, " exists already.")
        prediction_path = join(
            h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS,
            label_name)
        predicted_subtomos_names = list(file[prediction_path])
        subtomo_names = list(file[h5_internal_paths.RAW_SUBTOMOGRAMS])
        total_subtomos = len(subtomo_names)
    return predicted_subtomos_names, subtomo_names, total_subtomos


def segment_and_write(data_path: str, model: UNet3D, label_name: str) -> None:
    model = model.to(torch.float)
    device = get_device()
    subtomos_data = []
    with h5py.File(data_path, 'r') as data_file:
        flag = _check_segmentation_existence(data_file, label_name)
        predicted_subtomos_names, subtomo_names, total_subtomos = \
            _get_subtomos_names_to_segment(data_file, label_name, flag)

    with h5py.File(data_path, 'a') as data_file:
        for subtomo_name in subtomo_names:
            subtomo_h5_internal_path = join(h5_internal_paths.RAW_SUBTOMOGRAMS,
                                            subtomo_name)
            subtomos_data.append(data_file[subtomo_h5_internal_path][:])
        subtomos_data = np.array(subtomos_data)
        subtomos_data = subtomos_data - np.mean(subtomos_data)
        subtomos_data = subtomos_data / np.std(subtomos_data)
        print("data_mean = {}, data_std = {}".format(np.mean(subtomos_data), np.std(subtomos_data)))
        subtomos_data = list(subtomos_data)
        for index, subtomo_name, subtomo_data in zip(tqdm(range(total_subtomos)), subtomo_names, subtomos_data):
            if subtomo_name not in predicted_subtomos_names:
                subtomo_data = np.array([subtomo_data])
                subtomo_data = subtomo_data[:, None]
                segmented_data = model(torch.from_numpy(subtomo_data).to(device).to(torch.float))
                segmented_data = segmented_data.cpu().detach().numpy()
                _write_segmented_subtomo_data(data_file=data_file,
                                              segmented_data=segmented_data,
                                              label_name=label_name,
                                              subtomo_name=subtomo_name)
    return


def _write_segmented_subtomo_data(data_file: h5py.File,
                                  segmented_data: np.array,
                                  label_name: str,
                                  subtomo_name: str):
    subtomo_h5_internal_path = join(
        h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS,
        label_name)
    subtomo_h5_internal_path = join(subtomo_h5_internal_path,
                                    subtomo_name)
    data_file[subtomo_h5_internal_path] = segmented_data[0, :, :, :, :]
    return


def write_particle_mask_from_motl(path_to_motl: str,
                                  output_path: str,
                                  output_shape: tuple,
                                  sphere_radius=8,
                                  values_in_motl: bool = True,
                                  number_of_particles=None,
                                  z_shift=0,
                                  particles_in_tom_format=True) -> None:
    _, motl_extension = os.path.splitext(path_to_motl)
    assert motl_extension in [".csv", ".em"]

    _, mask_extension = os.path.splitext(output_path)
    assert mask_extension in [".hdf", ".mrc"]

    if motl_extension == ".csv" or motl_extension == ".em":
        if motl_extension == ".csv":
            motive_list = read_motl_from_csv(path_to_motl)
            if isinstance(number_of_particles, int):
                motive_list = motive_list[:number_of_particles]
                print("Only", str(number_of_particles),
                      " particles in the motive list will be pasted.")
            else:
                print("All particles in the motive list will be pasted.")
            if particles_in_tom_format:
                coordinates = [
                    np.array([int(row[9]) + z_shift, int(row[8]), int(row[7])])
                    for
                    row in motive_list]
            else:
                coordinates = [
                    np.array([int(row[7]) + z_shift, int(row[8]), int(row[9])])
                    for
                    row in motive_list]
            if values_in_motl:
                score_values = [row[0] for row in motive_list]
            else:
                score_values = np.ones(len(motive_list))
                print("The map will be binary.")
        else:
            _, motive_list = read_em(path_to_emfile=path_to_motl)
            if isinstance(number_of_particles, int):
                motive_list = motive_list[:number_of_particles]
                print("Only", str(number_of_particles),
                      " particles in the motive list will be pasted.")
            else:
                print("All particles in the motive list will be pasted.")
            coordinates = extract_coordinates_from_em_motl(motive_list)

            if particles_in_tom_format:
                print("coordinates already in tom format")
                coordinates = [[int(p[2]) + z_shift, int(p[1]), int(p[0])] for p
                               in coordinates]
            else:
                print("transforming coordinates to tom format")
                coordinates = [[int(p[2]) + z_shift, int(p[1]), int(p[0])] for p
                               in coordinates]
            score_values = np.ones(len(coordinates))

        predicted_dataset = np.zeros(output_shape)
        for center, value in zip(coordinates, score_values):
            paste_sphere_in_dataset(dataset=predicted_dataset, center=center,
                                    radius=sphere_radius, value=value)

        if mask_extension == ".hdf":
            write_dataset_hdf(output_path=output_path,
                              tomo_data=predicted_dataset)
        elif mask_extension == ".mrc":
            if np.max(score_values) == np.min(score_values):
                dtype = np.int8
            else:
                dtype = np.float16

            write_mrc_dataset(mrc_path=output_path, array=predicted_dataset,
                              dtype=dtype)

    return


def generate_particle_mask_from_motl(path_to_motl: str,
                                     output_shape: tuple,
                                     sphere_radius=8,
                                     values_in_motl: bool = True,
                                     number_of_particles=None,
                                     z_shift=0,
                                     particles_in_tom_format=True) -> None:
    _, motl_extension = os.path.splitext(path_to_motl)
    assert motl_extension in [".csv", ".em"]

    if motl_extension == ".csv" or motl_extension == ".em":
        if motl_extension == ".csv":
            motive_list = read_motl_from_csv(path_to_motl)
            if isinstance(number_of_particles, int):
                motive_list = motive_list[:number_of_particles]
                print("Only", str(number_of_particles),
                      " particles in the motive list will be pasted.")
            else:
                print("All particles in the motive list will be pasted.")
            if particles_in_tom_format:
                coordinates = [
                    np.array([int(row[9]) + z_shift, int(row[8]), int(row[7])])
                    for
                    row in motive_list]
            else:
                coordinates = [
                    np.array([int(row[7]) + z_shift, int(row[8]), int(row[9])])
                    for
                    row in motive_list]
            if values_in_motl:
                score_values = [row[0] for row in motive_list]
            else:
                score_values = np.ones(len(motive_list))
                print("The map will be binary.")
        else:
            _, motive_list = read_em(path_to_emfile=path_to_motl)
            if isinstance(number_of_particles, int):
                motive_list = motive_list[:number_of_particles]
                print("Only", str(number_of_particles),
                      " particles in the motive list will be pasted.")
            else:
                print("All particles in the motive list will be pasted.")
            coordinates = extract_coordinates_from_em_motl(motive_list)

            if particles_in_tom_format:
                print("coordinates already in tom format")
                coordinates = [[int(p[2]) + z_shift, int(p[1]), int(p[0])] for p
                               in coordinates]
            else:
                print("transforming coordinates to tom format")
                coordinates = [[int(p[2]) + z_shift, int(p[1]), int(p[0])] for p
                               in coordinates]
            score_values = np.ones(len(coordinates))

        predicted_dataset = np.zeros(output_shape)
        for center, value in zip(coordinates, score_values):
            paste_sphere_in_dataset(dataset=predicted_dataset, center=center,
                                    radius=sphere_radius, value=value)

    return predicted_dataset


def split_and_write_h5_partition(h5_partition_data_path: str,
                                 h5_train_patition_path: str,
                                 h5_test_patition_path: str,
                                 split=-1,
                                 label_name="particles",
                                 shuffle=True) -> None:
    with h5py.File(h5_partition_data_path, 'r') as f:
        raw_subtomo_names = list(f[h5_internal_paths.RAW_SUBTOMOGRAMS])
        if shuffle:
            random.shuffle(raw_subtomo_names)
        else:
            print("Splitting sets without shuffling")
        if 0 < split < 1:
            split = int(split * len(raw_subtomo_names))
            print("split = ", split)
        else:
            split = int(split)
            print("split = ", split)
            if split < 0:
                print("All the subtomos are considered in the training set.")
                with h5py.File(h5_train_patition_path, "w") as f_train:
                    for subtomo_name in raw_subtomo_names:
                        raw_subtomo_h5_internal_path \
                            = join(h5_internal_paths.RAW_SUBTOMOGRAMS,
                                   subtomo_name)
                        data_raw_train = f[raw_subtomo_h5_internal_path][:]

                        labels_subtomo_h5_internal_path = join(
                            h5_internal_paths.LABELED_SUBTOMOGRAMS, label_name)
                        labels_subtomo_h5_internal_path = join(
                            labels_subtomo_h5_internal_path,
                            subtomo_name)
                        data_label_train = f[labels_subtomo_h5_internal_path][:]

                        f_train[raw_subtomo_h5_internal_path] = data_raw_train
                        f_train[
                            labels_subtomo_h5_internal_path] = data_label_train
                print("The training set has been written in ",
                      h5_train_patition_path)
            else:
                with h5py.File(h5_train_patition_path, "w") as f_train:
                    for subtomo_name in raw_subtomo_names[:split]:
                        raw_subtomo_h5_internal_path \
                            = join(h5_internal_paths.RAW_SUBTOMOGRAMS,
                                   subtomo_name)
                        data_raw_train = f[raw_subtomo_h5_internal_path][:]

                        labels_subtomo_h5_internal_path = join(
                            h5_internal_paths.LABELED_SUBTOMOGRAMS, label_name)
                        labels_subtomo_h5_internal_path = join(
                            labels_subtomo_h5_internal_path,
                            subtomo_name)
                        data_label_train = f[labels_subtomo_h5_internal_path][:]

                        f_train[raw_subtomo_h5_internal_path] = data_raw_train
                        f_train[
                            labels_subtomo_h5_internal_path] = data_label_train
                print("The training set has been written in ",
                      h5_train_patition_path)
                with h5py.File(h5_test_patition_path, "w") as f_test:
                    for subtomo_name in raw_subtomo_names[split:]:
                        raw_subtomo_h5_internal_path \
                            = join(h5_internal_paths.RAW_SUBTOMOGRAMS,
                                   subtomo_name)
                        data_raw_test = f[raw_subtomo_h5_internal_path][:]

                        labels_subtomo_h5_internal_path = join(
                            h5_internal_paths.LABELED_SUBTOMOGRAMS, label_name)
                        labels_subtomo_h5_internal_path = join(
                            labels_subtomo_h5_internal_path,
                            subtomo_name)
                        data_label_test = f[labels_subtomo_h5_internal_path][:]

                        f_test[raw_subtomo_h5_internal_path] = data_raw_test
                        f_test[
                            labels_subtomo_h5_internal_path] = data_label_test
                print("The testing set has been written in ",
                      h5_test_patition_path)
    return


def split_and_write_h5_partition_dice_multi_class(h5_partition_data_path: str,
                                                  h5_train_patition_path: str,
                                                  h5_test_patition_path: str,
                                                  split: float,
                                                  segmentation_names: list,
                                                  shuffle=True) -> None:
    with h5py.File(h5_partition_data_path, 'r') as f:
        raw_subtomo_names = list(f[h5_internal_paths.RAW_SUBTOMOGRAMS])
        if shuffle:
            random.shuffle(raw_subtomo_names)
        else:
            print("Splitting sets without shuffling")
        split = int(split * len(raw_subtomo_names))
        with h5py.File(h5_train_patition_path, "w") as f_train:
            for subtomo_name in raw_subtomo_names[:split]:
                raw_subtomo_h5_internal_path \
                    = join(h5_internal_paths.RAW_SUBTOMOGRAMS, subtomo_name)
                data_raw_train = f[raw_subtomo_h5_internal_path][:]
                f_train[raw_subtomo_h5_internal_path] = data_raw_train
                for label_name in segmentation_names:
                    labels_subtomo_h5_internal_path = join(
                        h5_internal_paths.LABELED_SUBTOMOGRAMS, label_name)
                    labels_subtomo_h5_internal_path = join(
                        labels_subtomo_h5_internal_path,
                        subtomo_name)
                    data_label_train = f[labels_subtomo_h5_internal_path][:]
                    f_train[labels_subtomo_h5_internal_path] = data_label_train

        with h5py.File(h5_test_patition_path, "w") as f_test:
            for subtomo_name in raw_subtomo_names[split:]:
                raw_subtomo_h5_internal_path = join(
                    h5_internal_paths.RAW_SUBTOMOGRAMS, subtomo_name)
                data_raw_test = f[raw_subtomo_h5_internal_path][:]
                f_test[raw_subtomo_h5_internal_path] = data_raw_test
                for label_name in segmentation_names:
                    labels_subtomo_h5_internal_path = join(
                        h5_internal_paths.LABELED_SUBTOMOGRAMS, label_name)
                    labels_subtomo_h5_internal_path = join(
                        labels_subtomo_h5_internal_path,
                        subtomo_name)
                    data_label_test = f[labels_subtomo_h5_internal_path][:]
                    f_test[labels_subtomo_h5_internal_path] = data_label_test
    return


def write_strongly_labeled_subtomograms(output_path: str,
                                        padded_raw_dataset: np.array,
                                        padded_labels_list: list,
                                        segmentation_names: list,
                                        window_centers: list,
                                        crop_shape: tuple,
                                        min_label_fraction: float = 0):
    with h5py.File(output_path, 'w') as f:
        for window_center in window_centers:
            print("window_center", window_center)
            subtomo_name = "subtomo_{0}".format(str(window_center))
            subtomo_raw_h5_internal_path = join(
                h5_internal_paths.RAW_SUBTOMOGRAMS,
                subtomo_name)
            subtomo_raw_data = crop_window_around_point(
                input_array=padded_raw_dataset,
                crop_shape=crop_shape,
                window_center=window_center)
            volume = crop_shape[0] * crop_shape[1] * crop_shape[2]
            subtomo_label_data_list = []
            subtomo_label_h5_internal_path_list = []
            segmentation_max = 0
            label_fraction = 0
            # Getting label channels for our current raw subtomo
            for label_name, padded_label in zip(segmentation_names,
                                                padded_labels_list):
                subtomo_label_data = crop_window_around_point(
                    input_array=padded_label,
                    crop_shape=crop_shape,
                    window_center=window_center)
                subtomo_label_data_list += [subtomo_label_data]
                print("subtomo_max = ", np.max(subtomo_label_data))
                subtomo_label_h5_internal_path = join(
                    h5_internal_paths.LABELED_SUBTOMOGRAMS, label_name)
                subtomo_label_h5_internal_path = join(
                    subtomo_label_h5_internal_path,
                    subtomo_name)
                subtomo_label_h5_internal_path_list += [
                    subtomo_label_h5_internal_path]
                segmentation_max = np.max(
                    [segmentation_max, np.max(subtomo_label_data)])
                label_indicator = np.where(subtomo_label_data > 0)
                if len(label_indicator) > 0:
                    current_fraction = len(label_indicator[0]) / volume
                    label_fraction = np.max(
                        [label_fraction, current_fraction])
            print("window_center, label_fraction", window_center,
                  label_fraction)
            if segmentation_max > 0.5 and label_fraction > min_label_fraction:
                print("Saving window_center, subtomo_raw_h5_internal_path",
                      window_center, subtomo_raw_h5_internal_path)
                f[subtomo_raw_h5_internal_path] = subtomo_raw_data
                for subtomo_label_h5_internal_path, subtomo_label_data in zip(
                        subtomo_label_h5_internal_path_list,
                        subtomo_label_data_list):
                    f[subtomo_label_h5_internal_path] = subtomo_label_data
            else:
                print("subtomo ", subtomo_name, "discarded")
    return
