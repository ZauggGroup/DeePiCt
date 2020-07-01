from os.path import join

import h5py
import numpy as np
from skimage import morphology as morph
from tqdm import tqdm

from constants import h5_internal_paths
from tomogram_utils.coordinates_toolbox.subtomos import \
    get_subtomo_corner_side_lengths_and_padding
from tomogram_utils.coordinates_toolbox.utils import shift_coordinates_by_vector


def get_clusters_within_size_range(dataset: np.array, min_cluster_size: int,
                                   max_cluster_size: int, connectivity=1):
    assert min_cluster_size <= max_cluster_size

    labeled_clusters, num = morph.label(input=dataset, background=0,
                                        return_num=True,
                                        connectivity=connectivity)
    labels_list, cluster_size = np.unique(labeled_clusters, return_counts=True)
    maximum = np.max(cluster_size)
    print("number of clusters = ", len(labels_list))
    print("With size ranges: from", np.min(cluster_size), "to",
          maximum)
    next_maximum = np.max(list(cluster_size[cluster_size < maximum]))
    print("With size ranges: from", np.min(cluster_size), "to",
          next_maximum)
    labels_list_within_range = labels_list[(cluster_size > min_cluster_size) & (
            cluster_size <= max_cluster_size)]
    cluster_size_within_range = list(
        cluster_size[(cluster_size > min_cluster_size) & (
                cluster_size <= max_cluster_size)])
    print("Clusters in subtomo before size filtering =", num)
    return labeled_clusters, labels_list_within_range, cluster_size_within_range


def get_cluster_centroids(dataset: np.array, min_cluster_size: int,
                          max_cluster_size: int, connectivity=1) -> tuple:
    labeled_clusters, labels_list_within_range, cluster_size_within_range = \
        get_clusters_within_size_range(dataset=dataset,
                                       min_cluster_size=min_cluster_size,
                                       max_cluster_size=max_cluster_size,
                                       connectivity=connectivity)
    centroids_list = list()
    total_clusters = len(labels_list_within_range)
    print("Computing cluster centroids:")
    for index, label in zip(tqdm(range(total_clusters)),
                            labels_list_within_range):
        cluster = np.where(labeled_clusters == label)
        centroid = np.rint(np.mean(cluster, axis=1))
        centroids_list.append(centroid)
    return labeled_clusters, centroids_list, cluster_size_within_range


def get_cluster_centroids_from_partition(partition: str, label_name: str,
                                         min_cluster_size: int,
                                         max_cluster_size: int,
                                         output_shape: tuple,
                                         segmentation_class=0,
                                         overlap=12) -> tuple:
    with h5py.File(partition, "a") as f:
        internal_path = join(
            h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS,
            label_name)
        labels_path = join(h5_internal_paths.CLUSTERING_LABELS, label_name)
        subtomo_names = list(f[internal_path])
        print(len(subtomo_names), " subtomos in this partition.")
        full_centroids_list = list()
        full_cluster_size_list = list()
        for subtomo_name in list(subtomo_names):
            print("subtomo_name", subtomo_name)
            subtomo_path = join(internal_path, subtomo_name)
            subtomo_data = f[subtomo_path][segmentation_class, ...]
            subtomo_shape = subtomo_data.shape
            # Extract subtomo minus overlap

            # extract the subtomo data in the internal subtomo plus a bit more
            # (overlap//2), instead of extracting sharply
            subtomo_corner, subtomo_side_lengths, zero_border_thickness = \
                get_subtomo_corner_side_lengths_and_padding(subtomo_name,
                                                            subtomo_shape,
                                                            output_shape,
                                                            overlap // 2)

            shape_minus_overlap = tuple([dim - pad[0] - pad[1] for pad, dim in
                                         zip(zero_border_thickness,
                                             subtomo_data.shape)])

            mask_out_half_overlap = np.ones(shape_minus_overlap)
            mask_out_half_overlap = np.pad(mask_out_half_overlap,
                                           zero_border_thickness, "constant")

            subtomo_data = mask_out_half_overlap * subtomo_data

            # Threshold segmentation
            subtomo_data = 1 * (subtomo_data == 1)
            # Get centroids per subtomo

            subtomo_labels, subtomo_centroids_list, cluster_size_list = \
                get_cluster_centroids(dataset=subtomo_data,
                                      min_cluster_size=min_cluster_size,
                                      max_cluster_size=max_cluster_size)

            if subtomo_name not in list(f[labels_path]):
                subtomo_labels_path = join(labels_path, subtomo_name)
                f[subtomo_labels_path] = subtomo_labels[:]
            else:
                print("Clustering label already exists.")

            overlap_shift = np.array([overlap, overlap, overlap])
            shift_vector = np.array(subtomo_corner) - overlap_shift

            # Shift subtomo coordinates to global coordinate system
            subtomo_centroids_list = \
                shift_coordinates_by_vector(subtomo_centroids_list,
                                            shift_vector)

            full_centroids_list += subtomo_centroids_list
            full_cluster_size_list += cluster_size_list
    return full_centroids_list, full_cluster_size_list


def get_cluster_centroids_from_full_dataset(partition: str, label_name: str,
                                            min_cluster_size: int,
                                            max_cluster_size: int,
                                            output_shape: tuple,
                                            segmentation_class=0,
                                            overlap=12) -> list:
    with h5py.File(partition, "a") as f:
        internal_path = join(
            h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS,
            label_name)
        labels_path = join(h5_internal_paths.CLUSTERING_LABELS, label_name)
        subtomo_names = list(f[internal_path])
        print(len(subtomo_names), " subtomos in this partition.")
        full_centroids_list = list()
        for subtomo_name in list(subtomo_names):
            print("subtomo_name", subtomo_name)
            subtomo_path = join(internal_path, subtomo_name)
            subtomo_data = f[subtomo_path][segmentation_class, ...]
            subtomo_shape = subtomo_data.shape
            # Extract subtomo minus overlap

            # extract the subtomo data in the internal subtomo plus a bit more
            # (overlap//2), instead of extracting sharply
            subtomo_corner, subtomo_side_lengths, zero_border_thickness = \
                get_subtomo_corner_side_lengths_and_padding(subtomo_name,
                                                            subtomo_shape,
                                                            output_shape,
                                                            overlap // 2)

            shape_minus_overlap = tuple([dim - pad[0] - pad[1] for pad, dim in
                                         zip(zero_border_thickness,
                                             subtomo_data.shape)])

            mask_out_half_overlap = np.ones(shape_minus_overlap)
            mask_out_half_overlap = np.pad(mask_out_half_overlap,
                                           zero_border_thickness, "constant")

            subtomo_data = mask_out_half_overlap * subtomo_data

            # Threshold segmentation
            subtomo_data = 1 * (subtomo_data == 1)
            # Get centroids per subtomo

            subtomo_labels, subtomo_centroids_list = \
                get_cluster_centroids(dataset=subtomo_data,
                                      min_cluster_size=min_cluster_size,
                                      max_cluster_size=max_cluster_size)

            if subtomo_name not in list(f[labels_path]):
                subtomo_labels_path = join(labels_path, subtomo_name)
                f[subtomo_labels_path] = subtomo_labels[:]
            else:
                print("Clustering label already exists.")

            overlap_shift = np.array([overlap, overlap, overlap])
            shift_vector = np.array(subtomo_corner) - overlap_shift

            # Shift subtomo coordinates to global coordinate system
            subtomo_centroids_list = \
                shift_coordinates_by_vector(subtomo_centroids_list,
                                            shift_vector)

            full_centroids_list += subtomo_centroids_list
    return full_centroids_list
