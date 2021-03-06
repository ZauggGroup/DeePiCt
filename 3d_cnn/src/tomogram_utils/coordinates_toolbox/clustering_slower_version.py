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
                                   max_cluster_size: int or None, connectivity=1):
    if max_cluster_size is None:
        max_cluster_size = np.inf
    assert min_cluster_size <= max_cluster_size

    labeled_clusters, num = morph.label(input=dataset,
                                        background=0,
                                        return_num=True,
                                        connectivity=connectivity)
    labels_list, cluster_size = np.unique(labeled_clusters, return_counts=True)
    # excluding the background cluster:
    labels_list, cluster_size = labels_list[1:], cluster_size[1:]
    print("cluster_sizes:", cluster_size)
    maximum = np.max(cluster_size)
    print("number of clusters before size filtering = ", len(labels_list))
    print("size range before size filtering: ", np.min(cluster_size), "to",
          maximum)
    labels_list_within_range = labels_list[(cluster_size > min_cluster_size) & (
            cluster_size <= max_cluster_size)]
    cluster_size_within_range = list(
        cluster_size[(cluster_size > min_cluster_size) & (
                cluster_size <= max_cluster_size)])
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
    clusters_map_in_range = np.zeros(labeled_clusters.shape)
    for index, label, size in zip(tqdm(range(total_clusters)),
                                  labels_list_within_range, cluster_size_within_range):
        cluster = np.where(labeled_clusters == label)
        clusters_map_in_range[cluster] = size
        centroid = np.rint(np.mean(cluster, axis=1))
        centroids_list.append(centroid)
    return clusters_map_in_range, centroids_list, cluster_size_within_range


def get_cluster_centroids_in_contact(dataset: np.array, min_cluster_size: int,
                                     max_cluster_size: int, contact_mask: np.array,
                                     connectivity=1) -> tuple:
    labeled_clusters, labels_list_within_range, cluster_size_within_range = \
        get_clusters_within_size_range(dataset=dataset,
                                       min_cluster_size=min_cluster_size,
                                       max_cluster_size=max_cluster_size,
                                       connectivity=connectivity)
    centroids_list = list()
    centroids_size_list = list()
    total_clusters = len(labels_list_within_range)
    clusters_map_in_range = np.zeros(labeled_clusters.shape)
    for index, label, size in zip(tqdm(range(total_clusters)),
                                  labels_list_within_range, cluster_size_within_range):
        cluster_map = 1 * (labeled_clusters == label)
        cluster = np.where(labeled_clusters == label)
        centroid = np.rint(np.mean(cluster, axis=1))
        contact = len(np.where(cluster_map * contact_mask > 0)[0]) > 0
        if contact:
            clusters_map_in_range[cluster] = 1
            centroids_list.append(centroid)
            centroids_size_list.append(size)
    return clusters_map_in_range, centroids_list, centroids_size_list


def get_cluster_centroids_colocalization(dataset: np.array, min_cluster_size: int,
                                         max_cluster_size: int, contact_mask: np.array,
                                         tol_contact: float = 0,
                                         connectivity=1) -> tuple:
    labeled_clusters, labels_list_within_range, cluster_size_within_range = \
        get_clusters_within_size_range(dataset=dataset,
                                       min_cluster_size=min_cluster_size,
                                       max_cluster_size=max_cluster_size,
                                       connectivity=connectivity)
    centroids_list = list()
    centroids_size_list = list()
    total_clusters = len(labels_list_within_range)
    clusters_map_in_range = np.zeros(labeled_clusters.shape)
    for index, label, size in zip(tqdm(range(total_clusters)),
                                  labels_list_within_range, cluster_size_within_range):
        cluster = np.where(labeled_clusters == label)
        centroid = np.rint(np.mean(cluster, axis=1))
        cz, cy, cx = [int(c) for c in centroid]
        slicez = slice(cz - tol_contact, cz + tol_contact)
        slicey = slice(cy - tol_contact, cy + tol_contact)
        slicex = slice(cx - tol_contact, cx + tol_contact)
        contact = (contact_mask[slicez, slicey, slicex] > 0).any()
        if contact:
            clusters_map_in_range[cluster] = 1
            centroids_list.append(centroid)
            centroids_size_list.append(size)
    return clusters_map_in_range, centroids_list, centroids_size_list


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
