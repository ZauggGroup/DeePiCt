import h5py
import numpy as np
from os.path import join
from scipy import ndimage
from skimage import morphology as morph
from skimage.measure import regionprops_table
from tqdm import tqdm

from constants import h5_internal_paths
from tomogram_utils.coordinates_toolbox.subtomos import get_subtomo_corner_side_lengths_and_padding
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
    # Create binary mask of the labels within range
    clusters_map_in_range = np.zeros(labeled_clusters.shape)
    clusters_map_in_range[np.isin(labeled_clusters, labels_list_within_range)] = 1
    # Find out the centroids of the labels within range
    filtered_labeled_clusters = (labeled_clusters * clusters_map_in_range).astype(np.int)
    props = regionprops_table(filtered_labeled_clusters, properties=('label', 'centroid'))
    centroids_list = [np.rint([x, y, z]) for _, x, y, z in sorted(zip(props['label'].tolist(),
                                                                      props['centroid-0'].tolist(),
                                                                      props['centroid-1'].tolist(),
                                                                      props['centroid-2'].tolist()))]
    return clusters_map_in_range, centroids_list, cluster_size_within_range


def get_cluster_centroids_in_contact(dataset: np.array, min_cluster_size: int,
                                     max_cluster_size: int, contact_mask: np.array,
                                     connectivity=1) -> tuple:
    labeled_clusters, labels_list_within_range, cluster_size_within_range = \
        get_clusters_within_size_range(dataset=dataset,
                                       min_cluster_size=min_cluster_size,
                                       max_cluster_size=max_cluster_size,
                                       connectivity=connectivity)
    # Apply mask for labels within range and labels with contact mask
    labeled_clusters_in_contact = (labeled_clusters * contact_mask).astype(np.int)
    labels_list_in_contact = np.unique(labeled_clusters_in_contact)[1:]
    final_labels = np.intersect1d(labels_list_within_range, labels_list_in_contact)
    # Create binary mask of the labels within range and contact
    clusters_map_in_range = np.zeros(labeled_clusters.shape)
    clusters_map_in_range[np.isin(labeled_clusters, final_labels)] = 1
    # Find out the centroids of the labels within range
    filtered_labeled_clusters = (labeled_clusters * clusters_map_in_range).astype(int)
    props = regionprops_table(filtered_labeled_clusters, properties=('label', 'centroid'))
    centroids_list = [np.rint([x, y, z]) for _, x, y, z in sorted(zip(props['label'].tolist(),
                                                                      props['centroid-0'].tolist(),
                                                                      props['centroid-1'].tolist(),
                                                                      props['centroid-2'].tolist()))]
    _, cluster_size = np.unique(filtered_labeled_clusters, return_counts=True)
    centroids_size_list = cluster_size[1:].tolist()
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
    # Create binary mask of the labels within range
    clusters_map_in_range = np.zeros(labeled_clusters.shape)
    clusters_map_in_range[np.isin(labeled_clusters, labels_list_within_range)] = 1
    # Find out the centroids of the labels within range
    filtered_labeled_clusters = (labeled_clusters * clusters_map_in_range).astype(np.int)
    props = regionprops_table(filtered_labeled_clusters, properties=('label', 'centroid'))
    centroids_list = [np.rint([x, y, z]) for _, x, y, z in sorted(zip(props['label'].tolist(),
                                                                      props['centroid-0'].tolist(),
                                                                      props['centroid-1'].tolist(),
                                                                      props['centroid-2'].tolist()))]
    # Find centroids that in given radius have voxels of the contact mask
    inverted_mask = (contact_mask == 0).astype(int)
    distance_from_mask = ndimage.distance_transform_cdt(inverted_mask)
    thresh_distance_mask = (distance_from_mask <= tol_contact).astype(int)
    labels_list_filtered, cluster_size = np.unique(filtered_labeled_clusters, return_counts=True)
    labels_list_filtered, cluster_size = labels_list_filtered[1:], cluster_size[1:]
    mask = np.array([True if thresh_distance_mask[int(centroid[0]), int(centroid[1]), int(centroid[2])] == 1 else False
                     for centroid in centroids_list])
    labels_list_filtered = labels_list_filtered[mask]
    centroids_list = np.array(centroids_list)[mask].tolist()
    centroids_size_list = cluster_size[mask]
    clusters_map_in_range = np.zeros(labeled_clusters.shape)
    clusters_map_in_range[np.isin(labeled_clusters, labels_list_filtered)] = 1

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
