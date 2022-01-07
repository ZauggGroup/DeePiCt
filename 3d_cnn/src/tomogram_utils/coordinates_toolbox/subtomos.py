import re
from typing import List, Tuple

import h5py
import numpy as np

from constants import h5_internal_paths


def get_coord_from_name(subtomo_name: str) -> List[int]:
    str_coord = re.findall(r'\b\d+\b', subtomo_name)
    return [int(val) for val in str_coord]


def get_subtomo_corners(output_shape: tuple, subtomo_shape: tuple,
                        subtomo_center: tuple or list) -> tuple:
    l1radius = [sh//2 for sh in subtomo_shape]
    start_corners = [int(center_dim) - int(l1_rad) for
                     center_dim, l1_rad in zip(subtomo_center, l1radius)]
    end_corners = [center_dim + subtomo_dim for center_dim, subtomo_dim
                   in zip(subtomo_center, l1radius)]
    end_corners = [int(np.min((end_point, tomo_dim))) for end_point, tomo_dim
                   in zip(end_corners,
                          output_shape)]
    side_lengths = [int(end - start) for end, start in
                    zip(end_corners, start_corners)]
    return start_corners, end_corners, side_lengths


def get_particle_coordinates_grid_with_overlap(dataset_shape: Tuple,
                                               shape_to_crop_zyx: Tuple,
                                               overlap_thickness: int):
    dataset_without_overlap_shape = [tomo_dim - 2 * overlap_thickness for
                                     tomo_dim in dataset_shape]
    internal_shape_to_crop_zyx = [dim - 2 * overlap_thickness for
                                  dim in shape_to_crop_zyx]

    particle_coordinates = get_particle_coordinates_grid(
        dataset_without_overlap_shape,
        internal_shape_to_crop_zyx)
    overlap_shift = overlap_thickness * np.array([1, 1, 1])
    particle_coordinates_with_overlap = [point + overlap_shift
                                         for point in particle_coordinates]
    return particle_coordinates_with_overlap


def get_particle_coordinates_grid(dataset_shape: Tuple[int],
                                  shape_to_crop_zyx: Tuple[int]) -> List[
    List[int]]:
    particle_coordinates = []
    nz_coords, ny_coords, nx_coords = [tomo_dim // box_size for
                                       tomo_dim, box_size in
                                       zip(dataset_shape, shape_to_crop_zyx)]
    for z in range(nz_coords):
        for y in range(ny_coords):
            for x in range(nx_coords):
                particle_coordinates += [
                    np.array(shape_to_crop_zyx) * np.array([z, y, x])
                    + np.array(shape_to_crop_zyx) // 2]
    particle_coordinates = [list(coord) for coord in particle_coordinates]
    return particle_coordinates


def get_random_particle_coordinates(dataset_shape: tuple,
                                    shape_to_crop_zyx: tuple,
                                    n_total: int) -> List[List[int]]:
    half_subtomo_shape = [int(sh / 2) for sh in shape_to_crop_zyx]
    sh_z, sh_y, sh_x = [range(subt_sh, sh - subt_sh) for sh, subt_sh in
                        zip(dataset_shape, half_subtomo_shape)]

    nz_coords = np.random.choice(a=sh_z, size=n_total)
    ny_coords = np.random.choice(a=sh_y, size=n_total)
    nx_coords = np.random.choice(a=sh_x, size=n_total)

    particle_coordinates = []
    for x, y, z in zip(nx_coords, ny_coords, nz_coords):
        particle_coordinates.append((z, y, x))
    unique_coordinates = np.unique(particle_coordinates, axis=0)
    coordinates = [list(elem) for elem in list(unique_coordinates)]
    return coordinates


def read_subtomo_names(subtomo_file_path):
    with h5py.File(subtomo_file_path, 'r') as f:
        return list(f[h5_internal_paths.RAW_SUBTOMOGRAMS])


def get_subtomo_corners_within_dataset(dataset_shape: tuple or List[int],
                                       subtomo_shape: tuple or List[int],
                                       center: tuple or List[
                                           int]) -> tuple:
    subtomo_l1radius = subtomo_shape[0] // 2, subtomo_shape[1] // 2, \
                       subtomo_shape[2] // 2
    start_corners = [center_dim - subtomo_dim for center_dim, subtomo_dim
                     in zip(center, subtomo_l1radius)]
    end_corners = [center_dim + subtomo_dim for center_dim, subtomo_dim
                   in zip(center, subtomo_l1radius)]
    end_corners = [np.min((end_point, tomo_dim)) for end_point, tomo_dim
                   in zip(end_corners,
                          dataset_shape)]

    start_corners = [np.max([corner, 0]) for corner in start_corners]
    end_corners = [np.min([corner, data_sh]) for corner, data_sh in
                   zip(end_corners, dataset_shape)]
    side_lengths = [end - start for start, end in
                    zip(start_corners, end_corners)]
    return start_corners, end_corners, side_lengths


def get_subtomo_corner_and_side_lengths(subtomo_name: str,
                                        subtomo_shape: tuple,
                                        output_shape: tuple) -> tuple:
    subtomo_center = get_coord_from_name(subtomo_name)
    init_points, _, subtomo_side_lengths = \
        get_subtomo_corners(output_shape, subtomo_shape, subtomo_center)
    return init_points, subtomo_side_lengths


def get_subtomo_corner_side_lengths_and_padding(subtomo_name: str,
                                                subtomo_shape: tuple,
                                                output_shape: tuple,
                                                overlap: int) -> tuple:
    subtomo_center = get_coord_from_name(subtomo_name)
    init_points, end_points, subtomo_side_lengths = \
        get_subtomo_corners(output_shape, subtomo_shape, subtomo_center)
    padding = 3 * [[overlap, overlap]]
    for i_point, e_point, c_point, dim, pad in zip(init_points, end_points,
                                                   subtomo_center,
                                                   subtomo_shape, padding):
        if np.abs(i_point - c_point) < dim // 2:
            subtomo_cut = dim // 2 - np.abs(i_point - c_point)
            pad[0] = np.max((0, overlap - subtomo_cut))
        if np.abs(e_point - c_point) < dim // 2:
            subtomo_cut = dim // 2 - np.abs(e_point - c_point)
            pad[1] = np.max((0, overlap - subtomo_cut))
    return init_points, subtomo_side_lengths, padding
