import csv
import os
from os.path import join

import numpy as np

from file_actions.readers.em import read_em
from file_actions.readers.motl import read_motl_from_csv
from file_actions.readers.tomograms import load_tomogram
from osactions.filesystem import create_dir
from tomogram_utils.coordinates_toolbox.utils import \
    extract_coordinates_and_values_from_em_motl
from tomogram_utils.coordinates_toolbox.utils import \
    filtering_duplicate_coords_with_values


def _generate_unit_particle(radius: float):
    radius = int(radius)
    unit_particle = [(0, 0, 0)]
    for i in range(radius):
        for j in range(radius):
            for k in range(radius):
                if np.sqrt(i ** 2 + j ** 2 + k ** 2) <= radius:
                    unit_particle += [(i, j, k), (-i, j, k), (i, -j, k),
                                      (i, j, -k), (-i, -j, k), (-i, j, -k),
                                      (i, -j, -k), (-i, -j, -k)]
    return unit_particle


def paste_sphere_in_dataset(dataset: np.array, center: tuple or list,
                            radius: int, value: float = 1):
    dataset_shape = dataset.shape
    cx, cy, cz = center
    ball = _generate_unit_particle(radius)
    for point in ball:
        i, j, k = point
        i, j, k = int(i), int(j), int(k)
        rel_point = np.array([i + cx, j + cy, k + cz])
        inside_criterion = rel_point < np.array(dataset_shape)
        if np.min(rel_point) >= 0 and np.min(inside_criterion):
            dataset[i + cx, j + cy, k + cz] = value
    return dataset


def _get_next_max(dataset: np.array, coordinates_list: list, radius: int,
                  numb_peaks: int, global_min: float) -> tuple:
    unit_particle = _generate_unit_particle(radius)
    if len(coordinates_list) < numb_peaks:
        for p in coordinates_list:
            particle = [np.array(p) + np.array(dp) for dp in unit_particle]
            for coord in particle:
                x, y, z = coord.astype(int)
                if (coord < dataset.shape).all() and 0 <= np.min(coord):
                    dataset[x, y, z] = global_min - 1
        next_max = np.ndarray.max(dataset)
        next_max_coords = np.where(next_max == dataset)
        dataset[next_max_coords] = global_min - 1
        n = len(next_max_coords[0])
        next_max_coords_list = \
            list(np.transpose(np.array(next_max_coords)).astype(int))
        next_values_list = next_max * np.ones(n)
        _, unique_next_coords = filtering_duplicate_coords_with_values(
            motl_coords=next_max_coords_list,
            motl_values=next_values_list,
            min_peak_distance=radius,
            preference_by_score=False,
            max_num_points=numb_peaks)
        flag = "not_overloaded"
    else:
        next_max = global_min - 1
        unique_next_coords = []
        flag = "overloaded"
    return next_max, unique_next_coords, flag


def extract_peaks(dataset: np.array, numb_peaks: int, radius: int,
                  threshold: float = -np.inf):
    global_max = np.ndarray.max(dataset)
    if threshold == -np.inf:
        global_min = np.ndarray.min(dataset)
    else:
        global_min = threshold
    # print("global_min =", global_min)
    global_max_coords = np.where(dataset == global_max)
    # print("global_max_coords =", global_max_coords)
    coordinates_list = [(global_max_coords[0][0],
                         global_max_coords[1][0],
                         global_max_coords[2][0])]
    # print("coordinates_list =", coordinates_list)
    list_of_maxima = [global_max]
    list_of_maxima_coords = coordinates_list
    # for n in range(numb_peaks):
    flag = "not_overloaded"
    while flag != "overloaded":
        next_max, coordinates_list, flag = \
            _get_next_max(dataset, coordinates_list, radius, numb_peaks,
                          global_min)
        # print("next_max =", next_max)
        # print("len(coordinates_list) =", len(coordinates_list))
        # print("flag =", flag)
        if next_max < global_min:
            print("Either reached indicator == global_min - 1, or threshold.")
            flag = "overloaded"
        else:
            list_of_maxima += [next_max for _ in coordinates_list]
            list_of_maxima_coords += coordinates_list
    return list_of_maxima, list_of_maxima_coords


def write_csv_motl(list_of_maxima: list, list_of_maxima_coords: list,
                   motl_output_dir: str):
    create_dir(motl_output_dir)

    numb_peaks = len(list_of_maxima)
    motl_file_name = join(motl_output_dir, 'motl_' + str(numb_peaks) + '.csv')

    with open(motl_file_name, 'w', newline='') as csv_file:
        motl_writer = csv.writer(csv_file, delimiter=' ', quotechar='|',
                                 quoting=csv.QUOTE_MINIMAL)

        for val, p in zip(list_of_maxima, list_of_maxima_coords):
            motl_writer.writerow([str(val) + ',' + str(p[1]) + ',' + str(
                p[2]) + ',' + str(p[0]) + ',0,0,0,' + str(p[1]) + ',' + str(
                p[2]) + ',' + str(p[0]) + ',0,0,0,0,0,0,0,0,0,1'])
    print("motive list writen in ", motl_file_name)
    return


def extract_motl_coordinates_and_score_values(motl: list) -> tuple:
    coordinates = [np.array([row[7], row[8], row[9]]) for row in
                   motl]
    score_values = [row[0] for row in motl]
    return score_values, coordinates


def _generate_horizontal_disk(radius: int, thickness: int) -> list:
    disk = []
    for i in range(radius):
        for j in range(radius):
            for k in range(thickness // 2):
                if np.sqrt(i ** 2 + j ** 2) <= radius:
                    disk += [(k, i, j), (k, -i, j), (k, i, -j), (k, -i, -j)]
                    if k > 0:
                        disk += [(-k, i, j), (-k, -i, j), (-k, i, -j),
                                 (-k, -i, -j)]
                        # disk += [(i, j, k), (-i, j, k), (i, -j, k), (-i, -j, k)]
                        # if k > 0:
                        #     disk += [(i, j, -k), (-i, j, -k), (i, -j, -k),
                        #              (-i, -j, -k)]

    return disk


def paste_rotated_disk(dataset: np.array, center: tuple, radius: int,
                       thickness: int,
                       ZXZ_angles: tuple):
    cx, cy, cz = center
    psi, theta, sigma = ZXZ_angles
    to_radians = lambda theta: theta * np.pi / 180
    #
    psi = to_radians(psi)
    theta = to_radians(theta)
    sigma = to_radians(sigma)

    rot_z = lambda psi: np.array(
        [[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0],
         [0, 0, 1]])

    rot_x = lambda psi: np.array([[1, 0, 0], [0, np.cos(psi), -np.sin(psi)],
                                  [0, np.sin(psi), np.cos(psi)]])

    # To fit hdf coordinate system:
    hdf_coords = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    swap_coords = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    ZXZ_matrix = rot_z(psi).dot(rot_x(theta))
    ZXZ_matrix = ZXZ_matrix.dot(rot_z(sigma))
    print(ZXZ_matrix)
    ZXZ_matrix = hdf_coords.dot(ZXZ_matrix)
    ZXZ_matrix = ZXZ_matrix.dot(swap_coords)

    disk = _generate_horizontal_disk(radius, thickness)
    new_disk = []
    for point in disk:
        new_disk += [ZXZ_matrix.dot(np.array(point))]
    for point in new_disk:
        i, j, k = point
        i = int(i)
        j = int(j)
        k = int(k)
        dataset[i + cx, j + cy, k + cz] = 1

    return dataset


def read_motl_coordinates_and_values(path_to_motl: str) -> tuple:
    _, motl_extension = os.path.splitext(path_to_motl)

    assert motl_extension in [".em", ".csv"], "motl clean should be in a valid format .em or .csv"
    if motl_extension == ".em":
        print("motl in .em format")
        header, motl = read_em(path_to_emfile=path_to_motl)
        motl_values, motl_coords = extract_coordinates_and_values_from_em_motl(
            motl)
    else:
        print("motl in .csv format")
        motl = read_motl_from_csv(path_to_motl)
        motl_values, motl_coords = extract_motl_coordinates_and_score_values(
            motl)
        motl_coords = np.array(motl_coords, dtype=int)
    return motl_values, motl_coords


def read_motl_data(path_to_motl: str):
    motl = load_tomogram(path_to_dataset=path_to_motl)
    motl_values, motl_coords = read_motl_coordinates_and_values(
        path_to_motl)
    motl_coords = np.array(motl_coords)
    angles = motl[:, 16:19]
    return motl_values, motl_coords, angles


def union_of_motls(path_to_motl_1: str, path_to_motl_2: str):
    values_1, coordinates_1 = read_motl_coordinates_and_values(
        path_to_motl=path_to_motl_1)
    values_2, coordinates_2 = read_motl_coordinates_and_values(
        path_to_motl=path_to_motl_2)
    coordinates = np.concatenate((coordinates_1, coordinates_2), axis=0)
    values = list(values_1) + list(values_2)
    return values, coordinates
