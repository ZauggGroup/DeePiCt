import os
import sys

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
SRC_DIR = os.path.abspath(FILE_DIR + "/../3d_cnn/src")
print(SRC_DIR)
sys.path.append(SRC_DIR)

import argparse

import os.path

import numpy as np
import pandas as pd

from file_actions.readers.em import read_em
from file_actions.writers.mrc import write_mrc_dataset
from tomogram_utils.coordinates_toolbox.utils import \
    extract_coordinates_from_em_motl
from tomogram_utils.peak_toolbox.utils import paste_sphere_in_dataset
from file_actions.readers.motl import read_txt_list


def generate_particle_mask_from_motl(path_to_motl: str, output_shape: tuple,
                                     sphere_radius: int, value: int = 1 or str,
                                     mask=None or np.array) -> np.array:
    """
    Function to paste a sphere of a given radius at every voxel coordinate
    specified by a motif list.
    """
    motl_extension = os.path.basename(path_to_motl).split(".")[-1]
    assert motl_extension in ["csv", "em", "txt"]

    if motl_extension == "csv":
        motive_list = pd.read_csv(path_to_motl, header=None)
        if motive_list.shape[1] == 20:
            coordinates = list(motive_list[[9, 8, 7]].values)
            if isinstance(value, str):
                scores = list(motive_list[[0]].values)
            else:
                scores = [value for _ in coordinates]
        elif motive_list.shape[1] == 3:
            print("only coordinates in this list")
            print(motive_list.keys())
            coordinates = list(motive_list[[2, 1, 0]].values)
            scores = [value for _ in coordinates]
        else:
            coordinates = []
            scores = []
    elif motl_extension == "em":
        _, motive_list = read_em(path_to_emfile=path_to_motl)
        coordinates = extract_coordinates_from_em_motl(motive_list)
        coordinates = [[int(p[2]), int(p[1]), int(p[0])] for p
                       in coordinates]
        scores = [value for _ in coordinates]
    else:
        motive_list = read_txt_list(txt_path=path_to_motl)
        coordinates = motive_list[["x", "y", "z"]].values
        coordinates = [[int(p[2]), int(p[1]), int(p[0])] for p
                       in coordinates]
        scores = [value for _ in coordinates]

    predicted_dataset = np.zeros(output_shape)
    if mask is not None:
        shz, shy, shx = mask.shape
    else:
        shz, shy, shx = (np.inf, np.inf, np.inf)
    for val, center in zip(scores, coordinates):
        if mask is not None:
            z, y, x = center
            if z < shz and y < shy and x < shx:
                if mask[z, y, x] == 1:
                    paste_sphere_in_dataset(dataset=predicted_dataset, center=center,
                                            radius=sphere_radius, value=val)
        else:
            paste_sphere_in_dataset(dataset=predicted_dataset, center=center,
                                    radius=sphere_radius, value=val)
    return predicted_dataset


def main():
    parser = get_cli()
    args = parser.parse_args()
    radius = args.radius
    path_to_motl = args.motif_list
    output_path = args.output
    value = args.voxel_value
    output_shape = tuple([int(s) for s in args.tomo_shape])

    print("origin list:", path_to_motl)
    print("destination mask:", output_path)
    dataset = generate_particle_mask_from_motl(path_to_motl=path_to_motl,
                                               output_shape=output_shape,
                                               sphere_radius=radius,
                                               value=value, mask=None)
    write_mrc_dataset(mrc_path=output_path, array=dataset)


def get_cli():
    parser = argparse.ArgumentParser(
        description="Convert a coordinates list file into spherical masks."
    )

    parser.add_argument(
        "-r",
        "--radius",
        required=True,
        help="Sphere radius."
    )

    parser.add_argument(
        "-motl",
        "--motif_list",
        required=True,
        help="Coordinates file in csv, em or txt formats."
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output path for MRC file."
    )

    parser.add_argument(
        "-shape",
        "--tomo_shape",
        nargs=3,
        required=True,
        help="Output shape of mask in shx shy shz format."
    )

    parser.add_argument(
        "-value",
        "--voxel_value",
        required=False,
        default=1,
        help="Value assigned to voxels in the spheres, default is 1."
    )

    return parser


if __name__ == "__main__":
    main()
