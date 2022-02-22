import os
import sys

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
SRC_DIR = os.path.abspath(FILE_DIR + "/../3d_cnn/src")
print(SRC_DIR)
sys.path.append(SRC_DIR)

import argparse

import pandas as pd

from file_actions.readers.motl import read_motl
from motl_utils import merge_motls


def get_cli():
    parser = argparse.ArgumentParser(
        description="Merge several coordinate files into a single list, after "
                    "elliptical distance constraints."
    )

    parser.add_argument(
        "-f",
        "--coordinate_files",
        nargs="+",
        required=True,
        help="Lists of coordinates to merge."
    )

    parser.add_argument(
        "-o",
        "--output_file",
        required=True,
        help="Output csv file path."
    )

    parser.add_argument(
        "--abc",
        nargs=3,
        required=True,
        help="Coefficients of desired minimum voxel distances along x, y and z axis in format a b c."
    )

    return parser


def main():
    parser = get_cli()
    args = parser.parse_args()
    lists_paths = args.coordinate_files
    output_path = args.output_file
    a, b, c = [int(s) for s in args.abc]

    coordinates_lists = [read_motl(motl_path=path)[["x", "y", "z"]] for path in lists_paths]
    cumulative_list = pd.concat(coordinates_lists, axis=0)[["x", "y", "z"]].apply(pd.to_numeric)
    constrained_list = merge_motls(total_list=cumulative_list, a=a, b=b, c=c)
    constrained_list.to_csv(output_path, index=False, header=None)


if __name__ == "__main__":
    main()
