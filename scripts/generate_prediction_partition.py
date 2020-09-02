import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-config_file", "--config_file", type=str)
parser.add_argument("-pythonpath", "--pythonpath", type=str)
parser.add_argument("-fold", "--fold", type=str, default="None")
parser.add_argument("-tomo_name", "--tomo_name", type=str)

args = parser.parse_args()
pythonpath = args.pythonpath
sys.path.append(pythonpath)


import os
import ast

import numpy as np
import pandas as pd

from constants.dataset_tables import DatasetTableHeader
from file_actions.readers.tomograms import load_tomogram
from paths.pipeline_dirs import testing_partition_path
from tomogram_utils.volume_actions.actions import \
    partition_raw_intersecting_mask
from tomogram_utils.volume_actions.actions import partition_tomogram
from constants.config import Config

config_file = args.config_file
tomo_name = args.tomo_name
fold = ast.literal_eval(args.fold)
config = Config(args.config_file)

snakemake_pattern = config.work_dir + "/testing_data/" + tomo_name + \
                    "/.test_partition.{fold}.done".format(fold=str(fold))


print("tomo_name", tomo_name)
partition_output_dir, partition_path = testing_partition_path(output_dir=config.work_dir,
                                                              tomo_name=tomo_name,
                                                              fold=fold)

print("partition_path =", partition_path)
os.makedirs(partition_output_dir, exist_ok=True)

if os.path.exists(partition_path):
    print("Exiting, path exists.")
else:
    overlap = config.overlap
    box_size = config.box_size
    box_shape = (box_size, box_size, box_size)

    DTHeader = DatasetTableHeader(processing_tomo=config.processing_tomo)
    df = pd.read_csv(config.dataset_table)
    df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)

    tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
    path_to_raw = tomo_df.iloc[0][DTHeader.processing_tomo]
    intersecting_mask_path = tomo_df.iloc[0][DTHeader.filtering_mask]
    raw_dataset = load_tomogram(path_to_dataset=path_to_raw, dtype=float)
    if isinstance(intersecting_mask_path, float):
        print("No region mask file available.")
        partition_tomogram(dataset=raw_dataset,
                           output_h5_file_path=partition_path,
                           subtomo_shape=box_shape,
                           overlap=overlap)
    else:
        intersecting_mask_path = tomo_df.iloc[0][DTHeader.filtering_mask]
        intersecting_mask = load_tomogram(path_to_dataset=intersecting_mask_path)

        mask_shape = intersecting_mask.shape
        dataset_shape = raw_dataset.shape

        minimum_shape = [np.min([data_dim, mask_dim]) for
                         data_dim, mask_dim in zip(dataset_shape, mask_shape)]
        minz, miny, minx = minimum_shape

        intersecting_mask = intersecting_mask[:minz, :miny, :minx]
        raw_dataset = raw_dataset[:minz, :miny, :minx]

        partition_raw_intersecting_mask(dataset=raw_dataset,
                                        mask_dataset=intersecting_mask,
                                        output_h5_file_path=partition_path,
                                        subtomo_shape=box_shape,
                                        overlap=overlap)

# For snakemake
with open(snakemake_pattern, "w") as f:
    print("Creating snakemake pattern")