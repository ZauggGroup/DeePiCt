import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-tomo_name", "--tomo_name", type=str)
parser.add_argument("-overlap", "--overlap", type=int)
parser.add_argument("-pythonpath", "--pythonpath", type=str)
parser.add_argument("-partition_name", "--partition_name", type=str)
parser.add_argument("-segmentation_names", "--segmentation_names", nargs='+', type=str)  # todo change to list!!
parser.add_argument("-dataset_table", "--dataset_table", type=str)
parser.add_argument("-output_dir", "--output_dir", type=str)
parser.add_argument("-work_dir", "--work_dir", type=str)
parser.add_argument("-processing_tomo", "--processing_tomo", type=str)
parser.add_argument("-box_shape", "--box_shape", type=int)
parser.add_argument("-min_label_fraction", "--min_label_fraction", type=float)
parser.add_argument("-max_label_fraction", "--max_label_fraction", type=float)

args = parser.parse_args()
pythonpath = args.pythonpath
tomo_name = args.tomo_name
overlap = args.overlap
partition_name = args.partition_name
segmentation_names = args.segmentation_names
dataset_table = args.dataset_table
output_dir = args.output_dir
work_dir = args.work_dir
processing_tomo = args.processing_tomo
box_shape = args.box_shape
min_label_fraction = args.min_label_fraction
max_label_fraction = args.max_label_fraction
sys.path.append(pythonpath)

import os
from os import makedirs
from os.path import join

import yaml
import numpy as np
import pandas as pd

from tomogram_utils.volume_actions.actions import \
    generate_strongly_labeled_partition
from paths.pipeline_dirs import training_partition_path

df = pd.read_csv(dataset_table)
df['tomo_name'] = df['tomo_name'].astype(str)

print(tomo_name)
tomo_df = df[df['tomo_name'] == tomo_name]
path_to_raw = tomo_df.iloc[0][processing_tomo]
print(path_to_raw)
labels_dataset_list = list()
for semantic_class in segmentation_names:
    mask_name = semantic_class + '_mask'
    path_to_mask = tomo_df.iloc[0][mask_name]
    print("path_to_mask = {}".format(path_to_mask))
    labels_dataset_list.append(path_to_mask)

print("labels_dataset_list = ")
print(labels_dataset_list)

subtomogram_shape = (box_shape, box_shape, box_shape)
# output_h5_file_name = partition_name + ".h5"
# output_dir = os.path.join(output_dir, "training_data")
# output_path = join(output_dir, tomo_name)
#
# output_path = join(output_path, output_h5_file_name)

output_path_dir, output_path = training_partition_path(output_dir=work_dir, tomo_name=tomo_name,
                                                       partition_name=partition_name)
makedirs(name=output_path_dir, exist_ok=True)

label_fractions_list = generate_strongly_labeled_partition(
    path_to_raw=path_to_raw,
    labels_dataset_paths_list=labels_dataset_list,
    segmentation_names=segmentation_names,
    output_h5_file_path=output_path,
    subtomo_shape=subtomogram_shape,
    overlap=overlap,
    min_label_fraction=min_label_fraction,
    max_label_fraction=max_label_fraction)

selected_cubes = np.where(np.array(label_fractions_list) > min_label_fraction)[0].shape
if len(selected_cubes) == 0:
    selected_cubes = 0
else:
    selected_cubes = selected_cubes[0]
print("{} out of {} cubes in partition file.".format(selected_cubes, len(label_fractions_list)))

# print("path to training partition written on table: ", output_path)
# df.loc[df['tomo_name'] == tomo_name, config["training"]["partition_name"]] = [output_path]
# df.to_csv(path_or_buf=dataset_table, index=False)
