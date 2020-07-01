import argparse
import os
from os import makedirs
from os.path import join

import yaml
import numpy as np
import pandas as pd

from tomogram_utils.volume_actions.actions import \
    generate_strongly_labeled_partition

parser = argparse.ArgumentParser()
parser.add_argument("-config", "--config", help="yaml_file", type=str)
parser.add_argument("-set", "--set",
                    help="tomos set name to be used for training", type=int)

args = parser.parse_args()
yaml_file = args.config
config = yaml.safe_load(open(yaml_file))
tomos_set = args.set
tomo_training_list = config['tomos_sets'][tomos_set]['training_list']
overlap = config["training"]["overlap"]

partition_name = config["training"]['partition_name']
segmentation_names = config["training"]['semantic_classes']
dataset_table = config['dataset_table']
output_dir = config["output_dir"]
output_dir = os.path.join(output_dir, "training_data")
processing_tomo = config["training"]["processing_tomo"]
box_shape = config["training"]["box_shape"]
min_label_fraction = config["training"]["min_label_fraction"]
max_label_fraction = config["training"]["max_label_fraction"]

df = pd.read_csv(dataset_table)
df['tomo_name'] = df['tomo_name'].astype(str)

for tomo_name in tomo_training_list:
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
    output_h5_file_name = config["training"]["partition_name"] + ".h5"
    output_path = join(output_dir, tomo_name)
    makedirs(name=output_path, exist_ok=True)
    output_path = join(output_path, output_h5_file_name)

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

    print("path to training partition written on table: ", output_path)
    df.loc[df['tomo_name'] == tomo_name, config["training"]["partition_name"]] = [output_path]
    df.to_csv(path_or_buf=dataset_table, index=False)
