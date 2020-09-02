import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-tomo_name", "--tomo_name", type=str)
parser.add_argument("-config_file", "--config_file", type=str)
parser.add_argument("-fold", "--fold", type=str, default="None")
parser.add_argument("-pythonpath", "--pythonpath", type=str)

args = parser.parse_args()
pythonpath = args.pythonpath
tomo_name = args.tomo_name
config_file = args.config_file

sys.path.append(pythonpath)

import os
import ast
import numpy as np
import pandas as pd

from constants.config import Config
from tomogram_utils.volume_actions.actions import \
    generate_strongly_labeled_partition
from paths.pipeline_dirs import training_partition_path

config = Config(config_file)
df = pd.read_csv(config.dataset_table)
df['tomo_name'] = df['tomo_name'].astype(str)
fold = ast.literal_eval(args.fold)

print(tomo_name)
tomo_df = df[df['tomo_name'] == tomo_name]
path_to_raw = tomo_df.iloc[0][config.processing_tomo]
labels_dataset_list = list()
for semantic_class in config.semantic_classes:
    mask_name = semantic_class + '_mask'
    path_to_mask = tomo_df.iloc[0][mask_name]
    labels_dataset_list.append(path_to_mask)

box_shape = (config.box_size, config.box_size, config.box_size)
output_path_dir, output_path = training_partition_path(output_dir=config.work_dir,
                                                       tomo_name=tomo_name,
                                                       fold=fold)
print(output_path_dir)
os.makedirs(name=output_path_dir, exist_ok=True)
if os.path.isfile(output_path):
    print("Training partition already exists")
else:
    print("Training partition to be generated...")
    label_fractions_list = generate_strongly_labeled_partition(
        path_to_raw=path_to_raw,
        labels_dataset_paths_list=labels_dataset_list,
        segmentation_names=config.semantic_classes,
        output_h5_file_path=output_path,
        subtomo_shape=box_shape,
        overlap=config.overlap,
        min_label_fraction=config.min_label_fraction,
        max_label_fraction=config.max_label_fraction)

    selected_cubes = np.where(np.array(label_fractions_list) > config.min_label_fraction)[0].shape
    if len(selected_cubes) == 0:
        selected_cubes = 0
    else:
        selected_cubes = selected_cubes[0]
    print("{} out of {} cubes in partition file.".format(selected_cubes, len(label_fractions_list)))

# For snakemake
snakemake_pattern = "training_data/{tomo_name}/.train_partition.{fold}.done".format(tomo_name=tomo_name, fold=str(fold))
snakemake_pattern = os.path.join(config.work_dir, snakemake_pattern)
with open(snakemake_pattern, "w") as f:
    print("Creating snakemake pattern")
