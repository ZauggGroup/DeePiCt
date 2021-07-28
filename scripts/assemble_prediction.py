import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-pythonpath", "--pythonpath", type=str)
parser.add_argument("-config_file", "--config_file", type=str)
parser.add_argument("-tomo_name", "--tomo_name", type=str)
parser.add_argument("-fold", "--fold", type=str, default="None")

args = parser.parse_args()
pythonpath = args.pythonpath
sys.path.append(pythonpath)

import os
import ast

import pandas as pd

from constants import h5_internal_paths
from constants.config import Config
from constants.dataset_tables import DatasetTableHeader
from file_actions.writers.h5 import assemble_tomo_from_subtomos
from file_actions.readers.tomograms import load_tomogram
from paths.pipeline_dirs import get_probability_map_path, testing_partition_path
from constants.config import get_model_name

tomo_name = args.tomo_name
fold = ast.literal_eval(args.fold)
config_file = args.config_file
config = Config(user_config_file=config_file)

model_path, model_name = get_model_name(config, fold)

snakemake_pattern = config.output_dir + "/predictions/" + model_name + "/" + tomo_name + "/" + config.pred_class + \
                    "/.{fold}.probability_map.done".format(fold=str(fold))
from networks.utils import get_training_testing_lists

if isinstance(fold, int):
    tomo_training_list, tomo_testing_list = get_training_testing_lists(config=config, fold=fold)
    if tomo_name in tomo_testing_list:
        run_job = True
    else:
        run_job = False
else:
    run_job = True

if run_job:
    output_dir_tomo, data_partition = testing_partition_path(output_dir=config.work_dir,
                                                             tomo_name=tomo_name,
                                                             fold=fold)

    segmentation_label = model_name
    box_shape = [config.box_size, config.box_size, config.box_size]

    tomo_output_dir, output_path = get_probability_map_path(config.output_dir, model_name, tomo_name,
                                                            config.pred_class)
    os.makedirs(tomo_output_dir, exist_ok=True)

    DTHeader = DatasetTableHeader(processing_tomo=config.processing_tomo)
    df = pd.read_csv(config.dataset_table)
    df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)
    tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
    print("config.processing_tomo", config.processing_tomo)
    tomo_path = tomo_df.iloc[0][config.processing_tomo]
    tomo = load_tomogram(path_to_dataset=tomo_path)
    output_shape = tomo.shape
    del tomo

    subtomos_internal_path = os.path.join(
        h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS,
        segmentation_label)

    assemble_tomo_from_subtomos(output_path=output_path, partition_file_path=data_partition, output_shape=output_shape,
                                subtomo_shape=box_shape, subtomos_internal_path=subtomos_internal_path,
                                class_number=config.pred_class_number, overlap=config.overlap,
                                reconstruction_type="prediction", final_activation='sigmoid')

    print("Assembling prediction has finalized.")

# For snakemake:
snakemake_pattern_dir = os.path.dirname(snakemake_pattern)
os.makedirs(snakemake_pattern_dir, exist_ok=True)
with open(file=snakemake_pattern, mode="w") as f:
    print("Creating snakemake pattern", snakemake_pattern)
