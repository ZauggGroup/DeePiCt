import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-pythonpath", "--pythonpath", type=str)
parser.add_argument("-config_file", "--config_file", type=str)
parser.add_argument("-tomo_name", "--tomo_name", type=str)

args = parser.parse_args()
pythonpath = args.pythonpath
sys.path.append(pythonpath)

import os

import pandas as pd

from constants import h5_internal_paths
from constants.config import Config
from constants.dataset_tables import DatasetTableHeader
from file_actions.writers.h5 import assemble_tomo_from_subtomos
from file_actions.readers.tomograms import load_tomogram
from paths.pipeline_dirs import get_probability_map_path, testing_partition_path

tomo_name = args.tomo_name
config_file = args.config_file
config = Config(user_config_file=config_file)

output_dir_tomo, data_partition = testing_partition_path(output_dir=config.work_dir,
                                                         tomo_name=tomo_name,
                                                         model_name=config.model_name)

segmentation_label = config.model_name
box_shape = [config.box_size, config.box_size, config.box_size]

tomo_output_dir, output_path = get_probability_map_path(config.output_dir, config.model_name, tomo_name,
                                                        config.pred_class)
os.makedirs(tomo_output_dir, exist_ok=True)

DTHeader = DatasetTableHeader(processing_tomo=config.processing_tomo)
df = pd.read_csv(config.dataset_table)
df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)
tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
tomo_path = tomo_df.iloc[0][DTHeader.processing_tomo]
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
