import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-pythonpath", "--pythonpath", type=str)
parser.add_argument("-tomo_name", "--tomo_name", type=str)
parser.add_argument("-fold", "--fold", type=int or None, default=None)
parser.add_argument("-config_file", "--config_file", help="yaml_file", type=str)

args = parser.parse_args()
pythonpath = args.pythonpath
sys.path.append(pythonpath)

import os

import numpy as np
import pandas as pd
import torch

from constants.dataset_tables import DatasetTableHeader
from file_actions.readers.tomograms import load_tomogram
from pytorch_cnn.classes.loss import DiceCoefficient
from paths.pipeline_dirs import get_post_processed_prediction_path
from constants.config import Config
from constants.statistics import write_statistics_pp
from constants.config import model_descriptor_from_config

config_file = args.config_file
config = Config(user_config_file=config_file)
tomo_name = args.tomo_name


prediction_path = get_post_processed_prediction_path(output_dir=config.output_dir,
                                                     model_name=config.model_name,
                                                     tomo_name=tomo_name,
                                                     semantic_class=config.pred_class)
print(prediction_path)
assert os.path.isfile(prediction_path), "The prediction file does not exist!"

DTHeader = DatasetTableHeader(semantic_classes=config.semantic_classes, filtering_mask=config.region_mask)
df = pd.read_csv(config.dataset_table)
df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)
clean_mask_name = DTHeader.masks_names[config.pred_class_number]

tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
lamella_file = tomo_df.iloc[0][DTHeader.filtering_mask]
target_path = tomo_df.iloc[0][clean_mask_name]

if str(lamella_file) == "nan":
    prediction = load_tomogram(prediction_path)
else:
    lamella_indicator = load_tomogram(path_to_dataset=lamella_file)
    prediction = load_tomogram(path_to_dataset=prediction_path)
    shx, shy, shz = [np.min([shl, shp]) for shl, shp in
                     zip(lamella_indicator.shape, prediction.shape)]
    lamella_indicator = lamella_indicator[:shx, :shy, :shz]
    prediction = prediction[:shx, :shy, :shz]
    prediction = np.array(lamella_indicator, dtype=float) * np.array(prediction,
                                                                     dtype=float)

target = load_tomogram(path_to_dataset=target_path)

shx, shy, shz = [np.min([shl, shp]) for shl, shp in
                 zip(target.shape, prediction.shape)]

target = target[:shx, :shy, :shz]
prediction = prediction[:shx, :shy, :shz]

prediction = torch.from_numpy(prediction).float()
target = torch.from_numpy(target).float()

measure = DiceCoefficient()
dice_statistic = measure.forward(prediction=prediction, target=target)
dice_statistic = float(dice_statistic)

models_table = os.path.join(config.output_dir, "models")
models_table = os.path.join(models_table, "models.csv")
if args.fold is None:
    models_table_path = model_descriptor_from_config(config)
else:
    models_table_path = os.path.join(config.output_dir, "models/models.csv")

statistics_file = os.path.join(config.output_dir, "pp_statistics.csv")

write_statistics_pp(statistics_file=statistics_file, tomo_name=tomo_name, model_name=config.model_name,
                    model_parameters=models_table_path, statistic_variable="dice",
                    statistic_value=round(dice_statistic, 4), pr_radius=None,
                    min_cluster_size=config.min_cluster_size, max_cluster_size=config.max_cluster_size,
                    threshold=config.threshold, prediction_class=config.pred_class,
                    clustering_connectivity=config.clustering_connectivity)

print("Dice coefficient =", dice_statistic)

# For snakemake:
prediction_dir = os.path.dirname(prediction_path)
snakemake_pattern = os.path.join(prediction_dir, ".done_dice_eval_snakemake")
with open(file=snakemake_pattern, mode="w") as f:
    print("Creating snakemake pattern", snakemake_pattern)
