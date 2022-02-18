import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-pythonpath", "--pythonpath", type=str)
parser.add_argument("-tomo_name", "--tomo_name", type=str)
parser.add_argument("-fold", "--fold", type=str, default="None")
parser.add_argument("-config_file", "--config_file", help="yaml_file", type=str)

args = parser.parse_args()
pythonpath = args.pythonpath
sys.path.append(pythonpath)

import os
import ast

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
from constants.config import get_model_name
from networks.io import get_device

config_file = args.config_file
config = Config(user_config_file=config_file)
tomo_name = args.tomo_name
fold = ast.literal_eval(args.fold)

model_path, model_name = get_model_name(config, fold)
snakemake_pattern = config.output_dir + "/predictions/" + model_name + "/" + tomo_name + "/" + \
                    config.pred_class + "/.{fold}.done_dice_eval_snakemake".format(fold=str(fold))
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
    print("model_name", model_name)
    prediction_path = get_post_processed_prediction_path(output_dir=config.output_dir,
                                                         model_name=model_name,
                                                         tomo_name=tomo_name,
                                                         semantic_class=config.pred_class)
    print(prediction_path)
    assert os.path.isfile(prediction_path), "The prediction file does not exist!"

    DTHeader = DatasetTableHeader(semantic_classes=config.semantic_classes, filtering_mask=config.region_mask)
    df = pd.read_csv(config.dataset_table)
    df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)
    clean_mask_name = DTHeader.masks_names[config.pred_class_number]

    tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
    target_path = tomo_df.iloc[0][clean_mask_name]
    prediction = load_tomogram(path_to_dataset=prediction_path)

    contact_mode = config.contact_mode
    if contact_mode == "intersection":
        lamella_file = tomo_df.iloc[0][DTHeader.filtering_mask]

        if str(lamella_file) == "nan":
            prediction = load_tomogram(prediction_path)
        else:
            lamella_indicator = load_tomogram(path_to_dataset=lamella_file)
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

    statistics_file = os.path.join(config.output_dir, "pp_statistics.csv")

    device = get_device()
    checkpoint = torch.load(model_path, map_location=device)
    model_descriptor = checkpoint["model_descriptor"]
    # model_descriptor = model_descriptor_from_config(config)
    # model_descriptor.fold = fold
    print(statistics_file)
    write_statistics_pp(statistics_file=statistics_file, tomo_name=tomo_name, model_descriptor=model_descriptor,
                        statistic_variable="dice",
                        statistic_value=round(dice_statistic, 4), pr_radius=config.pr_tolerance_radius,
                        min_cluster_size=config.min_cluster_size, max_cluster_size=config.max_cluster_size,
                        threshold=config.threshold, prediction_class=config.pred_class,
                        clustering_connectivity=config.clustering_connectivity, processing_tomo=config.processing_tomo,
                        region_mask=config.region_mask)

    print("Dice coefficient =", dice_statistic)

# For snakemake:
snakemake_pattern_dir = os.path.dirname(snakemake_pattern)
os.makedirs(snakemake_pattern_dir, exist_ok=True)
with open(file=snakemake_pattern, mode="w") as f:
    print("Creating snakemake pattern", snakemake_pattern)
