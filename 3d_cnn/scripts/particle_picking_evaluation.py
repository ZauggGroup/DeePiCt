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
import torch
from os.path import join

import numpy as np
import pandas as pd

from constants.dataset_tables import DatasetTableHeader
from file_actions.writers.csv import motl_writer
from networks.utils import build_prediction_output_dir
from performance.statistics_utils import pr_auc_score, \
    f1_score_calculator, precision_recall_calculator, get_max_F1
from tomogram_utils.peak_toolbox.utils import read_motl_coordinates_and_values
from constants.statistics import write_statistics_pp
from constants.config import Config, model_descriptor_from_config
from plotting.statistics import generate_performance_plots
from constants.config import get_model_name
from networks.io import get_device

config_file = args.config_file
config = Config(user_config_file=config_file)
tomo_name = args.tomo_name
fold = ast.literal_eval(args.fold)

model_path, model_name = get_model_name(config, fold)
print("model_name", model_name)
snakemake_pattern = config.output_dir + "/predictions/" + model_name + "/" + tomo_name + "/" + config.pred_class + \
                    "/pr_radius_" + str(config.pr_tolerance_radius) + \
                    "/detected/.{fold}.done_pp_snakemake".format(fold=str(fold))
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
    DTHeader = DatasetTableHeader(semantic_classes=config.semantic_classes)
    df = pd.read_csv(config.dataset_table)
    df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)

    print("Processing tomo", tomo_name)
    pred_output_dir = os.path.join(config.output_dir, "predictions")
    tomo_output_dir = build_prediction_output_dir(base_output_dir=pred_output_dir,
                                                  label_name="", model_name=model_name,
                                                  tomo_name=tomo_name, semantic_class=config.pred_class)
    print(tomo_output_dir)
    os.makedirs(tomo_output_dir, exist_ok=True)
    motl_in_dir = [file for file in os.listdir(tomo_output_dir) if 'motl_' == file[:5]]
    assert len(motl_in_dir) == 1, "only one motive list can be filtered; we got {}.".format(len(motl_in_dir))
    path_to_motl_predicted = os.path.join(tomo_output_dir, motl_in_dir[0])

    tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
    path_to_motl_true = tomo_df.iloc[0][DTHeader.clean_motls[config.pred_class_number]]

    figures_dir = os.path.join(tomo_output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    predicted_values, predicted_coordinates = read_motl_coordinates_and_values(
        path_to_motl=path_to_motl_predicted)

    true_values, true_coordinates = read_motl_coordinates_and_values(
        path_to_motl=path_to_motl_true)
    unique_peaks_number = len(predicted_values)

    predicted_coordinates = np.array(predicted_coordinates)
    prec, recall, tp_true, tp_pred, fp_pred, tp_pred_scores, fp_pred_scores, fn, *_ = \
        precision_recall_calculator(
            predicted_coordinates=predicted_coordinates,
            value_predicted=predicted_values,
            true_coordinates=true_coordinates,
            radius=config.pr_tolerance_radius)

    F1_score = f1_score_calculator(prec, recall)
    auPRC = pr_auc_score(precision=prec, recall=recall)
    max_F1, optimal_peak_number = get_max_F1(F1_score)

    print("auPRC = ", auPRC, "; max_F1 = ", max_F1, "; final F1 = ", F1_score[-1])

    tomo_output_dir = os.path.join(tomo_output_dir, "pr_radius_" + str(config.pr_tolerance_radius))
    path_to_detected = join(tomo_output_dir, "detected")
    path_to_detected_true = join(path_to_detected, "in_true_motl")
    path_to_detected_predicted = join(path_to_detected, "in_pred_motl")
    path_to_undetected_predicted = join(tomo_output_dir, "undetected")

    os.makedirs(path_to_detected_predicted, exist_ok=True)
    os.makedirs(path_to_detected_true, exist_ok=True)
    os.makedirs(path_to_undetected_predicted, exist_ok=True)

    motl_writer(path_to_output_folder=path_to_detected_predicted,
                list_of_peak_coords=tp_pred,
                list_of_peak_scores=tp_pred_scores,
                in_tom_format=True)
    motl_writer(path_to_output_folder=path_to_detected_true,
                list_of_peak_coords=tp_true,
                list_of_peak_scores=[1 for n in tp_true],
                in_tom_format=True)
    motl_writer(path_to_output_folder=path_to_undetected_predicted,
                list_of_peak_coords=fp_pred,
                list_of_peak_scores=fp_pred_scores,
                in_tom_format=True)

    # save performance figures
    generate_performance_plots(recall=recall, prec=prec, F1_score=F1_score, predicted_values=predicted_values,
                               tp_pred_scores=tp_pred_scores, fp_pred_scores=fp_pred_scores, figures_dir=figures_dir)

    statistics_file = os.path.join(config.output_dir, "pp_statistics.csv")

    device = get_device()
    checkpoint = torch.load(model_path, map_location=device)

    if 'model_descriptor' in checkpoint.keys():
        model_descriptor = checkpoint["model_descriptor"]
    else:
        print("WARNING: model without model descriptor")
        model_descriptor = model_descriptor_from_config(config)
        checkpoint["model_descriptor"] = model_descriptor
        torch.save({
            'model_descriptor': model_descriptor,
            'epoch': checkpoint['epoch'],
            'model_state_dict': checkpoint['model_state_dict'],
            'optimizer_state_dict': checkpoint['optimizer_state_dict'],
            'loss': checkpoint['loss'],
        }, model_path)

    print(statistics_file)
    write_statistics_pp(statistics_file=statistics_file, tomo_name=tomo_name, model_descriptor=model_descriptor,
                        statistic_variable="auPRC",
                        statistic_value=round(auPRC, 4), pr_radius=config.pr_tolerance_radius,
                        min_cluster_size=config.min_cluster_size, max_cluster_size=config.max_cluster_size,
                        threshold=config.threshold, prediction_class=config.pred_class,
                        clustering_connectivity=config.clustering_connectivity, processing_tomo=config.processing_tomo,
                        region_mask=config.region_mask)

    write_statistics_pp(statistics_file=statistics_file, tomo_name=tomo_name, model_descriptor=model_descriptor,
                        statistic_variable="max_F1",
                        statistic_value=round(max_F1, 4), pr_radius=config.pr_tolerance_radius,
                        min_cluster_size=config.min_cluster_size, max_cluster_size=config.max_cluster_size,
                        threshold=config.threshold, prediction_class=config.pred_class,
                        clustering_connectivity=config.clustering_connectivity, processing_tomo=config.processing_tomo,
                        region_mask=config.region_mask)

    write_statistics_pp(statistics_file=statistics_file, tomo_name=tomo_name, model_descriptor=model_descriptor,
                        statistic_variable="F1",
                        statistic_value=round(F1_score[-1], 4), pr_radius=config.pr_tolerance_radius,
                        min_cluster_size=config.min_cluster_size, max_cluster_size=config.max_cluster_size,
                        threshold=config.threshold, prediction_class=config.pred_class,
                        clustering_connectivity=config.clustering_connectivity, processing_tomo=config.processing_tomo,
                        region_mask=config.region_mask)

# For snakemake:
snakemake_pattern_dir = os.path.dirname(snakemake_pattern)
os.makedirs(snakemake_pattern_dir, exist_ok=True)
with open(file=snakemake_pattern, mode="w") as f:
    print("Creating snakemake pattern", snakemake_pattern)
