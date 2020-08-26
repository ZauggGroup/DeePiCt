import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-pythonpath", "--pythonpath", type=str)
parser.add_argument("-dataset_table", "--dataset_table", type=str)
parser.add_argument("-pred_output_dir", "--pred_output_dir", type=str)
parser.add_argument("-work_dir", "--work_dir", type=str)
parser.add_argument("-model_path", "--model_path", type=str)
parser.add_argument("-test_partition", "--test_partition", type=str)
parser.add_argument("-processing_tomo", "--processing_tomo", type=str)
parser.add_argument("-cv_data_path", "--cv_data_path", type=str)
parser.add_argument("-fold", "--fold", type=str)
parser.add_argument("-gpu", "--gpu", help="cuda visible devices", type=str)
parser.add_argument("-class_number", "--class_number", type=int)
parser.add_argument("-calculate_motl", "--calculate_motl", type=str)
parser.add_argument("-statistics_file", "--statistics_file", type=str)
parser.add_argument("-radius", "--radius", type=int)
parser.add_argument("-min_cluster_size", "--min_cluster_size", type=int)
parser.add_argument("-max_cluster_size", "--max_cluster_size", type=int)
parser.add_argument("-threshold", "--threshold", type=float)
parser.add_argument("-region_mask", "--region_mask", type=str)

args = parser.parse_args()
pythonpath = args.pythonpath
sys.path.append(pythonpath)
import os

import numpy as np
import pandas as pd

from constants.dataset_tables import DatasetTableHeader, ModelsTableHeader
from file_actions.readers.tomograms import load_tomogram
from paths.pipeline_dirs import fold_testing_partition_path, get_probability_map_path, get_models_table_path
from tomogram_utils.volume_actions.actions import \
    partition_raw_intersecting_mask
from tomogram_utils.volume_actions.actions import partition_tomogram

import torch
import torch.nn as nn

from collections import OrderedDict
from file_actions.writers.h5 import segment_and_write
from networks.io import get_device
from networks.unet import UNet3D, UNet

from constants import h5_internal_paths
from file_actions.writers.h5 import assemble_tomo_from_subtomos

from os import listdir

from distutils.util import strtobool
from file_actions.writers.csv import build_tom_motive_list
from file_actions.writers.tomogram import write_tomogram
from tomogram_utils.coordinates_toolbox.clustering import get_cluster_centroids
from paths.pipeline_dirs import get_post_processed_prediction_path

from os.path import join

import matplotlib.pyplot as plt
import matplotlib

from file_actions.writers.csv import motl_writer
from file_actions.writers.csv import write_statistics
from networks.utils import build_prediction_output_dir
from performance.statistics_utils import pr_auc_score, \
    f1_score_calculator, precision_recall_calculator
from tomogram_utils.peak_toolbox.utils import read_motl_coordinates_and_values
from constants.statistics import write_statistics_pp

dataset_table = args.dataset_table
test_partition = args.test_partition
processing_tomo = args.processing_tomo
output_dir = args.output_dir
work_dir = args.work_dir
fold = args.fold
model_name_ext = args.model_name
model_name = model_name_ext[:-4] + "_" + fold
cv_data_path = args.cv_data_path
class_number = args.class_number
min_cluster_size = args.min_cluster_size
max_cluster_size = args.max_cluster_size
threshold = args.threshold
# calculate_motl = strtobool(args.calculate_motl)
filtering_mask = args.filtering_mask
radius = args.radius
statistics_file = args.statistics_file

models_table = os.path.join(output_dir, "models")
models_table = os.path.join(models_table, "models.csv")

cv_data = pd.read_csv(cv_data_path)
cv_data["fold"] = cv_data["fold"].apply(lambda x: str(x))
cv_data.set_index("fold", inplace=True)

tomo_evaluation_list_tmp = cv_data.loc[fold]["cv_evaluation_list"].split(" ")
tomo_evaluation_list = []
for tomo_name in tomo_evaluation_list_tmp:
    if len(tomo_name) > 0:
        tomo_evaluation_list.append(tomo_name)
print("fold", fold, "; tomo_evaluation_list:", tomo_evaluation_list)

for tomo_name in tomo_evaluation_list:

    partition_output_dir, partition_path = fold_testing_partition_path(output_dir=work_dir, tomo_name=tomo_name,
                                                                       model_name=model_name,
                                                                       partition_name=test_partition, fold=fold)
    os.makedirs(partition_output_dir, exist_ok=True)
    print("output path:", partition_path)

    if os.path.exists(partition_path):
        print("Exiting, path exists.")
    else:
        ModelsHeader = ModelsTableHeader()
        models_df = pd.read_csv(models_table, dtype=ModelsHeader.dtype_dict)
        model_df = models_df[models_df[ModelsHeader.model_name] == model_name]
        overlap = model_df.iloc[0][ModelsHeader.overlap]

        box_shape = int(model_df.iloc[0][ModelsHeader.box_size])

        subtomogram_shape = (box_shape, box_shape, box_shape)

        DTHeader = DatasetTableHeader(processing_tomo=processing_tomo)

        df = pd.read_csv(dataset_table)
        df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)

        print("Partitioning tomo", tomo_name)

        tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
        path_to_raw = tomo_df.iloc[0][DTHeader.processing_tomo]
        path_to_lamella = tomo_df.iloc[0][DTHeader.filtering_mask]
        raw_dataset = load_tomogram(path_to_dataset=path_to_raw)
        if isinstance(path_to_lamella, float):
            print("No filtering mask file available.")
            partition_tomogram(dataset=raw_dataset,
                               output_h5_file_path=partition_path,
                               subtomo_shape=subtomogram_shape,
                               overlap=overlap)
        else:
            path_to_lamella = tomo_df.iloc[0][DTHeader.filtering_mask]
            lamella_mask = load_tomogram(path_to_dataset=path_to_lamella)

            lamella_shape = lamella_mask.shape
            dataset_shape = raw_dataset.shape

            minimum_shape = [np.min([data_dim, lamella_dim]) for
                             data_dim, lamella_dim in zip(dataset_shape, lamella_shape)]
            minz, miny, minx = minimum_shape

            lamella_mask = lamella_mask[:minz, :miny, :minx]
            raw_dataset = raw_dataset[:minz, :miny, :minx]

            partition_raw_intersecting_mask(dataset=raw_dataset,
                                            mask_dataset=lamella_mask,
                                            output_h5_file_path=partition_path,
                                            subtomo_shape=subtomogram_shape,
                                            overlap=overlap)

gpu = args.gpu
if gpu is None:
    print("No CUDA_VISIBLE_DEVICES passed...")
    if torch.cuda.is_available():
        CUDA_VISIBLE_DEVICES = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

ModelsHeader = ModelsTableHeader()
models_df = pd.read_csv(models_table, dtype=ModelsHeader.dtype_dict)
model_df = models_df[models_df[ModelsHeader.model_name] == model_name]
print(model_df)
assert model_df.shape[0] == 1
overlap = model_df.iloc[0][ModelsHeader.overlap]
box_shape = int(model_df.iloc[0][ModelsHeader.box_size])
box_shape = [box_shape, box_shape, box_shape]
path_to_model = model_df.iloc[0][ModelsHeader.model_path]
initial_features = model_df.iloc[0][ModelsHeader.initial_features]
depth = model_df.iloc[0][ModelsHeader.depth]
output_classes = model_df.iloc[0][ModelsHeader.output_classes]
BN = model_df.iloc[0][ModelsHeader.batch_norm].astype(bool)
encoder_dropout = model_df.iloc[0][ModelsHeader.encoder_dropout]
decoder_dropout = model_df.iloc[0][ModelsHeader.decoder_dropout]
label_name = model_name
net_conf = {'final_activation': None, 'depth': depth,
            'initial_features': initial_features, "out_channels": output_classes,
            "BN": BN, "encoder_dropout": encoder_dropout,
            "decoder_dropout": decoder_dropout}
device = get_device()
model = UNet3D(**net_conf)
# model = UNet(**net_conf)
model.to(device)
checkpoint = torch.load(path_to_model, map_location=device)

if torch.cuda.device_count() > 1:
    print("actually it only works when the number of gpus used for"
          " testing and for training are the same")
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

    substring = 'module.'
    checkpoint_tmp = OrderedDict()
    for k in checkpoint['model_state_dict']:
        new_k = substring + k if not k.startswith(substring) else k
        checkpoint_tmp[new_k] = checkpoint['model_state_dict'][k]
    checkpoint['model_state_dict'] = checkpoint_tmp
else:
    substring = 'module.'
    checkpoint_tmp = OrderedDict()
    for k in checkpoint['model_state_dict']:
        new_k = k[len(substring):] if k.startswith(substring) else k
        checkpoint_tmp[new_k] = checkpoint['model_state_dict'][k]
    checkpoint['model_state_dict'] = checkpoint_tmp

model.load_state_dict(checkpoint['model_state_dict'])
model = model.eval()

for tomo_name in tomo_evaluation_list:
    output_dir_tomo, partition_path = fold_testing_partition_path(output_dir=work_dir, tomo_name=tomo_name,
                                                                  model_name=model_name, partition_name=test_partition,
                                                                  fold=fold)
    DTHeader = DatasetTableHeader(partition_name=test_partition)
    df = pd.read_csv(dataset_table, dtype={DTHeader.tomo_name: str})

    print("Segmenting tomo", tomo_name)

    print("test_partition", partition_path)
    segment_and_write(data_path=partition_path, model=model, label_name=label_name)
    print("The segmentation has finished!")

for tomo_name in tomo_evaluation_list:
    output_dir_tomo, data_partition = fold_testing_partition_path(output_dir=work_dir, tomo_name=tomo_name,
                                                                  model_name=model_name, partition_name=test_partition,
                                                                  fold=fold)

    segmentation_label = model_name
    models_table = get_models_table_path(output_dir)
    ModelsHeader = ModelsTableHeader()
    models_df = pd.read_csv(models_table,
                            dtype={ModelsHeader.model_name: str,
                                   ModelsHeader.segmentation_names: str})

    model_df = models_df[models_df[ModelsHeader.model_name] == model_name]
    assert model_df.shape[0] < 2, "several models have the same name in models.csv"
    assert model_df.shape[0] > 0, "no model with that name in models.csv"

    overlap = model_df.iloc[0][ModelsHeader.overlap]
    box_shape = int(model_df.iloc[0][ModelsHeader.box_size])
    box_shape = [box_shape, box_shape, box_shape]
    semantic_names = model_df.iloc[0]['semantic_classes'].split(',')
    semantic_class = semantic_names[class_number]

    tomo_output_dir, output_path = get_probability_map_path(output_dir, model_name, tomo_name, semantic_class)
    os.makedirs(tomo_output_dir, exist_ok=True)

    DTHeader = DatasetTableHeader(processing_tomo=processing_tomo)
    df = pd.read_csv(dataset_table)
    df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)
    tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
    tomo_path = tomo_df.iloc[0][DTHeader.processing_tomo]
    tomo = load_tomogram(path_to_dataset=tomo_path)
    output_shape = tomo.shape
    del tomo
    subtomos_internal_path = os.path.join(
        h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS,
        segmentation_label)
    assemble_tomo_from_subtomos(
        output_path=output_path,
        partition_file_path=data_partition,
        output_shape=output_shape,
        subtomo_shape=box_shape,
        subtomos_internal_path=subtomos_internal_path,
        class_number=class_number, overlap=overlap,
        reconstruction_type="prediction",
        final_activation='sigmoid')

    print("Assembling prediction has finalized.")

####### clustering and cleaning


ModelsHeader = ModelsTableHeader()
models_df = pd.read_csv(models_table, dtype=ModelsHeader.dtype_dict)
model_df = models_df[models_df[ModelsHeader.model_name] == model_name]
print(model_df)
assert model_df.shape[0] == 1
overlap = model_df.iloc[0][ModelsHeader.overlap]
box_shape = int(model_df.iloc[0][ModelsHeader.box_size])
box_shape = [box_shape, box_shape, box_shape]

segmentation_label = model_name
semantic_names = model_df.iloc[0][ModelsHeader.segmentation_names].split(',')
semantic_class = semantic_names[class_number]

DTHeader = DatasetTableHeader()
for tomo_name in tomo_evaluation_list:
    print("Processing tomo", tomo_name)
    tomo_output_dir, output_path = get_probability_map_path(args.output_dir, model_name, tomo_name, semantic_class)

    # for file in listdir(tomo_output_dir):
    #     if "motl" in file:
    #         print("Motive list already exists:", file)
    #         calculate_motl = False

    assert os.path.isfile(output_path)
    prediction_dataset = load_tomogram(path_to_dataset=output_path)
    output_shape = prediction_dataset.shape
    prediction_dataset = 1 * (prediction_dataset > threshold)

    DTHeader = DatasetTableHeader(filtering_mask=filtering_mask)
    print("filtering mask:", DTHeader.filtering_mask)
    df = pd.read_csv(dataset_table)
    df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)
    tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
    masking_file = tomo_df.iloc[0][DTHeader.filtering_mask]

    if str(masking_file) == "nan":
        print("No filtering mask...")
    else:
        mask_indicator = load_tomogram(path_to_dataset=masking_file)
        shx, shy, shz = [np.min([shl, shp]) for shl, shp in
                         zip(mask_indicator.shape, prediction_dataset.shape)]
        mask_indicator = mask_indicator[:shx, :shy, :shz]
        prediction_dataset = prediction_dataset[:shx, :shy, :shz]
        prediction_dataset = np.array(mask_indicator, dtype=float) * np.array(prediction_dataset,
                                                                              dtype=float)
    if np.max(prediction_dataset) > 0:
        clusters_labeled_by_size, centroids_list, cluster_size_list = \
            get_cluster_centroids(dataset=prediction_dataset,
                                  min_cluster_size=min_cluster_size,
                                  max_cluster_size=max_cluster_size,
                                  connectivity=3)
    else:
        clusters_labeled_by_size = prediction_dataset
        centroids_list = []
    clusters_output_path = get_post_processed_prediction_path(output_dir=output_dir, model_name=model_name,
                                                              tomo_name=tomo_name,
                                                              semantic_class=semantic_class)
    write_tomogram(output_path=clusters_output_path, tomo_data=clusters_labeled_by_size)
    os.makedirs(tomo_output_dir, exist_ok=True)

    motl_name = "motl_" + str(len(centroids_list)) + ".csv"
    motl_file_name = os.path.join(tomo_output_dir, motl_name)

    if len(centroids_list) > 0:
        motive_list_df = build_tom_motive_list(
            list_of_peak_coordinates=centroids_list,
            list_of_peak_scores=cluster_size_list, in_tom_format=False)
        motive_list_df.to_csv(motl_file_name, index=False, header=False)
        print("Motive list saved in", motl_file_name)
    else:
        print("Saving empty list!")
        motive_list_df = pd.DataFrame({})
        motive_list_df.to_csv(motl_file_name, index=False, header=False)

segmentation_label = model_name

ModelsHeader = ModelsTableHeader()
models_df = pd.read_csv(models_table,
                        dtype={ModelsHeader.model_name: str,
                               ModelsHeader.segmentation_names: str})

model_df = models_df[models_df[ModelsHeader.model_name] == model_name]
print(model_df)
assert model_df.shape[0] == 1
overlap = model_df.iloc[0][ModelsHeader.overlap]
box_shape = int(model_df.iloc[0][ModelsHeader.box_size])
box_shape = [box_shape, box_shape, box_shape]
semantic_classes = model_df.iloc[0]['semantic_classes'].split(',')
semantic_class = semantic_classes[class_number]

DTHeader = DatasetTableHeader(semantic_classes=semantic_classes)
df = pd.read_csv(dataset_table)
df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)
for tomo_name in tomo_evaluation_list:
    print("Processing tomo", tomo_name, "fold", fold)
    output_dir_tmp = os.path.join(output_dir, "predictions")
    tomo_output_dir = build_prediction_output_dir(base_output_dir=output_dir_tmp,
                                                  label_name="",
                                                  model_name=model_name,
                                                  tomo_name=tomo_name,
                                                  semantic_class=semantic_class)
    os.makedirs(tomo_output_dir, exist_ok=True)
    print("tomo_output_dir:", tomo_output_dir)
    motl_in_dir = [file for file in os.listdir(tomo_output_dir) if 'motl' == file[:4]]
    assert len(motl_in_dir) == 1, "only one motive list can be filtered."
    path_to_motl_predicted = os.path.join(tomo_output_dir, motl_in_dir[0])

    tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
    path_to_motl_true = tomo_df.iloc[0][DTHeader.clean_motls[class_number]]

    subtomo_shape = tuple([sh - overlap for sh in box_shape])

    class_name = semantic_classes[class_number]

    figures_dir = os.path.join(tomo_output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    predicted_values, predicted_coordinates = read_motl_coordinates_and_values(
        path_to_motl=path_to_motl_predicted)

    true_values, true_coordinates = read_motl_coordinates_and_values(
        path_to_motl=path_to_motl_true)
    unique_peaks_number = len(predicted_values)

    predicted_coordinates = np.array(predicted_coordinates)

    prec, recall, tp_true, tp_pred, fp_pred, tp_pred_scores, fp_pred_scores, *_ = \
        precision_recall_calculator(
            predicted_coordinates=predicted_coordinates,
            value_predicted=predicted_values,
            true_coordinates=true_coordinates,
            radius=radius)

    F1_score = f1_score_calculator(prec, recall)

    if len(F1_score) > 0:
        max_F1 = np.max(F1_score)
        optimal_peak_number = np.min(np.where(F1_score == max_F1)[0])
    else:
        max_F1 = 0
        optimal_peak_number = np.nan

    auPRC = pr_auc_score(precision=prec, recall=recall)
    print("auPRC = ", auPRC, "and max_F1 = ", max_F1)

    tomo_output_dir = os.path.join(tomo_output_dir, "pr_radius_" + str(radius))
    path_to_detected_predicted = join(tomo_output_dir, "detected")
    path_to_undetected_predicted = join(tomo_output_dir, "undetected")

    os.makedirs(path_to_detected_predicted, exist_ok=True)
    os.makedirs(path_to_undetected_predicted, exist_ok=True)

    motl_writer(path_to_output_folder=path_to_detected_predicted,
                list_of_peak_coords=tp_pred,
                list_of_peak_scores=tp_pred_scores,
                in_tom_format=True)
    motl_writer(path_to_output_folder=path_to_undetected_predicted,
                list_of_peak_coords=fp_pred,
                list_of_peak_scores=fp_pred_scores,
                in_tom_format=True)
    matplotlib.use('Agg')
    plt.figure(1)
    plt.hist(predicted_values, bins=45, label="predicted")
    plt.xlabel("score value")
    plt.ylabel("frequency")
    plt.title(str(len(predicted_values)) + " peaks")
    plt.legend()
    plt.gcf()
    fig_name = join(figures_dir, "histogram_predicted.png")
    plt.savefig(fname=fig_name, format="png")

    plt.figure(2)
    plt.hist(tp_pred_scores, bins=45, label="true positives", fc=(0, 0, 1, 0.5))
    plt.hist(fp_pred_scores, bins=45, label="false positives", fc=(1, 0, 0, 0.5))
    plt.xlabel("score value")
    plt.ylabel("frequency")
    plt.title(str(len(fp_pred)) + " peaks")
    plt.legend()
    plt.gcf()
    fig_name = join(figures_dir, "histogram-detected-undetected.png")
    plt.savefig(fname=fig_name, format="png")

    plt.figure(3)
    pr_legend_str = "auPRC = " + str(round(auPRC, 4))
    f1_legend_str = "(max_F1, best_peaks) = ({}, {})".format(round(max_F1, 4), optimal_peak_number)
    title_str = str(len(predicted_coordinates)) + " peaks"
    plt.plot(F1_score, label=f1_legend_str)
    plt.xlabel("number of peaks")
    plt.ylabel("F1 score")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title(title_str)
    plt.legend()
    plt.gcf()
    fig_name = join(figures_dir, "f1_score_" + title_str + ".png")
    plt.savefig(fname=fig_name, format="png")

    plt.figure(4)
    plt.plot(recall, prec, label=pr_legend_str)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title(title_str)
    plt.legend()
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.gcf()
    fig_name = join(figures_dir, "pr_" + title_str + ".png")
    plt.savefig(fname=fig_name, format="png")
    plt.close()

    print("statistics_file", args.statistics_file)
    if args.statistics_file != "None":
        statistics_label = segmentation_label + "_pr_radius_" + str(radius)

        write_statistics(statistics_file=args.statistics_file,
                         statistics_label="auPRC_" + statistics_label,
                         tomo_name=tomo_name,
                         stat_measure=round(auPRC, 4))

        write_statistics(statistics_file=args.statistics_file,
                         statistics_label="F1_" + statistics_label,
                         tomo_name=tomo_name,
                         stat_measure=round(max_F1, 4))

        if os.path.dirname(statistics_file) == "":
            statistics_file = "pp_statistics.csv"
            write_statistics_pp(statistics_file=statistics_file, tomo_name=tomo_name, model_name=model_name,
                                model_parameters=models_table, statistic_variable="auPRC",
                                statistic_value=round(auPRC, 4),
                                pr_radius=radius, min_cluster_size=min_cluster_size,
                                max_cluster_size=max_cluster_size, threshold=threshold, prediction_class=semantic_class)
            write_statistics_pp(statistics_file=statistics_file, tomo_name=tomo_name, model_name=model_name,
                                model_parameters=models_table, statistic_variable="maxF1",
                                statistic_value=round(max_F1, 4), pr_radius=radius,
                                min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size,
                                threshold=threshold, prediction_class=semantic_class)
        else:
            statistics_file = os.path.dirname(statistics_file) + "/pp_statistics.csv"
            write_statistics_pp(statistics_file=statistics_file, tomo_name=tomo_name, model_name=model_name,
                                model_parameters=models_table, statistic_variable="auPRC",
                                statistic_value=round(auPRC, 4),
                                pr_radius=radius, min_cluster_size=min_cluster_size,
                                max_cluster_size=max_cluster_size, threshold=threshold, prediction_class=semantic_class)
            write_statistics_pp(statistics_file=statistics_file, tomo_name=tomo_name, model_name=model_name,
                                model_parameters=models_table, statistic_variable="maxF1",
                                statistic_value=round(max_F1, 4), pr_radius=radius,
                                min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size,
                                threshold=threshold, prediction_class=semantic_class)

### For snakemake:
snakemake_pattern = os.path.join(".done_patterns/" + semantic_class, os.path.basename(path_to_model))
os.makedirs(os.path.dirname(snakemake_pattern), exist_ok=True)
snakemake_pattern = snakemake_pattern + ".done_pp_cv_snakemake"
with open(file=snakemake_pattern, mode="w") as f:
    print("Creating snakemake pattern", snakemake_pattern)
