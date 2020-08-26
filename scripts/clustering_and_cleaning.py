import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-pythonpath", "--pythonpath", type=str)
parser.add_argument("-tomo_name", "--tomo_name", type=str)
parser.add_argument("-config_file", "--config_file", type=str)

args = parser.parse_args()
pythonpath = args.pythonpath
sys.path.append(pythonpath)

import os
from os import listdir

import pandas as pd
import numpy as np
from file_actions.readers.tomograms import load_tomogram
from file_actions.writers.csv import build_tom_motive_list
from file_actions.writers.tomogram import write_tomogram
from tomogram_utils.coordinates_toolbox.clustering import get_cluster_centroids
from paths.pipeline_dirs import get_probability_map_path, get_post_processed_prediction_path
from constants.dataset_tables import DatasetTableHeader
from constants.config import Config

config_file = args.config_file
config = Config(user_config_file=config_file)
tomo_name = args.tomo_name
calculate_motl = True

print("Processing tomo", tomo_name)
tomo_output_dir, output_path = get_probability_map_path(config.output_dir, config.model_name, tomo_name,
                                                        config.pred_class)

for file in listdir(tomo_output_dir):
    if "motl" in file:
        print("Motive list already exists:", file)
        calculate_motl = False

assert os.path.isfile(output_path)
prediction_dataset = load_tomogram(path_to_dataset=output_path)
output_shape = prediction_dataset.shape
prediction_dataset_thr = 1 * (prediction_dataset > config.threshold)

DTHeader = DatasetTableHeader(filtering_mask=config.region_mask)
print("Intersecting mask:", DTHeader.filtering_mask)
df = pd.read_csv(config.dataset_table)
df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)
tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
masking_file = tomo_df.iloc[0][DTHeader.filtering_mask]
clusters_output_path = get_post_processed_prediction_path(output_dir=config.output_dir, model_name=config.model_name,
                                                          tomo_name=tomo_name,
                                                          semantic_class=config.pred_class)
os.makedirs(tomo_output_dir, exist_ok=True)

if str(masking_file) == "nan":
    print("No intersecting mask available of the type {} for tomo {}.".format(config.region_mask, tomo_name))
else:
    mask_indicator = load_tomogram(path_to_dataset=masking_file)
    shx, shy, shz = [np.min([shl, shp]) for shl, shp in
                     zip(mask_indicator.shape, prediction_dataset_thr.shape)]
    mask_indicator = mask_indicator[:shx, :shy, :shz]
    prediction_dataset_thr = prediction_dataset_thr[:shx, :shy, :shz]
    prediction_dataset_thr = np.array(mask_indicator, dtype=float) * np.array(prediction_dataset_thr,
                                                                              dtype=float)
if np.max(prediction_dataset_thr) > 0:
    clusters_labeled_by_size, centroids_list, cluster_size_list = \
        get_cluster_centroids(dataset=prediction_dataset_thr,
                              min_cluster_size=config.min_cluster_size,
                              max_cluster_size=config.max_cluster_size,
                              connectivity=config.clustering_connectivity)
else:
    clusters_labeled_by_size = prediction_dataset_thr
    centroids_list = []
    cluster_size_list = []
clusters_output_path = get_post_processed_prediction_path(output_dir=config.output_dir, model_name=config.model_name,
                                                          tomo_name=tomo_name, semantic_class=config.pred_class)
write_tomogram(output_path=clusters_output_path, tomo_data=clusters_labeled_by_size)
os.makedirs(tomo_output_dir, exist_ok=True)
if calculate_motl:
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
