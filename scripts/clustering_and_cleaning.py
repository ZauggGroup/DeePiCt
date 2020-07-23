import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-pythonpath", "--pythonpath", type=str)
parser.add_argument("-tomo_name", "--tomo_name", type=str)
parser.add_argument("-class_number", "--class_number", type=int)
parser.add_argument("-model_name", "--model_name", type=str)
parser.add_argument("-output_dir", "--output_dir", type=str)
parser.add_argument("-min_cluster_size", "--min_cluster_size", type=int)
parser.add_argument("-max_cluster_size", "--max_cluster_size", type=int)
parser.add_argument("-threshold", "--threshold", type=float)
parser.add_argument("-dataset_table", "--dataset_table", type=str)
parser.add_argument("-filtering_mask", "--filtering_mask", type=str)
parser.add_argument("-segmentation_names", "--segmentation_names", nargs='+', type=str)

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

class_number = args.class_number
model_name = args.model_name[:-4]
output_dir = args.output_dir
min_cluster_size = args.min_cluster_size
max_cluster_size = args.max_cluster_size
threshold = args.threshold
dataset_table = args.dataset_table
calculate_motl = True
tomo_name = args.tomo_name
filtering_mask = args.filtering_mask

segmentation_label = model_name
semantic_names = args.segmentation_names
semantic_class = semantic_names[class_number]

DTHeader = DatasetTableHeader()

print("Processing tomo", tomo_name)
tomo_output_dir, output_path = get_probability_map_path(output_dir, model_name, tomo_name, semantic_class)

for file in listdir(tomo_output_dir):
    if "motl" in file:
        print("Motive list already exists:", file)
        calculate_motl = False

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
                              connectivity=1, compute_centroids=calculate_motl)
else:
    clusters_labeled_by_size = prediction_dataset
    centroids_list = []
clusters_output_path = get_post_processed_prediction_path(output_dir=output_dir, model_name=model_name,
                                                          tomo_name=tomo_name,
                                                          semantic_class=semantic_class)
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
