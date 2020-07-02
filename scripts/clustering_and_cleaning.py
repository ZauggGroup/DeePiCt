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
parser.add_argument("-calculate_motl", "--calculate_motl", type=str)

args = parser.parse_args()
pythonpath = args.pythonpath
sys.path.append(pythonpath)

import os
from os import listdir

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from distutils.util import strtobool
from constants.dataset_tables import ModelsTableHeader, DatasetTableHeader
from file_actions.readers.tomograms import load_tomogram
from file_actions.writers.csv import build_tom_motive_list
from file_actions.writers.tomogram import write_tomogram
from networks.utils import build_prediction_output_dir
from tomogram_utils.coordinates_toolbox.clustering import get_cluster_centroids

class_number = args.class_number
model_name = args.model_name[:-4]
output_dir = args.output_dir
min_cluster_size = args.min_cluster_size
max_cluster_size = args.max_cluster_size
threshold = args.threshold
dataset_table = args.dataset_table
calculate_motl = strtobool(args.calculate_motl)
tomo_name = args.tomo_name

models_table = os.path.join(output_dir, "models")
models_table = os.path.join(models_table, "models.csv")
label_name = model_name

ModelsHeader = ModelsTableHeader()
models_df = pd.read_csv(models_table, dtype=ModelsHeader.dtype_dict)
model_df = models_df[models_df[ModelsHeader.model_name] == model_name]
print(model_df)
assert model_df.shape[0] == 1
overlap = model_df.iloc[0][ModelsHeader.overlap]
box_shape = int(model_df.iloc[0][ModelsHeader.box_size])
box_shape = [box_shape, box_shape, box_shape]

segmentation_label = model_name
semantic_names = model_df.iloc[0][ModelsHeader.semantic_classes].split(',')
semantic_class = semantic_names[class_number]

DTHeader = DatasetTableHeader()

print("Processing tomo", tomo_name)
tomo_output_dir = os.path.join(output_dir, "predictions")
tomo_output_dir = os.path.join(tomo_output_dir, model_name)
tomo_output_dir = os.path.join(tomo_output_dir, tomo_name)
tomo_output_dir = os.path.join(tomo_output_dir, semantic_class)
os.makedirs(tomo_output_dir, exist_ok=True)

# df = pd.read_csv(dataset_table)
# df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)
# tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
for file in listdir(tomo_output_dir):
    if "motl" in file:
        print("Motive list already exists:", file)
        calculate_motl = False

print(tomo_output_dir)
output_path = os.path.join(tomo_output_dir, "prediction.mrc")
assert os.path.isfile(output_path)
prediction_dataset = load_tomogram(path_to_dataset=output_path)
output_shape = prediction_dataset.shape
sigmoid = nn.Sigmoid()
prediction_dataset = sigmoid(
    torch.from_numpy(prediction_dataset).float())
prediction_dataset = 1 * (prediction_dataset > threshold).float()
prediction_dataset = prediction_dataset.numpy()
prediction_dataset.astype(int)
if np.max(prediction_dataset) > 0:
    clustering_labels, centroids_list, cluster_size_list = \
        get_cluster_centroids(dataset=prediction_dataset,
                              min_cluster_size=min_cluster_size,
                              max_cluster_size=max_cluster_size,
                              connectivity=1, compute_centroids=calculate_motl)
else:
    clustering_labels = prediction_dataset
    centroids_list = []
clusters_output_path = os.path.join(tomo_output_dir, "clusters.mrc")
write_tomogram(output_path=clusters_output_path, tomo_data=clustering_labels)
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
