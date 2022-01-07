import argparse
import os
from os import listdir

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import yaml

from constants.dataset_tables import ModelsTableHeader, DatasetTableHeader
from file_actions.readers.tomograms import load_tomogram
from file_actions.writers.csv import build_tom_motive_list
from file_actions.writers.tomogram import write_tomogram
from networks.utils import build_prediction_output_dir
from tomogram_utils.coordinates_toolbox.clustering import get_cluster_centroids

parser = argparse.ArgumentParser()
parser.add_argument("-config", "--config", help="yaml_file", type=str)
parser.add_argument("-set", "--set",
                    help="tomos set name to be used for training", type=int)
args = parser.parse_args()
yaml_file = args.config
config = yaml.safe_load(open(yaml_file))
tomos_set = args.set
tomo_list = config['tomos_sets'][tomos_set]['test_list']

class_number = config['prediction']['class_number']
model_name = config["model_path"][:-4]
output_dir = config["pred_output_dir"]
models_table = os.path.join(output_dir, "models")
models_table = os.path.join(models_table, "models.csv")

label_name = model_name

motl_parameters = config['clustering_parameters']
min_cluster_size = motl_parameters['min_cluster_size']
max_cluster_size = motl_parameters['max_cluster_size']
threshold = motl_parameters['threshold']
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

dataset_table = config['dataset_table']

DTHeader = DatasetTableHeader()

output_dir = os.path.join(config['pred_output_dir'], "predictions")
output_dir = os.path.join(output_dir, model_name)
for tomo_name in tomo_list:
    print("Processing tomo", tomo_name)
    tomo_output_dir = os.path.join(output_dir, tomo_name)
    tomo_output_dir = os.path.join(tomo_output_dir, semantic_class)
    os.makedirs(tomo_output_dir, exist_ok=True)

    df = pd.read_csv(dataset_table)
    df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)
    tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
    x_dim = int(tomo_df.iloc[0][DTHeader.x_dim])
    y_dim = int(tomo_df.iloc[0][DTHeader.y_dim])
    z_dim = int(tomo_df.iloc[0][DTHeader.z_dim])
    output_shape = (z_dim, y_dim, x_dim)
    calculate_motl = config["clustering_parameters"]["calculate_motl"]
    for file in listdir(tomo_output_dir):
        if "motl" in file:
            print("Motive list already exists:", file)
            calculate_motl = False
    if calculate_motl:
        output_path = os.path.join(tomo_output_dir, "prediction.mrc")
        assert os.path.isfile(output_path)
        prediction_dataset = load_tomogram(path_to_dataset=output_path)

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
                                      connectivity=1)
        else:
            clustering_labels = prediction_dataset
            centroids_list = []
        clusters_output_path = os.path.join(tomo_output_dir, "clusters.mrc")

        if not os.path.isfile(clusters_output_path):
            write_tomogram(output_path=clusters_output_path,
                              tomo_data=clustering_labels)

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
