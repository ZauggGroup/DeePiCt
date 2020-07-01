import argparse
import os
from os.path import join

import pandas as pd
import yaml

from constants import h5_internal_paths
from constants.dataset_tables import ModelsTableHeader, DatasetTableHeader
from file_actions.writers.h5 import \
    assemble_tomo_from_subtomos
from networks.utils import build_prediction_output_dir

parser = argparse.ArgumentParser()
parser.add_argument("-yaml_file", "--yaml_file", help="yaml_file", type=str)
parser.add_argument("-tomos_set", "--tomos_set",
                    help="tomos set name to be used for training", type=int)
args = parser.parse_args()
yaml_file = args.yaml_file
config = yaml.safe_load(open(yaml_file))
tomos_set = args.tomos_set
tomo_list = config['tomos_sets'][tomos_set]['test_list']
# unet_hyperparameters = config['unet_hyperparameters']

output_dir = config["output_dir"]
models_table = os.path.join(output_dir, "models")
models_table = os.path.join(models_table, "models.csv")
model_name = config["model_name"][:-4]
segmentation_label = model_name
class_number = config['prediction']['class_number']

ModelsHeader = ModelsTableHeader()
models_df = pd.read_csv(models_table,
                        dtype={ModelsHeader.model_name: str,
                               ModelsHeader.semantic_classes: str})

model_df = models_df[models_df[ModelsHeader.model_name] == model_name]
print(model_df)
assert model_df.shape[0] == 1
overlap = model_df.iloc[0][ModelsHeader.overlap]
box_shape = int(model_df.iloc[0][ModelsHeader.box_size])
box_shape = [box_shape, box_shape, box_shape]

semantic_names = model_df.iloc[0]['segmentation_names'].split(',')
semantic_class = semantic_names[class_number]
dataset_table = config['dataset_table']
test_partition = config["prediction"]['partition_name']

DTHeader = DatasetTableHeader(partition_name=test_partition)
df = pd.read_csv(dataset_table)
df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)

output_dir = os.path.join(output_dir, "predictions")
output_dir = os.path.join(output_dir, model_name)
for tomo_name in tomo_list:
    tomo_output_dir = os.path.join(output_dir, tomo_name)
    tomo_output_dir = os.path.join(tomo_output_dir, semantic_class)

    os.makedirs(tomo_output_dir, exist_ok=True)
    output_path = os.path.join(tomo_output_dir, "prediction.mrc")
    if os.path.isfile(output_path):
        print("The prediction file exists")
    else:
        tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
        x_dim = int(tomo_df.iloc[0][DTHeader.x_dim])
        y_dim = int(tomo_df.iloc[0][DTHeader.y_dim])
        z_dim = int(tomo_df.iloc[0][DTHeader.z_dim])
        output_shape = (z_dim, y_dim, x_dim)
        data_partition = tomo_df.iloc[0][DTHeader.partition_name]
        subtomos_internal_path = join(
            h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS,
            segmentation_label)
        assemble_tomo_from_subtomos(
            output_path=output_path,
            partition_file_path=data_partition,
            output_shape=output_shape,
            subtomo_shape=box_shape,
            subtomos_internal_path=subtomos_internal_path,
            class_number=class_number, overlap=overlap,
            reconstruction_type="prediction")
