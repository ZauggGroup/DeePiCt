import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-pythonpath", "--pythonpath", type=str)
parser.add_argument("-dataset_table", "--dataset_table", type=str)
parser.add_argument("-output_dir", "--output_dir", type=str)
parser.add_argument("-work_dir", "--work_dir", type=str)
parser.add_argument("-model_name", "--model_name", type=str)
parser.add_argument("-test_partition", "--test_partition", type=str)
parser.add_argument("-tomo_name", "--tomo_name", type=str)
parser.add_argument("-class_number", "--class_number", type=int)

args = parser.parse_args()
pythonpath = args.pythonpath
sys.path.append(pythonpath)

import os

import pandas as pd

from constants import h5_internal_paths
from constants.dataset_tables import ModelsTableHeader, DatasetTableHeader
from file_actions.writers.h5 import \
    assemble_tomo_from_subtomos
from paths.pipeline_dirs import testing_partition_path


tomo_name = args.tomo_name
output_dir = args.output_dir
model_name = args.model_name[:-4]
class_number = args.class_number
dataset_table = args.dataset_table
test_partition = args.test_partition
work_dir = args.work_dir
# output_dir_tomo = os.path.join(output_dir, "test_partitions")
# output_dir_tomo = os.path.join(output_dir_tomo, tomo_name)
# data_partition = os.path.join(output_dir_tomo, test_partition + ".h5")
output_dir_tomo, data_partition = testing_partition_path(output_dir=work_dir, tomo_name=tomo_name,
                                                         model_name=model_name, partition_name=test_partition)


segmentation_label = model_name
models_table = os.path.join(output_dir, "models")
models_table = os.path.join(models_table, "models.csv")

ModelsHeader = ModelsTableHeader()
models_df = pd.read_csv(models_table,
                        dtype={ModelsHeader.model_name: str,
                               ModelsHeader.semantic_classes: str})

model_df = models_df[models_df[ModelsHeader.model_name] == model_name]
assert model_df.shape[0] < 2, "several models have the same name in models.csv"
assert model_df.shape[0] > 0, "no model with that name in models.csv"

overlap = model_df.iloc[0][ModelsHeader.overlap]
box_shape = int(model_df.iloc[0][ModelsHeader.box_size])
box_shape = [box_shape, box_shape, box_shape]
semantic_names = model_df.iloc[0]['segmentation_names'].split(',')
semantic_class = semantic_names[class_number]

tomo_output_dir = os.path.join(output_dir, "predictions")
tomo_output_dir = os.path.join(tomo_output_dir, model_name)
tomo_output_dir = os.path.join(tomo_output_dir, tomo_name)
tomo_output_dir = os.path.join(tomo_output_dir, semantic_class)
os.makedirs(tomo_output_dir, exist_ok=True)
output_path = os.path.join(tomo_output_dir, "prediction.mrc")

DTHeader = DatasetTableHeader()
df = pd.read_csv(dataset_table)
df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)
tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
x_dim = int(tomo_df.iloc[0][DTHeader.x_dim])
y_dim = int(tomo_df.iloc[0][DTHeader.y_dim])
z_dim = int(tomo_df.iloc[0][DTHeader.z_dim])
output_shape = (z_dim, y_dim, x_dim)
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
    reconstruction_type="prediction")

print("Assembling prediction has finalized.")
