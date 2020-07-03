import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-pythonpath", "--pythonpath", type=str)
parser.add_argument("-dataset_table", "--dataset_table", type=str)
parser.add_argument("-output_dir", "--output_dir", type=str)
parser.add_argument("-model_name", "--model_name", type=str)
parser.add_argument("-test_partition", "--test_partition", type=str)
parser.add_argument("-processing_tomo", "--processing_tomo", type=str)
parser.add_argument("-tomo_name", "--tomo_name", type=str)

args = parser.parse_args()
pythonpath = args.pythonpath
sys.path.append(pythonpath)
import os

import numpy as np
import pandas as pd

from constants.dataset_tables import DatasetTableHeader, ModelsTableHeader
from file_actions.readers.tomograms import load_tomogram
from tomogram_utils.volume_actions.actions import \
    partition_raw_intersecting_mask
from tomogram_utils.volume_actions.actions import partition_tomogram

dataset_table = args.dataset_table
model_name = args.model_name[:-4]
test_partition = args.test_partition
processing_tomo = args.processing_tomo
output_dir = args.output_dir
tomo_name = args.tomo_name
models_table = os.path.join(output_dir, "models")
models_table = os.path.join(models_table, "models.csv")
write_on_table = True

ModelsHeader = ModelsTableHeader()
models_df = pd.read_csv(models_table, dtype=ModelsHeader.dtype_dict)
model_df = models_df[models_df[ModelsHeader.model_name] == model_name]
overlap = model_df.iloc[0][ModelsHeader.overlap]

box_shape = int(model_df.iloc[0][ModelsHeader.box_size])

subtomogram_shape = (box_shape, box_shape, box_shape)

DTHeader = DatasetTableHeader(processing_tomo=processing_tomo,
                              partition_name=test_partition)

df = pd.read_csv(dataset_table)
df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)

print("Partitioning tomo", tomo_name)

output_dir_tomo = os.path.join(output_dir, "test_partitions")
output_dir_tomo = os.path.join(output_dir_tomo, tomo_name)
output_dir_tomo = os.path.join(output_dir_tomo, model_name)
os.makedirs(output_dir_tomo, exist_ok=True)
partition_path = os.path.join(output_dir_tomo, test_partition + ".h5")

print("output path:", partition_path)
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
                     data_dim, lamella_dim
                     in zip(dataset_shape, lamella_shape)]
    minz, miny, minx = minimum_shape

    lamella_mask = lamella_mask[:minz, :miny, :minx]
    raw_dataset = raw_dataset[:minz, :miny, :minx]

    partition_raw_intersecting_mask(dataset=raw_dataset,
                                    mask_dataset=lamella_mask,
                                    output_h5_file_path=partition_path,
                                    subtomo_shape=subtomogram_shape,
                                    overlap=overlap)

# df.loc[df[
#            DTHeader.tomo_name] == tomo_name, DTHeader.partition_name] = \
#     [partition_path]
# df.to_csv(dataset_table, index=False)
