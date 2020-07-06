import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-gpu", "--gpu", help="cuda visible devices", type=str)
parser.add_argument("-pythonpath", "--pythonpath", type=str)
parser.add_argument("-dataset_table", "--dataset_table", type=str)
parser.add_argument("-output_dir", "--output_dir", type=str)
parser.add_argument("-work_dir", "--work_dir", type=str)
parser.add_argument("-model_name", "--model_name", type=str)
parser.add_argument("-test_partition", "--test_partition", type=str)
parser.add_argument("-processing_tomo", "--processing_tomo", type=str)
parser.add_argument("-tomo_name", "--tomo_name", type=str)

args = parser.parse_args()
pythonpath = args.pythonpath
sys.path.append(pythonpath)

import os

import pandas as pd
import torch
import torch.nn as nn

from constants.dataset_tables import ModelsTableHeader, DatasetTableHeader
from file_actions.writers.h5 import segment_and_write
from networks.io import get_device
from networks.unet import UNet3D, UNet
from paths.pipeline_dirs import testing_partition_path

gpu = args.gpu
if gpu is None:
    print("No CUDA_VISIBLE_DEVICES passed...")
    if torch.cuda.is_available():
        CUDA_VISIBLE_DEVICES = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

dataset_table = args.dataset_table
output_dir = args.output_dir
work_dir = args.work_dir
model_name = args.model_name[:-4]
test_partition = args.test_partition
processing_tomo = args.processing_tomo
tomo_name = args.tomo_name

models_table = os.path.join(output_dir, "models")
models_table = os.path.join(models_table, "models.csv")

# output_dir_tomo = os.path.join(output_dir, "test_partitions")
# output_dir_tomo = os.path.join(output_dir_tomo, tomo_name)
# output_dir_tomo = os.path.join(output_dir_tomo, model_name)
# partition_path = os.path.join(output_dir_tomo, test_partition + ".h5")
output_dir_tomo, partition_path = testing_partition_path(output_dir=work_dir, tomo_name=tomo_name,
                                                         model_name=model_name, partition_name=test_partition)
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
BN = model_df.iloc[0][ModelsHeader.batch_normalization].astype(bool)
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

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

checkpoint = torch.load(path_to_model, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.eval()

DTHeader = DatasetTableHeader(partition_name=test_partition)
df = pd.read_csv(dataset_table, dtype={DTHeader.tomo_name: str})

print("Segmenting tomo", tomo_name)

print("test_partition", partition_path)
segment_and_write(data_path=partition_path, model=model, label_name=label_name)
print("The segmentation has finished!")
