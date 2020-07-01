import os
import argparse

import pandas as pd
import torch
import torch.nn as nn
import yaml

from constants.dataset_tables import ModelsTableHeader, DatasetTableHeader
from file_actions.writers.h5 import segment_and_write
from networks.io import get_device
from networks.unet import UNet3D, UNet

parser = argparse.ArgumentParser()
parser.add_argument("-gpu", "--gpu", help="cuda visible devices", type=str)
parser.add_argument("-yaml_file", "--yaml_file", help="yaml_file", type=str)
parser.add_argument("-tomos_set", "--tomos_set",
                    help="tomos set name to be used for training", type=int)

args = parser.parse_args()
gpu = args.gpu
if gpu is None:
    print("No CUDA_VISIBLE_DEVICES passed...")
    if torch.cuda.is_available():
        CUDA_VISIBLE_DEVICES = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
yaml_file = args.yaml_file
config = yaml.safe_load(open(yaml_file))
tomos_set = args.tomos_set
dataset_table = config['dataset_table']
tomo_list = config['tomos_sets'][tomos_set]['test_list']
output_dir = config["output_dir"]
models_table = os.path.join(output_dir, "models")
models_table = os.path.join(models_table, "models.csv")
model_name = config["model_name"][:-4]
test_partition = config["prediction"]['partition_name']
processing_tomo = config["prediction"]['processing_tomo']

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
for tomo_name in tomo_list:
    print("Segmenting tomo", tomo_name)
    tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
    data_path = tomo_df.iloc[0][DTHeader.partition_name]
    print("test_partition", data_path)
    segment_and_write(data_path=data_path, model=model, label_name=label_name)
    print("The segmentation has finished!")
