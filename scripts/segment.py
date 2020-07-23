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
parser.add_argument("-depth", "--depth", type=int)
parser.add_argument("-decoder_dropout", "--decoder_dropout", type=float)
parser.add_argument("-encoder_dropout", "--encoder_dropout", type=float)
parser.add_argument("-batch_size", "--batch_size", type=int)
parser.add_argument("-batch_norm", "--batch_norm", type=bool)
parser.add_argument("-initial_features", "--initial_features", type=int)
parser.add_argument("-segmentation_names", "--segmentation_names", nargs='+', type=str)
parser.add_argument("-overlap", "--overlap", type=int)
parser.add_argument("-box_shape", "--box_shape", type=int)

args = parser.parse_args()
pythonpath = args.pythonpath
sys.path.append(pythonpath)

import os

import pandas as pd
import torch
import torch.nn as nn

from collections import OrderedDict
from constants.dataset_tables import DatasetTableHeader
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

output_dir_tomo, partition_path = testing_partition_path(output_dir=work_dir, tomo_name=tomo_name,
                                                         model_name=model_name, partition_name=test_partition)
overlap = args.overlap
box_shape = args.box_shape
box_shape = [box_shape, box_shape, box_shape]
path_to_model = args.model_name #model_df.iloc[0][ModelsHeader.model_path]
depth = args.depth
decoder_dropout = args.decoder_dropout
encoder_dropout = args.encoder_dropout
batch_size = args.batch_size
BN = args.batch_norm
initial_features = args.initial_features
# output_classes = model_df.iloc[0][ModelsHeader.output_classes]
label_name = model_name
segmentation_names = args.segmentation_names
output_classes = len(segmentation_names)

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

substring = 'module.'
checkpoint_tmp = OrderedDict()
for k in checkpoint['model_state_dict']:
    new_k = k[len(substring):] if k.startswith(substring) else k
    checkpoint_tmp[new_k] = checkpoint['model_state_dict'][k]
checkpoint['model_state_dict'] = checkpoint_tmp

model.load_state_dict(checkpoint['model_state_dict'])
model = model.eval()

DTHeader = DatasetTableHeader(partition_name=test_partition)
df = pd.read_csv(dataset_table, dtype={DTHeader.tomo_name: str})

print("Segmenting tomo", tomo_name)

print("test_partition", partition_path)
segment_and_write(data_path=partition_path, model=model, label_name=label_name)
print("The segmentation has finished!")

### For snakemake:
snakemake_pattern = ".done_patterns/" + path_to_model + "." + tomo_name + ".segmentation.done"
snakemake_pattern_dir = os.path.dirname(snakemake_pattern)
os.makedirs(snakemake_pattern_dir, exist_ok=True)
with open(file=snakemake_pattern, mode="w") as f:
    print("Creating snakemake pattern", snakemake_pattern)
