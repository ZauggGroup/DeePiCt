import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-gpu", "--gpu", help="cuda visible devices", type=str)
parser.add_argument("-pythonpath", "--pythonpath", type=str)
parser.add_argument("-config_file", "--config_file", type=str)
parser.add_argument("-tomo_name", "--tomo_name", type=str)
parser.add_argument("-fold", "--fold", type=str, default="None")

args = parser.parse_args()
pythonpath = args.pythonpath
sys.path.append(pythonpath)

import os
import ast

import torch
import torch.nn as nn

from collections import OrderedDict
from file_actions.writers.h5 import segment_and_write
from constants.config import Config, get_model_name
from networks.io import get_device
from networks.unet import UNet3D
from paths.pipeline_dirs import testing_partition_path

gpu = args.gpu
if gpu is None:
    print("No CUDA_VISIBLE_DEVICES passed...")
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


config_file = args.config_file
config = Config(user_config_file=config_file)
tomo_name = args.tomo_name
fold = ast.literal_eval(args.fold)

model_path, model_name = get_model_name(config, fold)

snakemake_pattern = ".done_patterns/" + model_name + "." + tomo_name + ".{fold}.segmentation.done".format(fold=str(fold))

output_dir_tomo, partition_path = testing_partition_path(output_dir=config.work_dir,
                                                         tomo_name=tomo_name,
                                                         fold=fold)
box_shape = [config.box_size, config.box_size, config.box_size]

output_classes = len(config.semantic_classes)

net_conf = {'final_activation': None, 'depth': config.depth,
            'initial_features': config.initial_features, "out_channels": output_classes,
            "BN": config.batch_norm, "encoder_dropout": config.encoder_dropout,
            "decoder_dropout": config.decoder_dropout}

device = get_device()
model = UNet3D(**net_conf)
model.to(device)
checkpoint = torch.load(model_path, map_location=device)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

    substring = 'module.'
    checkpoint_tmp = OrderedDict()
    for k in checkpoint['model_state_dict']:
        new_k = substring + k if not k.startswith(substring) else k
        checkpoint_tmp[new_k] = checkpoint['model_state_dict'][k]
    checkpoint['model_state_dict'] = checkpoint_tmp
else:
    substring = 'module.'
    checkpoint_tmp = OrderedDict()
    for k in checkpoint['model_state_dict']:
        new_k = k[len(substring):] if k.startswith(substring) else k
        checkpoint_tmp[new_k] = checkpoint['model_state_dict'][k]
    checkpoint['model_state_dict'] = checkpoint_tmp

model.load_state_dict(checkpoint['model_state_dict'])
model = model.eval()

print("Segmenting tomo", tomo_name)
segment_and_write(data_path=partition_path, model=model, label_name=model_name)
print("The segmentation has finished!")

# For snakemake:
snakemake_pattern_dir = os.path.dirname(snakemake_pattern)
os.makedirs(snakemake_pattern_dir, exist_ok=True)
with open(file=snakemake_pattern, mode="w") as f:
    print("Creating snakemake pattern", snakemake_pattern)
