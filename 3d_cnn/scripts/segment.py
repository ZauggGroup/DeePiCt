import argparse
import sys

import numpy as np

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
import warnings
import pandas as pd

from collections import OrderedDict
from file_actions.writers.h5 import segment_and_write
from constants.config import Config, get_model_name, model_descriptor_from_config
from networks.io import get_device
from networks.unet import UNet3D
from constants.dataset_tables import DatasetTableHeader
from file_actions.readers.tomograms import load_tomogram
from paths.pipeline_dirs import testing_partition_path
from networks.utils import get_training_testing_lists

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

snakemake_pattern = ".done_patterns/" + model_name + "." + tomo_name + ".{fold}.segmentation.done".format(
    fold=str(fold))

if isinstance(fold, int):
    tomo_training_list, tomo_testing_list = get_training_testing_lists(config=config, fold=fold)
    if tomo_name in tomo_testing_list:
        run_job = True
    else:
        run_job = False
else:
    run_job = True

if run_job:
    output_dir_tomo, partition_path = testing_partition_path(output_dir=config.work_dir,
                                                             tomo_name=tomo_name,
                                                             fold=fold)
    box_shape = [config.box_size, config.box_size, config.box_size]
    output_classes = len(config.semantic_classes)

    device = get_device()
    checkpoint = torch.load(model_path, map_location=device)

    if 'model_descriptor' not in checkpoint.keys():
        warnings.warn("Model without model descriptor... it will be added")
        print("WARNING: model without model descriptor... it will be added")
        model_descriptor = model_descriptor_from_config(config)
        checkpoint["model_descriptor"] = model_descriptor
        torch.save({
            'model_descriptor': model_descriptor,
            'epoch': checkpoint['epoch'],
            'model_state_dict': checkpoint['model_state_dict'],
            'optimizer_state_dict': checkpoint['optimizer_state_dict'],
            'loss': checkpoint['loss'],
        }, model_path)
    else:
        print("Model trained under the following original settings:",
              checkpoint['model_descriptor'])

    model_descriptor = checkpoint['model_descriptor']

    net_conf = {'final_activation': None,
                'depth': model_descriptor.depth,
                'initial_features': model_descriptor.initial_features,
                "out_channels": model_descriptor.output_classes,
                "BN": model_descriptor.batch_norm,
                "encoder_dropout": model_descriptor.encoder_dropout,
                "decoder_dropout": model_descriptor.decoder_dropout}

    model = UNet3D(**net_conf)
    model.to(device)

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

    DTHeader = DatasetTableHeader(processing_tomo=config.processing_tomo, filtering_mask=config.region_mask)
    df = pd.read_csv(config.dataset_table, dtype={"tomo_name": str})
    df[DTHeader.tomo_name] = df[DTHeader.tomo_name].astype(str)
    tomo_df = df[df[DTHeader.tomo_name] == tomo_name]
    path_to_raw = tomo_df.iloc[0][config.processing_tomo]
    intersecting_mask_path = tomo_df.iloc[0][config.region_mask]
    raw_dataset = load_tomogram(path_to_dataset=path_to_raw, dtype=float)
    mean_val = np.mean(raw_dataset)
    std_val = np.std(raw_dataset)
    del raw_dataset

    print("Segmenting tomo", tomo_name)
    segment_and_write(data_path=partition_path, model=model, label_name=model_name, mean_value=mean_val,
                      std_value=std_val)
    print("The segmentation has finished!")

# For snakemake:
snakemake_pattern_dir = os.path.dirname(snakemake_pattern)
os.makedirs(snakemake_pattern_dir, exist_ok=True)
with open(file=snakemake_pattern, mode="w") as f:
    print("Creating snakemake pattern", snakemake_pattern)
