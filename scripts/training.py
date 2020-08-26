import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-config_file", "--config_file", type=str)
parser.add_argument("-pythonpath", "--pythonpath", type=str)
parser.add_argument("-fold", "--fold", type=str, default=None)
parser.add_argument("-gpu", "--gpu", help="cuda visible devices", type=str)

args = parser.parse_args()
pythonpath = args.pythonpath
config_file = args.config_file
sys.path.append(pythonpath)

import os

import numpy as np

import torch.nn as nn
import torch.optim as optim

from networks.io import get_device, to_device
from networks.utils import generate_data_loaders
from networks.loss import DiceCoefficientLoss
from networks.routines import train, validate

from networks.unet import UNet3D
from networks.utils import save_unet_model
from networks.visualizers import TensorBoard_multiclass

from constants.config import Config, record_model

config = Config(args.config_file)
gpu = args.gpu
device = get_device()

# Generate relevant dirs
logging_dir = os.path.join(config.output_dir, "logging")
model_dir = os.path.join(config.output_dir, "models")
models_table = os.path.join(model_dir, "models.csv")
log_path = os.path.join(logging_dir, config.model_name)
os.makedirs(log_path, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

train_loader, val_loader = generate_data_loaders(config=config)

net_conf = {'final_activation': nn.Sigmoid(), 'depth': config.depth,
            'initial_features': config.initial_features,
            "out_channels": len(config.semantic_classes), "BN": config.batch_norm,
            "encoder_dropout": config.encoder_dropout,
            "decoder_dropout": config.decoder_dropout}

net = UNet3D(**net_conf)
net = to_device(net=net, gpu=gpu)

loss = DiceCoefficientLoss()
loss = loss.to(device)
optimizer = optim.Adam(net.parameters())
metric = loss

fold = args.fold
record_model(config, fold)

lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,
                                                    patience=10, verbose=True)

validation_loss = np.inf
best_epoch = -1
old_epoch = 0

logger = TensorBoard_multiclass(log_dir=log_path, log_image_interval=1)

for epoch in range(config.epochs):
    current_epoch = epoch + old_epoch

    train(model=net, loader=train_loader, optimizer=optimizer, loss_function=loss,
          epoch=current_epoch, device=device, log_interval=1, tb_logger=logger,
          log_image=False, lr_scheduler=lr_scheduler)

    step = current_epoch * len(train_loader.dataset)

    current_validation_loss = validate(model=net, loader=val_loader, loss_function=loss,
                                       metric=metric, device=device, step=step, tb_logger=logger,
                                       log_image_interval=None)

    # save best epoch
    if current_validation_loss <= validation_loss:
        best_epoch = current_epoch
        print("Best epoch! -->", best_epoch, "with validation loss:", current_validation_loss)
        validation_loss = current_validation_loss
        save_unet_model(path_to_model=config.model_path, epoch=current_epoch,
                        net=net, optimizer=optimizer, loss=current_validation_loss)
    else:
        print("Epoch =", current_epoch, " was not the best.")
        print("The current best is epoch =", best_epoch)

print("We have finished the training!")
print("Best validation loss: {} of epoch {}".format(validation_loss, best_epoch))

# For snakemake:
snakemake_pattern = os.path.join(".done_patterns", config.model_path + ".done")
os.makedirs(os.path.dirname(snakemake_pattern), exist_ok=True)
with open(file=snakemake_pattern, mode="w") as f:
    print("Creating snakemake pattern", snakemake_pattern)
