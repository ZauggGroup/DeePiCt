import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-pythonpath", "--pythonpath", type=str)
parser.add_argument("-segmentation_names", "--segmentation_names", nargs='+', type=str)
parser.add_argument("-dataset_table", "--dataset_table", type=str)
parser.add_argument("-box_shape", "--box_shape", type=int)
parser.add_argument("-gpu", "--gpu", help="cuda visible devices", type=str)
parser.add_argument("-output_dir", "--output_dir", type=str)
parser.add_argument("-tomo_training_list", "--tomo_training_list", nargs="+", type=str)
parser.add_argument("-split", "--split", type=float)
parser.add_argument("-n_epochs", "--n_epochs", type=int)
parser.add_argument("-depth", "--depth", type=int)
parser.add_argument("-decoder_dropout", "--decoder_dropout", type=float)
parser.add_argument("-encoder_dropout", "--encoder_dropout", type=float)
parser.add_argument("-batch_size", "--batch_size", type=int)
parser.add_argument("-batch_norm", "--batch_norm", type=bool)
parser.add_argument("-initial_features", "--initial_features", type=int)
parser.add_argument("-overlap", "--overlap", type=int)
parser.add_argument("-processing_tomo", "--processing_tomo", type=str)
parser.add_argument("-partition_name", "--partition_name", type=str)
parser.add_argument("-model_name", "--model_name", type=str)

args = parser.parse_args()
pythonpath = args.pythonpath
sys.path.append(pythonpath)

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as du

from constants.dataset_tables import DatasetTableHeader
from file_actions.writers.csv import write_on_models_notebook
from networks.io import get_device
from networks.loss import DiceCoefficientLoss
from networks.routines import train, validate

from networks.unet import UNet3D, UNet
from networks.utils import save_unet_model, generate_model_name
from networks.visualizers import TensorBoard_multiclass
from tomogram_utils.volume_actions.actions import \
    load_and_normalize_dataset_list

gpu = args.gpu
if gpu is None:
    print("No CUDA_VISIBLE_DEVICES passed... checking if there is an available gpu")
    if torch.cuda.is_available():
        CUDA_VISIBLE_DEVICES = "0"
        print("CUDA_VISIBLE_DEVICES = {}".format(gpu))
else:
    print("CUDA_VISIBLE_DEVICES = {}".format(gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

dataset_table = args.dataset_table
output_dir = args.output_dir
tomo_training_list = args.tomo_training_list
partition_name = args.partition_name
segmentation_names = args.segmentation_names
split = args.split
n_epochs = args.n_epochs
depth = args.depth
decoder_dropout = args.decoder_dropout
encoder_dropout = args.encoder_dropout
batch_size = args.batch_size
batch_norm = args.batch_norm
initial_features = args.initial_features
output_classes = len(segmentation_names)
overlap = args.overlap
processing_tomo = args.processing_tomo
box_size = args.box_shape
model_name = args.model_name

logging_dir = os.path.join(output_dir, "logging")
model_dir = os.path.join(output_dir, "models")
models_table = os.path.join(model_dir, "models.csv")

box_shape = [box_size, box_size, box_size]

DTHeader = DatasetTableHeader(partition_name=partition_name,
                              semantic_classes=segmentation_names)
df = pd.read_csv(dataset_table, dtype={DTHeader.tomo_name: str})

training_partition_paths = list()
data_aug_rounds_list = list()
for tomo_name in tomo_training_list:
    print(tomo_name)
    partition_path = os.path.join(output_dir, "training_data")
    partition_path = os.path.join(partition_path, tomo_name)
    partition_path = os.path.join(partition_path, partition_name + ".h5")
    training_partition_paths += [partition_path]
    data_aug_rounds_list += [0]

device = get_device()
train_data, train_labels, val_data, val_labels = \
    load_and_normalize_dataset_list(training_partition_paths,
                                    data_aug_rounds_list,
                                    segmentation_names, split)
print("Train data: mean = {}, std = {}".format(np.mean(train_data), np.std(train_data)))
print("unique labels = {}".format(np.unique(train_labels)))
print("training data shape =", train_data.shape)
print("validation data shape =", val_data.shape)

train_set = du.TensorDataset(torch.from_numpy(train_data),
                             torch.from_numpy(train_labels))
val_set = du.TensorDataset(torch.from_numpy(val_data),
                           torch.from_numpy(val_labels))

train_loader = du.DataLoader(train_set, shuffle=True, batch_size=batch_size)
val_loader = du.DataLoader(val_set, batch_size=batch_size)

final_activation = nn.Sigmoid()
net_conf = {'final_activation': final_activation, 'depth': depth,
            'initial_features': initial_features,
            "out_channels": output_classes, "BN": batch_norm,
            "encoder_dropout": encoder_dropout,
            "decoder_dropout": decoder_dropout}

# net = UNet_dropout(**net_conf)
net = UNet3D(**net_conf)
# net = UNet(**net_conf)
net = net.to(device)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    net = nn.DataParallel(net)
net.to(device)  # currently the model is in one single node

loss = DiceCoefficientLoss()
loss = loss.to(device)
optimizer = optim.Adam(net.parameters())
old_epoch = 0
metric = loss


model_path = os.path.join(model_dir, model_name)
log_model = os.path.join(logging_dir, model_name[:-4])
os.makedirs(log_model, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

write_on_models_notebook(model_name=model_name[:-4], label_name=model_name[:-4], model_dir=model_dir,
                         log_dir=log_model, depth=depth,
                         initial_features=initial_features, n_epochs=n_epochs,
                         training_paths_list=training_partition_paths,
                         split=split, output_classes=output_classes,
                         segmentation_names=segmentation_names, box_size=box_size,
                         overlap=overlap,
                         processing_tomo=processing_tomo,
                         partition_name=partition_name,
                         retrain=False,
                         path_to_old_model="",
                         models_notebook_path=models_table,
                         encoder_dropout=encoder_dropout,
                         decoder_dropout=decoder_dropout,
                         BN=batch_norm)

lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,
                                                    patience=10, verbose=True)

print("The neural network training is now starting...")
validation_loss = np.inf
best_epoch = -1
logger = TensorBoard_multiclass(log_dir=log_model, log_image_interval=1)
for epoch in range(n_epochs):
    new_epoch = epoch + old_epoch
    train(net, train_loader, optimizer=optimizer, loss_function=loss,
          epoch=new_epoch, device=device, log_interval=1, tb_logger=logger,
          log_image=False, lr_scheduler=lr_scheduler)
    step = new_epoch * len(train_loader.dataset)
    # run validation after training epoch
    current_validation_loss = validate(net, val_loader, loss, metric,
                                       device=device, step=step,
                                       tb_logger=logger)
    if current_validation_loss <= validation_loss:
        best_epoch = new_epoch
        print("Best epoch! -->", best_epoch)
        validation_loss = current_validation_loss
        save_unet_model(path_to_model=model_path, epoch=new_epoch,
                        net=net, optimizer=optimizer, loss=current_validation_loss)
    else:
        print("Epoch =", new_epoch, " was not the best.")
        print("The current best one is epoch =", best_epoch)

print("We have finished the training!")
print("The best validation loss was", validation_loss)
print("The best epoch was", best_epoch)
