import os
from os.path import join

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data as du

from constants import h5_internal_paths
from file_actions.readers.h5 import read_training_data
from image.filters import preprocess_data
from networks.io import get_device
from networks.unet import UNet
from tomogram_utils.volume_actions.actions import split_and_preprocess_dataset


def load_unet_model(path_to_model: str, confs: dict, net: nn.Module = UNet,
                    mode="eval"):
    net = net(**confs)
    checkpoint = torch.load(path_to_model)
    net.load_state_dict(checkpoint['model_state_dict'])
    device = get_device()
    net.to(device)
    optimizer = optim.Adam(net.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    if "eval" == mode:
        net.eval()
        return net, optimizer, epoch, loss
    elif "train" == mode:
        net.train()
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        return net, optimizer, epoch, loss
    else:
        print("The loading mode requested is not supported.")


def save_unet_model(path_to_model: str, epoch, net, optimizer, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path_to_model)


def get_testing_and_training_sets_from_partition(training_data_path: str,
                                                 label_name: str,
                                                 split=0.8) -> tuple:
    print("The training data path is ", training_data_path)
    raw_data, labels = read_training_data(training_data_path,
                                          label_name=label_name)
    print("Initial unique labels", np.unique(labels))

    # Normalize data
    preprocessed_data = preprocess_data(raw_data)

    # add a channel dimension
    preprocessed_data = np.array(preprocessed_data)[:, None]
    labels = np.array(labels)[:, None]

    train_data, train_labels, val_data, val_labels, data_order = \
        split_and_preprocess_dataset(preprocessed_data, labels, split)
    return train_data, train_labels, val_data, val_labels, data_order


def generate_train_val_loaders(raw_array, label_array, split, batch_size=5,
                               shuffle=False):
    train_data, train_labels = raw_array[:split, ...], label_array[:split, ...]
    val_data, val_labels = raw_array[split:, ...], label_array[split:, ...]
    train_set = du.TensorDataset(torch.from_numpy(train_data),
                                 torch.from_numpy(train_labels))
    val_set = du.TensorDataset(torch.from_numpy(val_data),
                               torch.from_numpy(val_labels))
    train_loader = du.DataLoader(train_set, batch_size, shuffle)
    val_loader = du.DataLoader(val_set, batch_size)
    return train_loader, val_loader


def data_loader(data_path: str, semantic_class: str,
                number_vols: int = -1, labeled_only: bool = False):
    raw_array = list()
    label_array = list()
    with h5py.File(data_path, 'r') as f:
        if len(list(f)) > 0:
            h5_path_raw = h5_internal_paths.RAW_SUBTOMOGRAMS
            h5_path_label = join(h5_internal_paths.LABELED_SUBTOMOGRAMS,
                                 semantic_class)
            source_raw_names = list(f[h5_path_raw])
            source_label_names = list(f[h5_path_label])
            assert set(source_raw_names) == set(source_label_names)
            if number_vols == -1:
                number_vols = len(source_raw_names)
            for vol_name in source_raw_names[:number_vols]:
                src_raw_path = join(h5_path_raw, vol_name)
                src_label_path = join(h5_path_label, vol_name)
                if labeled_only:
                    # Only read if at least one label is positive:
                    if np.max(f[src_label_path][:]) > 0:
                        raw_array += [f[src_raw_path][:]]
                        label_array += [f[src_label_path][:]]
                    else:
                        print(
                            "Partition element with no label data: not included.")
                else:
                    # Read all of the data.
                    raw_array += [f[src_raw_path][:]]
                    label_array += [f[src_label_path][:]]
            # Add channel dimension
    raw_array, label_array = np.array(raw_array)[:, None], np.array(
        label_array)[:, None]
    return raw_array, label_array


def generate_model_name(box_shape: tuple or list, semantic_classes: list,
                        DA_tag: str, net_conf: dict, tomos_set: str) -> str:
    if np.max(box_shape) == np.min(box_shape):
        box_size = str(np.min(box_shape))
    else:
        bz, by, bx = box_shape
        box_size = str(bx) + "_" + str(by) + "_" + str(bz)
    classes_string = "_"
    for semantic_class in semantic_classes:
        classes_string += semantic_class + "_"

    net_name = box_size + "pix_encoder_dropout" + str(
        net_conf['encoder_dropout']) + "_decoder_dropout" + str(
        net_conf['decoder_dropout']) + "_DA_" + DA_tag + "_BN_" + str(
        net_conf['BN']) + classes_string + "_D_" + str(
        net_conf['depth']) + "_IF_" + str(net_conf['initial_features']) + \
               "_set_" + tomos_set
    return net_name


def build_prediction_output_dir(base_output_dir: str, label_name: str,
                                model_name: str,
                                tomo_name: str, semantic_class: str):
    output_dir = os.path.join(base_output_dir, label_name)
    output_dir = os.path.join(output_dir, model_name)
    output_dir = os.path.join(output_dir, tomo_name)
    output_dir = os.path.join(output_dir, semantic_class)
    return output_dir
