from os.path import join

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data as du

from src.python.constants import h5_internal_paths
from src.python.networks.io import get_device
from src.python.networks.unet import UNet
from src.python.tomogram_utils.actions import split_and_preprocess_dataset
from file_actions.readers.h5 import read_training_data
from src.python.image.filters import preprocess_data


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
                number_vols: int = -1):
    raw_array = list()
    label_array = list()
    with h5py.File(data_path, 'r') as f:
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
            raw_array += [f[src_raw_path][:]]
            label_array += [f[src_label_path][:]]
        # Add channel dimension
        raw_array, label_array = np.array(raw_array)[:, None], np.array(
            label_array)[:, None]
    return raw_array, label_array
