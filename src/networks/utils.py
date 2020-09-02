import os
from os.path import join

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as du

from constants import h5_internal_paths
from constants.config import Config, CV_DATA_FILE
from file_actions.readers.h5 import read_training_data
from image.filters import preprocess_data
from networks.io import get_device
from networks.unet import UNet
from paths.pipeline_dirs import training_partition_path
from tensors.transformations import apply_transformation_iteration
from tomogram_utils.volume_actions.actions import \
    load_and_normalize_dataset_list
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


def get_training_testing_lists(config: Config, fold: int):
    if fold is not None:
        cv_data = pd.read_csv(CV_DATA_FILE)
        print(cv_data)
        cv_data["fold"] = cv_data["fold"].apply(lambda x: str(x))
        cv_data.set_index("fold", inplace=True)

        def split_list(tomo_list_str: str, sep: str = " "):
            tomo_list = tomo_list_str.split(sep)
            tomo_list_spl = []
            for tomo in tomo_list:
                if len(tomo) > 0:
                    tomo_list_spl.append(tomo)
            return tomo_list_spl

        tomo_training_list = split_list(tomo_list_str=cv_data.loc[str(fold)]["cv_training_list"])
        tomo_testing_list = split_list(tomo_list_str=cv_data.loc[str(fold)]["cv_validation_list"])
    else:
        tomo_training_list = config.training_tomos
        tomo_testing_list = []

    return tomo_training_list, tomo_testing_list


def generate_data_loaders(config: Config, tomo_training_list: list, fold: int or None = None):
    training_partition_paths = list()
    data_aug_rounds_list = list()
    for tomo_name in tomo_training_list:
        print(tomo_name)
        _, partition_path = training_partition_path(output_dir=config.work_dir,
                                                    tomo_name=tomo_name,
                                                    fold=fold)
        training_partition_paths += [partition_path]
        data_aug_rounds_list += [0]

    train_data, train_labels, val_data, val_labels = \
        load_and_normalize_dataset_list(training_partition_paths,
                                        data_aug_rounds_list,
                                        config.semantic_classes, config.split)

    print("Train data: mean = {}, std = {}".format(np.mean(train_data), np.std(train_data)))
    print("unique labels = {}".format(np.unique(train_labels)))
    print("training data shape =", train_data.shape)
    print("validation data shape =", val_data.shape)
    train_set = du.TensorDataset(torch.from_numpy(train_data),
                                 torch.from_numpy(train_labels))
    val_set = du.TensorDataset(torch.from_numpy(val_data),
                               torch.from_numpy(val_labels))

    train_loader = du.DataLoader(train_set, shuffle=True, batch_size=config.batch_size)
    val_loader = du.DataLoader(val_set, batch_size=config.batch_size)
    return train_loader, val_loader


def generate_data_loaders_data_augmentation(config: Config, tomo_training_list: list, fold: int or None = None):
    training_partition_paths = list()
    data_aug_rounds_list = list()
    for tomo_name in tomo_training_list:
        print(tomo_name)
        _, partition_path = training_partition_path(output_dir=config.work_dir,
                                                    tomo_name=tomo_name,
                                                    fold=fold)
        training_partition_paths += [partition_path]
        data_aug_rounds_list += [0]

    train_data, train_labels, val_data, val_labels = \
        load_and_normalize_dataset_list(training_partition_paths,
                                        data_aug_rounds_list,
                                        config.semantic_classes, config.split)

    # Data augmentation rounds:
    if train_data.shape[0] > 0:
        data_aug_rounds = config.da_rounds
        rot_angle = config.da_rot_angle
        elastic_alpha = config.da_elastic_alpha
        sigma_gauss = config.da_sigma_gauss
        salt_pepper_p = config.da_salt_pepper_p
        salt_pepper_ampl = config.da_salt_pepper_ampl
        print(train_labels.shape)
        assert len(train_labels.shape) in [5, 6]
        if len(train_labels.shape) == 6:
            src_label_data = train_labels
        else:
            print("one single label")
            src_label_data = train_labels[None, :]
        train_data_aug = train_data.copy()
        train_labels_aug = src_label_data.copy()
        for iteration in range(data_aug_rounds):
            transf_raw_tensor, transf_label_tensors = \
                apply_transformation_iteration(
                    src_raw=train_data,
                    src_label_data=src_label_data,
                    rot_range=rot_angle,
                    elastic_alpha=elastic_alpha,
                    sigma_noise=sigma_gauss,
                    salt_pepper_p=salt_pepper_p,
                    salt_pepper_ampl=salt_pepper_ampl)
            train_data_aug = np.append(train_data_aug, np.array(transf_raw_tensor), axis=0)
            train_labels_aug = np.append(train_labels_aug, np.array(transf_label_tensors), axis=1)
        print("train_labels_aug.shape", train_labels_aug.shape)
        print("train_data_aug.shape", train_data_aug.shape)
    train_set = du.TensorDataset(torch.from_numpy(train_data),
                                 torch.from_numpy(train_labels))
    val_set = du.TensorDataset(torch.from_numpy(val_data),
                               torch.from_numpy(val_labels))

    train_loader = du.DataLoader(train_set, shuffle=True, batch_size=config.batch_size)
    val_loader = du.DataLoader(val_set, batch_size=config.batch_size)
    return train_loader, val_loader


if __name__ == "__main__":
    from constants.config import Config
    from networks.utils import get_training_testing_lists

    config_file = "test_config.yaml"
    src_data_path = "test_work/training_data/induced/partition.h5"
    config = Config(config_file)

    tomo_training_list, tomo_testing_list = get_training_testing_lists(config=config, fold=None)

    train_loader, val_loader = generate_data_loaders_data_augmentation(config=config,
                                                                       tomo_training_list=tomo_training_list,
                                                                       fold=None)


