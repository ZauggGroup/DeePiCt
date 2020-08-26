import os
import torch

import torch.nn as nn


def get_device():
    if torch.cuda.is_available():
        print("GPU is available")
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cuda")
    else:
        print("GPU is not available")
        device = torch.device("cpu")
    return device


def to_device(net, gpu: str or None):
    if gpu is None:
        print("No CUDA_VISIBLE_DEVICES passed... checking if there is an available gpu")
        if torch.cuda.is_available():
            print("Should pass: CUDA_VISIBLE_DEVICES = {}".format(gpu))
    else:
        print("CUDA_VISIBLE_DEVICES = {}".format(gpu))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = get_device()
    net = net.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        net = nn.DataParallel(net)
    return net
