import torch


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
