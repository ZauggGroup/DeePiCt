import numpy as np


def preprocess_data(data: np.array):
    data_mean = data.mean()
    data_std = data.std()
    print("The data mean value is", data_mean)
    print("The data std value is", data_std)

    data = data - data_mean
    data = data/data_std
    # check again to double check
    print("After normalization the data has mean value", data.mean())
    print("After normalization the data has standard deviation", data.std())
    return data.astype('float32')


def normalize_image_stack(images):
    normalized = np.zeros_like(images, dtype='float32')
    for ii in range(len(images)):
        im = images[ii].astype('float32')
        im -= im.min()
        im /= im.max()
        normalized[ii] = im
    return normalized
