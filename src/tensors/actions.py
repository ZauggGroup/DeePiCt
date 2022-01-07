import numpy as np


def crop_tensor(input_array: np.array, shape_to_crop: tuple) -> np.array:
    """
    Function from A. Kreshuk to crop tensors of order 3, starting always from
    the origin.
    :param input_array: the input np.array image
    :param shape_to_crop: a tuple (cz, cy, cx), where each entry corresponds
    to the size of the  cropped region along each axis.
    :return: np.array of size (cz, cy, cx)
    """
    input_shape = input_array.shape
    assert all(ish >= csh for ish, csh in zip(input_shape, shape_to_crop)), \
        "Input shape must be larger equal crop shape"
    # get the difference between the shapes
    shape_diff = tuple((ish - csh) // 2
                       for ish, csh in zip(input_shape, shape_to_crop))
    # calculate the crop
    crop = tuple(slice(sd, sh - sd)
                 for sd, sh in zip(shape_diff, input_shape))
    return input_array[crop]


def crop_window(input_array: np.array, shape_to_crop: tuple or list,
                window_corner: tuple or list):
    """
    Function from A. Kreshuk to crop tensors of order 3, starting always
    from a given corner.
    :param input_array: the input np.array image
    :param shape_to_crop: a tuple (cz, cy, cx), where each entry corresponds
    to the size of the  cropped region along each axis.
    :param window_corner: point from where the window will be cropped.
    :return: np.array of size (cz, cy, cx)
    """
    input_shape = input_array.shape
    assert all(ish >= csh for ish, csh in zip(input_shape, shape_to_crop)), \
        "Input shape must be larger equal crop shape"
    # get the difference between the shapes
    crop = tuple(slice(wc, wc + csh)
                 for wc, csh in zip(window_corner, shape_to_crop))
    # print(crop)
    return input_array[crop]


def crop_window_around_point(input_array: np.array, crop_shape: tuple or list,
                             window_center: tuple or list) -> np.array:
    # The window center is not in tom_coordinates, it is (z, y, x)
    input_shape = input_array.shape
    assert all(ish - csh // 2 - center >= 0 for ish, csh, center in
               zip(input_shape, crop_shape, window_center)), \
        "Input shape must be larger or equal than crop shape"
    assert all(center - csh // 2 >= 0 for csh, center in
               zip(crop_shape, window_center)), \
        "Input shape around window center must be larger equal than crop shape"
    # get the difference between the shapes
    crop = tuple(slice(center - csh // 2, center + csh // 2)
                 for csh, center in zip(crop_shape, window_center))
    return input_array[crop]
