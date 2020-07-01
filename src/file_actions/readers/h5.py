from os.path import join
from random import shuffle

import h5py
import numpy as np

from constants import h5_internal_paths


def get_subtomos_and_labels(path_to_output_h5, label):
    with h5py.File(path_to_output_h5, 'r') as f:
        internal_path = h5_internal_paths.LABELED_SUBTOMOGRAMS
        internal_path = join(internal_path, label)
        subtomos_names = list(f[internal_path])
        subtomos_number = len(subtomos_names)
        subtomos_list = []
        for subtomo_name in subtomos_names:
            subtomo_path = join(internal_path, subtomo_name)
            subtomo = f[subtomo_path][:]
            subtomos_list.append(subtomo)
        labels = np.ones(subtomos_number)
    return subtomos_list, labels


def fill_multiclass_labels(semantic_classes, co_labeling_dict, labels_data):
    # format of co_labeling_dict: if class1 \subset class2 then class2
    # in co_labeling[class1]
    for class_index, semantic_class in enumerate(semantic_classes):
        for other_index, other_class in enumerate(semantic_classes):
            if other_class in co_labeling_dict[semantic_class]:
                print(semantic_class, " is a subset of ", other_class)
                where_semantic_class = np.where(labels_data[class_index, :] > 0)
                labels_data[other_index, where_semantic_class] = np.ones(
                    (1, len(where_semantic_class)))
    return labels_data


def load_classification_training_set(semantic_classes, path_to_output_h5):
    subtomos_data = []
    # classes_number = len(semantic_classes)
    labels_data = []  # np.zeros((classes_number, 0))
    for class_number, semantic_class in enumerate(semantic_classes):
        subtomos_list, labels = get_subtomos_and_labels(path_to_output_h5,
                                                        semantic_class)
        labels_semantic_class = [class_number for _ in range(len(labels))]

        subtomos_data += subtomos_list
        labels_data += labels_semantic_class
        # labels_data = np.concatenate((labels_data, labels_semantic_class),
        #                              axis=1)
    # labels_data = labels_data.transpose()
    # labels_data = list(labels_data)
    # Shuffle both lists subtomos_data and labels_data together
    zipped_data = list(zip(subtomos_data, labels_data))
    shuffle(zipped_data)
    subtomos_data = list(map(lambda p: p[0], zipped_data))
    labels_data = list(map(lambda p: p[1], zipped_data))

    labels_data = np.array(labels_data)
    subtomos_data = np.array(subtomos_data)[:, None]
    return subtomos_data, labels_data


def read_training_data(training_data_path: str,
                       label_name="ribosomes",
                       split=-1) -> tuple:
    data = []
    labels = []
    if split < 0:
        print("split = ", split)
        with h5py.File(training_data_path, 'r') as f:
            raw_subtomo_names = list(f[h5_internal_paths.RAW_SUBTOMOGRAMS])
            for subtomo_name in raw_subtomo_names:
                raw_subtomo_h5_internal_path = join(
                    h5_internal_paths.RAW_SUBTOMOGRAMS, subtomo_name)
                data += [f[raw_subtomo_h5_internal_path][:]]
                labels_subtomo_h5_internal_path = join(
                    h5_internal_paths.LABELED_SUBTOMOGRAMS, label_name)
                labels_subtomo_h5_internal_path = join(
                    labels_subtomo_h5_internal_path,
                    subtomo_name)
                labels += [f[labels_subtomo_h5_internal_path][:]]
    else:
        with h5py.File(training_data_path, 'r') as f:
            raw_subtomo_names = list(f[h5_internal_paths.RAW_SUBTOMOGRAMS])
            if 1 > split > 0:
                split = int(split * len(raw_subtomo_names))
            else:
                split = int(split)
            for subtomo_name in raw_subtomo_names[:split]:
                raw_subtomo_h5_internal_path = join(
                    h5_internal_paths.RAW_SUBTOMOGRAMS, subtomo_name)
                data += [f[raw_subtomo_h5_internal_path][:]]
                labels_subtomo_h5_internal_path = join(
                    h5_internal_paths.LABELED_SUBTOMOGRAMS, label_name)
                labels_subtomo_h5_internal_path = join(
                    labels_subtomo_h5_internal_path,
                    subtomo_name)
                labels += [f[labels_subtomo_h5_internal_path][:]]

    data = np.array(data)
    labels = np.array(labels)
    assert data.shape == labels.shape
    print("Loaded data and labels of shape", labels.shape)

    return data, labels


def read_training_data_dice_multi_class(training_data_path: str,
                                        segmentation_names: list,
                                        split: int = -1) -> tuple:
    """
    Loads partition training sets from h5 files, where the following are
    internal paths for labels and raw files, respectively:
    volumes/raw/
    volumes/labels/<label name>/
    :param training_data_path: 
    :param segmentation_names: 
    :param split: 
    :return: (data, labels)
    data: an array of shape N x 1 x Dx x Dy x Dz, where N is the total number
    of raw volumes
    labels: an array of shape N x S x Dx x Dy x Dz, where S is the number of
    semantic classes (or labels) in segmentation_names

    """
    data = []
    labels = []
    with h5py.File(training_data_path, 'r') as f:
        if len(list(f)) > 0:
            raw_subtomo_names = list(f[h5_internal_paths.RAW_SUBTOMOGRAMS])
            if split == -1:
                subtomo_list = raw_subtomo_names
            else:
                subtomo_list = raw_subtomo_names[:split]
            for subtomo_name in subtomo_list:
                raw_subtomo_h5_internal_path = join(
                    h5_internal_paths.RAW_SUBTOMOGRAMS, subtomo_name)
                labels_current_subtomo = []
                for label_name in segmentation_names:
                    labels_subtomo_h5_internal_path = join(
                        h5_internal_paths.LABELED_SUBTOMOGRAMS, label_name)
                    labels_subtomo_h5_internal_path = join(
                        labels_subtomo_h5_internal_path,
                        subtomo_name)
                    labels_current_subtomo += [
                        f[labels_subtomo_h5_internal_path][:]]
                segm_max = [np.max(label_data) for label_data in
                            labels_current_subtomo]
                if np.max(segm_max) > 0.5:
                    data += [[f[raw_subtomo_h5_internal_path][:]]]
                    labels += [np.array(labels_current_subtomo)]
                else:
                    print("Due to lack of annotations, discarding",
                          subtomo_name)
        else:
            print("Empty training set")

    data = np.array(data)
    labels = np.array(labels)
    print("Data shape from the reader", data.shape)
    print("Labels shape from the reader", labels.shape)
    return data, labels


# Todo: remove after testing transformations
def dummy_read_training_data_dice_multi_class(training_data_path: str,
                                              segmentation_names: list,
                                              split=-1,
                                              N=100) -> tuple:
    data = []
    labels = []
    with h5py.File(training_data_path, 'r') as f:
        raw_subtomo_names = list(f[h5_internal_paths.RAW_SUBTOMOGRAMS])[:N]
        raw_subtomo_names += list(f[h5_internal_paths.RAW_SUBTOMOGRAMS])[
                             int(2 * N):int(3 * N)]
        for subtomo_name in raw_subtomo_names[:split]:
            raw_subtomo_h5_internal_path = join(
                h5_internal_paths.RAW_SUBTOMOGRAMS, subtomo_name)
            data += [f[raw_subtomo_h5_internal_path][:]]
            labels_current_subtomo = []
            for label_name in segmentation_names:
                labels_subtomo_h5_internal_path = join(
                    h5_internal_paths.LABELED_SUBTOMOGRAMS, label_name)
                labels_subtomo_h5_internal_path = join(
                    labels_subtomo_h5_internal_path,
                    subtomo_name)
                labels_current_subtomo += [
                    f[labels_subtomo_h5_internal_path][:]]
            labels += [np.array(labels_current_subtomo)]

    data = np.array(data)
    labels = np.array(labels)
    print("Loaded data of shape", data.shape)
    print("Loaded labels of shape", labels.shape)
    return data, labels


def read_raw_data_from_h5(data_path: str) -> np.array:
    data = []
    with h5py.File(data_path, 'r') as f:
        for subtomo_name in list(f[h5_internal_paths.RAW_SUBTOMOGRAMS]):
            subtomo_h5_internal_path = join(h5_internal_paths.RAW_SUBTOMOGRAMS,
                                            subtomo_name)
            data += [f[subtomo_h5_internal_path][:]]
    return np.array(data)
