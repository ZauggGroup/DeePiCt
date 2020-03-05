import os
import argparse
import yaml

import numpy as np
import pandas as pd
import h5py

from keras.callbacks import TensorBoard
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Cropping2D
from keras.layers.core import Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras import backend as K

import datetime



def main():
    parser = get_cli()
    args = parser.parse_args()

    if tf.test.is_gpu_available():
        print("GPU is available 😩👌")
    else:
        print("GPU is not available 💩")

    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

        train_cfg = config["training"]
        timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%y%m%d-%H%M")
        if not config["general"]["run_name"]:
            RUN_NAME = f"{timestamp}_training"
        else:
            RUN_NAME = config["general"]["run_name"].format(timestamp = timestamp)
        print("RUN NAME:", RUN_NAME)

    dataset_paths = args.datasets

    print(f"{f' DATA PREPARATION ':#^50}")
    datasets = []

    for p in dataset_paths: 
        print(f"Reading {p}...")
        with h5py.File(p, 'r') as f:
            features = f['features'][:]
            labels = f['labels'][:]
            sample_id = f.attrs["sample_id"]

            features = np.expand_dims(features, -1)
            labels = np.expand_dims(labels, -1)

            print("Normalizing...")
            mean = features.mean()
            std = features.std()
            print(f"Before normalization: {mean: .2} +/-{std:.2}", end="\t")

            features -= mean
            features /= std
            print(f"After normalization: {features.mean(): .2} +/-{features.std():.2}")

            datasets.append([sample_id, features, labels])

    ids, features, labels = zip(*datasets)
    ids = np.array(ids)
    del datasets

    comb_idx = np.hstack([np.full(d.shape[0], i) for i, d in enumerate(features)])
    comb_features = np.vstack(features)
    comb_labels = np.vstack(labels)

    del features, labels # Free up memory

    print(f"{f' MODEL TRAINING ':#^50}")

    train_features = comb_features
    train_labels = comb_labels
    train_labels[train_labels == 1] = 0
    train_labels[train_labels > 1] = 1

    # Filter out fraction of all-empty patches
    if train_cfg["general"]["drop_empty"]:
        drop_idx = np.array([np.any(slice) for slice in train_labels]) | (np.random.random(train_labels.shape[0]) > train_cfg["general"]["drop_empty"])
        train_features = train_features[drop_idx]
        train_labels = train_labels[drop_idx]

    # Create model
    input_img = Input((comb_features.shape[1], comb_features.shape[2], 1), name='img')
    model = get_unet(input_img, n_filters=train_cfg["general"]["n_filters"], target_shape=train_labels.shape)
    model.compile(
        optimizer=Adam(learning_rate=train_cfg["general"]["lr"]),
        loss=neg_dice_coefficient,
        metrics=[dice_coefficient, "binary_crossentropy"]
    )

    # Fitting the model 
    results = model.fit(
        train_features, 
        train_labels, 
        batch_size=train_cfg["general"]["batch_size"],
        epochs=train_cfg["production"]["epochs"]
    )

    print(f"{f' SAVING MODEL ':#^50}")
    model.save(train_cfg["production"]["model_output"])

def conv2d_block(input_tensor, n_filters, kernel_size=3):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               padding="same")(input_tensor)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), 
               padding="same")(x)
    x = Activation("relu")(x)
    return x


def get_unet(input_img, n_filters, target_shape):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*4, kernel_size=3) #The first block of U-net
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = conv2d_block(p1, n_filters=n_filters*8, kernel_size=3)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = conv2d_block(p2, n_filters=n_filters*16, kernel_size=3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = conv2d_block(p3, n_filters=n_filters*32, kernel_size=3)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*64, kernel_size=3)
     
    # expansive path
    u6 = Conv2DTranspose(n_filters*32, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = conv2d_block(u6, n_filters=n_filters*32, kernel_size=3)

    u7 = Conv2DTranspose(n_filters*16, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = conv2d_block(u7, n_filters=n_filters*16, kernel_size=3)

    u8 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = conv2d_block(u8, n_filters=n_filters*8, kernel_size=3)

    u9 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = conv2d_block(u9, n_filters=n_filters*4, kernel_size=3)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


def dice_coefficient(y_true, y_pred):
    eps = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + eps)


def neg_dice_coefficient(y_true, y_pred):
    eps = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return -((2. * intersection) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + eps))


def get_cli():
    # TODO: CLI documentation
    parser = argparse.ArgumentParser(
        description="Process tomogram-label pairs into 2D training datasets."
    )

    parser.add_argument( 
        "-c",
        "--config",
        required=True
    )

    parser.add_argument( 
        "-d",
        "--datasets",
        required=True,
        nargs="+"
    )
    
    return parser


if __name__ == "__main__":
    main()