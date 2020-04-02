import os
import argparse
import yaml

import numpy as np
import pandas as pd
import h5py

from tensorflow import test as tft
from keras.callbacks import TensorBoard
from keras.layers import Input
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

import datetime

from ConfigUtil import assemble_config
from UNet import *



def main():
    # Configuration
    srcdir = os.path.dirname(os.path.realpath(__file__))
    parser = get_cli()
    args = parser.parse_args()

    config = assemble_config(
        f"{srcdir}/defaults.yaml",
        args.config,
        subconfig_paths = [("training", "general"), ("training", "evaluation")],
        cli_args = args
    )
    timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%y%m%d-%H%M")
    if not config["run_name"]:
        RUN_NAME = f"{timestamp}_training"
    else:
        RUN_NAME = config["run_name"].format(timestamp = timestamp)
    print("RUN NAME:", RUN_NAME)

    dataset_paths = args.datasets
    np.random.shuffle(dataset_paths)

    assert config["cv_folds"] > 1, \
        "Cannot evaluate model with less than 2 cross-validation folds"
    assert config["cv_folds"] <= len(dataset_paths), \
        f"Cannot perform {config['cv_folds']}-fold cross-validation with only {len(dataset_paths)} datasets"

    # Check GPU
    if tft.is_gpu_available():
        print("GPU is available 😩👌")
    else:
        print("GPU is not available 💩")


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

            if config["normalize"]:
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

    cv_sets = np.arange(len(features))
    comb_idx = np.hstack([np.full(d.shape[0], i) for i, d in enumerate(features)])
    comb_features = np.vstack(features)
    comb_labels = np.vstack(labels)

    del features, labels # Free up memory

    print(f"{f' MODEL TRAINING ':#^50}")

    cv_df = pd.DataFrame()

    for cv_idx, cv_ids in enumerate(np.array_split(cv_sets, config["cv_folds"])):

        print(f"{f' CV fold {cv_idx} ':#^50}")

        # Data splitting
        cv_mask = np.ones(len(ids), dtype=np.bool)
        cv_mask[cv_ids] = False

        train_ids = ids[~cv_mask]
        train_features = comb_features[~np.isin(comb_idx, cv_ids)]
        train_labels = comb_labels[~np.isin(comb_idx, cv_ids)]
        
        # Filter out fraction of all-empty patches
        if config["drop_empty"]:
            drop_idx = np.array([np.any(slice) for slice in train_labels]) | (np.random.random(train_labels.shape[0]) > config["drop_empty"])
            train_features = train_features[drop_idx]
            train_labels = train_labels[drop_idx]

        test_ids = ids[cv_mask]
        test_features = comb_features[np.isin(comb_idx, cv_ids)]
        test_labels = comb_labels[np.isin(comb_idx, cv_ids)]
        
        print("Datasets for training:", *train_ids)
        print("Datasets for testing:", *test_ids)

        # Create model
        input_img = Input((comb_features.shape[1], comb_features.shape[2], 1), name='img')
        model = get_unet(input_img, n_filters=config["n_filters"], target_shape=train_labels.shape)
        model.compile(
            optimizer=Adam(learning_rate=config["lr"]),
            loss=neg_dice_coefficient,
            metrics=[dice_coefficient, "binary_crossentropy"]
        )

        # Saving the log and show it by tensorboard
        callbacks = []

        if config["tensorboard"]:
            print("TensorBoard name:", RUN_NAME)
            callbacks.append(TensorBoard(log_dir=f"{config['tf_logdir']}/{RUN_NAME}_CV-{cv_idx}"))

        if config["stopping_patience"]:
            callbacks.append(EarlyStopping(patience=config["stopping_patience"]))

        # Fitting the model 
        results = model.fit(
            train_features, 
            train_labels, 
            batch_size=config["batch_size"], 
            epochs=config["epochs"],
            callbacks=callbacks,
            validation_data=(test_features, test_labels),
            verbose=2
        )

        results.history["cv_fold"] = cv_idx
        results.history["train_ids"] = ",".join(train_ids)
        results.history["val_ids"] = ",".join(test_ids)
        results.history["epoch"] = np.arange(len(results.history["val_loss"]))

        hist_df = pd.DataFrame(results.history)

        cv_df = cv_df.append(hist_df)

    print(f"{f' GATHERING METRICS ':#^50}")

    cv_df.to_csv(f"{config['tf_logdir']}/{RUN_NAME}_metrics.csv", index=False, header=True)

    def select_best_row(df):
        return df.sort_values("val_dice_coefficient", ascending=False).iloc[0,:][["val_dice_coefficient", "epoch"]]

    metrics = cv_df.groupby("cv_fold").apply(select_best_row).reset_index()
    metrics = metrics.agg({"val_dice_coefficient": ["mean", "std"], "epoch": ["mean", "std"]})

    print(f"Mean validation score: {metrics.iloc[0, 0]:.2} ± {metrics.iloc[1, 0]:.2}")
    print(f"Mean epochs required to reach maximum validation score: {metrics.iloc[0, 1]:.2} ± {metrics.iloc[1, 1]:.2}")


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