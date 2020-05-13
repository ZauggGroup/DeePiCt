import argparse
from keras.models import Model, load_model
import keras.backend as K
import numpy as np
import mrcfile
import os

from PatchUtil import *
from ConfigUtil import assemble_config, csv_list

from UNet import dice_coefficient, neg_dice_coefficient

def main():

    srcdir = os.path.dirname(os.path.realpath(__file__))
    parser = get_cli()
    args = parser.parse_args()

    # Configuration
    config = assemble_config(
        f"{srcdir}/defaults.yaml",
        args.config,
        subconfig_paths = [("prediction",)],
        cli_args = args
    )

    tomo_file = args.features
    out_file = args.output
    model_file = args.model

    if len(config["patch_dim"]) == 1:
        config["patch_dim"]*=2

    if len(config["patch_size"]) == 1:
        config["patch_size"]*=2

    assert len(config["patch_dim"]) == 2, "patch_dim needs to be a single int or comma-separated pair of ints"
    assert len(config["patch_size"]) == 2, "patch_size needs to be a single int or comma-separated pair of ints"

    # Load model
    model = load_model(
        model_file,
        custom_objects={
            'neg_dice_coefficient':neg_dice_coefficient,
            "dice_coefficient":dice_coefficient
        }
    )

    # Preprocess tomogram
    tomo = read_mrc(tomo_file).astype(np.float32)
    orig_shape = tomo.shape

    if config["z_cutoff"]:
        config["z_cutoff"] = min(tomo.shape[0], config["z_cutoff"])
        z_center = tomo.shape[0] // 2
        z_idx = slice(z_center-(config["z_cutoff"] // 2), z_center+(config["z_cutoff"] // 2))

        tomo = tomo[z_idx]

    # Normalization
    if config["normalize"]:
        mean = tomo.mean()
        std = tomo.std()

        tomo -= mean
        tomo /= std

    if config["crop"]:
        padding = [(0, 0)] + [(config["crop"],)*2]*2
        tomo = np.pad(tomo, padding, mode="reflect")

    # Slice & predict
    tomo_patches = np.expand_dims(into_patches_3d(tomo, config["patch_size"], config["patch_dim"]), -1) # Add channel dim
    tomo_pred = model.predict(tomo_patches)
    rec = from_patches_3d(tomo_pred[...,0], config["patch_dim"], tomo.shape, pad=config["crop"])

    if config["compensate_crop"]:
        padding = [((i - o)//2,)*2 for i, o in zip(orig_shape, rec.shape)]
        rec = np.pad(rec, padding)

    # Save output
    mrcfile.new(out_file, data=rec.astype(np.float32), overwrite=True).close()


def read_mrc(file):
    with mrcfile.open(file, permissive=True) as f:
        return f.data

def get_cli():
    parser = argparse.ArgumentParser(
        description="Predict organelle segmentation from tomogram data using a trained model."
    )

    parser.add_argument( 
        "-f",
        "--features",
        required=True,
        help="Input features to predict from."
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="save location for predicted segmentation."
    )

    parser.add_argument( 
        "-m",
        "--model",
        required=False,
        help="Location of pretrained model in HDF5 format."
    )

    parser.add_argument( 
        "-c",
        "--config",
        required=False,
        help="Configuration YAML file. Overrides defaults, overridden by CLI arguments."
    )

    parser.add_argument(
        "-x",
        "--crop",
        required=False,
        help="Crop patches on reassembly to counter artifacts around patch edges."
    )

    parser.add_argument(
        "-C",
        "--compensate_crop",
        action="store_true",
        default=None,
        help="Compensate patch cropping and z-axis cutoff to preserve the source tomogram's shape."
    )

    parser.add_argument(
        "-N",
        "--dont_compensate_crop",
        action="store_true",
        default=None,
        dest="compensate_crop",
        help="Don't compensate patch cropping and z-axis cutoff. Used to override config file."
    )

    parser.add_argument(
        "-p",
        "--patch_size",
        type=csv_list,
        required=False,
        help="Comma-separated height and width for patches (has to be consistent with patch size used for training the model)."
    )

    parser.add_argument(
        "-d",
        "--patch_dim",
        type=csv_list,
        required=False,
        help="Comma-separated numbers of row and columns of patches to be evenly-spaced out along tomogram slices."
    )

    parser.add_argument(
        "-z",
        "--z_cutoff",
        required=False,
        type=int,
        help="Only predict for this many z-slices from z center."
    )

    return parser

if __name__ == "__main__":
    main()