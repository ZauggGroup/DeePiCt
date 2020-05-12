import argparse
import numpy as np
import mrcfile
import h5py
import os

from PatchUtil import into_patches
from ConfigUtil import assemble_config, csv_list

def main():

    # Configuration
    srcdir = os.path.dirname(os.path.realpath(__file__))
    parser = get_cli()
    args = parser.parse_args()

    config = assemble_config(
        f"{srcdir}/defaults.yaml",
        args.config,
        subconfig_paths = [("preprocessing", "slicing")],
        cli_args = args
    )

    features = args.features
    labels = args.labels
    out_file = args.output

    z_stride = config["z_stride"]
    crop = config["crop"]
    patch_size = config["patch_size"]
    patch_dim = config["patch_dim"]
    flip_y = config["flip_y"]

    dataset_id = os.path.splitext(os.path.basename(features))[0]

    if len(patch_dim) == 1:
        patch_dim*=2

    if len(patch_size) == 1:
        patch_size*=2

    assert len(patch_dim) == 2, "patch_dim needs to be a single int or comma-separated pair of ints"
    assert len(patch_size) == 2, "patch_size needs to be a single int or comma-separated pair of ints"


    # Load data + labels
    features = read_mrc(features)
    labels = read_mrc(labels)
    if flip_y:
        labels = np.flip(labels, 1) # For some reason MRC file orientation likes to get messed up

    assert labels.shape == features.shape, "Tomogram data and labels have mismatching shape"

    # Stack features and labels, trim unlabeled slices, select n-th slices
    stack = np.stack([features, labels])

    if config["z_cutoff"]:
        config["z_cutoff"] = min(stack.shape[1], config["z_cutoff"])
        z_center = stack.shape[1] // 2
        z_idx = slice(z_center-(config["z_cutoff"] // 2), z_center+(config["z_cutoff"] // 2))    
    else:
        z_idx = np.array([np.any(slice) for slice in stack[1]])

    stack = stack[:,z_idx]
    stack = np.moveaxis(stack, 0, -1)
    stack = stack[::z_stride]

    # Crop images
    if crop:
        stack = stack[:, crop:-crop, crop:-crop] # Crop images

    # Process image into patches
    patch_stack = [into_patches(image, patch_size, patch_dim) for image in stack]

    # Stack slice patches
    patch_stack = np.vstack(patch_stack)
    patch_stack = patch_stack.astype(np.float32)

    # Split data into features and labels again
    processed_features = patch_stack[...,0]
    processed_labels = patch_stack[...,1]

    print(f"Created {patch_stack.shape[0]} patches across {stack.shape[0]} z-slices.")

    with h5py.File(out_file, "w") as h:
        h.attrs["sample_id"] = dataset_id
        h.create_dataset("features", data=processed_features)
        h.create_dataset("labels", data=processed_labels)

def read_mrc(file):
    with mrcfile.open(file, permissive=True) as f:
        return f.data

def get_cli():
    # TODO: CLI documentation
    parser = argparse.ArgumentParser(
        description="Process tomogram-label pairs into 2D training datasets."
    )

    parser.add_argument( 
        "-f",
        "--features",
        required=True
    )

    parser.add_argument( 
        "-l",
        "--labels",
        required=True
    )

    parser.add_argument( 
        "-o",
        "--output",
        required=True
    )

    parser.add_argument( 
        "-c",
        "--config",
        required=False
    )

    parser.add_argument(
        "-z",
        "--z_stride",
        required=False
    )

    parser.add_argument(
        "-x",
        "--crop",
        required=False
    )

    parser.add_argument(
        "-s",
        "--patch_size",
        type=csv_list,
        required=False
    )

    parser.add_argument(
        "-n",
        "--patch_dim",
        type=csv_list,
        required=False
    )

    parser.add_argument(
        "-y",
        "--flip_y",
        action="store_true",
        default=None
    )

    parser.add_argument(
        "--dont_flip_y",
        action="store_false",
        dest="flip_y",
        default=None
    )

    return parser

if __name__ == "__main__":
    main()