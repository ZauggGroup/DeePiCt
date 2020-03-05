import argparse
import numpy as np
import mrcfile
import h5py
import os

from PatchUtil import into_patches

def main():

    parser = get_cli()
    args = parser.parse_args()

    features = args.features
    labels = args.labels
    out_file = args.output

    z_stride = args.z_stride # 5 Z-distance between slices
    crop = args.crop # 64 # Crop image before extracting patches
    patch_size = args.patch_size # 288 # arg
    patch_n = args.patch_n # 25 # replace w/ patch_n
    rotate = args.rotate # True # Arg
    flip_y = args.flip_y

    dataset_id = os.path.splitext(os.path.basename(features))[0]

    if isinstance(patch_n, int):
        patch_n = (patch_n,)*2
    else:
        patch_n = tuple(int(dim) for dim in patch_n.split(","))

    if isinstance(patch_size, int):
        patch_size = (patch_size,)*2
    else:
        patch_size = tuple(int(dim) for dim in patch_size.split(","))

    assert (len(patch_n) == 2) & (all(isinstance(i, int) for i in patch_n)), \
        "patch_n needs to be a single int or comma-separated pair of ints"
    assert (len(patch_size) == 2) & (all(isinstance(i, int) for i in patch_size)), \
        "patch_size needs to be a single int or comma-separated pair of ints"

    # Load data + labels
    features = read_mrc(features)
    labels = read_mrc(labels)
    if flip_y:
        labels = np.flip(labels, 1) # For some reason MRC file orientation likes to get messed up

    assert labels.shape == features.shape, "Tomogram data and labels have mismatching shape"

    # Stack features and labels, trim unlabeled slices, select n-th slices
    stack = np.stack([features, labels])
    nonzero_idx = np.array([np.any(slice) for slice in stack[1]])
    stack = stack[:,nonzero_idx]
    stack = np.moveaxis(stack, 0, -1)
    stack = stack[::z_stride]

    # Crop images
    if crop:
        stack = stack[:, crop:-crop, crop:-crop] # Crop images

    # Process image into patches
    patch_stack = [into_patches(image, patch_size, patch_n) for image in stack]

    # Stack slice patches
    patch_stack = np.vstack(patch_stack)
    patch_stack = patch_stack.astype(np.float32)

    # Rotate patches
    if rotate:
        patch_stack = np.stack([np.rot90(patch, np.random.randint(4)) for patch in patch_stack])

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
        "-f",
        "--labels",
        required=True
    )

    parser.add_argument( 
        "-o",
        "--output",
        required=True
    )

    parser.add_argument(
        "-z",
        "--z_stride",
        default=1
    )

    parser.add_argument(
        "-c",
        "--crop",
        default=0
    )

    parser.add_argument(
        "-s",
        "--patch_size",
        required=True
    )

    parser.add_argument(
        "-n",
        "--patch_n",
        required=True
    )

    parser.add_argument(
        "-r",
        "--rotate",
        action="store_true"
    )

    parser.add_argument(
        "-f",
        "--flip_y",
        action="store_true"
    )

    return parser

if __name__ == "__main__":
    main()