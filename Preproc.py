import pandas as pd
import numpy as np
import mrcfile
import h5py
import os
import argparse # TODO: CLI

from PatchUtil import *


meta = pd.read_csv("~/mattausc/data/200207_3d_segmentation/metadata.csv", dtype=str) # No zero paddin
meta.id = meta.id.apply(lambda x: f"{x:>03}")
meta.from_em = meta.from_em.astype(int)

a = True
for file in meta[["data", "labels"]].values.flat:
    if not os.path.exists(file):
        print("File not found:", file) # TODO: raise exception instead
        a = False
else:
    if a:
        print("All files found!")

# TODO: ove this to YAML file
z_stride = 5 # Z-distance between slices
crop = 64 # Crop image before extracting patches
patch_size = 288 
max_patches = 25
rotate = True
out_basepath = f"/struct/mahamid/mattausc/data/200207_3d_segmentation/"

processed_tomos = []
patch_shape = (patch_size,)*2
patch_n = (int(np.sqrt(max_patches)),)*2

for idx, sample_meta in meta.iloc[-2:,:].iterrows():
    sample_id = sample_meta["date"] + "_" + sample_meta["id"]
    print(f"Processing tomogram {sample_id}...")

    # Load data + labels
    labels = read_mrc(sample_meta["labels"])
    if not sample_meta["from_em"]:
        labels = np.flip(labels, 1) # For some reason MRC file orientation likes to get messed up
    features = read_mrc(sample_meta["data"])

    assert labels.shape == features.shape, "Tomogram data and labels have mismatching shape!"

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
    patch_stack = [into_patches(image, patch_shape, patch_n) for image in stack]
    
    # Stack slice patches
    patch_stack = np.vstack(patch_stack)
    patch_stack = patch_stack.astype(np.float32)

    # Rotate patches
    if rotate:
        patch_stack = np.stack([np.rot90(patch, np.random.randint(4)) for patch in patch_stack])

    # Finish padding by cutting back labels again
    processed_features = patch_stack[...,0]
    processed_labels = patch_stack[...,1]
    
    print(f"Created {patch_stack.shape[0]} chunks across {stack.shape[0]} slices.")
    
    processed_tomos.append((sample_id, patch_stack[...,0], patch_stack[...,1]))



for sample_id, features, labels in processed_tomos:
    # TODO: do this straightaway
    out_file = f"{sample_id}_zstride-{z_stride}_size-{patch_size}.h5"
    print("Writing to", out_file)
    with h5py.File(out_basepath + out_file, "w") as h:
        h.attrs["sample_id"] = sample_id
        h.create_dataset("features", data=features)
        h.create_dataset("labels", data=labels)