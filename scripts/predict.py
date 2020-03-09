import argparse
from keras.models import Model, load_model
import keras.backend as K
import numpy as np
import mrcfile

from PatchUtil import *
from ConfigUtil import assemble_config

from UNet import dice_coefficient, neg_dice_coefficient

# TODO: main()

srcdir = os.path.dirname(os.path.realpath(__file__))
parser = get_cli()
args = parser.parse_args()

config = assemble_config(
    f"{srcdir}/defaults.yaml",
    args.config,
    subconfig_paths = [("prediction")],
    cli_args = args
)


# TODO CLI
tomo_file = args.input
out_file = args.output

# TODO from config:
patch_size = (288, 288)
pad = 48
compensate_pad
z_mask_n = 250
patch_n = (5,5)
out_file
model_file

model = load_model(
    model_file
    custom_objects={
        'neg_dice_coefficient':neg_dice_coefficient,
        "dice_coefficient":dice_coefficient
    }
)

tomo = read_mrc(tomo_in).astype(np.float32)

z_mask_n = min(tomo.shape[0], z_mask_n)
z_center = tomo.shape[0] // 2
z_idx = slice(z_center-(z_count // 2), z_center+(z_count // 2))


tomo = tomo[z_idx]

mean = tomo.mean()
std = tomo.std()

tomo -= mean
tomo /= std

tomo_patches = np.expand_dims(into_patches_3d(tomo, patch_size, patch_n), -1) # Add channel dim
tomo_pred = model.predict(tomo_patches)
rec = from_patches_3d(tomo_pred[...,0], (5, 5), tomo.shape, pad=pad)

mrcfile.new(out_file, data=rec.astype(np.float32), overwrite=True).close()


def read_mrc(file):
    with mrcfile.open(file, permissive=True) as f:
        return f.data

def get_cli():
    # TODO: CLI documentation
    parser = argparse.ArgumentParser(
        description="Predict organelle segmentation from tomogram data."
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