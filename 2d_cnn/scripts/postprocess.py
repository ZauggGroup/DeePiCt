import argparse
import numpy as np
import os
import mrcfile
from ConfigUtil import assemble_config, csv_list
from scipy.ndimage import gaussian_filter1d

def main():

    srcdir = os.path.dirname(os.path.realpath(__file__))
    parser = get_cli()
    args = parser.parse_args()

    # Configuration
    config = assemble_config(
        f"{srcdir}/defaults.yaml",
        args.config,
        subconfig_paths = [("postprocessing",)],
        cli_args = args
    )

    pred = mrcfile.open(args.input, permissive=True).data

    pred_gauss = gaussian_filter1d(pred, axis=0, sigma=config["sigma"])
    
    if config["threshold"]:
        pred_gauss = (pred_gauss > config["threshold"]).astype(np.float32)

    mrcfile.new(args.output, data=pred_gauss, overwrite=True).close()    


def get_cli():
    parser = argparse.ArgumentParser(
        description="Post-process organelle segmentation by using a 1D gaussian filter along the z-axis."
    )

    parser.add_argument( 
        "-i",
        "--input",
        required=True,
        help="Input segmentation."
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Save location for processed segmentation."
    )

    parser.add_argument( 
        "-s",
        "--sigma",
        required=False,
        help="Sigma for 1D gaussian filter."
    )

    parser.add_argument( 
        "-t",
        "--threshold",
        required=False,
        help="Threshold to apply to convert probabilities into binary labels."
    )

    parser.add_argument( 
        "-c",
        "--config",
        required=False,
        help="Configuration YAML file. Overrides defaults, overridden by CLI arguments."
    )

    return parser

if __name__ == "__main__":
    main()
