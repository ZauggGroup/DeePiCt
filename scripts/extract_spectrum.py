import numpy as np
import numpy.fft as fft
import pandas as pd

import argparse

from FilterUtil import rad_avg, read_mrc


def extract_spectrum(tomo):
    """
    Extract radially averaged amplitude spectrum from a tomogram
    """
    # Scale intensitites pre-fft
    tomo -= tomo.min()
    tomo /= tomo.max()

    t = fft.fftn(tomo)
    t = fft.fftshift(t)
    t = np.abs(t)

    spectrum = rad_avg(t)
    spectrum = pd.Series(spectrum, index = np.arange(len(spectrum)))

    return spectrum


def main():
    
    parser = get_cli()
    args = parser.parse_args()
    
    tomo = args.input

    spectrum = extract_spectrum(tomo)

    spectrum.to_csv(args.output, sep="\t", header=["intensity"], index_label="freq")


def get_cli():
    parser = argparse.ArgumentParser(
        description="Extract radially averaged amplitude spectrum from cryo-ET data."
    )

    parser.add_argument( 
        "-i",
        "--input",
        required=True,
        type=read_mrc,
        help="Tomogram to extract spectrum from (.mrc/.rec format)."
    )

    parser.add_argument( 
        "-o",
        "--output",
        required=True,
        help="Output destination for extracted spectrum (.tsv format)."
    )
    
    return parser


if __name__ == "__main__":
    main()

    