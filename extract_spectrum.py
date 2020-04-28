import numpy as np
import numpy.fft as fft
import pandas as pd

import argparse

from FilterUtils import rad_avg, read_mrc


def extract_spectrum(tomo):
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
        type=read_mrc
    )

    parser.add_argument( 
        "-o",
        "--output",
        required=True,
        type=str
    )
    
    return parser


if __name__ == "__main__":
    main()

    