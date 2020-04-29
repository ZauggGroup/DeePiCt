import numpy as np
import numpy.fft as fft
import pandas as pd
import mrcfile

import argparse

from FilterUtils import rad_avg, rot_kernel


def match_spectrum(tomo, target_spectrum):
    
    tomo -= tomo.min()
    tomo /= tomo.max()

    t = fft.fftn(tomo)
    t = fft.fftshift(t)

    del tomo

    input_spectrum = rad_avg(np.abs(t))
    target_spectrum.resize(len(input_spectrum))
    equal_v = target_spectrum / input_spectrum

    # TODO: make this adjustable!!
    slope = 20
    offset = 10
    sigmoid = 1/(1 + np.exp(np.linspace(-slope, slope, len(equal_v)) - offset))

    equal_v *= sigmoid
    equal_kernel = rot_kernel(equal_v, t.shape)
    
    t *= equal_kernel
    t = fft.ifftn(t)
    t = np.abs(t)

    return t


def main():
    
    parser = get_cli()
    args = parser.parse_args()
    
    with mrcfile.open(args.input, permissive = True) as m:
        tomo = m.data.astype("f4")
        tomo_h = m.header

    target_spectrum = pd.read_csv(args.target, sep="\t")["intensity"].values()

    filtered_tomo = match_spectrum(tomo, target_spectrum)

    m = mrcfile.new(args.output)
    m.set_data(filtered_tomo)
    m.set_extended_header(tomo_h)
    m.close()


def get_cli():
    parser = argparse.ArgumentParser(
        description="Match tomogram to another tomogram's amplitude spectrum."
    )

    parser.add_argument( 
        "-i",
        "--input",
        required=True,
        help="Tomogram to match (.mrc/.rec)"
    )

    parser.add_argument( 
        "-t",
        "--target",
        required=True,
        help="Target spectrum to match the input tomogram to (.tsv)"
    )

    parser.add_argument( 
        "-o",
        "--output",
        required=True,
        help="Output location for matched tomogram"
    )
    
    return parser


if __name__ == "__main__":
    main()
