import argparse
import os
import sys
import warnings

import mrcfile
import numpy as np
import numpy.fft as fft
import pandas as pd
from FilterUtil import rad_avg, rot_kernel
from constants.config import Config


def match_spectrum(tomo, target_spectrum, cutoff=None, smooth=0):
    """
    Adjust a tomogram's amplitude spectrum to match the extracted spectrum of another tomogram.
    Arguments:
        tomo: the input tomogram as a 3D numpy array.
        target_spectrum: the target spectrum as a 1D numpy array.
        cutoff: apply a cutoff at this frequency (default: no cutoff)
        smooth: smoothen the cutoff into a sigmoid. Value roughly resembles width of sigmoid.
    """

    # Normalize tomogram
    target_spectrum = target_spectrum.copy()
    tomo -= tomo.min()
    tomo /= tomo.max()

    # Do FFT
    t = fft.fftn(tomo)
    t = fft.fftshift(t)

    del tomo

    # Get input tomogram's radially averaged amplitude spectrum, get equalization vector
    input_spectrum = rad_avg(np.abs(t))
    target_spectrum.resize(len(input_spectrum))
    equal_v = target_spectrum / input_spectrum

    if cutoff:
        if smooth:
            slope = len(equal_v) / smooth
            offset = 2 * slope * ((cutoff - len(equal_v) / 2) / len(equal_v))

            cutoff_v = 1 / (1 + np.exp(np.linspace(-slope, slope, len(equal_v)) - offset))

        else:
            cutoff_v = np.ones_like(equal_v)
            try:
                equal_v[cutoff:] = 0
            except IndexError:
                warnings.warn("Flat cutoff is higher than maximum frequency")

        equal_v *= cutoff_v

    # Create and apply equalization kernel
    equal_kernel = rot_kernel(equal_v, t.shape)

    t *= equal_kernel
    del equal_kernel

    # Inverse FFT
    t = fft.ifftn(t)
    t = np.abs(t).astype("f4")

    return t


def main():
    parser = get_cli()
    args = parser.parse_args()
    pythonpath = args.pythonpath
    sys.path.append(pythonpath)
    # gpu = args.gpu
    # if gpu is None:
    #     print("No CUDA_VISIBLE_DEVICES passed...")
    #     if torch.cuda.is_available():
    #         os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # else:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    config_file = args.config_file
    config = Config(user_config_file=config_file)
    df = pd.read_csv(config.dataset_table, dtype={"tomo_name": str})
    df.set_index('tomo_name', inplace=True)
    tomo_name = args.tomo_name
    input_tomo = df[config.processing_tomo][tomo_name]
    target_tomo = os.path.join(config.work_dir, tomo_name)
    target_tomo = os.path.join(target_tomo, "match_spectrum_filtered.mrc")
    with mrcfile.open(input_tomo, permissive=True) as m:
        tomo = m.data.astype("f4")
        tomo_h = m.header

    target_spectrum = pd.read_csv(target_tomo, sep="\t")["intensity"].values

    filtered_tomo = match_spectrum(tomo, target_spectrum, config.cutoff, config.smoothen)

    m = mrcfile.new(args.output, overwrite=True)
    m.set_data(filtered_tomo)
    m.set_extended_header(tomo_h)
    m.close()


def get_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pythonpath", "--pythonpath", type=str)
    parser.add_argument("-tomo_name", "--tomo_name", type=str)
    parser.add_argument("-fold", "--fold", type=str, default="None")
    parser.add_argument("-config_file", "--config_file", help="yaml_file", type=str)
    return parser


if __name__ == "__main__":
    main()
