import mrcfile

import sys

sys.path.append("/g/scb/mahamid/trueba/3d-unet/src")

import os
import os.path

import numpy as np
import pandas as pd
from scipy import spatial

from file_actions.writers.csv import motl_writer
from file_actions.readers.em import read_em
from file_actions.readers.tomograms import load_tomogram
from file_actions.writers.mrc import write_mrc_dataset
# from file_actions.writers.h5 import write_particle_mask_from_motl_in_score_range
from tomogram_utils.coordinates_toolbox.utils import \
    extract_coordinates_from_em_motl
from tomogram_utils.peak_toolbox.utils import paste_sphere_in_dataset
from file_actions.writers.csv import new_motl_writer

def read_mrc(file):
    with mrcfile.open(file, permissive=True) as f:
        return f.data


# dataset = '/g/scb/mahamid/trueba/data/hela/rotated_data/t2_forsegmentation.mrc'
# dataset = '/g/scb/mahamid/trueba/data/hela/rotated_data/Nuclear_envelope.mrc'
# dataset = '/g/scb/mahamid/trueba/data/hela/rotated_data/NPCs.mrc'
data = read_mrc(dataset).copy()
# data = 1*(data < 254)
# subtomo = data[150:250, 675:825, 300:450].copy()
# output_path = '/g/scb/mahamid/trueba/data/hela/rotated_data/subtomo_NPC.mrc'
# output_path = '/g/scb/mahamid/trueba/data/hela/rotated_data/NE.mrc'
# write_mrc_dataset(mrc_path=output_path, array=data)

from scipy.ndimage import interpolation
angle_y = 80
volume = interpolation.rotate(data, angle_y, order=0, mode='reflect', axes=(0, 2),
                              reshape=True)
output_path = '/g/scb/mahamid/trueba/data/hela/rotated_data/NE_rot.mrc'
# output_path = '/g/scb/mahamid/trueba/data/hela/rotated_data/t2_forsegmentation_rot.mrc'
# output_path = '/g/scb/mahamid/trueba/data/hela/rotated_data/NPCs_rot.mrc'
write_mrc_dataset(mrc_path=output_path, array=volume)




















