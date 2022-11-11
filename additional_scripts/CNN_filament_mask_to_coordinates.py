#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import mrcfile
from skimage.morphology import skeletonize_3d, medial_axis, remove_small_objects
from skimage.color import label2rgb
from scipy.ndimage import morphology, label, binary_erosion, convolve
from scipy.spatial import distance
from scipy.interpolate import splprep, splev
import math
import csv
import os
import sys
import utils

data_dict = { #Dictionary that stores the information of files in the batch processing
    'tomo_name': [], #name of tomogram
    'working_dir': [], #working directory that contains all tomo masks and ground truth (if needed)
    'tomo_file': [], #path to the CNN mask
    'voxel_size': [], #voxel size of tomogram
    'erode_size': [], #size of erosion in voxels, usually around filament diameter/3
    'min_fil_length': [], #minimum length to be considered a filament, in nm
    'resampling_steps':[], #distance for resampling, in nm
    'maxdist': [], #max dist for filaments to be joined
    'maxang': [], #max ang for filaments to be joined
    'cnn_statistics': [], #whether or not to calculate CNN statistics aka false positive and negative rates
    'ground_truth_file': [], #path to segmentation ground truth for calculating CNN 
    'output_dir': [] #output directory
}
keys = list(data_dict)

#Read the input config file and output log file
config_path = sys.argv[1]
log_path = sys.argv[2]
with open(config_path, 'r') as config_file: #reads the config file and adds the information to the dictionary
    line_num = 0
    for line in config_file:
        line = line.rstrip('\n')
        if line_num == 0:
            line_num += 1
        else:
            fields = line.split(', ')
            for i in range(len(fields)):
                data_dict[keys[i]].append(fields[i])
config_file.close()

#Open log file for writing
log_file = open(log_path, 'a')
#Analyze each tomogram
for i in range(len(data_dict['tomo_name'])):
    
    print('Analyzing tomogram mask ' + data_dict['tomo_name'][i])
    log_file.write('Analyzing tomogram mask ' + data_dict['tomo_name'][i] + '\n')

    #read data
    log_file.write('Reading tomogram mask...\n')
    data = utils.readmrc(data_dict['working_dir'][i] + data_dict['tomo_file'][i]) #read the mrc file of the mask
    
    #if calculation of false pos/neg rate is not needed, can activate to speed up processing
    #log_file.write('Reducing data size...\n')
    #reduced_data = reduce_data(data) 
    
    #2d hole-filling in all directions to account for segmentation artefacts
    log_file.write('Filling holes in data...\n')
    filled_data = utils.fill_holes(data) #can replace data w/ reduced_data if the reduction command is activated

    #erosion to get rid of artificial filament connections
    log_file.write('Eroding...\n')
    erode_size = int(data_dict['erode_size'][i])
    eroded_data = utils.erode(filled_data, np.ones((erode_size, erode_size, erode_size)))
    eroded_data = eroded_data.astype('float32')
#    writemrc(eroded_data, data_dict['output_dir'][i] + data_dict['tomo_name'][i] + '_test_erosion.mrc')
    
    #skeletonization to get midline
    log_file.write('Skeletonizing filaments...\n')
    skeleton_data = utils.skeletonize_filaments(eroded_data) #skeletonizing filaments, neighbourhood = 2
   
    log_file.write('Cleaning skeleton...\n')
#    cleaned_skeleton = clean_branches(skeleton_data)
    cleaned_skeleton = utils.clean_branches2(skeleton_data)
    cleaned_skeleton_write = cleaned_skeleton.astype('float32')
    
    log_file.write('Identifying individual filaments...\n')
    labelled_data, labelnum = utils.labeldata(cleaned_skeleton)
    log_file.write('A total of '+ str(labelnum) + ' filaments are identified.\n')

    cleaned_labels, labelnum = utils.init_clean(labelled_data, 4)

    log_file.write('Identifying points in each filament...\n')
    binned_data = utils.ID_filaments(cleaned_labels)

    log_file.write('Sorting points in each filament...\n')
    sorted_data = utils.sort_all_lines(binned_data)

    resampled_data = utils.resample_all(sorted_data, float(data_dict['resampling_steps'][i])/float(data_dict['voxel_size'][i]))

    cleaned_resampled_data = utils.clean_small(resampled_data, math.floor(float(data_dict['min_fil_length'][i])/(2*float(data_dict['resampling_steps'][i]))))
    data_orientation = utils.orientation_all(cleaned_resampled_data)
    df = utils.find_same_line(cleaned_resampled_data, data_orientation, float(data_dict['maxdist'][i]), float(data_dict['maxang'][i]))
    joined_data = utils.join_lines(cleaned_resampled_data, df)

    log_file.write('Calculating the length of each filament...\n')
    filament_lengths = utils.filament_length_all(joined_data)

    log_file.write('Filtering out small filaments...\n')
    filtered_joined_data, total_fil_length = utils.filter_length(joined_data, filament_lengths, float(data_dict['min_fil_length'][i]))
    log_file.write('A total of ' + str(len(filtered_joined_data)) + ' are kept after filtering.\n')
    log_file.write('The total filament length in the tomogram is ' + str(total_fil_length*float(data_dict['voxel_size'][i])) + ' nm.\n')

    resampled_filtered_data = utils.resample_all(filtered_joined_data, float(data_dict['resampling_steps'][i])/float(data_dict['voxel_size'][i]))

    log_file.write('Creating image after processing...\n')
#    val_image = create_grow_image(filtered_joined_data, np.shape(data)) #activate if needed to create an mrc file that contains a image grown 2 voxels in each direction from the skeleton for inspection
    val_image = utils.image_from_dict(resampled_filtered_data, np.shape(data))
    new_val_image = val_image.astype('float32')
    utils.writemrc(new_val_image, data_dict['output_dir'][i] + data_dict['tomo_name'][i] + '_after_cleaning_test.mrc')
    
    if data_dict['cnn_statistics'][i] == 'y': #calculating CNN statistics
        ground_truth = utils.readmrc(data_dict['working_dir'][i] + data_dict['ground_truth_file'][i])
        false_neg, false_pos = utils.cnn_statistics(data, ground_truth, data_dict['output_dir'][i] + data_dict['tomo_name'][i] + '_false_neg.mrc', data_dict['output_dir'][i] + data_dict['tomo_name'][i] + '_false_pos.mrc')
        log_file.write('The false negative rate is ' + str(false_neg) + '\n')
        log_file.write('The false positive rate is ' + str(false_pos) + '\n')

    log_file.write('Writing coordinates as csv file...\n')
    output_file = data_dict['output_dir'][i] + data_dict['tomo_name'][i] + '_coord.csv'
    voxel_size = float(data_dict['voxel_size'][i])
    with open(output_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Point ID", "X", "Y", "Z", "Filament ID"])
        k = 1
        for a in range(1, len(resampled_filtered_data)+1):
            for b in range(len(resampled_filtered_data[a])):
                writer.writerow([k, resampled_filtered_data[a][b][2]*voxel_size, resampled_filtered_data[a][b][1]*voxel_size, resampled_filtered_data[a][b][0]*voxel_size, a, b+1])
                k = k+1
    log_file.write('Analysis complete.\n')
    log_file.write('\n')

log_file.close()
