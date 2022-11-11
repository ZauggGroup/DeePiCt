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

def readmrc(filename): #read mrc file
    with mrcfile.open(filename) as file:
        data = file.data
    return (data)

def writemrc(data, newfile): #write mrc file
    with mrcfile.new(newfile, overwrite = True) as file:
        file.set_data(data)

def reduce_data(data): #delete all Z-stacks that have no filaments
    x = np.nonzero(data)
    reduced_data = data [min(x[0]):max(x[0])+1,:,:]
    return (reduced_data)

def fill_holes(data): #fill holes in all 3 dimensions using 2D slices
    filled_data = data.astype('float32')
    for i in range(np.shape(filled_data)[0]):
        filled_data [i,:,:] = morphology.binary_fill_holes(filled_data[i,:,:]).astype('float32')
    for i in range(np.shape(filled_data)[1]):
        filled_data [:,i,:] = morphology.binary_fill_holes(filled_data[:,i,:]).astype('float32')
    for i in range(np.shape(filled_data)[2]):
        filled_data [:,:,i] = morphology.binary_fill_holes(filled_data[:,:,i]).astype('float32')
    return (filled_data)

def erode(data, kernel): #binary erosion
    return binary_erosion(data, structure=kernel)

def skeletonize_filaments(data): #skeletonize data and set to binary
    skeleton_data = skeletonize_3d(data)
    #set all non-zero voxels to 1
    skeleton_data_bin = (skeleton_data > 0).astype(int)
    return (skeleton_data_bin)

def clean_edge(point, arr, data):
    #pick one randomy and delete all other pixels
    for i in range(len(arr)-1):
        data[point[0] + arr[i][0],point[1] + arr[i][1],point[2] + arr[i][2]] = 0

def clean_not_edge(point, arr, data):
    #find longest pairwise distance
    #To do: change to find largest pairwise angle using cos
    small = 1
    keep_one = np.array([])
    keep_two = np.array([])
    print(point, len(arr))
    for i in range(len(arr)):
        for j in range(i):
            a_points = [[0,0,0], arr[i]]
            b_points = [[0,0,0], arr[j]]
            c_points = [arr[i], arr[j]]
            a = distance.pdist(a_points)
            b = distance.pdist(b_points)
            c = distance.pdist(c_points)
            cos_C = (a*a+b*b-c*c)/(2*a*b)
            if cos_C < small:
                small = cos_C
                keep_one = arr[i]
                keep_two = arr[j]
                print(small, keep_one, keep_two)
    #everything pixel is not the two furthest points connected to a junction gets deleted
    for i in arr:
        if (np.array_equal(i, keep_one)) or (np.array_equal(i, keep_two)):
            continue
        data[point[0]+i[0],point[1]+i[1],point[2]+i[2]] = 0

def get_neighbours(kernel):
    neighbours = np.argwhere(kernel == 1)
    neighbours = np.delete(neighbours, np.argwhere(np.all(neighbours == 1, axis = 1)), axis = 0)
    return (neighbours-[1,1,1])

def clean_branches(data): #clean branches in skeleton
    #change data type and define initial values
    #maybe implement memory cache and numba.njit
    new_data = np.pad(data.astype(int), ((1,1), (1,1), (1,1)))
    convolve_kernel = np.ones((3,3,3))
    convolve_kernel[1,1,1] = 0

    #define position coordinates for neighbours
    for i in range(1, np.shape(new_data)[0]-1):
        print('Cleaning slice ' + str(i) + ' ...')
        for j in range(1, np.shape(new_data)[1]-1):
            for k in range(1, np.shape(new_data)[2]-1):
                if new_data[i, j, k] == 1: #only check filaments to reduce runtime
                    neighbours = get_neighbours(new_data[i-1:i+2, j-1:j+2, k-1:k+2])
                    if (i == 1) or (i == np.shape(new_data)[0]-2) or (j == 1) or (j == np.shape(new_data)[1]-2) or (k == 1) or (k == np.shape(new_data)[2]-2): 
                        if len(neighbours) > 1:
                            clean_edge([i,j,k], neighbours, new_data) #delete neighbours until 1 is left
                    else:
                        if len(neighbours) > 2:
                            clean_not_edge([i,j,k], neighbours, new_data) #delete neighbours until 2 is left

    new_data = new_data.astype('float32')
    return (new_data[1:-1, 1:-1, 1:-1])

def clean_branches2(data):
    convolve_kernel = np.ones((3,3,3))
    convolve_kernel[1,1,1] = 0
    convolved_data = convolve(data, convolve_kernel, mode = 'constant', cval = 0)
    cleaned_data = data
    cleaned_data[np.logical_and(data == 1, convolved_data>=3)] = 0
    return(cleaned_data)

def labeldata(data):
    #label each individual filmanet after cleaning skeleton, connectivity = 2 (26-connect in 3D)
    str_3D=np.ones((3,3,3), dtype = int)
    labelled_data, num = label(data, structure = str_3D)
    return (labelled_data, num)

def init_clean(data, init_threshold):
    #Input parameters that determine filter threshold
    #This is only a rough initial filtering to decrease processing time, finer filterings will come later
    cleaned_labels = remove_small_objects(data, min_size=init_threshold, connectivity=2)
    labelled_data, num = labeldata(cleaned_labels) 
    return (labelled_data, num)

def ID_filaments(labelled_data):
    #create a dictioary of the coordinates that belong to each label

    idx = np.nonzero(labelled_data) #identify coordinates of non-background points
    vals = labelled_data[idx] #values of non-background points
    sort_idx = np.argsort(vals, kind = 'mergesort') #sort values of non-background points
    cuts, = np.nonzero(np.diff(vals[sort_idx], prepend = 0)) #count number of points in each label (by taking non-zero of difference)
    groups = np.split(np.stack(idx, axis = 1)[sort_idx], cuts[1:]) 
    #organize coordinates of non-background points in [z,y,x] (np.stack, axis = 1)
    #re-organiize them in ascending order of their labels ([sort_idx])
    #split the array by number of points in each label, so each label is stored in separate arrays
    binned_points = dict(zip(vals[sort_idx[cuts]], groups)) #create output dictionary
    return (binned_points)

def are_neighbours(pt1, pt2):
    #Check if pt1 and pt2 are neighbours, in the 26-point sense
    return (np.abs(pt1[0]-pt2[0]) < 2) and (np.abs(pt1[1]-pt2[1]) < 2) and (np.abs(pt1[2]-pt2[2]) < 2)

def sort_to_form_line(unsorted_list):
    #Given a list of neighboring points which forms a line, but in random order, 
    #sort them to the correct order.
    #IMPORTANT: Each point must be a neighbor (26-point sense) 
    #to a least one other point!
    copy_list = unsorted_list.tolist()
    sorted_list = [copy_list.pop(0)]

    while len(copy_list) > 0:
        i = 0
        while i < len(copy_list):
            if are_neighbours(sorted_list[0], copy_list[i]):
                #neighbours at front of list
                sorted_list.insert(0, copy_list.pop(i))
            elif are_neighbours(sorted_list[-1], copy_list[i]):
                #neighbours at rear of list
                sorted_list.append(copy_list.pop(i))
            else:
                i = i+1

    return sorted_list

def sort_all_lines(dict_of_lines):
    #Given a dictionary, where each entry is an array of unsorted points along a line, sort all arrays in dictiomnary
    sorted_data = {}   
    for i in range(1,len(dict_of_lines)+1):
        sorted_data[i] = sort_to_form_line(dict_of_lines[i])
    return(sorted_data)

def dist(pt1, pt2):
    #Euclidean distance of 2 points in 3D
    return math.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2+(pt1[2]-pt2[2])**2)

def spline_resample_line(array_of_points, undersample):
    tck, u = splprep(np.stack(array_of_points, axis = 1), s=0) #create spline fit of lines
    oversample_param = np.linspace(0, 1, num = len(array_of_points)*10) #oversample the line
    oversampled_points = np.stack(splev(oversample_param, tck), axis = 1) #change back to [x,y,z] format
    length = 0
    resampled_points = [oversampled_points[0]]
    for i in range(len(oversampled_points)-1): #delete points in oversampled line until points are equidistant with dist similar to undersampling threshold
        length = length + dist(oversampled_points[i], oversampled_points[i+1])
        if length < undersample:
            continue
        else:
            resampled_points.append(oversampled_points[i+1])
            length = 0
    if (resampled_points[-1] != oversampled_points[-1]).any:
        resampled_points[-1] = oversampled_points[-1]
    return (resampled_points)

def resample_all(dict_of_lines, undersample):
    resampled_lines = {}
    for i in range(1,len(dict_of_lines)+1):
        resampled_lines[i] = spline_resample_line(dict_of_lines[i], undersample)
    return(resampled_lines)

def clean_small(dict_of_lines, min_size):
    new_dict = {}
    curr = 1
    for i in range(1,len(dict_of_lines)+1):
        if len(dict_of_lines[i]) < min_size:
            continue
        new_dict[curr] = dict_of_lines[i]
        curr += 1
    return(new_dict)

def image_from_dict(dict_of_lines, dim):
    #Will create a 3D image from the filaments remaining after processing, that is grown 2 pixels in each direction. 
    image = np.zeros(dim).astype('float32')
    for i in range(1, len(dict_of_lines)+1):
        for j in range(len(dict_of_lines[i])):
            image[round(dict_of_lines[i][j][0]),round(dict_of_lines[i][j][1]),round(dict_of_lines[i][j][2])] = 1
    return image    

def orientation_line(array_of_points):
    orientation = []
    for i in range(len(array_of_points)):
        if i == 0:
            orientation.append(array_of_points[i+2]-array_of_points[i])
        elif i == len(array_of_points)-1:
            orientation.append(array_of_points[i]-array_of_points[i-2])
        else:
            orientation.append(array_of_points[i+1]-array_of_points[i-1])
    for i in range(len(orientation)):
        orientation[i] = orientation[i]/np.linalg.norm(orientation[i])
    return (orientation)

def orientation_all(dict_of_lines):
    dict_of_orientation = {}
    for i in range(1, len(dict_of_lines)+1):
        dict_of_orientation[i] = orientation_line(dict_of_lines[i])
    return(dict_of_orientation)

def relative_orientation(ori1, ori2):
    #relative angle given two orientation vectors, from 0 to 90 degrees
    angle = np.arccos(np.clip(np.dot(ori1, ori2), -1.0, 1.0))*180/np.pi
    if angle > 90:
        angle = 180 - angle
    return (angle)

def find_neighbours(pt, maxdist, array_of_points):
    #given a origin point, a search dist and an array of points to search, return boolean array (with length array_of_points) with True if point is in search dist from origin pt
    neighbours = np.zeros(len(array_of_points), dtype = bool)
    for i in range(len(array_of_points)):
        if dist(pt, array_of_points[i]) <= maxdist: #if distance < maxdist
            if np.any(np.not_equal(pt,array_of_points[i])) == True: #exclude if point is within array
                neighbours[i] = True
    return(neighbours)

def is_parallel(pt, vector, maxang, array_of_points, array_of_vectors):
    #given a unit vector and an array of other unit vectors, return boolean array (with length array_of_vector) with True if the angles between the vectors is less than maxang
    parallel = np.zeros(len(array_of_vectors), dtype = bool)
    for i in range(len(array_of_vectors)):
        if relative_orientation(vector, array_of_vectors[i]) <= maxang:
            relative_orientation_vector = (pt - array_of_points[i])/np.linalg.norm(pt - array_of_points[i])
            if (relative_orientation(vector, relative_orientation_vector) <= maxang) or (relative_orientation(array_of_vectors[i], relative_orientation_vector) <= maxang):
                parallel[i] = True
    return(parallel)

def find_same_line(dict_of_lines, dict_of_orientation, maxdist, maxang):
    #create list of all endpoints and endpoint vectors
    endpoints = [dict_of_lines[1][0], dict_of_lines[1][-1]]
    line_num = [1,1]
    for i in range(2, len(dict_of_lines)+1):
        endpoints = np.concatenate((endpoints, [dict_of_lines[i][0]]))
        endpoints = np.concatenate((endpoints, [dict_of_lines[i][-1]]))
        line_num.append(i)
        line_num.append(i)
    endpoint_vectors = [dict_of_orientation[1][0], dict_of_orientation[1][-1]]
    for i in range(2, len(dict_of_orientation)+1):
        endpoint_vectors = np.concatenate((endpoint_vectors, [dict_of_orientation[i][0]]))
        endpoint_vectors = np.concatenate((endpoint_vectors, [dict_of_orientation[i][-1]]))
    #initialize dictionaries
    join_points_dict = [] #index of points to join
    join_line_dict = [] #index of filaments to join
    start_end = []
    #for each point, search for neighbours
    for i in range(len(endpoints)):
        if i%2 == 0:
            same_filament = i+1
            start_end.append(0)
        else:
            same_filament = i-1
            start_end.append(1)
        neighbours_list = find_neighbours(endpoints[i], maxdist, endpoints)
        neighbours_list[same_filament] = False
        parallel_list = is_parallel(endpoints[i], endpoint_vectors[i], maxang, endpoints, endpoint_vectors)
        condition_list = np.logical_and(neighbours_list, parallel_list)
        join_points_list = np.flatnonzero(condition_list)
        join_line_list = np.ndarray.flatten(np.argwhere(condition_list == True)//2 + 1)
        join_points_dict.append(join_points_list)
        join_line_dict.append(join_line_list)
    is_paired = np.zeros(len(join_points_dict)).astype('bool')
    for i in range(len(join_points_dict)):
        if len(join_points_dict[i])==1:
            partner = join_points_dict[i][0]
            if len(join_points_dict[partner]==1) and (join_points_dict[partner][0]==i):
                is_paired[i] = True 
    #create df of all endpoints
    df = pd.DataFrame({'coord':list(endpoints), 'orientation':list(endpoint_vectors), 'filament': line_num, 'start_end':start_end, 'neighbours':list(join_points_dict), 'neighbour_filaments':list(join_line_dict), 'is_paired':is_paired})
    #coord is coordinates in z, x, y
    #orientation is unit orientation vector in dz, dx, dy
    #line_num is filament number the endpoint belongs to
    #start_end is 0 if point is "start" of the line and 1 if "end"
    #neighbours is endpoint index of potential pairing partners for joining lines
    #neighbour_filaments is line index of potential pairing partners for joining lines
    #is_paired means the endpoint has 1 non-ambiguious pairing partner
    return df

def other_end(i):
    if i%2 == 0:
        other_end = i+1
    else:
        other_end = i-1
    return (other_end)

def extend_start(dict_of_lines, df, i, line, processed_line):
    line_idx = df.loc[i, 'filament']
    if line_idx not in processed_line:
        processed_line.append(line_idx)
    if df.loc[i, 'is_paired'] == False:
        return(line, processed_line)
    partner = df.loc[i, 'neighbours'][0]
    partner_line_idx = df.loc[partner, 'filament']
    if df.loc[partner, 'start_end'] == 1:
        new_line = np.concatenate((dict_of_lines[partner_line_idx], line))
    if df.loc[partner, 'start_end'] == 0:
        new_line = np.concatenate((np.flip(dict_of_lines[partner_line_idx], axis = 0), line))
    new_start = other_end(partner)
    new_line, processed_line = extend_start(dict_of_lines, df, new_start, new_line, processed_line)
    return(new_line, processed_line)

def extend_end(dict_of_lines, df, i, line, processed_line):
    line_idx = df.loc[i, 'filament']
    if line_idx not in processed_line:
        processed_line.append(line_idx)
    if df.loc[i, 'is_paired'] == False:
        return(line, processed_line)
    partner = df.loc[i, 'neighbours'][0]
    partner_line_idx = df.loc[partner, 'filament']
    if df.loc[partner, 'start_end'] == 0:
        new_line = np.concatenate((line, dict_of_lines[partner_line_idx]))
    if df.loc[partner, 'start_end'] == 1:
        new_line = np.concatenate((line, np.flip(dict_of_lines[partner_line_idx], axis = 0)))
    new_end = other_end(partner)
    processed_line.append(partner_line_idx)
    new_line, processed_line = extend_end(dict_of_lines, df, new_end, new_line, processed_line)
    return(new_line, processed_line)

def join_lines(dict_of_lines, df):
    curr = 1
    processed = np.zeros(len(dict_of_lines)).astype('bool')
    new_dict = {}
    for i in range(len(processed)):
        if processed[i]:
            continue
        joined_start, processed_idx = extend_start(dict_of_lines, df, 2*i, dict_of_lines[i+1], [])
        joined, processed_idx = extend_end(dict_of_lines, df, 2*i+1, joined_start, processed_idx)
        for j in processed_idx:
            processed[j-1] = True
        new_dict[curr] = joined
        curr += 1
    return(new_dict)


def filament_length(list_of_points):
    #Calculate filament length given sorted list of points
    length = 0
    for i in range(len(list_of_points)-1):
        length = length + dist(list_of_points[i], list_of_points[i+1])
    return (length)

def filament_length_all (dict_of_lines):
    #Calculate filament length for all filaments, taking note that the output index starts from 0 while the dict index starts from 1
    filament_lengths = []
    for i in range(1, len(dict_of_lines)+1):
        filament_lengths.append(filament_length(dict_of_lines[i]))
    return (filament_lengths)

def filter_length(dict_of_lines, filament_lengths, threshold):
    #remove entries that have lengths smaller than the threshold, and re-organize so the indexes are continuous
    total_fil_length = 0
    filtered_dict = {}
    i = 1
    for j in range(1, len(dict_of_lines)+1):
        if filament_lengths[j-1] >= threshold:
            total_fil_length = total_fil_length + filament_lengths[j-1]
            filtered_dict[i] = dict_of_lines[j]
            i = i+1
    return (filtered_dict, total_fil_length)

#The following are functions used for visualizing the false positive and negative rate of the CNN

def create_grow_image(dict_of_lines, dim):
    #Will create a 3D image from the filaments remaining after processing, that is grown 2 pixels in each direction. 
    image = np.zeros(dim).astype(int)
    for i in range(1, len(dict_of_lines)+1):
        for j in range(len(dict_of_lines[i])):
            put_as_one(image, dict_of_lines[i][j], 2)
    return image            

def point_is_in_bounds(point, shape): #check point is within 3D array boundaries
    for i in range(3):
        if (point[i] < 0):
            return False
        if (point[i] > shape[i]-1):
            return False
    return True

def put_as_one(image, point, growth_size):
    for x in range(-growth_size, growth_size+1):
        for y in range(-growth_size,growth_size+1):
            for z in range(-growth_size, growth_size+1):
                if (point_is_in_bounds([point[0]+x,point[1]+y,point[2]+z], np.shape(image))):
                    image[point[0]+x, point[1]+y, point[2]+z] = 1

def put_as_zero(image, point, growth_size): #substract ground truth from CNN output, or vice versa, when they are within a range of growth size in each direction
    for x in range(-growth_size, growth_size+1):
        for y in range(-growth_size,growth_size+1):
            for z in range(-growth_size, growth_size+1):
                if (point_is_in_bounds([point[0]+x,point[1]+y,point[2]+z], np.shape(image))):
                    image[point[0]+x, point[1]+y, point[2]+z] = 0

def cnn_statistics(data, ground_truth, false_neg_file, false_pos_file): #for calculating false positive and negative rates
    int_data = data.astype('int')
    false_pos = data.astype('int')
    int_ground_truth = ground_truth.astype('int')
    false_neg = ground_truth.astype('int')
    #subtract CNN output from ground truth for calculating false negative
    for i in range(np.shape(false_neg)[0]):
        for j in range(np.shape(false_neg)[1]):
            for k in range(np.shape(false_neg)[2]):
                if int_data[i, j, k] == 1:
                    put_as_zero(false_neg, [i,j,k], 2)
    #subtract ground truth from CNN output for calculating false positive
    for i in range(np.shape(false_pos)[0]):
        for j in range(np.shape(false_pos)[1]):
            for k in range(np.shape(false_pos)[2]):
                if int_ground_truth[i, j, k] == 1:
                    put_as_zero(false_pos, [i,j,k], 2)
    #calculation of false positive and negative rates
    false_neg_CNN = np.sum(false_neg)/np.sum(int_ground_truth)
    false_pos_CNN = np.sum(false_pos)/np.sum(int_data)
    writemrc(false_neg.astype('float32'), false_neg_file)
    writemrc(false_pos.astype('float32'), false_pos_file)
    return (false_neg_CNN, false_pos_CNN)