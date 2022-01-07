import numpy as np
import warnings 

def into_patches(image, patch_shape, patch_n):
    """
    Process a 2D image into evenly spaced-out 2D patches.
    
    Arguments:
        image: image to process into patches as a 2D numpy array.
        patch_size: target size of patches: (height, width).
        patch_n: number of rows and columns of patches: (rows, columns).
    
    Returns:
        A stack of patches as a numpy array of shape (patch_n[0]*patch_n[1], y, x).
    """
    
    y_stride = (image.shape[0] - patch_shape[0]) / (patch_n[0] - 1) if patch_n[0] > 1 else 0
    x_stride = (image.shape[1] - patch_shape[1]) / (patch_n[1] - 1) if patch_n[1] > 1 else 0
    
    out = np.stack([
            image[
                int(row*y_stride):int(row*y_stride)+patch_shape[0],
                int(col*x_stride):int(col*x_stride)+patch_shape[1]
            ] for col in range(patch_n[1]) for row in range(patch_n[0])
        ])
    return out

def from_patches(patches, patch_n, target_shape, pad=0):
    """
    Assemble a 2D image stack from evenly spaced-out 2D patches.
    Overlapping areas will be averaged.

    Arguments:
        patches: stack of patches as a numpy array of shape (patch_n[0]*patch_n[1], y, x).
        patch_n: number of rows and columns of patches: (rows, columns).
        target_shape: target shape in which the patches shall be assembled into.
        pad: cropping to apply to patches on all sides.

    Returns:
        A "D assembly of the patches as a numpy array in the target shape.
    """
    
    y_stride = (target_shape[0] - patches.shape[1]) / (patch_n[0] - 1) if patch_n[0] > 1 else 0
    x_stride = (target_shape[1] - patches.shape[2]) / (patch_n[1] - 1) if patch_n[1] > 1 else 0
     
    canvas_shape = list(target_shape)+[2]
    if pad:
        patches = patches[:, pad:-pad, pad:-pad]
        canvas_shape[0] -= 2*pad
        canvas_shape[1] -= 2*pad

    coords = [
        (
            slice(int(y*y_stride), int(y*y_stride) + patches.shape[1]),
            slice(int(x*x_stride), int(x*x_stride) + patches.shape[2])
        ) for x in range(patch_n[1]) for y in range(patch_n[0]) 
    ]
        
    canvas = np.zeros(canvas_shape)
    
    for patch, coord in zip(patches, coords):
        canvas[coord] += np.stack([patch, np.ones(patch.shape)], -1)
    
    if np.any(canvas[...,-1] == 0):
        warnings.warn("zero-coverage regions detected")
    
    return canvas[...,~-1]/canvas[...,-1]

def into_patches_3d(image, patch_shape, patch_n):
    """
    Process a 3D image stack into evenly spaced-out 2D patches.
    
    Arguments:
        image: image stack to process into patches as a 3D numpy array.
        patch_size: target size of patches: (height, width).
        patch_n: number of rows and columns of patches: (rows, columns).
    
    Returns:
        A stack of patches as a numpy array of shape (patch_n[0]*patch_n[1]*z, y, x).
    """

    assert len(patch_shape) == len(patch_n), "Rank of patch shape and patch number need to match number of selected axis"
    
    y_stride = (image.shape[1] - patch_shape[0]) / (patch_n[0] - 1) if patch_n[0] > 1 else 0
    x_stride = (image.shape[2] - patch_shape[1]) / (patch_n[1] - 1) if patch_n[1] > 1 else 0
    
    out = np.vstack([
            image[
                :,
                int(row*y_stride):int(row*y_stride)+patch_shape[0],
                int(col*x_stride):int(col*x_stride)+patch_shape[1]
            ] for col in range(patch_n[1]) for row in range(patch_n[0])
        ])
    return out

def from_patches_3d(patches, patch_n, target_shape, pad=0):
    """
    Assemble a 3D image stack from evenly spaced-out 2D patches.
    Patches need to be grouped along first array axis by patch position, not by Z-slice; 
    this can be ensured by using PatchUtil.into_patches_3d to create patches.
    Overlapping areas will be averaged.

    Arguments:
        patches: stack of patches as a numpy array of shape (patch_n[0]*patch_n[1]*z, y, x).
        patch_n: number of rows and columns of patches: (rows, columns).
        target_shape: target shape in which the patches shall be assembled into.
        pad: cropping to apply to patches on all sides.

    Returns:
        A 3D assembly of the patches as a numpy array in the target shape.
    """
    # TODO: check whether optimizing this function is viable, counter channel could also just be 2D.
    
    y_stride = (target_shape[1] - patches.shape[1]) / (patch_n[0] - 1) if patch_n[0] > 1 else 0
    x_stride = (target_shape[2] - patches.shape[2]) / (patch_n[1] - 1) if patch_n[1] > 1 else 0
    
    canvas_shape = list(target_shape)+[2]
    if pad:
        patches = patches[:, pad:-pad, pad:-pad]
        canvas_shape[1] -= 2*pad
        canvas_shape[2] -= 2*pad
    
    unstacked = np.split(patches, patch_n[0]*patch_n[1])
        
    coords = [
        (
            slice(0, canvas_shape[0]),
            slice(int(y*y_stride), int(y*y_stride) + patches.shape[1]),
            slice(int(x*x_stride), int(x*x_stride) + patches.shape[2])
        ) for x in range(patch_n[1]) for y in range(patch_n[0]) 
    ]
    
    canvas = np.zeros(canvas_shape)
    
    for patch, coord in zip(unstacked, coords):
        canvas[coord] += np.stack([patch, np.ones(patch.shape)], -1)
    
    if np.any(canvas[...,-1] == 0):
        warnings.warn("zero-coverage regions detected")

    return canvas[...,~-1]/canvas[...,-1]