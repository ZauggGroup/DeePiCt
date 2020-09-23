import numpy as np
import numpy.fft as fft
import mrcfile
from scipy import ndimage
from scipy.interpolate import interp1d
import scipy.stats as st

def read_mrc(file):
    """
    Read in a tomogram in MRC or REC format and return its data as a numpy array.
    """

    with mrcfile.open(file, permissive=True) as m:
        return m.data.astype(np.float32)

def resize(arr, target_shape, **kwargs):
    """
    Resize an array to a target shape.
    The input will be centered in the new shape and cropped or padded to fit the new shape.
    The target shape can be larger and smaller in any dimension.
    Kwargs will be forwarded to np.pad().
    """
    idx = []
    for a, b in zip(arr.shape, target_shape):
        if b == -1:
            idx.append(
                [
                    slice(None),
                    (0, 0)
                ]
            )
            continue

        dif = a-b
        
        before = dif//2
        after = dif-before
        
        if dif > 0:
            idx.append(
                [
                    slice(before, -after),
                    (0, 0)
                ]
            )
        else:
            idx.append(
                [
                    slice(None),
                    (-before, -after)
                ]
            )
       
    slice_idx, pad_idx = zip(*idx)
    
    arr = arr[slice_idx]
    arr = np.pad(arr, pad_idx, **kwargs)
        
    return arr


def hypot_nd(axes, offset=0.5):
    """
    Recursive wrapper for np.hypot to get n-dimensional distances to an origin.
    Offset determines where in the axes the origin is located.
    """
    if len(axes) == 2:
        return np.hypot(
            axes[0] - max(axes[0].shape) * offset,
            axes[1] - max(axes[1].shape) * offset,
        )
    else:
        return np.hypot(
            hypot_nd(axes[1:], offset),
            axes[0] - max(axes[0].shape) * offset,
        )

    
def rad_avg(image):
    """
    Radially average an array around the array's center.
    """
    bins = np.max(image.shape)/2

    axes = np.ogrid[tuple(slice(0,s) for s in image.shape)]
    r = hypot_nd(axes)
    
    rbin = (bins*r/r.max()).astype(np.int)
    radial_mean = ndimage.mean(image, labels=rbin, index=np.arange(1, rbin.max()+1))
    
    return radial_mean


def rot_kernel(arr, shape):
    """
    Create a centered n-dimensional rotational kernel from a 1D array.
    """
    func = interp1d(np.arange(len(arr)), arr, bounds_error=False, fill_value=0)

    axes = np.ogrid[tuple(slice(0, np.ceil(s/2)) for s in shape)]
    kernel = hypot_nd(axes, offset=0).astype("f4")
    kernel = func(kernel).astype("f4")
    for idx, s in enumerate(shape):
        padding = [(0,0)]*len(shape)
        padding[idx] = (int(np.floor(s/2)), 0)
        
        mode = "reflect" if s % 2 else "symmetric"
        
        kernel = np.pad(kernel, padding, mode=mode)
    
    return kernel
