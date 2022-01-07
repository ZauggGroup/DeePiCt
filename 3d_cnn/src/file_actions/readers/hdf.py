import h5py
import numpy as np
from constants import h5_internal_paths


def _load_hdf_dataset(hdf_file_path: str) -> np.array:
    """
    Example:
        path_to_file: 'eman2/ribo3/image.txt'
        internal_path: 'MDF/images/0/image'

        NOTE:
        The internal path can be found by:
        import h5py
        hdf_file = h5py.File(path_to_file)
        key0 = list(hdf_file)[0]
        key1 = list(hdf_file[0])[0]
        ...
        successfully taking outputs as keys, e.g.:
        list(hdf_file[key0][key1][key2][key3])
        until one gets the voxel values
        The internal path has the structure:
        internal_path = 'key0/key1/.../keyn'
    """
    # hdf_file = h5py.File(path_to_file)
    # dataset = hdf_file['MDF']['images']['0']['image']
    with h5py.File(hdf_file_path, 'r') as f:
        dataset = f[h5_internal_paths.HDF_INTERNAL_PATH][:]
    return dataset


def _convert_to_array(dataset):
    return np.array(dataset)


def load_eman2_ds_as_array(path_to_file: str):
    ds = _load_hdf_dataset(path_to_file)
    return _convert_to_array(ds)
