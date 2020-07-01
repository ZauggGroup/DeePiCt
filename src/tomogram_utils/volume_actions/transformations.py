import numpy as np
import scipy.ndimage as scimg


def rotate_ref(ref: np.array, zxz_angles_in_degrees: tuple,
               axis_in_tom_format=False, mode = 'constant'):
    """
    Following tom_rotate conventions, in this script we rotate a reference image
    according to an Euler angle tuple (phi, psi, theta), where:
    1. a rot around z of angle phi is applied first
    2. a rot around x of angle theta is applied second
    3. a rot around z of angle psi is applied third

    :param ref: reference array to be rotated
    :param zxz_angles_in_degrees: Euler angles (phi, psi theta)
    :param axis_in_tom_format: Boolean, if True, a reference coordinate system
    XYZ is assumed.
    :param mode: str, describing mode in scimg rotation interpolation. For more
    info check please scipy.ndimage.interpolation.rotate
    :return: zxz_rotated_ref rotated reference
    """
    phi, psi, theta = zxz_angles_in_degrees

    z_rotated_ref = np.zeros(ref.shape)
    xz_rotated_ref = np.zeros(ref.shape)
    zxz_rotated_ref = np.zeros(ref.shape)

    if axis_in_tom_format:
        scimg.rotate(input=ref, angle=phi, axes=(0, 1), reshape=False,
                     output=z_rotated_ref, order=1, mode=mode, cval=0.0,
                     prefilter=False)
        scimg.rotate(input=z_rotated_ref, angle=theta, axes=(1, 2),
                     reshape=False, output=xz_rotated_ref, order=1,
                     mode=mode, cval=0.0, prefilter=False)
        scimg.rotate(input=xz_rotated_ref, angle=psi, axes=(0, 1),
                     reshape=False, output=zxz_rotated_ref, order=1,
                     mode=mode, cval=0.0, prefilter=False)
    else:
        zxz_rotated_ref_inverted = np.zeros(ref.shape)
        scimg.rotate(input=ref, angle=phi, axes=(2, 1), reshape=False,
                     output=z_rotated_ref, order=1, mode=mode, cval=0.0,
                     prefilter=False)
        scimg.rotate(input=z_rotated_ref, angle=theta, axes=(1, 0),
                     reshape=False, output=xz_rotated_ref, order=1, mode=mode,
                     cval=0.0, prefilter=False)
        scimg.rotate(input=xz_rotated_ref, angle=psi, axes=(2, 1),
                     reshape=False, output=zxz_rotated_ref_inverted, order=1,
                     mode=mode, cval=0.0, prefilter=False)
        zxz_rotated_ref = zxz_rotated_ref_inverted[:, ::-1, :]
    return zxz_rotated_ref


def paste_reference(dataset: np.array, ref: np.array, center: tuple,
                    axis_in_tom_format=False):
    """

    :param dataset: np.array where the reference will be pasted
    :param ref: np.array to be pasted
    :param center: tuple (cx, cy, cz), indices of the center where ref will
    be pasted in dataset
    :param axis_in_tom_format: if True, then an XYZ coordinate system is
    assumed. Otherwise, a ZYX is considered.
    """
    cx, cy, cz = center
    ref_center = [sh // 2 for sh in ref.shape]
    index_0, index_1, index_2 = np.where(ref > 0)
    if axis_in_tom_format:
        r_cx, r_cy, r_cz = ref_center
        for x, y, z in zip(index_0, index_1, index_2):
            point_index = np.array(
                [x - r_cx + cx, y - r_cy + cy, z - r_cz + cz])
            if (np.array([0, 0, 0]) <= point_index).all() and (
                point_index < np.array(dataset.shape)).all():
                dataset[x - r_cx + cx, y - r_cy + cy, z - r_cz + cz] = np.max(
                    [ref[x, y, z],
                     dataset[x - r_cx + cx, y - r_cy + cy, z - r_cz + cz]])
    else:
        r_cz, r_cy, r_cx = ref_center
        for z, y, x in zip(index_0, index_1, index_2):
            point_index = np.array(
                [z - r_cz + cz, y - r_cy + cy, x - r_cx + cx])
            if (np.array([0, 0, 0]) <= point_index).all() and (
                point_index < np.array(dataset.shape)).all():
                dataset[z - r_cz + cz, y - r_cy + cy, x - r_cx + cx] = np.max(
                    [ref[z, y, x],
                     dataset[z - r_cz + cz, y - r_cy + cy, x - r_cx + cx]])
    return
