import numpy as np
import pandas as pd
import scipy.spatial as spatial

from file_actions.readers.motl import read_motl


def make_motl(points, xyz=True):
    if len(points) > 0:
        if not xyz:
            points = [(p[2], p[1], p[0]) for p in points]
        coords_array = np.array(points)
        mydict = {
            "x": coords_array[:, 0],
            "y": coords_array[:, 1],
            "z": coords_array[:, 2],
        }
    else:
        mydict = {
            "x": [],
            "y": [],
            "z": [],
        }
    motl_df = pd.DataFrame(mydict)
    return motl_df


def mask_coordinates(coordinates, mask):
    masked_points = []
    sz, sy, sx = mask.shape
    print("initial coordinates = ", len(coordinates))
    for point in coordinates:
        x, y, z = point
        if np.min([sz - z, sy - y, sx - x]) >= 0:
            if mask[z, y, x] == 1:
                masked_points.append(point)
    print("after masking = ", len(masked_points))
    return masked_points


def motl_writer(path, points, xyz=True):
    motl_df = make_motl(points, xyz=xyz)
    motl_df.to_csv(path, index=False, header=None)
    return


def mask_motl(input_motl, mask):
    coordinates = list(input_motl[["x", "y", "z"]].values)
    masked_coordinates = mask_coordinates(coordinates=coordinates, mask=mask)
    output_motl = make_motl(points=masked_coordinates, xyz=True)
    return output_motl


def data_from_motl(motl_path):
    motl = read_motl(motl_path)
    if "score" in motl.keys():
        vals = motl["score"].values
    else:
        vals = [0 for _ in range(motl.shape[0])]
    coords = list(motl[["x", "y", "z"]].values)
    return vals, coords


def compute_distances(elliptic_points, points):
    """
    calculate distances after elliptic transformation
    """
    keep = []
    elliptic_keep = []
    keep_index = []
    throw = []
    for index, point, elliptic_point in zip(range(len(points)), points, elliptic_points):
        if len(keep) == 0:
            keep.append(point)
            elliptic_keep.append(point)
            keep_index.append(index)
        else:
            dist = spatial.distance.cdist([elliptic_point], elliptic_keep, metric='euclidean')
            if np.min(dist) < 1:
                throw.append(index)
            else:
                keep.append(point)
                elliptic_keep.append(elliptic_point)
                keep_index.append(index)

    return keep_index, throw


def merge_motls(total_list: pd.DataFrame, a: int, b: int, c: int) -> pd.DataFrame:
    """
    According to elliptic constraints
    """
    total_list = total_list.copy()
    total_list["x_elliptic"] = total_list["x"].apply(lambda val: val / a)
    total_list["y_elliptic"] = total_list["y"].apply(lambda val: val / b)
    total_list["z_elliptic"] = total_list["z"].apply(lambda val: val / c)
    elliptic_points = total_list[["x_elliptic", "y_elliptic", "z_elliptic"]].values
    points = total_list[["x", "y", "z"]].values
    keep_index, _ = compute_distances(elliptic_points, points)
    kept_points = total_list.iloc[keep_index, :][["x", "y", "z"]]
    return kept_points
