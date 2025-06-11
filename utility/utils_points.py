import numpy as np
from scipy.spatial.distance import cdist
from TPTBox import NII
from TPTBox.core.vert_constants import COORDINATE
from utility.utils_raycast import max_distance_ray_cast_convex


def get_raycasted_point(rib_nii: NII, point_arr: np.ndarray, cur_point_idx: int, direction_vector: COORDINATE) -> int:
    end_point_coords = max_distance_ray_cast_convex(
        rib_nii,
        point_arr[cur_point_idx],
        direction_vector,
    )
    end_point_coords = np.asarray([round(i) for i in end_point_coords])
    end_point_idx = get_idx_point_closest_to_point(point_arr, end_point_coords)
    return end_point_idx


def unit_vector(v):
    return v / np.linalg.norm(v)


def get_point_arr(arr):
    X, Y, Z = np.where(arr)
    point_arr = np.asarray([[X[i], Y[i], Z[i]] for i in range(len(X))])
    return point_arr


def get_point_arr_mm(arr, resolution):
    point_arr = get_point_arr(arr)
    zms_stacked = np.vstack([resolution] * point_arr.shape[0])
    point_arr_mm = np.multiply(point_arr, zms_stacked)
    return point_arr_mm


def cdist_to_point(point, a):
    return fast_cdist([point], a)[0]


def fast_cdist(a, b):
    return cdist(a, b)


def np_index(arr: np.ndarray, entry) -> np.ndarray:
    bool_arr = entry == arr
    idxs = np.flatnonzero((bool_arr).all(1))
    return idxs


def get_idx_point_closest_to_point(point_arr: np.ndarray, point: COORDINATE) -> int:
    point = tuple(round(i) for i in point)
    idxs = np_index(point_arr, point)
    if len(idxs) == 0:
        distance_vectors = cdist(point_arr, np.asarray([point]))
        idxs = [np.argmin(distance_vectors)]
    return idxs[0]
