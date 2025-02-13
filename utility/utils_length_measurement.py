from TPTBox import np_utils, NII
import numpy as np
from scipy.spatial.distance import cdist
from utility.utils_points import get_idx_point_closest_to_point, get_raycasted_point


def array_slice(a, axis, start, end, step=1):
    return a[(slice(None),) * (axis % a.ndim) + (slice(start, end, step),)]


def array_get_slice(a, axis, start, end, step=1):
    return (slice(None),) * (axis % a.ndim) + (slice(start, end, step),)


def change_one_slice(s: slice, change: tuple[int, int], shape: int) -> slice:
    return slice(
        max(s.start + change[0], 0),
        min(s.stop + change[1], shape),
    )


def change_slice_tuple(
    slices: tuple[slice, slice, slice], change: int | tuple[tuple[int, int], tuple[int, int], tuple[int, int]], shape: tuple[int, int, int]
) -> tuple[slice, slice, slice]:
    if isinstance(change, int):
        change = ((-change, change), (-change, change), (-change, change))
    return tuple(change_one_slice(slices[d], change[d], shape[d]) for d in range(3))


def slices_border_shape(slices: tuple[slice, slice, slice], shp: tuple[int, int, int], voxel_tolerance: int = 2):
    seg_at_border = False
    for d in range(3):
        if slices[d].start <= voxel_tolerance or slices[d].stop - 1 >= shp[d] - voxel_tolerance:
            seg_at_border = True
            break
    return seg_at_border


def refine_start_points(rib_cropped, start_point_idx, point_arr, logger):
    """Refines start point by slicing at its location and using the center point of that slices segmentation

    Args:
        rib_cropped (_type_): _description_
        start_point_idx (_type_): _description_
        point_arr (_type_): _description_
        logger (_type_): _description_

    Returns:
        _type_: _description_
    """
    start_point_coord = point_arr[start_point_idx]
    # refine start point coord by using center of mass of this L/R slice
    start_slice = array_slice(rib_cropped.get_seg_array(), 2, start_point_coord[2], start_point_coord[2] + 1)
    # dim 0
    start_slice[0 : max(start_point_coord[0] - 20, 0)] = 0
    start_slice[min(start_point_coord[0] + 21, rib_cropped.shape[0] - 1) : rib_cropped.shape[0] - 1] = 0
    # dim 1
    start_slice[:, 0 : max(start_point_coord[1] - 20, 0)] = 0
    start_slice[:, min(start_point_coord[1] + 21, rib_cropped.shape[1] - 1) : rib_cropped.shape[1] - 1] = 0
    start_com = [round(i) for i in np_utils.np_center_of_mass(start_slice)[1]]
    start_coord = np.asarray([start_com[0], start_com[1], start_point_coord[2]])
    try:
        start_point_idx = get_idx_point_closest_to_point(point_arr, start_coord)
    except Exception as e:
        return None
    # refinement done
    return start_point_idx


def find_all_candidate_points(
    point_arr,
    cur_point_idx,
    prior_point_idx,
    prior_prior_point_idx,
    interpolation_distance_mm,
    interpolation_distance_mm_tol,
    distance_row,
):
    # Get all candidate points
    point_candidate_idxs = np.where(
        (interpolation_distance_mm - interpolation_distance_mm_tol <= distance_row)
        & (distance_row <= interpolation_distance_mm + interpolation_distance_mm_tol)
    )[0]

    # if prior points exist, remove candidates closer to prior point
    # remove all candidates that are closer to the prior point than the current
    point_candidate_idxs, removed_idxs = remove_candidates_idxs_closer_to_prior_points(
        prior_point_idx,
        point_candidate_idxs,
        interpolation_distance_mm,
        interpolation_distance_mm_tol,
        point_arr,
        cur_point_idx,
    )
    point_candidate_idxs, removed_idxs = remove_candidates_idxs_closer_to_prior_points(
        prior_prior_point_idx,
        point_candidate_idxs,
        interpolation_distance_mm,
        interpolation_distance_mm_tol,
        point_arr,
        cur_point_idx,
    )
    point_candidates_coords = [point_arr[idx] for idx in point_candidate_idxs]
    return point_candidate_idxs, point_candidates_coords


def remove_candidates_idxs_closer_to_prior_points(
    prior_point_idx,
    point_candidates_idxs,
    interpolation_distance_mm,
    interpolation_distance_mm_tol,
    point_arr,
    cur_point_idx,
):
    before_remove = point_candidates_idxs.copy()
    if prior_point_idx is not None:
        point_distance_tolerance_circle = (interpolation_distance_mm / 2) - (2 * interpolation_distance_mm_tol)
        point_candidates_idxs = list(
            [
                p
                for p in point_candidates_idxs
                if np.linalg.norm(point_arr[p] - point_arr[cur_point_idx])
                < np.linalg.norm(point_arr[p] - point_arr[prior_point_idx]) - point_distance_tolerance_circle
            ]
        )
    removed = [p for p in before_remove if p not in point_candidates_idxs]
    return point_candidates_idxs, removed


def remove_candidate_coords_closer_to_prior_points(
    prior_point_idx,
    point_candidates_coords,
    interpolation_distance_mm,
    interpolation_distance_mm_tol,
    point_arr,
    cur_point_idx,
):
    if prior_point_idx is not None:
        point_distance_tolerance_circle = interpolation_distance_mm + interpolation_distance_mm_tol
        point_candidates_coords = list(
            [
                p
                for p in point_candidates_coords
                if np.linalg.norm(p - point_arr[cur_point_idx])
                < np.linalg.norm(p - point_arr[prior_point_idx]) - point_distance_tolerance_circle
            ]
        )
    return point_candidates_coords


def find_end_point(
    point_arr,
    rib_seg_cropped,
    cur_point_idx,
    prior_point_idx,
    precision_resolution,
    interpolation_distance_mm,
    interpolation_distance_mm_tol,
    distance_row,
):
    # handle end of path
    # if prior point exists, just raycast it
    if prior_point_idx is not None:
        end_point_idxs = get_possible_end_points(
            rib_seg_cropped,
            point_arr,
            # distance_matrix_mm,
            cur_point_idx,
            prior_point_idx,
            precision_resolution,
            interpolation_distance_mm,
        )
        end_point_idxs = list(set(end_point_idxs))
        prior_distances = [distance_row[i] for i in end_point_idxs]
        end_point_idx = end_point_idxs[np.argmax(prior_distances)]
        prior_distance = distance_row[end_point_idx]
        #
        end_point_coords = point_arr[end_point_idx]
    else:
        # find farthest point that is also not closer to previous point
        distance_row = np.asarray([i if i < interpolation_distance_mm + interpolation_distance_mm_tol else 0 for i in distance_row])
        end_point_idx = np.argmax(distance_row)
        end_point_coords = point_arr[end_point_idx]
        prior_distance = distance_row[end_point_idx]
    return end_point_idx, end_point_coords, prior_distance


def get_possible_end_points(
    rib_nii: NII,
    point_arr: np.ndarray,
    cur_point_idx: int,
    prior_point_idx: int,
    resolution_precision: float,
    interpolation_distance_mm: float,
) -> list[int]:
    initial_direction_vector = point_arr[cur_point_idx] - point_arr[prior_point_idx]
    end_points = [get_raycasted_point(rib_nii, point_arr, cur_point_idx, initial_direction_vector)]

    for d in range(3):
        for change in range(-8, 9, 2):
            d_vector = initial_direction_vector.copy()
            d_vector[d] += change * resolution_precision
            end_points.append(get_raycasted_point(rib_nii, point_arr, cur_point_idx, d_vector))
    return end_points
