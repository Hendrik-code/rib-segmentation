from TPTBox import NII
import numpy as np
from utility.utils_points import get_point_arr_mm, cdist_to_point, fast_cdist


def calc_rib_features(
    start_point,
    rib_seg_cropped: NII,
    sem_vert_cropped: NII,
    fixed_points_along_path,
):
    features: dict = {}
    resolution = rib_seg_cropped.zoom

    path_points_relative_to_start = [tuple(np.multiply(g["coord"] - start_point, resolution)) for i, g in fixed_points_along_path.items()]

    bbox = rib_seg_cropped.compute_crop(dist=0)
    relative_voxel_change_PIR = []
    for d in range(3):  # dimensions
        bbox_d = bbox[d]
        start = start_point[d]
        relative_voxel_change_PIR.append((bbox_d.start - start, bbox_d.stop - start))

    relative_mm_change_PIR = [(i * resolution[idx], j * resolution[idx]) for idx, (i, j) in enumerate(relative_voxel_change_PIR)]

    features["path_points_relative_to_start"] = path_points_relative_to_start
    features["relative_mm_change_PIR"] = relative_mm_change_PIR

    start_point_mm = np.multiply(start_point, resolution)
    start_circle = tuple(slice(max(i - 40, 0), min(i + 40, rib_seg_cropped.shape[idx] - 1)) for idx, i in enumerate(start_point))
    newarr = np.zeros(rib_seg_cropped.shape, dtype=rib_seg_cropped.dtype)
    newarr[start_circle] = rib_seg_cropped[start_circle]
    rib_point_arr_mm = get_point_arr_mm(newarr, resolution)
    relevant_labels = [i for i in sem_vert_cropped.unique() if i in [41, 42, 47, 48, 49]]
    for s in relevant_labels:
        # extract subregion
        vert_s = sem_vert_cropped.extract_label(s)
        # calc distance and angle between subregions and rib
        #
        # closest distance from any s to start point
        #
        subreg_point_arr_mm = get_point_arr_mm(vert_s, resolution)
        distances = cdist_to_point(start_point_mm, subreg_point_arr_mm)
        subreg_rel_point_mm = subreg_point_arr_mm[np.argmin(distances)]
        to_start_direction = subreg_rel_point_mm - start_point_mm  # start to subregion s
        # print(to_start_direction)
        to_start_distance = np.linalg.norm(to_start_direction)
        # print(s, ":", to_start_distance)
        features[f"{s}_start_to_subregion_direction"] = to_start_direction
        features[f"{s}_start_to_subregion_distance"] = to_start_distance
        #
        # closest distance from any s to any rib voxel
        #
        distances = fast_cdist(rib_point_arr_mm, subreg_point_arr_mm)
        idxs = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
        rib_point = rib_point_arr_mm[idxs[0]]
        subreg_point = subreg_point_arr_mm[idxs[1]]
        closest_s_direction = subreg_point - rib_point  # rib point to subregion s
        closest_s_distance = np.linalg.norm(closest_s_direction)
        # print(s, ":", closest_s_distance)
        features[f"{s}_closest_to_subregion_direction"] = closest_s_direction
        features[f"{s}_closest_to_subregion_distance"] = closest_s_distance
        #
        # closest distance to center of that structure (to start point)
        #
        com_s_point = vert_s.center_of_masses()[1]
        com_s_point_mm = np.multiply(com_s_point, resolution)
        com_to_start_direction = com_s_point_mm - start_point_mm
        com_to_start_distance = np.linalg.norm(com_to_start_direction)
        features[f"{s}_start_to_com_direction"] = com_to_start_direction
        features[f"{s}_start_to_com_distance"] = com_to_start_distance
        #
        # closest distance to com of structure to any rib voxel
        #
        distances = cdist_to_point(com_s_point_mm, rib_point_arr_mm)
        rib_point = rib_point_arr_mm[np.argmin(distances)]
        closest_to_com_direction = com_s_point_mm - rib_point
        closest_to_com_distance = np.linalg.norm(closest_to_com_direction)
        features[f"{s}_closest_to_com_direction"] = closest_to_com_direction
        features[f"{s}_closest_to_com_distance"] = closest_to_com_distance
    return features
