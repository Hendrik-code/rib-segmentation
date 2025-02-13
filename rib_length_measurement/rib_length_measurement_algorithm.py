from TPTBox import NII, No_Logger, Log_Type, Location, Vertebra_Instance
from TPTBox.core.vert_constants import vert_subreg_labels, COORDINATE
import numpy as np
from utility.utils_length_measurement import (
    change_slice_tuple,
    slices_border_shape,
    refine_start_points,
    find_all_candidate_points,
    find_end_point,
)
from utility.utils_points import cdist, cdist_to_point, get_idx_point_closest_to_point, get_point_arr, get_point_arr_mm, get_raycasted_point

logger = No_Logger(prefix="RibLengthMeasurementAlgorithm")

vidx2name = Vertebra_Instance.idx2name()
vname2idx = Vertebra_Instance.name2idx()


def rib_length_algorithm(
    sem_vr: NII,
    stump_rib_threshold_in_mm: int = 38,
    interpolation_distance_mm: int = 15,
    max_iterations: int = 150,
    do_dilateerode: bool = True,
    return_debug_data: bool = False,
    verbose: bool | int = 0,
) -> dict:
    """The Rib Length Measurement Algorithm, calculating the length of the provided segmentation

    Args:
        sem_vr (NII): Semantic Mask
        stump_rib_threshold_in_mm (int, optional): Threshold for stump rib length. Defaults to 38.
        interpolation_distance_mm (int, optional): The circular distance for each iteration. Defaults to 15.
        max_iterations (int, optional): Maximum number of iteration until it should crash. Defaults to 150.
        round_digits (int, optional): _description_. Defaults to 5.
        return_debug_data (bool, optional): If true, will return debug data. Defaults to False.
        verbose (bool | int, optional): Verbosity level for the algorithm. Defaults to 0.

    Returns:
        dict: A data dictionary
    """
    verbose = int(verbose)
    precision_resolution = sem_vr.zoom[0]
    assert 0 < precision_resolution <= 1.0, f"precision_resolution must be in (0, 1.0], got {precision_resolution}"
    #
    debug_data: dict = {}
    with logger:
        sem_vr.reorient_()
        sem_vr_labels = sem_vr.unique()
        assert Location.Vertebra_Corpus_border.value in sem_vr_labels, f"no corpus ({Location.Vertebra_Corpus_border.value}) in input"
        non_vert_subreg_labels = [i for i in sem_vr_labels if not (40 < i < 51)]

        assert len(non_vert_subreg_labels) == 1, f"Not exactly one non-subregion label present as rib label, got {non_vert_subreg_labels}"
        rib_label = non_vert_subreg_labels[0]
        expected_riblabels = [Location.Rib_Left.value, Location.Rib_Right.value]
        if rib_label not in expected_riblabels:
            logger.print(f"Unusual rib label {rib_label}, expected {expected_riblabels}", Log_Type.STRANGE)

        # Extract label
        init_shp = sem_vr.shape
        rib_seg = sem_vr.extract_label(rib_label)
        rib_crop = rib_seg.compute_crop(dist=0)
        seg_at_border = slices_border_shape(rib_crop, init_shp, voxel_tolerance=2)
        # Crop down
        init_crop = sem_vr.compute_crop(dist=0)
        init_crop = change_slice_tuple(init_crop, change=6, shape=init_shp)  # TODO change to < 6 changes rib length
        sem_vr.apply_crop_(init_crop)
        rib_seg_cropped = rib_seg.apply_crop_(init_crop)
        logger.print(f"Cropped from {init_shp} to {sem_vr.shape}", verbose=verbose > 1)
        sem_vert = sem_vr.extract_label(vert_subreg_labels(True), keep_label=True)
        sem_corpus = sem_vert.extract_label(Location.Vertebra_Corpus_border.value)
        zooms = rib_seg_cropped.zoom

        if do_dilateerode:
            rib_seg_cropped = (
                rib_seg_cropped.dilate_msk_(n_pixel=1, connectivity=1, labels=1, verbose=verbose > 0)
                .erode_msk_(n_pixel=1, connectivity=1, labels=1, verbose=verbose > 0)
                .dilate_msk_(n_pixel=1, connectivity=1, labels=1, verbose=verbose > 0)
                .erode_msk_(n_pixel=1, connectivity=1, labels=1, verbose=verbose > 0)
            )

        # calculate center of vertebra in question
        vert_com = sem_corpus.center_of_masses()[1]
        debug_data["sem_corpus_crop"] = sem_corpus
        #####################
        # intialize path
        fixed_points_along_path: dict = {}
        rib_volume = rib_seg_cropped.volumes()[1]
        logger.print(f"rib_volume = {rib_volume}", verbose=verbose > 0)

        # get all points on segmentation mask
        point_arr = get_point_arr(rib_seg_cropped)
        # stack resolution
        try:
            zms_stacked = np.vstack([zooms] * point_arr.shape[0])
        except Exception as e:
            logger.print("could not compute zms_stacked, arr must be empty", Log_Type.FAIL)
            return {"debug": debug_data}
        # get points in mm
        point_arr_mm = np.multiply(point_arr, zms_stacked)
        # get center of vertebra in mm
        vert_com_mm = np.multiply(vert_com, zooms)

        distance_to_vertebra = cdist_to_point(vert_com_mm, point_arr_mm)
        # minimum distance to vertebra corpus becomes start point of path
        start_point_idx = np.argmin(distance_to_vertebra)
        refined_start_idx = refine_start_points(rib_seg_cropped, start_point_idx, point_arr, logger)
        if refined_start_idx is None:
            logger.print("Start point refinement failed", Log_Type.FAIL)
        else:
            start_point_idx = refined_start_idx
        start_point_coord = point_arr[start_point_idx]
        # add startpoint to path
        fixed_points_along_path[start_point_idx] = {"coord": start_point_coord, "prior": None, "prior_distance": None, "endpoint": True}
        ################# START ALGORITHM #################
        # initialize algorithm values
        start_interpolation_distance_mm = interpolation_distance_mm
        interpolation_distance_mm = start_interpolation_distance_mm
        interpolation_distance_mm_tol = precision_resolution
        min_interpolation_distance_mm = (interpolation_distance_mm / 2) - (2 * interpolation_distance_mm_tol)
        # Moving variables
        prior_prior_point_idx = None
        prior_point_idx = None
        prior_distance = None
        cur_point_idx = start_point_idx
        #
        end_point = None

        # Get array
        darr = rib_seg_cropped.get_seg_array()
        darr[start_point_coord[0], start_point_coord[1], start_point_coord[2]] = 100
        debug_arr = darr.copy()

        # Loop counts
        loop_count = 0
        iter_count = 1

        ################# ITERATION LOOP #################
        while True:
            loop_count += 1
            #
            logger.print(f"ITERATION {iter_count}", verbose=verbose > 1)

            # Get distances to current point
            distance_row = cdist_to_point(point_arr_mm[cur_point_idx], point_arr_mm)
            # Get all candidate points
            point_candidate_idxs, point_candidates_coords = find_all_candidate_points(
                point_arr,
                cur_point_idx,
                prior_point_idx,
                prior_prior_point_idx,
                interpolation_distance_mm,
                interpolation_distance_mm_tol,
                distance_row,
            )
            n_candidates = len(point_candidate_idxs)
            logger.print("Possible_candidates", n_candidates, verbose=verbose > 1)
            #

            if len(point_candidates_coords) == 0:
                logger.print("No point candidates, move to end sequence", verbose=verbose > 1)

                end_point_idx, end_point_coords, prior_distance = find_end_point(
                    point_arr,
                    rib_seg_cropped,
                    cur_point_idx,
                    prior_point_idx,
                    precision_resolution,
                    interpolation_distance_mm,
                    interpolation_distance_mm_tol,
                    distance_row,
                )
                if end_point_idx not in fixed_points_along_path:
                    # delete prior point if too close to end point
                    if prior_point_idx is not None and prior_distance < min_interpolation_distance_mm:
                        logger.print("Deleted point prior to end because of proximity", verbose=verbose > 1)
                        prior_distance = np.linalg.norm(point_arr_mm[prior_point_idx] - point_arr_mm[end_point_idx])
                        fixed_points_along_path.pop(prior_point_idx, None)
                    #
                    fixed_points_along_path[end_point_idx] = {
                        "coord": end_point_coords,
                        "prior": cur_point_idx,
                        "prior_distance": prior_distance,
                        "endpoint": True,
                    }
                else:
                    fixed_points_along_path[end_point_idx]["endpoint"] = True
                #
                end_point = end_point_coords
                logger.print(f"Found end point at {end_point_coords}", verbose=verbose > 0)
                darr[end_point[0], end_point[1], end_point[2]] = 200
                debug_data[f"iterations_{iter_count}"] = rib_seg_cropped.set_array(darr.copy())
                #
                # Update moving values
                prior_prior_point_idx = prior_point_idx
                prior_point_idx = cur_point_idx
                cur_point_idx = end_point_coords
                ################# End loop because we found end #################
                break
                #################
            else:
                # there are multiple candidates
                # take average as new point
                avg_candidate_coord = [round(i) for i in np.sum(point_candidates_coords, axis=0) / n_candidates]
                # move half-distance in that direction
                cur_coord = point_arr[cur_point_idx]
                avg_candidate_coord = tuple(np.add(cur_coord, (np.subtract(avg_candidate_coord, cur_coord) * 0.5)))
                #
                new_point_idx = get_idx_point_closest_to_point(point_arr, avg_candidate_coord)
                new_point_coord = point_arr[new_point_idx]

                prior_distance = distance_row[new_point_idx]
                # if it didn't move far enough, then the circle is not wide enough
                if prior_distance < min_interpolation_distance_mm or len(point_candidates_coords) < 3:
                    interpolation_distance_mm += 2 * precision_resolution
                    logger.print(
                        f"Increased interpolation distance, distance={prior_distance} and threshold={min_interpolation_distance_mm}",
                        verbose=verbose > 1,
                    )
                    continue
                else:
                    interpolation_distance_mm = start_interpolation_distance_mm

                # else we found a good new point
                for p in point_candidates_coords:
                    darr[p[0], p[1], p[2]] = iter_count + 1
                darr[new_point_coord[0], new_point_coord[1], new_point_coord[2]] = iter_count + 100
                debug_arr[new_point_coord[0], new_point_coord[1], new_point_coord[2]] = iter_count + 100
                if new_point_idx in fixed_points_along_path:
                    logger.print("New point already in the path, something went wrong", Log_Type.FAIL, verbose=True)
                    break
                fixed_points_along_path[new_point_idx] = {
                    "coord": new_point_coord,
                    "prior": cur_point_idx,
                    "prior_distance": prior_distance,
                    "endpoint": False,
                }
                ###########
                # Update moving values
                prior_prior_point_idx = prior_point_idx
                prior_point_idx = cur_point_idx
                cur_point_idx = new_point_idx

            debug_data[f"iterations_{iter_count}"] = rib_seg_cropped.set_array(darr.copy())
            iter_count += 1
            loop_count = 0
            if iter_count >= max_iterations:
                logger.print(f"Did not converge after max_iterations={max_iterations}", Log_Type.FAIL)
                end_point = start_point_coord
                break
        ################# End Algorithm #################
        if verbose > 2:
            for idx, (i, g) in enumerate(fixed_points_along_path.items()):
                logger.print(idx, i, g)

        # Calculating rib length with piece-wise linear interpolation
        rib_length = sum([g["prior_distance"] for i, g in fixed_points_along_path.items() if g["prior"] is not None])

        # stump rib detection
        if rib_length > stump_rib_threshold_in_mm:
            stump_rib = False
        else:
            # stump_rib if not segmentation at border, else None (not determinable)
            stump_rib = True if not seg_at_border else None

        start_point = start_point_coord
        debug_data["final_algo"] = rib_seg_cropped.set_array(darr.copy())
        if not return_debug_data:
            debug_data = None
        return {
            "debug": debug_data,
            "rib_length": rib_length,
            "sr": stump_rib,
            "start_point": start_point,
            "end_point": end_point,
            "fixed_points_along_path": fixed_points_along_path,
            "seg_at_border": seg_at_border,
            "darr": darr.copy(),
            "rib_volume": rib_volume,
            "rib_seg_cropped": rib_seg_cropped,
            "sem_vert_cropped": sem_vert,
        }
