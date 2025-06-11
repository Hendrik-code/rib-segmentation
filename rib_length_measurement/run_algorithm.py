from TPTBox import NII, No_Logger, Log_Type, Location, Vertebra_Instance, POI, calc_poi_labeled_buffered
from rib_length_measurement.calc_orientation import calc_orientation_from_poi
from rib_length_measurement.rib_length_measurement_algorithm import rib_length_algorithm

logger = No_Logger(prefix="measure_rib_length")


def measure_rib_length_subject(
    sem_seg: NII,
    inst_seg: NII,
    poi: POI | None = None,
    calc_orientation: bool = False,
) -> list[dict]:
    """Calculates the length of the lowest two ribs in a subject

    Args:
        sem_seg (NII): Semantic Mask
        inst_seg (NII): Instance Mask
        poi (POI | None, optional): Center of Corpus and Direction Points if available. Defaults to None.
        calc_orientation (bool, optional): If true, will compute the orientation of the vertebra and return that. Defaults to False.

    Returns:
        list[dict]: A list of datapoints, where each datapoint is a dictionary mapping keys to values.
    """
    results = []
    # Orientation consistent
    sem_seg.reorient_().map_labels_({50: 49}, verbose=False)
    inst_seg.reorient_()
    try:
        sem_seg.assert_affine(other=inst_seg)
    except AssertionError:
        return results
    # get last 2 vertebra
    rib_labels = [v for v in inst_seg.unique() if v in Vertebra_Instance.rib_label()]
    last_vertebrae = [i for i in inst_seg.unique() if (7 < i and i < 21) or i == 28]
    last_vertebrae.sort()
    last_two_vertebrae: list[Vertebra_Instance] = [
        Vertebra_Instance(v) for v in last_vertebrae[-3:] if Vertebra_Instance(v).RIB in rib_labels
    ]
    if len(last_two_vertebrae) == 0:
        logger.print("No last rib visible")
        return results
    last_v = last_two_vertebrae[-1]
    if len(last_two_vertebrae) == 0 or last_two_vertebrae[0].value <= 16:
        logger.print("Not last BWK/LWK visible")
        return results

    # Loop over vertebra
    for vert in last_two_vertebrae:
        is_last_v = vert == last_v
        rib_label = vert.RIB
        if rib_label not in rib_labels:
            logger.print(f"has no rib at vertebra {vert.name}")
            continue
        # vertebra level
        logger.print(f"Process vertebra {vert}")
        vert_vr = inst_seg.extract_label([vert.value, rib_label], keep_label=True)

        # Orientation of vertebra in image space
        rel_to_corpus, PIR_angle_degrees = None, None
        if calc_orientation:
            if poi is None:
                poi = calc_poi_labeled_buffered(
                    inst_seg,
                    sem_seg,
                    subreg_id=[Location.Vertebra_Corpus, Location.Vertebra_Direction_Posterior],
                    out_path=__file__.join("poi.json"),
                )
            poi_vr = poi.extract_vert(vert.value)
            _, _, rel_to_corpus, PIR_angle_degrees = calc_orientation_from_poi(poi_vr, vert.value)
            rel_to_corpus = {k: list(v) for k, v in rel_to_corpus.items()}
        # For both left and right side rib
        for leftside in [False, True]:  #
            leftsidestr = "left" if leftside else "right"
            sem_rib_label = Location.Rib_Left.value if leftside else Location.Rib_Right.value

            if sem_rib_label not in sem_seg.unique():
                logger.print(f"has no rib at vertebra {vert.name} on {leftsidestr} side", Log_Type.STRANGE)
                continue

            # extract correct label
            sem_vr = sem_seg.extract_label([sem_rib_label, 41, 42, 43, 44, 45, 46, 47, 48, 49], keep_label=True)
            sem_vr[vert_vr == 0] = 0

            # hand it over to one rib function handle
            data_dict = measure_one_rib_length(
                sem_vr,
                leftside=leftside,
                v=vert,
            )
            data_dict["vertebra"] = vert.value
            data_dict["side"] = leftsidestr
            data_dict["is_last_v"] = is_last_v
            data_dict["vert_ori_rel_to_corpus"] = rel_to_corpus
            data_dict["vert_ori_angle_degrees_PIR"] = PIR_angle_degrees
            results.append(data_dict)
    return results


def measure_one_rib_length(
    sem_vr: NII,
    v: Vertebra_Instance,
    leftside: bool,
    resolution: float = 0.5,
):
    leftsidestr = "left" if leftside else "right"
    init_shp = sem_vr.shape
    # crop
    sem_vr_crop = sem_vr.compute_crop(dist=8)
    sem_vr_cropped = sem_vr.apply_crop(sem_vr_crop)
    logger.print(f"Cropped down from {init_shp}, to {sem_vr_cropped.shape}")
    # then rescale
    orig_zoom = sem_vr.zoom
    sem_vr2 = sem_vr_cropped.rescale((resolution, resolution, resolution), verbose=False, mode="nearest")
    #
    logger.print(f"Calc rib stats, vertebra {v}, {leftsidestr} side, resolution={resolution}", verbose=True)
    #########################
    # Call to Rib length measurement algorithm
    try:
        data_dict = rib_length_algorithm(sem_vr2)
    except Exception as e:
        raise e
    #########################
    data_dict["orig_zoom"] = list(orig_zoom)

    logger.print("is stump rib=", data_dict["sr"], "length=", data_dict["rib_length"])

    # TODO calc rib features

    return data_dict
