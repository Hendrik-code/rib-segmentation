from TPTBox import NII, No_Logger, Log_Type, Location, Vertebra_Instance

vidx2name = Vertebra_Instance.idx2name()
vname2idx = Vertebra_Instance.name2idx()

logger = No_Logger(prefix="RibAssignment")


def assign_ribs_to_vert_segmentation(
    vert_seg: NII,
    sem_seg: NII,
    rib_seg: NII,
    verbose: bool = False,
) -> tuple[NII, NII]:
    """Assigns binary rib connected components to the corresponding vertebra by proximity and bounding box overlap

    Args:
        vert_seg (NII): The instance segmentation mask of the vertebrae
        sem_seg (NII): The semantic segmentation mask of the vertebrae (subregions)
        rib_seg (NII): The binary rib segmentation mask.
        verbose (bool, optional): If true, logs more stuff. Defaults to False.

    Returns:
        sem_seg,inst_seg:
        sem_seg: NII = The combined semantic mask, \\
        inst_seg: NII = The combined instance mask
    """
    rib_seg.assert_affine(other=vert_seg, verbose=verbose)
    rib_seg.assert_affine(other=sem_seg)

    ori = vert_seg.orientation
    rib_seg.reorient_(verbose=verbose)
    vert_seg.reorient_()
    sem_seg.reorient_()

    rib_arr = rib_seg.get_seg_array()
    rib_arr[rib_arr != 0] = 1
    rib_arr[rib_arr == 0] = 2
    rib_arr[rib_arr == 1] = 0
    rib_background = rib_seg.set_array(rib_arr)

    vert_seg.apply_mask(rib_background, inplace=True)
    sem_seg.apply_mask(rib_background, inplace=True)

    cost_sup = sem_seg.extract_label(
        [
            Location.Arcus_Vertebrae,
            Location.Costal_Process_Left,
            Location.Costal_Process_Right,
            Location.Superior_Articular_Left,
            Location.Superior_Articular_Right,
        ],
        keep_label=True,
    ).map_labels_(
        {
            Location.Costal_Process_Left.value: 1,
            Location.Costal_Process_Right.value: 2,
            Location.Superior_Articular_Left.value: 1,
            Location.Superior_Articular_Right.value: 2,
            Location.Arcus_Vertebrae.value: 3,
        },
        verbose=verbose,
    )

    if len(rib_seg.unique()) == 1:
        rib_cc = rib_seg.get_connected_components(labels=1, connectivity=3)
        assert isinstance(rib_cc, NII)
    else:
        rib_cc = rib_seg.copy()
    cc_labels = rib_cc.unique()
    rib_vertmap = {i: 255 for i in cc_labels}
    rib_subregmap = {i: 0 for i in cc_labels}
    list_candidates = []
    #
    vert_labels = vert_seg.unique()
    # Go over each vertebra and check the rib ccs
    for v in vert_labels:
        if v >= 21 or v < 7:
            continue
        vert_l = vert_seg.extract_label(v)
        cost_sup_l = cost_sup.apply_mask(vert_l)

        for leftside in [False, True]:
            leftsidestring = "Left" if leftside else "Right"
            cos_side_l = (1 - int(leftside)) + 1
            cost_sup_ls = cost_sup_l.extract_label([cos_side_l, 3])
            arcus_ls = cost_sup_l.extract_label(3)
            try:
                bbox_crop = cost_sup_ls.compute_crop(dist=0)
            except ValueError:
                logger.print(f"Vertebra {v} on {leftsidestring} side has no costalis/superior")
                continue
            crop_margin = [0, 0]
            crop_margin[1 - int(leftside)] = 10
            bbox_crop = (
                bbox_crop[0],
                slice(bbox_crop[1].start, bbox_crop[1].stop - (bbox_crop[1].stop - bbox_crop[1].start) // 2),
                slice(max(bbox_crop[2].start - crop_margin[0], 0), min(bbox_crop[2].stop + crop_margin[1], cost_sup_ls.shape[2] - 1)),
            )
            rib_cc_c = rib_cc.apply_crop(bbox_crop)
            volumes = rib_cc_c.volumes()

            if len(volumes) == 0:
                logger.print(f"Vertebra {v} on {leftsidestring} side has no rib", Log_Type.STRANGE)
                continue

            # calc all dominances first
            for rib_cc_label in volumes.keys():
                #
                rib_cc_cl = rib_cc_c.extract_label(rib_cc_label)
                arcus_ls_cc = arcus_ls.apply_crop(bbox_crop)
                # overlap of both crops is distance

                def dist_x(idx: int):
                    rib_cl_crop = rib_cc_cl.compute_crop()[idx]
                    arcus_crop = arcus_ls_cc.compute_crop()[idx]
                    if idx == 2:
                        arcus_end = arcus_crop.stop if not leftside else arcus_crop.start
                        rib_start = rib_cl_crop.stop if leftside else rib_cl_crop.start
                        dist_s = rib_start - arcus_end if not leftside else arcus_end - rib_start
                    elif idx == 1:
                        arcus_end = arcus_crop.start
                        rib_start = rib_cl_crop.start
                        dist_s = arcus_end - rib_start
                    dist = max(0, dist_s)
                    return dist

                dist = sum([dist_x(i) for i in [1, 2]])
                dist = max(1, dist)
                assert dist > 0, (dist, bbox_crop[2])

                logger.print("Can", rib_cc_label, leftside, "vol", volumes[rib_cc_label], "dist", dist)
                dom = volumes[rib_cc_label] / (dist)
                list_candidates.append((dom, v, rib_cc_label, leftside))
    # sort dominances by decreasing order. then loop over and assign
    list_candidates.sort(key=lambda x: x[0], reverse=True)
    logger.print(list_candidates)
    vert_already_assigned = []
    # make labelmap
    for c in list_candidates:
        # sorted by dominance
        _, v, ribl, leftside = c
        va = str(v) + f"_{leftside}"
        if rib_vertmap[ribl] != 255 or va in vert_already_assigned:
            continue
        rib_vertmap[ribl] = Vertebra_Instance(v).RIB
        rib_subregmap[ribl] = Location.Rib_Left.value if leftside else Location.Rib_Right.value
        vert_already_assigned.append(va)
    #
    # apply masks to get rib sem and rib inst
    rib_sem = rib_cc.map_labels(rib_subregmap, verbose=False)
    rib_inst = rib_cc.map_labels(rib_vertmap, verbose=False)

    undefined_rib_counter = 0
    for i, g in rib_vertmap.items():
        if g == 255:
            undefined_rib_counter += 1
    logger.print(f"Found {undefined_rib_counter} Rib CCs that could not be matched")

    vert_seg[rib_inst != 0] = rib_inst[rib_inst != 0]
    sem_seg[rib_sem != 0] = rib_sem[rib_sem != 0]
    return sem_seg.reorient_(ori), vert_seg.reorient_(ori)
