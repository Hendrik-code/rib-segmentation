from TPTBox import NII, POI
from instance_rib_assignment.assign_ribs_to_vert import assign_ribs_to_vert_segmentation
from rib_length_measurement.run_algorithm import measure_rib_length_subject
from rib_features.calc_rib_features import calc_rib_features


def run_all_steps(
    rib_mask: NII,
    vertebra_instance_mask: NII,
    vertebra_semantic_mask: NII,
    poi: POI | None = None,
    calc_orientation: bool = False,
    verbose: bool = False,
):
    rib_labels = rib_mask.unique()
    assert 1 in rib_labels, "not foreground in rib mask"
    # ensure binary
    rib_mask[rib_mask != 1] = 0

    # run assign ribs
    sem_seg, inst_seg = assign_ribs_to_vert_segmentation(
        vertebra_instance_mask,
        vertebra_semantic_mask,
        rib_mask,
        verbose=verbose,
    )

    # run measure rib length
    results = measure_rib_length_subject(
        sem_seg,
        inst_seg,
        poi=poi,
        calc_orientation=calc_orientation,
    )

    # run rib features based on the rib length measurement results
    for d in results:
        d["features"] = calc_rib_features(
            d["start_point"],
            d["rib_seg_cropped"],
            d["sem_vert_cropped"],
            d["fixed_points_along_path"],
        )

    return results


if __name__ == "__main__":
    import numpy as np

    rib_in = "<Path-to-your-data>"
    vertebra_instance_in = "<Path-to-your-data>"
    vertebra_semantic_in = "<Path-to-your-data>"

    rib_mask = NII.load(rib_in, seg=True)
    vertebra_instance_mask = NII.load(vertebra_instance_in, seg=True)
    vertebra_semantic_mask = NII.load(vertebra_semantic_in, seg=True)

    results = run_all_steps(
        rib_mask,
        vertebra_instance_mask,
        vertebra_semantic_mask,
        poi=None,
        calc_orientation=False,
    )

    d0 = results[0]
    for i, g in d0.items():
        if not isinstance(g, np.ndarray):
            if not isinstance(g, dict):
                print(i, ":", g)
            else:
                print(i)
                for fi, fg in g.items():
                    print("- ", fi, ":", fg)
