import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))
from TPTBox import BIDS_FILE, BIDS_Global_info, No_Logger, NII, np_utils, Log_Type, Vertebra_Instance, Location, POI
import numpy as np
from utility.utils_points import unit_vector
from utility.utils_orientation import angle_between, radian_to_degrees


def calc_orientation_from_poi(poi: POI, region: int):
    poi_v: POI = poi.extract_vert(region)

    point_keys = [
        Location.Vertebra_Corpus,
        Location.Vertebra_Direction_Posterior,
        Location.Vertebra_Direction_Inferior,
        Location.Vertebra_Direction_Right,
    ]
    for p in point_keys:
        assert p in poi.keys_region(), f"POI {p} not found, got {poi.keys_region()}"

    point_keys = [i.value for i in point_keys]
    points = {s: np.asarray(v) for r, s, v in poi_v.items() if s in point_keys}
    # calc corpus - three other to get directional vectors (and normalize)
    rel_to_corpus = {
        s: unit_vector(v - points[Location.Vertebra_Corpus.value]) for s, v in points.items() if s != Location.Vertebra_Corpus.value
    }
    pir_global_vectors = {
        Location.Vertebra_Direction_Posterior.value: np.array([1, 0, 0]),
        Location.Vertebra_Direction_Inferior.value: np.array([0, 1, 0]),
        Location.Vertebra_Direction_Right.value: np.array([0, 0, 1]),
    }
    PIR_angles = [angle_between(v, pir_global_vectors[s]) for s, v in rel_to_corpus.items()]
    PIR_angle_degrees = [radian_to_degrees(i) for i in PIR_angles]

    # R = [x_x, y_x, z_x; x_y, y_y, y_z; z_x, z_y, z_z]
    R = np.asarray([[v[idx] for v in rel_to_corpus.values()] for idx in range(3)])
    corpus_com = points[Location.Vertebra_Corpus.value]

    return R, corpus_com, rel_to_corpus, PIR_angle_degrees


def rotate_3darray(array, rotation):
    # rotate the 3D numpy array using given parameters around a defined center
    # create meshgrid
    from scipy.ndimage import map_coordinates

    dim = array.shape
    ax = np.arange(dim[0])
    ay = np.arange(dim[1])
    az = np.arange(dim[2])
    coords = np.meshgrid(ax, ay, az)

    # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
    xyz = np.vstack(
        [
            coords[0].reshape(-1) - float(dim[0]) / 2,  # x coordinate, centered
            coords[1].reshape(-1) - float(dim[1]) / 2,  # y coordinate, centered
            coords[2].reshape(-1) - float(dim[2]) / 2,
        ]
    )  # z coordinate, centered

    r = rotation
    mat = r.copy()  # r.as_matrix()

    # apply transformation
    transformed_xyz = np.dot(mat, xyz)

    # extract coordinates
    x = transformed_xyz[0, :] + float(dim[0]) / 2
    y = transformed_xyz[1, :] + float(dim[1]) / 2
    z = transformed_xyz[2, :] + float(dim[2]) / 2

    x = x.reshape((dim[1], dim[0], dim[2]))
    y = y.reshape((dim[1], dim[0], dim[2]))
    z = z.reshape((dim[1], dim[0], dim[2]))
    new_xyz = [x, y, z]

    # sample
    # arrayR = map_coordinates(array, new_xyz, order=0, mode='nearest')
    arrayR = map_coordinates(array, new_xyz, order=0, mode="constant")
    arrayR = np.swapaxes(arrayR, 0, 1)
    return arrayR
