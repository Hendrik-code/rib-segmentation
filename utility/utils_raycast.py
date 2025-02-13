from TPTBox import NII
from TPTBox.core.vert_constants import COORDINATE
import numpy as np
from numpy.linalg import norm
from scipy.interpolate import RegularGridInterpolator


def max_distance_ray_cast_convex(
    region: NII,
    start_coord: COORDINATE,
    direction_vector: COORDINATE,
    acc_delta: float = 0.00005,
):
    start_point_np = np.asarray(start_coord)
    if start_point_np is None:
        return None

    """Convex assumption!"""
    # Compute a normal vector, that defines the plane direction
    normal_vector = np.asarray(direction_vector)
    normal_vector = normal_vector / norm(normal_vector)
    # Create a function to interpolate within the mask array
    interpolator = RegularGridInterpolator([np.arange(region.shape[i]) for i in range(3)], region.get_array())

    def is_inside(distance):
        coords = [start_point_np[i] + normal_vector[i] * distance for i in [0, 1, 2]]
        if any(i < 0 for i in coords):
            return 0
        if any(coords[i] > region.shape[i] - 1 for i in range(len(coords))):
            return 0
        # Evaluate the mask value at the interpolated coordinates
        mask_value = interpolator(coords)
        return mask_value > 0.5

    if not is_inside(0):
        return start_point_np
    count = 0
    min_v = 0
    max_v = sum(region.shape)
    delta = max_v * 2
    while acc_delta < delta:
        bisection = (max_v - min_v) / 2 + min_v
        if is_inside(bisection):
            min_v = bisection
        else:
            max_v = bisection
        delta = max_v - min_v
        count += 1
    return start_point_np + normal_vector * ((min_v + max_v) / 2)
