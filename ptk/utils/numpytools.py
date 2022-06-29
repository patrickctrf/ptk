import numpy as np

__all__ = ["find_nearest", "axis_angle_into_quaternion",
           "quaternion_into_axis_angle", "skew_matrix_from_array",
           "array_from_skew_matrix",
           "rotation_matrix_into_axis_angle", "axis_angle_into_rotation_matrix",
           "hamilton_product"]

from scipy.linalg import expm


def hamilton_product(q1, q2):
    """
Performs composition of two quaternions by Hamilton product. This is equivalent
of a rotation descried by quaternion_1 (q1), followed by quaternion_2 (q2).

https://en.wikipedia.org/wiki/Quaternion#Hamilton_product

    :param q1: 4-item iterable representing unit quaternion.
    :param q2: 4-item iterable representing unit quaternion.
    :return: Resulting quaternion.
    """
    a1 = q1[0]
    b1 = q1[1]
    c1 = q1[2]
    d1 = q1[3]

    a2 = q2[0]
    b2 = q2[1]
    c2 = q2[2]
    d2 = q2[3]

    return a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2, \
           a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2, \
           a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2, \
           a1 * b2 + b1 * c2 - c1 * b2 + d1 * a2


def find_nearest(array_to_search, value):
    """
This function takes 1 array as first argument and a value to find the element
in array whose value is the closest. Returns the closest value element and its
index in the original array.

    :param array_to_search: Reference array.
    :param value: Value to find closest element.
    :return: Tuple (Element value, element index).
    """
    idx = (np.absolute(array_to_search - value)).argmin()
    return array_to_search[idx], idx


def axis_angle_into_quaternion(normalized_axis, angle):
    """
Takes an axis-angle rotation and converts into quaternion rotation.
https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    :param normalized_axis: Axis of rotation (3-element array).
    :param angle: Simple rotation angle (float or 1-element array).
    :return: 4-element array, containig quaternion (q0,q1,q2,q3).
    """
    # From axis-angle notation into quaternion notation.
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    quaternion_orientation_r = np.zeros((4,))
    quaternion_orientation_r[0] = np.cos(angle / 2)
    quaternion_orientation_r[1] = np.sin(angle / 2) * normalized_axis[0]
    quaternion_orientation_r[2] = np.sin(angle / 2) * normalized_axis[1]
    quaternion_orientation_r[3] = np.sin(angle / 2) * normalized_axis[2]

    return quaternion_orientation_r


def quaternion_into_axis_angle(quaternion):
    """
Takes an quaternion rotation and converts into axis-angle rotation.
https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    :param quaternion: 4-element tensor, containig quaternion (q0,q1,q2,q3).
    :return: (Axis of rotation (3-element tensor), Simple rotation angle (float or 1-element tensor))
    """
    # Simple rotation angle
    angle = np.nan_to_num(np.arccos(quaternion[0])) * 2

    # Avoids recalculating this sin.
    sin_angle_2 = np.sin(angle / 2)

    # Replace zero values. Avoid numerical issues.
    sin_angle_2 = np.where(sin_angle_2 == 0, 0.0001, sin_angle_2)

    # Rotation axis
    normalized_axis = np.zeros((3,))
    normalized_axis[0] = quaternion[1] / sin_angle_2
    normalized_axis[1] = quaternion[2] / sin_angle_2
    normalized_axis[2] = quaternion[3] / sin_angle_2

    return normalized_axis, angle


def skew_matrix_from_array(x):
    """
Receives a 3-element array and return its respective skew matrix.

    :param x: 3-element array.
    :return: Respective skew-matrix (3x3)
    """
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0],
    ])


def array_from_skew_matrix(x):
    """
Receives a skew matrix and returns its associated 3-element vector (array).

    :param x: Skew matrix (3x3)
    :return: Associated array (3-element).
    """
    return np.array([x[2][1], x[0][2], x[1][0]])


# # ERRADA!!!!!!!!!!!!!!!!!#
# def exp_matrix(skew_matrix):
#     # ERRADA!!!!!!!!!!!!!!!!!#
#     norma = np.linalg.norm(skew_matrix)
#
#     return np.eye(N=3, M=3) + \
#            (np.sin(norma) / norma) * skew_matrix + \
#            (1 - np.cos(norma)) / (norma ** 2) * np.matmul(skew_matrix, skew_matrix)


def rotation_matrix_into_axis_angle(r_matrix):
    """
Converts a 3x3 rotation matrix into equivalent axis-angle rotation.

    :param r_matrix: 3x3 rotation matrix (array).
    :return: Tuple -> (normalized_axis (3-element array), rotation angle)
    """
    # Converts R orientation matrix into equivalent skew matrix. SO(3) -> so(3)
    # phi is a simple rotation angle (the value in radians of the angle of rotation)
    if (np.trace(r_matrix) - 1) / 2 > 1 or (np.trace(r_matrix) - 1) / 2 < -1:
        print("valor fora do range: ", (np.trace(r_matrix) - 1) / 2)
    phi = np.nan_to_num(np.arccos((np.trace(r_matrix) - 1) / 2))

    # Skew "orientation" matrix into axis-angles tensor (3-element).
    # we do not multiply by phi, so we have a normalized rotation AXIS (in a SKEW matrix yet)
    # normalized because we didnt multiply the axis by the rotation angle (phi)
    return array_from_skew_matrix((r_matrix - r_matrix.T) / (2 * np.sin(phi))), phi


def axis_angle_into_rotation_matrix(normalized_axis, angle):
    return expm(skew_matrix_from_array(normalized_axis * angle))
