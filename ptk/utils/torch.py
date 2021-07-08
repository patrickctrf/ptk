import torch


def find_nearest(array_to_search, value, device=torch.device("cpu"), dtype=torch.float32):
    """
This function takes 1 array as first argument and a value to find the element
in array whose value is the closest. Returns the closest value element and its
index in the original array.

    :param array_to_search: Reference array.
    :param value: Value to find closest element.
    :param device: Device to allocate new tensors. Default: torch.device("cpu").
    :param dtype: Data type for new tensors. Default: torch.float32.
    :return: Tuple (Element value, element index).
    """
    idx = (torch.absolute(array_to_search - value)).argmin()
    return array_to_search[idx], idx


def axis_angle_into_quaternion(normalized_axis, angle, device=torch.device("cpu"), dtype=torch.float32):
    """
Takes an axis-angle rotation and converts into quaternion rotation.
https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    :param normalized_axis: Axis of rotation (3-element array).
    :param angle: Simple rotation angle (float or 1-element array).
    :param device: Device to allocate new tensors. Default: torch.device("cpu").
    :param dtype: Data type for new tensors. Default: torch.float32.
    :return: 4-element array, containig quaternion (q0,q1,q2,q3).
    """
    # From axis-angle notation into quaternion notation.
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    quaternion_orientation_r = torch.zeros((4,), device=device, dtype=dtype)
    quaternion_orientation_r[0] = torch.cos(angle / 2)
    quaternion_orientation_r[1] = torch.sin(angle / 2) * normalized_axis[0]
    quaternion_orientation_r[2] = torch.sin(angle / 2) * normalized_axis[1]
    quaternion_orientation_r[3] = torch.sin(angle / 2) * normalized_axis[2]

    return quaternion_orientation_r


def quaternion_into_axis_angle(quaternion, device=torch.device("cpu"), dtype=torch.float32):
    """
Takes an quaternion rotation and converts into axis-angle rotation.
https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    :param quaternion: 4-element tensor, containig quaternion (q0,q1,q2,q3).
    :param device: Device to allocate new tensors. Default: torch.device("cpu").
    :param dtype: Data type for new tensors. Default: torch.float32.
    :return: (Axis of rotation (3-element tensor), Simple rotation angle (float or 1-element tensor))
    """
    # Simple rotation angle
    angle = torch.nan_to_num(torch.arccos(quaternion[0])) * 2

    # Avoids recalculating this sin.
    sin_angle_2 = torch.sin(angle / 2)

    # Rotation axis
    normalized_axis = torch.zeros((3,), device=device, dtype=dtype)
    normalized_axis[0] = quaternion[1] / sin_angle_2
    normalized_axis[1] = quaternion[2] / sin_angle_2
    normalized_axis[2] = quaternion[3] / sin_angle_2

    return normalized_axis, angle


def skew_matrix_from_array(x, device=torch.device("cpu"), dtype=torch.float32):
    """
Receives a 3-element array and return its respective skew matrix.

    :param x: 3-element array.
    :return: Respective skew-matrix (3x3).
    :param device: Device to allocate new tensors. Default: torch.device("cpu").
    :param dtype: Data type for new tensors. Default: torch.float32.
    """
    return torch.tensor([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0],
    ], device=device, dtype=dtype)


def array_from_skew_matrix(x, device=torch.device("cpu"), dtype=torch.float32):
    """
Receives a skew matrix and returns its associated 3-element vector (array).

    :param x: Skew matrix (3x3)
    :return: Associated array (3-element).
    :param device: Device to allocate new tensors. Default: torch.device("cpu").
    :param dtype: Data type for new tensors. Default: torch.float32.
    """
    return torch.tensor([x[2][1], x[0][2], x[1][0]],
                        device=device, dtype=dtype)


def exp_matrix(skew_matrix, device=torch.device("cpu"), dtype=torch.float32):
    norma = torch.linalg.norm(skew_matrix)

    return torch.eye(n=3, m=3, device=device, dtype=dtype) + \
           (torch.sin(norma) / norma) * skew_matrix + \
           (1 - torch.cos(norma)) / (norma ** 2) * torch.matmul(skew_matrix, skew_matrix)


def rotation_matrix_into_axis_angle(r_matrix, device=torch.device("cpu"), dtype=torch.float32):
    """
Converts a 3x3 rotation matrix into equivalent axis-angle rotation.


    :param device: Device to allocate new tensors. Default: torch.device("cpu").
    :param dtype: Data type for new tensors. Default: torch.float32.
    :param r_matrix: 3x3 rotation matrix (array).
    :return: Tuple -> (normalized_axis (3-element array), rotation angle)
    """
    # Converts R orientation matrix into equivalent skew matrix. SO(3) -> so(3)
    # phi is a simple rotation angle (the value in radians of the angle of rotation)
    phi = torch.nan_to_num(torch.arccos((torch.trace(r_matrix) - 1) / 2))

    # Skew "orientation" matrix into axis-angles tensor (3-element).
    # we do not multiply by phi, so we have a normalized rotation AXIS (in a SKEW matrix yet)
    # normalized because we didnt multiply the axis by the rotation angle (phi)
    return array_from_skew_matrix((r_matrix - r_matrix.T) / (2 * torch.sin(phi)), device=device, dtype=dtype), phi


def axis_angle_into_rotation_matrix(normalized_axis, angle, device=torch.device("cpu"), dtype=torch.float32):
    return exp_matrix(skew_matrix_from_array(normalized_axis * angle, device=device, dtype=dtype), device=device, dtype=dtype)
