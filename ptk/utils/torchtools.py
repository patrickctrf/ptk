import torch

__all__ = ["find_nearest", "axis_angle_into_quaternion",
           "quaternion_into_axis_angle", "skew_matrix_from_array",
           "array_from_skew_matrix",
           "rotation_matrix_into_axis_angle", "axis_angle_into_rotation_matrix",
           "hamilton_product"]


def hamilton_product(q1, q2):
    """
Performs composition of two quaternions by Hamilton product. This is equivalent
of a rotation descried by quaternion_1 (q1), followed by quaternion_2 (q2).

https://en.wikipedia.org/wiki/Quaternion#Hamilton_product

    :param q1: 4-item tensor representing unit quaternion.
    :param q2: 4-item tensor representing unit quaternion.
    :return: Resulting quaternion.
    """
    a1 = q1[:, 0]
    b1 = q1[:, 1]
    c1 = q1[:, 2]
    d1 = q1[:, 3]

    a2 = q2[:, 0]
    b2 = q2[:, 1]
    c2 = q2[:, 2]
    d2 = q2[:, 3]

    return a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2, \
           a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2, \
           a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2, \
           a1 * b2 + b1 * c2 - c1 * b2 + d1 * a2


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
    quaternion_orientation_r = torch.zeros((angle.shape[0], 4,), device=device, dtype=dtype)
    quaternion_orientation_r[:, 0] = torch.cos(angle / 2)
    quaternion_orientation_r[:, 1] = torch.sin(angle / 2) * normalized_axis[:, 0]
    quaternion_orientation_r[:, 2] = torch.sin(angle / 2) * normalized_axis[:, 1]
    quaternion_orientation_r[:, 3] = torch.sin(angle / 2) * normalized_axis[:, 2]

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
    angle = torch.nan_to_num(torch.arccos(quaternion[:, 0])) * 2

    # Avoids recalculating this sin.
    sin_angle_2 = torch.sin(angle / 2)

    # Replace zero values. Avoid numerical issues.
    sin_angle_2[sin_angle_2 == 0] = 0.0001

    # Rotation axis
    normalized_axis = torch.zeros((quaternion.shape[0], 3,), device=device, dtype=dtype)
    normalized_axis[:, 0] = quaternion[:, 1] / sin_angle_2
    normalized_axis[:, 1] = quaternion[:, 2] / sin_angle_2
    normalized_axis[:, 2] = quaternion[:, 3] / sin_angle_2

    return normalized_axis, angle.view(-1, 1)  # add feature dimension


def skew_matrix_from_array(x, device=torch.device("cpu"), dtype=torch.float32):
    """
Receives a 3-element array and return its respective skew matrix.

    :param x: 3-element array.
    :return: Respective skew-matrix (3x3).
    :param device: Device to allocate new tensors. Default: torch.device("cpu").
    :param dtype: Data type for new tensors. Default: torch.float32.
    """

    # zeros with the same batch size
    zeros = torch.zeros((x.shape[0], 1), device=device, dtype=dtype)

    return torch.stack((
        torch.hstack((zeros, -x[:, 2:], x[:, 1:2])),
        torch.hstack((x[:, 2:], zeros, -x[:, 0:1])),
        torch.hstack((-x[:, 1:2], x[:, 0:1], zeros)),
    ), dim=1)


def array_from_skew_matrix(x, device=torch.device("cpu"), dtype=torch.float32):
    """
Receives a skew matrix and returns its associated 3-element vector (array).

    :param x: Skew matrix (3x3)
    :return: Associated array (3-element).
    :param device: Device to allocate new tensors. Default: torch.device("cpu").
    :param dtype: Data type for new tensors. Default: torch.float32.
    """
    # We are only slicing the last index in order to keep last dimension.
    return torch.hstack((x[:, 2, 1:2], x[:, 0, 2:], x[:, 1, 0:1]))


# # ERRADA!!!!!!!!!!!!!!!!!#
# def exp_matrix(skew_matrix, device=torch.device("cpu"), dtype=torch.float32):
#     # ERRADA!!!!!!!!!!!!!!!!!#
#     # Reshape necessary to accept multiplication by matrix.
#     norma = torch.linalg.norm(skew_matrix, dim=(-2, -1)).view(-1, 1, 1)
#
#     x = 9
#
#     returno = torch.eye(n=3, m=3, device=device, dtype=dtype) + \
#               (torch.sin(norma) / norma) * skew_matrix + \
#               (1 - torch.cos(norma)) / (norma ** 2) * torch.matmul(skew_matrix, skew_matrix)
#
#     return returno


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
    # torch.einsum('bii->b', a) gives r_matrix trace in a batch of matrices.
    phi = torch.nan_to_num(torch.arccos((torch.einsum('bii->b', r_matrix) - 1) / 2).view(-1, 1, 1))

    # Skew "orientation" matrix into axis-angles tensor (3-element).
    # we do not multiply by phi, so we have a normalized rotation AXIS (in a SKEW matrix yet)
    # normalized because we didnt multiply the axis by the rotation angle (phi)
    aux = 2 * torch.sin(phi)
    return array_from_skew_matrix((r_matrix - r_matrix.movedim(1, 2)) / torch.where(aux == 0, torch.tensor(0.000001, device=device, dtype=dtype), aux), device=device, dtype=dtype), phi[:, 0, 0]


def axis_angle_into_rotation_matrix(normalized_axis, angle, device=torch.device("cpu"), dtype=torch.float32):
    return torch.matrix_exp(skew_matrix_from_array(normalized_axis * angle + 0.00001, device=device, dtype=dtype))
