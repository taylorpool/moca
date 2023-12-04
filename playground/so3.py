import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm

np.set_printoptions(suppress=True, precision=4)


def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def aa2quat(aa):
    theta = np.linalg.norm(aa)
    w = aa / theta
    quat = w * np.sin(theta / 2)
    quat = np.append(quat, np.cos(theta / 2))
    return quat


def quaternion_multiply(w, x):
    # Extract the values from w
    x0, y0, z0, w0 = w

    # Extract the values from x
    x1, y1, z1, w1 = x

    # Computer the product of the two quaternions, term by term
    wx_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    wx_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    wx_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    wx_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    # Create a 4 element array containing the final quaternion
    return np.array([wx_x, wx_y, wx_z, wx_w])


def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (w,x,y,z)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    w = Q[0]
    x = Q[1]
    y = Q[2]
    z = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (w * w + x * x) - 1
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)

    # Second row of the rotation matrix
    r10 = 2 * (x * y + w * z)
    r11 = 2 * (w * w + y * y) - 1
    r12 = 2 * (y * z - w * x)

    # Third row of the rotation matrix
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 2 * (w * w + z * z) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])

    return rot_matrix


# delta = np.array([0.1, 0.1, 0.1])
# print(expm(skew(delta)))
# print(R.from_quat(aa2quat(delta)).as_matrix())

# x y z w
quat = np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
print(R.from_quat(quat).as_matrix())
quat = [quat[-1], quat[0], quat[1], quat[2]]
print(quaternion_rotation_matrix(quat))
