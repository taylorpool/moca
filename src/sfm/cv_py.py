import numpy as np

# ------------------------- Helpers ------------------------- #
np.set_printoptions(suppress=True)


def homogenize(x):
    if x.ndim == 1:
        return np.append(x, 1)
    else:
        return np.hstack((x, np.ones((x.shape[0], 1))))


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[2], w[0], 0]])


# ------------------------- Implementations ------------------------- #
def PnP_mat(pts_img, pts_3d):
    pts_3d = homogenize(pts_3d)
    w = 1
    A = []
    Z = np.zeros(4)
    for (x, y), X in zip(pts_img, pts_3d):
        A.append([Z, -w * X, y * X])
        A.append([w * X, Z, -x * X])

    A = np.block(A)
    U, s, VT = np.linalg.svd(A)
    p = VT[-1]
    P = p.reshape((3, 4))
    return P


def PnP(K, pts2d, pts3d):
    # Get projection matrix
    P = PnP_mat(pts2d, pts3d)
    Rt = np.linalg.inv(K) @ P

    # Orthogonalize R to make sure it's a rotation matrix
    u, s, vh = np.linalg.svd(Rt[:3, :3])
    R = u @ vh

    t = Rt[:, -1]
    t /= np.sqrt(t @ t)

    return R, Rt[:, -1]


def findFundamentalMat(x1, x2):
    w = 1
    wp = 1
    A = []
    for (u, v), (up, vp) in zip(x1, x2):
        A.append(
            [up * u, up * v, up * w, vp * u, vp * v, vp * w, wp * u, wp * v, wp * w]
        )

    A = np.array(A)
    U, s, V = np.linalg.svd(A)
    f = V[-1]
    F = f.reshape((3, 3)) / f[-1]

    # Enforce rank-2 degrades results
    U, s, V = np.linalg.svd(F)
    s[2] = 0
    F = (U * s) @ V

    return F


def findEssentialMat(x1, x2, K1, K2):
    F = findFundamentalMat(x1, x2)
    E = K1.T @ F @ K2
    return E


def triangulate(pt1s, pt2s, P1, P2):
    pt1s = homogenize(pt1s)
    pt2s = homogenize(pt2s)
    X = np.zeros((pt1s.shape[0], 3))
    for i, (pt1, pt2) in enumerate(zip(pt1s, pt2s)):
        A = np.vstack(
            (
                (skew(pt1) @ P1)[:2],
                (skew(pt2) @ P2)[:2],
            )
        )

        U, s, V = np.linalg.svd(A)
        X[i] = V[-1, :3] / V[-1, -1]

    return X


def decomposeEssentialMat(E):
    u, s, vh = np.linalg.svd(E)

    w = np.zeros((3, 3))
    w[0, 1] = -1
    w[1, 0] = 1
    w[2, 2] = 1

    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        w *= -1

    R1 = u @ w.T @ vh
    R2 = u @ w @ vh
    t = u[:, -1]

    return R1, R2, t


def Rt2mat(R, t):
    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, -1] = t
    return Rt


def recoverPose(
    E,
    kp1,
    kp2,
    K1,
    K2,
):
    R1, R2, t = decomposeEssentialMat(E)
    T1 = np.eye(4)

    options = [Rt2mat(R1, t), Rt2mat(R1, -t), Rt2mat(R2, t), Rt2mat(R2, -t)]

    num_points = kp1.shape[0]
    best_in_front = 0
    best_option = np.eye(4)

    for T2 in options:
        num_in_front = 0
        pts3d = triangulate(kp1, kp2, K1 @ T1, K2 @ T2)
        for i in range(num_points):
            p = pts3d[i]

            if p[2] > 0:
                num_in_front += 1

        if num_in_front > best_in_front:
            best_in_front = num_in_front
            best_option = T2

    return best_option
