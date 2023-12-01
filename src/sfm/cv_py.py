import numpy as np

# ------------------------- Helpers ------------------------- #


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


def findEssentialMat(x1, x2, K):
    F = findFundamentalMat(x1, x2)
    E = K.T @ F @ K
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
