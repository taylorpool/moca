from src.variables import PinholeCamera, SE3, Landmark, SO3
import src.sfm.cv_util as cv_util
import src.moca as mc

from utils.index import Index
from memory import memset_zero


# fn RANSAC[
#     error: fn () -> Int32
# ](max_iter: Int = 2048, threshold: Float64 = 1.5) -> Tensor[DType.float64]:
#     var best_num = 0
#     var best_result = 0

#     for i in range(max_iter):
#         var e = error()
#         if e < best_result


fn PnP(
    pts2d: Tensor[DType.float64], pts3d: Tensor[DType.float64]
) -> Tensor[DType.float64]:
    let n = pts3d.dim(1)

    # build action matrix
    var A = Tensor[DType.float64](n * 2, 12)
    for i in range(n):
        A[Index(2 * i, 0)] = pts3d[i, 0]
        A[Index(2 * i, 1)] = pts3d[i, 1]
        A[Index(2 * i, 2)] = pts3d[i, 2]
        A[Index(2 * i, 3)] = 1.0

        A[Index(2 * i, 8)] = -pts2d[i, 0] * pts3d[i, 0]
        A[Index(2 * i, 9)] = -pts2d[i, 0] * pts3d[i, 1]
        A[Index(2 * i, 10)] = -pts2d[i, 0] * pts3d[i, 2]
        A[Index(2 * i, 11)] = -pts2d[i, 0]

        A[Index(2 * i + 1, 4)] = pts3d[i, 0]
        A[Index(2 * i + 1, 5)] = pts3d[i, 1]
        A[Index(2 * i + 1, 6)] = pts3d[i, 2]
        A[Index(2 * i + 1, 7)] = 1.0

        A[Index(2 * i + 1, 8)] = -pts2d[i, 1] * pts3d[i, 0]
        A[Index(2 * i + 1, 9)] = -pts2d[i, 1] * pts3d[i, 1]
        A[Index(2 * i + 1, 10)] = -pts2d[i, 1] * pts3d[i, 2]
        A[Index(2 * i + 1, 11)] = -pts2d[i, 1]

    # Solve P
    # var p = mc.svd(A)
    var p = Tensor[DType.float64](12)
    var P = Tensor[DType.float64](3, 4)
    P[Index(0, 0)] = p[0]
    P[Index(0, 1)] = p[1]
    P[Index(0, 2)] = p[2]
    P[Index(0, 3)] = p[3]
    P[Index(1, 0)] = p[4]
    P[Index(1, 1)] = p[5]
    P[Index(1, 2)] = p[6]
    P[Index(1, 3)] = p[7]
    P[Index(2, 0)] = p[8]
    P[Index(2, 1)] = p[9]
    P[Index(2, 2)] = p[10]
    P[Index(2, 3)] = p[11]

    return P


fn PnP(
    K: PinholeCamera, pts2d: Tensor[DType.float64], pts3d: Tensor[DType.float64]
) -> SE3:
    var P = PnP(pts2d, pts3d)
    var Rt = mc.mat_mat(mc.inv3(K.as_mat(True)), P)
    # TODO: Orthogonalize R to make sure it's a rotation matrix
    return SE3(SO3(Rt), mc.Vector3d(Rt[0, 3], Rt[1, 3], Rt[2, 3], 0))


fn triangulate(
    K1: PinholeCamera,
    T1: SE3,
    pts1: Tensor[DType.float64],
    K2: PinholeCamera,
    T2: SE3,
    pts2: Tensor[DType.float64],
) -> DynamicVector[Landmark]:
    debug_assert(
        pts1.dim(0) == pts2.dim(0), "[TRIANGULATE] Got varying number of points"
    )
    debug_assert(pts1.dim(1) == 2, "[TRIANGULATE]  Too many columns")
    debug_assert(pts2.dim(1) == 2, "[TRIANGULATE]  Too many columns")
    # TODO Cheriality check?

    var out = DynamicVector[Landmark](pts1.dim(0))
    for i in range(pts1.dim(1)):
        var p1 = mc.get_row[DType.float64, 4](pts1, i)
        var p2 = mc.get_row[DType.float64, 4](pts2, i)
        p1[2] = 1
        p2[2] = 1
        let tens1 = mc.mat_mat(SO3.skew(p1), (K1 * T1))
        let tens2 = mc.mat_mat(SO3.skew(p2), (K2 * T2))

        # TODO: More efficient way to do this?
        var A = Tensor[DType.float64](4, 4)
        for i in range(2):
            for j in range(4):
                A[Index(i, j)] = tens1[Index(i, j)]
                A[Index(i + 1, j)] = tens2[Index(i, j)]

        let svd = mc.svd(A)
        let z = svd.vh[3, 3]
        let p3d = mc.Vector3d(svd.vh[3, 0] / z, svd.vh[3, 1] / z, svd.vh[3, 2] / z, 0)
        out.push_back(Landmark(p3d))

    return out


fn findFundamentalMat(
    kp1: Tensor[DType.float64], kp2: Tensor[DType.float64]
) -> Tensor[DType.float64]:
    let num_correspondences = kp1.shape()[0]
    var A = Tensor[DType.float64](num_correspondences, 9)
    for i in range(num_correspondences):
        A[Index(i, 0)] = kp2[i, 0] * kp1[i, 0]
        A[Index(i, 1)] = kp2[i, 0] * kp1[i, 1]
        A[Index(i, 2)] = kp2[i, 0]
        A[Index(i, 3)] = kp2[i, 1] * kp1[i, 0]
        A[Index(i, 4)] = kp2[i, 1] * kp1[i, 1]
        A[Index(i, 5)] = kp2[i, 1]
        A[Index(i, 6)] = kp1[i, 0]
        A[Index(i, 7)] = kp1[i, 1]
        A[Index(i, 8)] = 1.0

    let svd = mc.svd(A)
    # print(A)

    var F = Tensor[DType.float64](3, 3)
    let z = svd.vh[8, 8]
    F[Index(0, 0)] = svd.vh[8, 0] / z
    F[Index(0, 1)] = svd.vh[8, 1] / z
    F[Index(0, 2)] = svd.vh[8, 2] / z
    F[Index(1, 0)] = svd.vh[8, 3] / z
    F[Index(1, 1)] = svd.vh[8, 4] / z
    F[Index(1, 2)] = svd.vh[8, 5] / z
    F[Index(2, 0)] = svd.vh[8, 6] / z
    F[Index(2, 1)] = svd.vh[8, 7] / z
    F[Index(2, 2)] = 1

    # var svd2 = mc.svd(F)
    # svd2.s[2] = 0
    # F = mc.mat_mat(mc.mat_mat(svd2.u, mc.diag(svd2.s)), svd2.vh)

    return F


fn findEssentialMat(
    kp1: Tensor[DType.float64],
    kp2: Tensor[DType.float64],
    K1: PinholeCamera,
    K2: PinholeCamera,
) -> Tensor[DType.float64]:
    let num_correspondences = kp1.shape()[0]
    var A = Tensor[DType.float64](num_correspondences, 9)
    let F = findFundamentalMat(kp1, kp2)
    let E = mc.mat_mat(mc.matT_mat(K2.as_mat(), F), K1.as_mat())
    return E


fn recoverPose(
    E: Tensor[DType.float64], kp1: Tensor[DType.float64], kp2: Tensor[DType.float64]
) -> SE3:
    return SE3.identity()
