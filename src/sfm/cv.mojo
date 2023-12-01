from src.variables import PinholeCamera, SE3, Landmark, SO3
import src.sfm.cv_util as cv_util
import src.moca as mc

from utils.index import Index
from memory import memset_zero
import math

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
    let n = pts3d.dim(0)

    # build action matrix
    var A = Tensor[DType.float64](n * 2, 12)
    memset_zero(A.data(), A.num_elements())
    for i in range(n):
        A[Index(2 * i, 4)] = -pts3d[i, 0]
        A[Index(2 * i, 5)] = -pts3d[i, 1]
        A[Index(2 * i, 6)] = -pts3d[i, 2]
        A[Index(2 * i, 7)] = -1.0

        A[Index(2 * i, 8)] = pts2d[i, 1] * pts3d[i, 0]
        A[Index(2 * i, 9)] = pts2d[i, 1] * pts3d[i, 1]
        A[Index(2 * i, 10)] = pts2d[i, 1] * pts3d[i, 2]
        A[Index(2 * i, 11)] = pts2d[i, 1]

        A[Index(2 * i + 1, 0)] = pts3d[i, 0]
        A[Index(2 * i + 1, 1)] = pts3d[i, 1]
        A[Index(2 * i + 1, 2)] = pts3d[i, 2]
        A[Index(2 * i + 1, 3)] = 1.0

        A[Index(2 * i + 1, 8)] = -pts2d[i, 0] * pts3d[i, 0]
        A[Index(2 * i + 1, 9)] = -pts2d[i, 0] * pts3d[i, 1]
        A[Index(2 * i + 1, 10)] = -pts2d[i, 0] * pts3d[i, 2]
        A[Index(2 * i + 1, 11)] = -pts2d[i, 0]

    # Solve P
    let svd = mc.svd(A)

    var P = Tensor[DType.float64](3, 4)
    P[Index(0, 0)] = svd.vh[11, 0]
    P[Index(0, 1)] = svd.vh[11, 1]
    P[Index(0, 2)] = svd.vh[11, 2]
    P[Index(0, 3)] = svd.vh[11, 3]
    P[Index(1, 0)] = svd.vh[11, 4]
    P[Index(1, 1)] = svd.vh[11, 5]
    P[Index(1, 2)] = svd.vh[11, 6]
    P[Index(1, 3)] = svd.vh[11, 7]
    P[Index(2, 0)] = svd.vh[11, 8]
    P[Index(2, 1)] = svd.vh[11, 9]
    P[Index(2, 2)] = svd.vh[11, 10]
    P[Index(2, 3)] = svd.vh[11, 11]

    return P


fn PnP(
    K: PinholeCamera, pts2d: Tensor[DType.float64], pts3d: Tensor[DType.float64]
) -> SE3:
    # Get projection matrix
    let P = PnP(pts2d, pts3d)
    let Rt = mc.mat_mat(mc.inv3(K.as_mat(True)), P)

    # Orthogonalize R to make sure it's a rotation matrix
    var R = Tensor[DType.float64](3, 3)
    for i in range(3):
        for j in range(3):
            R[Index(i, j)] = Rt[Index(i, j)]
    let svd = mc.svd(R)
    R = mc.mat_mat(svd.u, svd.vh)

    var t = mc.Vector3d(Rt[0, 3], Rt[1, 3], Rt[2, 3], 0)
    t /= math.sqrt((t * t).reduce_add())

    return SE3(SO3(R), t)


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
    for i in range(pts1.dim(0)):
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
    let F = findFundamentalMat(kp1, kp2)
    let E = mc.mat_mat(mc.matT_mat(K2.as_mat(True), F), K1.as_mat(True))
    return E


fn decomposeEssentialMat(E: Tensor[DType.float64]) -> Tuple[SO3, SO3, mc.Vector3d]:
    let svd = mc.svd(E)
    let u = svd.u
    let vh = svd.vh

    var w = Tensor[DType.float64](3, 3)
    w[Index(0, 1)] = -1
    w[Index(1, 0)] = 1
    w[Index(2, 2)] = 1

    let R1 = mc.mat_mat(mc.mat_matT(u, w), vh)
    let R2 = mc.mat_mat(mc.mat_mat(u, w), vh)
    let t = mc.Vector3d(u[0, 2], u[1, 2], u[2, 2], 0)

    return SO3(R1), SO3(R2), t


fn recoverPose(
    E: Tensor[DType.float64],
    kp1: Tensor[DType.float64],
    kp2: Tensor[DType.float64],
    K1: PinholeCamera,
    K2: PinholeCamera,
) -> SE3:
    let tuple = decomposeEssentialMat(E)
    let R1 = tuple.get[0, SO3]()
    let R2 = tuple.get[1, SO3]()
    let t = tuple.get[2, mc.Vector3d]()

    let T1 = SE3.identity()

    var options = InlinedFixedVector[SE3, 4](0)
    options.append(SE3(R1, t))
    options.append(SE3(R1, -t))
    options.append(SE3(R2, t))
    options.append(SE3(R2, -t))

    let num_points = kp1.dim(0)
    var best_in_front = 0
    var best_option = SE3.identity()

    for T2 in options:
        var num_in_front = 0
        let pts3d = triangulate(K1, T1, kp1, K2, T2, kp2)
        for i in range(num_points):
            let p = pts3d[i]

            if p.val[2] > 0:
                num_in_front += 1

        if num_in_front > best_in_front:
            best_in_front = num_in_front
            best_option = T2

    return best_option
