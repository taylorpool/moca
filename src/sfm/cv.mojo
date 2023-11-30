from src.variables import PinholeCamera, SE3, Landmark, SO3
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


fn PnP(K: PinholeCamera, pts2d: Tensor[DType.float64], pts3d: Tensor[DType.float64]):
    pass


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

    var out = DynamicVector[Landmark](pts1.dim(0))
    for i in range(pts1.dim(0)):
        # TODO: Set via slices or via stacking
        let p1 = mc.get_row[DType.float64, 4](pts1, i)
        let p2 = mc.get_row[DType.float64, 4](pts2, i)
        let tens1 = mc.mat_mat(SO3.skew(p1), (K1 * T1))
        let tens2 = mc.mat_mat(SO3.skew(p2), (K2 * T2))
        # TODO: SVD here
        let p3d = SIMD[DType.float64, 4](1)
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

    var f0 = Tensor[DType.float64](A.shape()[1])
    memset_zero(f0.data(), f0.num_elements())
    f0[0] = 1.0
    let f = mc.solve_homogeneous_equation(A, f0)

    var F = Tensor[DType.float64](3, 3)
    F[Index(0, 0)] = f[0]
    F[Index(0, 1)] = f[1]
    F[Index(0, 2)] = f[2]
    F[Index(1, 0)] = f[3]
    F[Index(1, 1)] = f[4]
    F[Index(1, 2)] = f[5]
    F[Index(2, 0)] = f[6]
    F[Index(2, 1)] = f[7]
    F[Index(2, 2)] = f[8]

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
