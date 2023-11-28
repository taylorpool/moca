from src.variables import PinholeCamera, SE3, Landmark, SO3
import src.moca as mc


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
        let tens1 = mc.matrix_matrix_multiply(SO3.skew(p1), (K1 * T1))
        let tens2 = mc.matrix_matrix_multiply(SO3.skew(p2), (K2 * T2))
        # TODO: SVD here
        let p3d = SIMD[DType.float64, 4](1)
        out.push_back(Landmark(p3d))

    return out


fn findEssentialMat(
    kp1: Tensor[DType.float64],
    kp2: Tensor[DType.float64],
    K1: PinholeCamera,
    K2: PinholeCamera,
) -> Tensor[DType.float64]:
    return Tensor[DType.float64](4, 4)


fn findFundamentalMat(
    kp1: Tensor[DType.float64], kp2: Tensor[DType.float64]
) -> Tensor[DType.float64]:
    return Tensor[DType.float64](4, 4)


fn recoverPose(
    E: Tensor[DType.float64], kp1: Tensor[DType.float64], kp2: Tensor[DType.float64]
) -> SE3:
    return SE3.identity()
