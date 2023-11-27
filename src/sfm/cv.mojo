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
) -> InlinedFixedVector[Landmark]:
    debug_assert(
        pts1.dim(0) == pts2.dim(0), "[TRIANGULATE] Got varying number of points"
    )
    debug_assert(pts1.dim(1) == 2, "[TRIANGULATE]  Too many columns")
    debug_assert(pts2.dim(1) == 2, "[TRIANGULATE]  Too many columns")

    var out = Tensor[DType.float64](pts1.dim(0), 4)
    for i in range(pts1.dim(0)):
        # TODO: Set via slices or via stacking
        var p1 = mc.get_row[DType.float64, 4](pts1, i)
        var p2 = mc.get_row[DType.float64, 4](pts2, i)
        var tens1 = mc.matrix_matrix_multiply(SO3.skew(p1), (K1 * T1))
        var tens2 = mc.matrix_matrix_multiply(SO3.skew(p2), (K2 * T2))
        # TODO: SVD here
        var p3d = SIMD[DType.float64, 4](1)
        mc.set_row[DType.float64, 4](out, i, p3d)


fn findEssentialMat(
    kp1: Tensor[DType.float64],
    kp2: Tensor[DType.float64],
    K1: PinholeCamera,
    K2: PinholeCamera,
) -> Tensor[DType.float64]:
    pass


fn findFundamentalMat(
    kp1: Tensor[DType.float64], kp2: Tensor[DType.float64]
) -> Tensor[DType.float64]:
    pass


fn recoverPose(
    E: Tensor[DType.float64], kp1: Tensor[DType.float64], kp2: Tensor[DType.float64]
) -> SE3:
    pass
