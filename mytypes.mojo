@value
struct Image:
    var id_cam: Int32
    var kp: Tensor[DType.float32]


@value
struct MatchPair:
    var id1: Int32
    var id2: Int32
    var matches: Tensor[DType.uint32]


@register_passable("trivial")
struct PinholeCamera:
    # TODO: Include extrinsics here too?
    # See: https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h#L206
    var cal: SIMD[DType.float64, 4]

    fn __init__(fx: Float64, fy: Float64, px: Float64, py: Float64) -> Self:
        return PinholeCamera {cal: SIMD[DType.float64, 4](fx, fy, px, py)}

    fn __init__(cal: SIMD[DType.float64, 4]) -> Self:
        return PinholeCamera {cal: cal}

    fn project(inout self, X: Tensor[DType.float64]) -> Tensor[DType.float64]:
        let w = X[2]
        var x = Tensor[DType.float64](2)
        x[0] = (self.fx() * X[0] + self.px() * X[2]) / w
        x[1] = (self.fy() * X[1] + self.py() * X[2]) / w
        return x

    @always_inline
    @staticmethod
    fn identity() -> PinholeCamera:
        return PinholeCamera(0, 0, 0, 0)

    @always_inline
    fn __add__(self, other: PinholeCamera) -> PinholeCamera:
        return self.cal + other.cal

    @always_inline
    fn fx(self) -> Float64:
        return self.cal[0]

    @always_inline
    fn fy(self) -> Float64:
        return self.cal[1]

    @always_inline
    fn px(self) -> Float64:
        return self.cal[2]

    @always_inline
    fn py(self) -> Float64:
        return self.cal[3]
