@register_passable("trivial")
struct PinholeCamera:
    # See: https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h#L206
    var cal: SIMD[DType.float64, 4]

    fn __init__(fx: Float64, fy: Float64, px: Float64, py: Float64) -> Self:
        return Self {cal: SIMD[DType.float64, 4](fx, fy, px, py)}

    fn __init__(cal: SIMD[DType.float64, 4]) -> Self:
        return Self {cal: cal}

    fn project(inout self, X: Tensor[DType.float64]) -> Tensor[DType.float64]:
        let w = X[2]
        var x = Tensor[DType.float64](2)
        x[0] = (self.fx() * X[0] + self.px() * X[2]) / w
        x[1] = (self.fy() * X[1] + self.py() * X[2]) / w
        return x

    @always_inline
    @staticmethod
    fn identity() -> Self:
        return PinholeCamera(0, 0, 0, 0)

    @always_inline
    fn __add__(self, other: PinholeCamera) -> Self:
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