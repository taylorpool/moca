from tensor import Tensor
from utils.index import Index
import src.moca as mc
from src.variables import SE3


@register_passable("trivial")
struct PinholeCamera:
    # See: https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h#L206
    var cal: mc.Vector4d

    fn __init__(fx: Float64, fy: Float64, px: Float64, py: Float64) -> Self:
        return Self {cal: mc.Vector4d(fx, fy, px, py)}

    fn __init__(cal: mc.Vector4d) -> Self:
        return Self {cal: cal}

    fn project(self, X: mc.Vector3d) -> mc.Vector2d:
        let w = X[2]
        var x = mc.Vector2d(0, 0)
        x[0] = (self.fx() * X[0] + self.px() * X[2]) / w
        x[1] = (self.fy() * X[1] + self.py() * X[2]) / w
        return x

    fn project3(self, X: mc.Vector3d) -> mc.Vector3d:
        var x = mc.Vector3d(0, 0, 0, 0)
        x[0] = self.fx() * X[0] + self.px() * X[2]
        x[1] = self.fy() * X[1] + self.py() * X[2]
        x[2] = X[2]
        return x

    @always_inline
    @staticmethod
    fn identity() -> Self:
        return Self(0, 0, 0, 0)

    @always_inline
    fn __add__(self, other: Self) -> Self:
        return self.cal + other.cal

    @always_inline
    fn __mul__(self, other: mc.Vector3d) -> mc.Vector3d:
        return self.project3(other)

    @always_inline
    fn __mul__(self, other: SE3) -> Tensor[DType.float64]:
        # TODO: Probably a faster way to do this
        return mc.matrix_matrix_multiply(self.as_mat(), other.as_mat())

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

    @always_inline
    fn as_mat(self, small: Bool = False) -> Tensor[DType.float64]:
        var out: Tensor[DType.float64]
        if small:
            out = Tensor[DType.float64](3, 3)
        else:
            out = Tensor[DType.float64](3, 4)
        out[Index(0, 0)] = self.fx()
        out[Index(1, 1)] = self.fy()
        out[Index(2, 2)] = 1
        out[Index(0, 2)] = self.px()
        out[Index(1, 2)] = self.py()
        return out
