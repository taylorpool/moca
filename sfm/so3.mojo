from math import sqrt, sin, cos
from utils.index import Index
import .moca as mc


@register_passable("trivial")
struct SO3:
    # TODO: w last so we can do expmap easier?
    var quat: mc.Vector4d

    fn __init__(w: Float64, x: Float64, y: Float64, z: Float64) -> Self:
        return Self {quat: mc.Vector4d(w, x, y, z)}

    fn __init__(quat: mc.Vector4d) -> Self:
        return Self {quat: quat}

    @always_inline
    @staticmethod
    fn identity() -> Self:
        return Self(1, 0, 0, 0)

    @always_inline
    @staticmethod
    fn expmap(vec: mc.Vector3d) -> Self:
        # TODO Ensure fourth element of vec is 0
        var theta = sqrt[DType.float64]((vec * vec).reduce_add())
        var xyz = sin(theta / 2) * vec / theta
        return Self(cos(theta / 2), xyz[0], xyz[1], xyz[2])

    @always_inline
    fn as_mat(self) -> Tensor[DType.float64]:
        var mat = Tensor[DType.float64](4, 4)

        # Extract the values from Q
        let w = self.quat[0]
        let x = self.quat[1]
        let y = self.quat[2]
        let z = self.quat[3]

        # First row of the rotation matrix
        mat[Index(0, 0)] = 2 * (w * w + x * x) - 1
        mat[Index(0, 1)] = 2 * (x * y - w * z)
        mat[Index(0, 2)] = 2 * (x * z + w * y)

        # Second row of the rotation matrix
        mat[Index(1, 0)] = 2 * (x * y + w * z)
        mat[Index(1, 1)] = 2 * (w * w + y * y) - 1
        mat[Index(1, 2)] = 2 * (y * z - w * x)

        # Third row of the rotation matrix
        mat[Index(2, 0)] = 2 * (x * z - w * y)
        mat[Index(2, 1)] = 2 * (y * z + w * x)
        mat[Index(2, 2)] = 2 * (w * w + z * z) - 1

        return mat

    @always_inline
    fn __mul__(self, other: SO3) -> SO3:
        let w0 = self.quat[0]
        let x0 = self.quat[1]
        let y0 = self.quat[2]
        let z0 = self.quat[3]

        let w1 = other.quat[0]
        let x1 = other.quat[1]
        let y1 = other.quat[2]
        let z1 = other.quat[3]

        # Computer the product of the two quaternions, term by term
        let out_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
        let out_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
        let out_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
        let out_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

        return SO3(out_w, out_x, out_y, out_z)

    @always_inline
    fn __add__(self, other: mc.Vector3d) -> SO3:
        """Not really addition, just syntatic sugar for "retract" operation.
        Adds in an vector from the tangent space.
        """
        return self * self.expmap(other)
