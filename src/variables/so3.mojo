from math import sqrt, sin, cos
from utils.index import Index
import src.moca as mc


@register_passable("trivial")
struct SO3:
    var quat: mc.Vector4d

    fn __init__(
        x: Float64,
        y: Float64,
        z: Float64,
        w: Float64,
    ) -> Self:
        return Self {quat: mc.Vector4d(x, y, z, w)}

    fn __init__(quat: mc.Vector4d) -> Self:
        let norm = (quat * quat).reduce_add()
        # TODO Change to warning?
        if (norm - 1.0) ** 2 > 1e-3:
            print("Initialized with quaternion norm", norm)
        return Self {quat: quat}

    @always_inline
    @staticmethod
    fn identity() -> Self:
        return Self(0, 0, 0, 1)

    @always_inline
    @staticmethod
    fn expmap(xi: mc.Vector3d) -> Self:
        # Zero out the last element
        var vec = xi
        vec[3] = 0

        let theta = sqrt[DType.float64]((vec * vec).reduce_add())
        vec = sin(theta / 2) * vec / theta
        vec[3] = cos(theta / 2)
        return Self(vec)

    @always_inline
    fn as_mat(self) -> Tensor[DType.float64]:
        var mat = Tensor[DType.float64](3, 3)

        # Extract the values from Q
        let x = self.quat[0]
        let y = self.quat[1]
        let z = self.quat[2]
        let w = self.quat[3]

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

    fn __invert__(self) -> Self:
        let x = self.quat[0]
        let y = self.quat[1]
        let z = self.quat[2]
        let w = self.quat[3]
        return Self(-x, -y, -z, w)

    fn inv(self) -> Self:
        return ~self

    @always_inline
    fn __mul__(self, p: mc.Vector3d) -> mc.Vector3d:
        let p_rot = SO3(p[0], p[1], p[2], 0)
        let inv = ~self
        return (self * p_rot * inv).quat

    @always_inline
    fn __mul__(self, other: SO3) -> SO3:
        let x0 = self.quat[0]
        let y0 = self.quat[1]
        let z0 = self.quat[2]
        let w0 = self.quat[3]

        let x1 = other.quat[0]
        let y1 = other.quat[1]
        let z1 = other.quat[2]
        let w1 = other.quat[3]

        # Computer the product of the two quaternions, term by term
        let out_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
        let out_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
        let out_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
        let out_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

        return SO3(out_x, out_y, out_z, out_w)

    @always_inline
    fn __add__(self, other: mc.Vector3d) -> SO3:
        """Not really addition, just syntatic sugar for "retract" operation.
        Adds in an vector from the tangent space.
        """
        return self * self.expmap(other)
