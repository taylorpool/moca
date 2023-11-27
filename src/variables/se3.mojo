from utils.index import Index
import src.moca as mc
from src.variables import SO3


fn cross(a: mc.Vector3d, b: mc.Vector4d) -> mc.Vector3d:
    return mc.Vector3d(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
        0,
    )


@register_passable("trivial")
struct SE3:
    var rot: SO3
    var trans: mc.Vector3d

    fn __init__(quat: mc.Vector4d, trans: mc.Vector4d) -> Self:
        return Self {rot: SO3(quat), trans: trans}

    fn __init__(rot: SO3, trans: mc.Vector4d) -> Self:
        return Self {rot: rot, trans: trans}

    @always_inline
    @staticmethod
    fn identity() -> Self:
        return Self(SO3(0, 0, 0, 1), mc.Vector3d(0, 0, 0))

    @always_inline
    @staticmethod
    fn expmap(xi: mc.Vector6d) -> Self:
        # TODO Full exponential or just chained here?
        var omega = mc.Vector3d(xi[0], xi[1], xi[2], 0)
        var v = mc.Vector3d(xi[3], xi[4], xi[5], 0)
        let R = SO3.expmap(omega)

        let theta2 = (omega * omega).reduce_add()
        let t: mc.Vector3d
        # Return simple chained expmap
        if theta2 < 1e-2:
            t = v
        # full expmap
        # https://github.com/borglab/gtsam/blob/develop/gtsam/geometry/Pose3.cpp#L177
        else:
            let t_parallel = omega * (omega * v).reduce_add()
            let omega_cross_v = cross(omega, v)
            t = (omega_cross_v - R * omega_cross_v + t_parallel) / theta2

        return Self(R, t)

    @always_inline
    fn as_mat(self) -> Tensor[DType.float64]:
        var mat = Tensor[DType.float64](4, 4)
        let R = self.rot.as_mat()

        # TODO: There's gotta be a better way to do this...
        for i in range(3):
            for j in range(3):
                mat[Index(i, j)] = R[Index(i, j)]

        for i in range(3):
            mat[Index(i, 3)] = self.trans[i]

        mat[Index(3, 3)] = 1

        return mat

    fn __invert__(self) -> Self:
        let invR = ~self.rot
        let invT = -(invR * self.trans)
        return Self(invR, invT)

    fn inv(self) -> Self:
        return ~self

    @always_inline
    fn __mul__(self, p: mc.Vector3d) -> mc.Vector3d:
        return self.rot * p + self.trans

    @always_inline
    fn __mul__(self, other: SE3) -> SE3:
        let R = self.rot * other.rot
        let t = self.rot * other.trans + self.trans
        return SE3(R, t)

    @always_inline
    fn __add__(self, other: mc.Vector6d) -> SE3:
        """Not really addition, just syntatic sugar for "retract" operation.
        Adds in an vector from the tangent space.
        """
        return self * self.expmap(other)
