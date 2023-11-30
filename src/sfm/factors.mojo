from utils.index import Index
from src.variables.pinhole import PinholeCamera
from src.variables.se3 import SE3
from src.variables.so3 import SO3
from src.variables.landmark import Landmark
import src.moca as mc


@value
@register_passable("trivial")
struct ProjectionFactor:
    var id_pose: Int64  # aka image #
    var id_cam: Int64
    var id_lm: Int64
    var measured: SIMD[DType.float64, 2]

    fn residual(self, K: PinholeCamera, T: SE3, p: Landmark) -> SIMD[DType.float64, 2]:
        return K.project(T * p.val) - self.measured

    fn jacobian(
        self,
        K: PinholeCamera,
        T: SE3,
        p: Landmark,
        inout H_K: Tensor[DType.float64],
        inout H_T: Tensor[DType.float64],
        inout H_p: Tensor[DType.float64],
    ):
        # TODO: Optimize all of this function
        let meas = K.project3(T * p.val)
        let pp = T * p.val
        let Kmat = K.as_mat()

        # Get normalization jacobian
        var h_f = Tensor[DType.float64](2, 3)
        h_f[Index(0, 0)] = 1.0 / meas[2]
        h_f[Index(0, 2)] = -meas[0] / (meas[2] * meas[2])
        h_f[Index(1, 1)] = 1.0 / meas[2]
        h_f[Index(1, 2)] = -meas[1] / (meas[2] * meas[2])

        # Get all other jacobians
        var f_K = Tensor[DType.float64](3, 4)
        f_K[Index(0, 0)] = pp[0]
        f_K[Index(0, 2)] = pp[2]
        f_K[Index(1, 1)] = pp[1]
        f_K[Index(1, 3)] = pp[2]

        var f_T_temp = Tensor[DType.float64](4, 6)
        f_T_temp[Index(2, 1)] = -p.val[0]
        f_T_temp[Index(1, 2)] = p.val[0]
        f_T_temp[Index(0, 2)] = -p.val[1]
        f_T_temp[Index(2, 0)] = p.val[1]
        f_T_temp[Index(1, 0)] = -p.val[2]
        f_T_temp[Index(0, 1)] = p.val[2]
        f_T_temp[Index(0, 3)] = 1.0
        f_T_temp[Index(1, 4)] = 1.0
        f_T_temp[Index(2, 5)] = 1.0
        let KT = mc.mat_mat(K.as_mat(), T.as_mat())
        let f_T = mc.mat_mat(KT, f_T_temp)

        let f_p = mc.mat_mat(K.as_mat(True), T.rot.as_mat())

        H_K = mc.mat_mat(h_f, f_K)
        H_T = mc.mat_mat(h_f, f_T)
        H_p = mc.mat_mat(h_f, f_p)
