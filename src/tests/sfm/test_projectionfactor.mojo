from src.variables import SO3, SE3, PinholeCamera, Landmark
from src.sfm.factors import ProjectionFactor
from utils.index import Index
from python import Python
import math
from src.util import (
    np2simd,
    np2tensor2d,
    assert_almost_equal,
    assert_almost_equal_tensor,
    nder,
)
import src.moca as mc


def test_Kjac() -> NoneType:
    print("# projection Kjac")

    var H_K = Tensor[DType.float64](2, 3)
    var H_T = Tensor[DType.float64](2, 6)
    var H_p = Tensor[DType.float64](2, 3)
    let factor = ProjectionFactor(0, 0, 0, mc.Vector2d(10, 20))
    let pose = SE3(SO3(0.2, 0.2, 0.2, 0.2, True), mc.Vector3d(1.0, 2.0, 3.0))
    let cam = PinholeCamera(320, 320, 160, 160)
    let lm = Landmark(SIMD[DType.float64, 4](1.0, 2.0, 3.0))
    factor.jacobian(cam, pose, lm, H_K, H_T, H_p)

    @parameter
    fn error(eps: mc.Vector4d) -> mc.Vector2d:
        return factor.residual(cam + eps, pose, lm)

    var H_K_num = nder[2, 4](error)

    assert_almost_equal_tensor[DType.float64, 8](H_K, H_K_num)


def test_Tjac() -> NoneType:
    print("# projection Tjac")

    var H_K = Tensor[DType.float64](2, 3)
    var H_T = Tensor[DType.float64](2, 6)
    var H_p = Tensor[DType.float64](2, 3)
    let factor = ProjectionFactor(0, 0, 0, mc.Vector2d(10, 20))
    let pose = SE3(SO3(0.2, 0.2, 0.2, 0.2, True), mc.Vector3d(1.0, 2.0, 3.0))
    let cam = PinholeCamera(320, 320, 160, 160)
    let lm = Landmark(SIMD[DType.float64, 4](1.0, 2.0, 3.0))
    factor.jacobian(cam, pose, lm, H_K, H_T, H_p)

    @parameter
    fn error(eps: mc.Vector8d) -> mc.Vector2d:
        return factor.residual(cam, pose + eps, lm)

    var H_T_num_temp = nder[2, 8](error)
    var H_T_num = Tensor[DType.float64](2, 6)
    for i in range(2):
        for j in range(6):
            H_T_num[Index(i, j)] = H_T_num_temp[Index(i, j)]

    assert_almost_equal_tensor[DType.float64, 16](H_T, H_T_num)


def test_Pjac() -> NoneType:
    print("# projection Pjac")

    var H_K = Tensor[DType.float64](2, 3)
    var H_T = Tensor[DType.float64](2, 6)
    var H_p = Tensor[DType.float64](2, 3)
    let factor = ProjectionFactor(0, 0, 0, mc.Vector2d(10, 20))
    let pose = SE3(SO3(0.2, 0.2, 0.2, 0.2, True), mc.Vector3d(1.0, 2.0, 3.0))
    let cam = PinholeCamera(320, 320, 160, 160)
    let lm = Landmark(SIMD[DType.float64, 4](1.0, 2.0, 3.0))
    factor.jacobian(cam, pose, lm, H_K, H_T, H_p)

    @parameter
    fn error(eps: mc.Vector4d) -> mc.Vector2d:
        return factor.residual(cam, pose, lm + eps)

    var H_p_num_temp = nder[2, 4](error)
    var H_p_num = Tensor[DType.float64](2, 3)
    for i in range(2):
        for j in range(3):
            H_p_num[Index(i, j)] = H_p_num_temp[Index(i, j)]

    assert_almost_equal_tensor[DType.float64, 8](H_p, H_p_num)


fn main() raises:
    test_Kjac()
    test_Tjac()
    test_Pjac()
