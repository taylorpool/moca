from src.variables import SO3, SE3
from utils.index import Index
from python import Python
import math
from src.util import (
    assert_almost_equal,
    assert_almost_equal_tensor,
)
import src.moca as mc

# TODO: Use more interesting quaternion, edges cases might be being missed


def rot32quat(rot3: PythonObject) -> mc.Vector4d:
    return mc.np2simd[4](rot3.toQuaternion().coeffs())


def quat2rot3(q: mc.Vector4d) -> PythonObject:
    gtsam = Python.import_module("gtsam")
    return gtsam.Rot3(q[3], q[0], q[1], q[2])


def test_expmap() -> NoneType:
    print("# se3 expmap")
    gtsam = Python.import_module("gtsam")
    np = Python.import_module("numpy")

    xipy = np.array([0.1, 0.1, 0.1, 1, 2, 3])
    xi = mc.np2simd[8](xipy)

    T_des = mc.np2tensor2d[DType.float64](gtsam.Pose3().Expmap(xipy).matrix())
    T_mine = SE3.expmap(xi).as_mat()

    assert_almost_equal_tensor[DType.float64, 16](T_des, T_mine)


def test_mult() -> NoneType:
    print("# se3 mult")
    gtsam = Python.import_module("gtsam")
    np = Python.import_module("numpy")

    qpy = np.array([0, 0, 1, 0])
    tpy = np.array([1, 2, 3])

    q = mc.np2simd[4](qpy)
    t = mc.np2simd[4](tpy)

    Tpy = gtsam.Pose3(quat2rot3(q), tpy)
    T = SE3(q, t)

    T_des = mc.np2tensor2d[DType.float64]((Tpy * Tpy).matrix())
    T_mine = (T * T).as_mat()

    assert_almost_equal_tensor[DType.float64, 16](T_des, T_mine)


def test_identity() -> NoneType:
    print("# se3 identity")
    I = Tensor[DType.float64](4, 4)
    for i in range(4):
        I[Index(i, i)] = 1

    T = SE3.identity().as_mat()
    assert_almost_equal_tensor[DType.float64, 16](I, T)


def test_matrix() -> NoneType:
    print("# se3 matrix")
    gtsam = Python.import_module("gtsam")
    np = Python.import_module("numpy")

    qpy = np.array([0, 0, 1, 0])
    tpy = np.array([1, 2, 3])

    q = mc.np2simd[4](qpy)
    t = mc.np2simd[4](tpy)

    Tpy = gtsam.Pose3(quat2rot3(q), tpy)
    T = SE3(q, t)

    T_des = mc.np2tensor2d[DType.float64](Tpy.matrix())
    T_mine = T.as_mat()

    assert_almost_equal_tensor[DType.float64, 16](T_des, T_mine)


def test_retract() -> NoneType:
    print("# se3 retract")
    gtsam = Python.import_module("gtsam")
    np = Python.import_module("numpy")

    xipy = np.array([0.1, 0.1, 0.1, 1, 2, 3])
    qpy = np.array([0, 0, 1, 0])
    tpy = np.array([1, 2, 3])

    xi = mc.np2simd[8](xipy)
    q = mc.np2simd[4](qpy)
    t = mc.np2simd[4](tpy)

    Tpy = gtsam.Pose3(quat2rot3(q), tpy)
    T = SE3(q, t)

    T_des = mc.np2tensor2d[DType.float64](Tpy.retract(xipy).matrix())
    T_mine = (T + xi).as_mat()

    assert_almost_equal_tensor[DType.float64, 16](T_des, T_mine)


def test_inv() -> NoneType:
    print("# se3 inverse")
    gtsam = Python.import_module("gtsam")
    np = Python.import_module("numpy")

    qpy = np.array([0, 0, 1, 0])
    tpy = np.array([1, 2, 3])

    q = mc.np2simd[4](qpy)
    t = mc.np2simd[4](tpy)

    Tpy = gtsam.Pose3(quat2rot3(q), tpy)
    T = SE3(q, t)

    T_des = mc.np2tensor2d[DType.float64](Tpy.inverse().matrix())
    T_mine = T.inv().as_mat()

    assert_almost_equal_tensor[DType.float64, 16](T_des, T_mine)


def test_rotate() -> NoneType:
    print("# se3 rotate")

    gtsam = Python.import_module("gtsam")
    np = Python.import_module("numpy")

    qpy = np.array([0, 0, 1, 0])
    tpy = np.array([1, 2, 3])
    ppy = np.array([4, 5, 6])

    q = mc.np2simd[4](qpy)
    t = mc.np2simd[4](tpy)
    p = mc.np2simd[4](ppy)

    Tpy = gtsam.Pose3(quat2rot3(q), tpy)
    T = SE3(q, t)

    p_des = mc.np2simd[4](Tpy.transformFrom(ppy))
    p_mine = T * p

    assert_almost_equal[DType.float64, 4](p_des, p_mine)


fn main() raises:
    test_expmap()
    test_mult()
    test_identity()
    test_matrix()
    test_retract()
    test_inv()
    test_rotate()
