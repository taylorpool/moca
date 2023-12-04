from src.variables import SO3
from utils.index import Index
from python import Python
import math
import src.moca as mc
from src.util import (
    assert_almost_equal,
    assert_almost_equal_tensor,
)


# TODO: Use more interesting quaternion, edges cases might be being missed
def test_from_mat() -> NoneType:
    print("# so3 from_mat")
    R = Python.import_module("scipy.spatial.transform").Rotation
    np = Python.import_module("numpy")

    Rpy = R.from_rotvec([0.1, 0.3, 0.4])
    qpy = Rpy.as_quat()
    matpy = Rpy.as_matrix()

    q = mc.np2simd[4](qpy)
    mat = mc.np2tensor2d[DType.float64](matpy)

    qmine = SO3(mat).quat

    assert_almost_equal(q, qmine)


def test_expmap() -> NoneType:
    print("# so3 expmap")
    R = Python.import_module("scipy.spatial.transform").Rotation
    np = Python.import_module("numpy")

    xipy = np.array([0.1, 0.1, 0.1])
    xi = mc.np2simd[4](xipy)

    q_des = mc.np2simd[4](R.from_rotvec(xipy).as_quat())
    q_mine = SO3.expmap(xi).quat

    assert_almost_equal(q_des, q_mine)


def test_mult() -> NoneType:
    print("# so3 mult")
    R = Python.import_module("scipy.spatial.transform").Rotation
    np = Python.import_module("numpy")

    q1py = np.array([0, 0, 1, 0])
    q2py = np.array([0, 1, 0, 0])

    q1 = SO3(mc.np2simd[4](q1py))
    q2 = SO3(mc.np2simd[4](q2py))

    q_des = mc.np2simd[4]((R.from_quat(q1py) * R.from_quat(q2py)).as_quat())
    q_mine = (q1 * q2).quat

    assert_almost_equal(q_des, q_mine)


def test_identity() -> NoneType:
    print("# so3 identity")
    I = Tensor[DType.float64](3, 3)
    for i in range(3):
        I[Index(i, i)] = 1

    q = SO3.identity().as_mat()
    assert_almost_equal_tensor[DType.float64, 16](I, q)


def test_matrix() -> NoneType:
    print("# so3 matrix")
    R = Python.import_module("scipy.spatial.transform").Rotation
    np = Python.import_module("numpy")

    two = math.rsqrt(SIMD[DType.float64, 1](2))
    q1py = np.array([0, 0, two, two])
    q1 = mc.np2simd[4](q1py)

    R_des = mc.np2tensor2d[DType.float64](R.from_quat(q1py).as_matrix())
    R_mine = SO3(q1).as_mat()

    assert_almost_equal_tensor[DType.float64, 16](R_des, R_mine)


def test_retract() -> NoneType:
    print("# so3 retract")
    gtsam = Python.import_module("gtsam")
    np = Python.import_module("numpy")

    two = math.rsqrt(SIMD[DType.float64, 1](2))
    q1py = np.array([0, 0, two, two])
    xipy = np.array([0.1, 0.1, 0.1])

    q1 = mc.np2simd[4](q1py)
    xi = mc.np2simd[4](xipy)

    # gtsam takes in as w, x, y, z returns as x, y, z, w
    R_des = gtsam.Rot3(q1py[3], q1py[0], q1py[1], q1py[2]).retract(xipy)
    q_des = mc.np2simd[4](R_des.toQuaternion().coeffs())

    q_mine = (SO3(q1) + xi).quat

    assert_almost_equal(q_des, q_mine)


def test_inv() -> NoneType:
    print("# so3 inv")
    R = Python.import_module("scipy.spatial.transform").Rotation
    np = Python.import_module("numpy")

    two = math.rsqrt(SIMD[DType.float64, 1](2))
    q1py = np.array([0, 0, two, two])
    q1 = mc.np2simd[4](q1py)

    q_des = mc.np2simd[4](R.from_quat(q1py).inv().as_quat())
    q_mine = SO3(q1).inv().quat

    assert_almost_equal(q_des, q_mine)


def test_rotate() -> NoneType:
    print("# so3 rotate")
    R = Python.import_module("scipy.spatial.transform").Rotation
    np = Python.import_module("numpy")

    two = math.rsqrt(SIMD[DType.float64, 1](2))
    q1py = np.array([0, 0, two, two])
    q1 = mc.np2simd[4](q1py)
    ppy = np.array([1, 2, 3])
    p = mc.np2simd[4](ppy)

    p_des = mc.np2simd[4](R.from_quat(q1py).apply(ppy))
    p_mine = SO3(q1) * p

    assert_almost_equal(p_des, p_mine)


fn main() raises:
    test_from_mat()
    test_expmap()
    test_mult()
    test_identity()
    test_matrix()
    test_retract()
    test_inv()
    test_rotate()
