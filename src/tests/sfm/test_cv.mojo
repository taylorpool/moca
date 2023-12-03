from utils.index import Index
from python import Python
import random

import src.sfm.cv as cv
import src.moca as mc
from src.util import (
    assert_almost_equal_tensor,
    assert_almost_equal,
    assert_true,
)
from src.variables import PinholeCamera, SE3, SO3
import src.sfm.cv_util as cv_util

from src.moca.test_moca import test_matT_matT


fn randf() -> SIMD[DType.float64, 1]:
    return random.randn_float64(0, 1)


def test_findFundamentalMat() -> NoneType:
    print("# cv fundamental")
    np = Python.import_module("numpy")

    # Define test data
    rng = np.random.default_rng(0)
    kp1py = rng.standard_normal((8, 2))
    kp2py = kp1py + 5

    kp1 = mc.np2tensor2d_f64(kp1py)
    kp2 = mc.np2tensor2d_f64(kp2py)

    # Get mine
    F_mine = cv.findFundamentalMat(kp1, kp2)

    for i in range(8):
        let k1 = mc.Vector3d(kp1[i, 0], kp1[i, 1], 1)
        let k2 = mc.Vector3d(kp2[i, 0], kp2[i, 1], 1)
        let out = mc.vecT_mat_vec(k2, F_mine, k1)
        assert_almost_equal[DType.float64](0, out)

    all_zero = True
    for i in range(3):
        for j in range(3):
            all_zero = all_zero and F_mine[i, j] == 0

    assert_true(not all_zero, "All of fundamental matrix was zero!")


def test_findEssentialMat() -> NoneType:
    print("# cv essential")
    cvpy = Python.import_module("cv2")
    np = Python.import_module("numpy")

    # Define test data
    np.random.seed(0)
    kp1py = np.random.rand(8, 2)
    kp2py = kp1py + 5

    kp1 = mc.np2tensor2d_f64(kp1py)
    kp2 = mc.np2tensor2d_f64(kp2py)
    K = PinholeCamera(8, 6, 4, 3)

    # Get mine
    E_mine = cv.findEssentialMat(kp1, kp2, K, K)

    let Kinv = mc.inv3(K.as_mat())
    for i in range(8):
        var k1 = mc.Vector3d(kp1[i, 0], kp1[i, 1], 1)
        k1 = mc.mat_vec(Kinv, k1)
        var k2 = mc.Vector3d(kp2[i, 0], kp2[i, 1], 1)
        k2 = mc.mat_vec(Kinv, k2)
        let out = mc.vecT_mat_vec(k2, E_mine, k1)
        assert_almost_equal[DType.float64](0, out)

    all_zero = True
    for i in range(3):
        for j in range(3):
            all_zero = all_zero and E_mine[i, j] == 0

    assert_true(not all_zero, "All of fundamental matrix was zero!")


def test_triangulate() -> NoneType:
    print("# cv triangulate")
    p = mc.Vector3d(0.1, 0.1, 1)
    K = PinholeCamera(8, 6, 4, 3)
    T1 = SE3.identity()
    T2 = SE3(SO3.expmap(mc.Vector3d(0.01, 0.01, 0.01)), mc.Vector3d(0.2, 0.2, 0.1))

    kp1 = K.project(T1 * p)
    kp2 = K.project(T2 * p)

    kp1_tens = Tensor[DType.float64](1, 2)
    kp1_tens.simd_store(0, kp1)
    kp2_tens = Tensor[DType.float64](1, 2)
    kp2_tens.simd_store(0, kp2)

    p_est = cv.triangulate(K, T1, kp1_tens, K, T2, kp2_tens)

    assert_almost_equal(p, p_est[0].val, atol=1e-4, rtol=1e-5)


def test_PnP() -> NoneType:
    print("# cv PnP")
    n = 45
    ps = Tensor[DType.float64](n, 3)
    random.randn(ps.data(), n * 3, 0, 1)
    K = PinholeCamera(8, 6, 4, 3)
    T = SE3(SO3.expmap(mc.Vector3d(0.05, 0.05, 0.05)), mc.Vector3d(0, 0, 1))

    kps = Tensor[DType.float64](n, 2)
    for i in range(ps.dim(0)):
        p = mc.Vector3d(ps[i, 0], ps[i, 1], ps[i, 2])
        kp = K.project(T * p)
        kps[Index(i, 0)] = kp[0]
        kps[Index(i, 1)] = kp[1]

    T_est = cv.PnP(K, kps, ps)

    assert_almost_equal(T.rot.quat, T_est.rot.quat, atol=1e-3, rtol=1e-5)
    assert_almost_equal(T.trans, T_est.trans)


def test_decomposeEssential() -> NoneType:
    print("# cv decomposeEssential")
    cvpy = Python.import_module("cv2")
    np = Python.import_module("numpy")

    K = PinholeCamera(8, 6, 4, 3)
    T1 = SE3.identity()
    T2 = SE3(SO3.expmap(mc.Vector3d(0.01, 0.01, 0.01)), mc.Vector3d(1, 0, 0))

    n = 100
    kp1 = Tensor[DType.float64](n, 2)
    kp2 = Tensor[DType.float64](n, 2)
    for i in range(n):
        p = mc.Vector3d(randf(), randf(), randf() + 5)
        k1 = K.project(T1 * p)
        k2 = K.project(T2 * p)
        kp1[Index(i, 0)] = k1[0]
        kp1[Index(i, 1)] = k1[1]
        kp2[Index(i, 0)] = k2[0]
        kp2[Index(i, 1)] = k2[1]

    kp1py = mc.tensor2np(kp1)
    kp2py = mc.tensor2np(kp2)

    # Get mine
    E_mine = cv.findEssentialMat(kp1, kp2, K, K)

    let mytuple = cv.decomposeEssentialMat(E_mine)
    let t_mine = mytuple.get[2, mc.Vector3d]()
    let r1_mine = mytuple.get[0, SO3]()
    let r2_mine = mytuple.get[1, SO3]()

    let theirtuple = cvpy.decomposeEssentialMat(mc.tensor2np(E_mine))
    var t_des = mc.np2simd[4](theirtuple[2])
    t_des[3] = 0
    let r1_des = SO3(mc.np2tensor2d_f64(theirtuple[0]))
    let r2_des = SO3(mc.np2tensor2d_f64(theirtuple[1]))

    assert_almost_equal(t_mine, t_des)
    assert_almost_equal(r1_mine.quat, r1_des.quat)
    assert_almost_equal(r2_mine.quat, r2_des.quat)


def test_recoverPose() -> NoneType:
    print("# cv decomposeEssential")
    K = PinholeCamera(8, 6, 4, 3)
    T1 = SE3.identity()
    T2 = SE3(SO3.expmap(mc.Vector3d(0.01, 0.01, 0.01)), mc.Vector3d(1, 0, 0))

    n = 100
    kp1 = Tensor[DType.float64](n, 2)
    kp2 = Tensor[DType.float64](n, 2)
    for i in range(n):
        p = mc.Vector3d(randf(), randf(), randf() + 5)
        k1 = K.project(T1 * p)
        k2 = K.project(T2 * p)
        kp1[Index(i, 0)] = k1[0]
        kp1[Index(i, 1)] = k1[1]
        kp2[Index(i, 0)] = k2[0]
        kp2[Index(i, 1)] = k2[1]

    # Get mine
    E_mine = cv.findEssentialMat(kp1, kp2, K, K)

    let T_mine = cv.recoverPose(E_mine, kp1, kp2, K, K)

    assert_almost_equal(T_mine.rot.quat, T2.rot.quat, atol=1e-4, rtol=1e-5)
    assert_almost_equal(T_mine.trans, T2.trans)


# # Run the test
fn main() raises:
    test_findFundamentalMat()
    test_findEssentialMat()
    test_triangulate()
    test_PnP()
