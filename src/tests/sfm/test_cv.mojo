from utils.index import Index
from python import Python
import random

import src.sfm.cv as cv
import src.moca as mc
from src.util import np2tensor2d_f64, assert_almost_equal_tensor, assert_almost_equal
from src.variables import PinholeCamera, SE3, SO3
import src.sfm.cv_util as cv_util

from src.moca.test_moca import test_matT_matT


def test_findFundamentalMat() -> NoneType:
    print("# cv fundamental")
    cvpy = Python.import_module("cv2")
    np = Python.import_module("numpy")

    # Define test data
    np.random.seed(0)
    kp1py = np.random.rand(8, 2) * 5
    kp2py = kp1py + 15

    kp1 = np2tensor2d_f64(kp1py)
    kp2 = np2tensor2d_f64(kp2py)

    # Get actual one
    F_des_py = cvpy.findFundamentalMat(kp2py, kp1py, cvpy.FM_8POINT)[0]
    F_des = np2tensor2d_f64(F_des_py)

    # Get mine
    F_mine = cv.findFundamentalMat(kp1, kp2)

    for i in range(8):
        let k1 = mc.Vector3d(kp1[i, 0], kp1[i, 1], 1)
        let k2 = mc.Vector3d(kp2[i, 0], kp2[i, 1], 1)
        let out = mc.vecT_mat_vec(k2, F_mine, k1)
        assert_almost_equal[DType.float64](0, out)


def test_findEssentialMat() -> NoneType:
    print("# cv essential")
    cvpy = Python.import_module("cv2")
    np = Python.import_module("numpy")

    # Define test data
    kp1py = np.random.rand(8, 2)
    kp2py = kp1py + 5
    Kpy = np.eye(3)

    kp1 = np2tensor2d_f64(kp1py)
    kp2 = np2tensor2d_f64(kp2py)
    K = PinholeCamera(1, 1, 0, 0)

    # Get actual one
    E_des_py = cvpy.findEssentialMat(kp1py, kp2py, Kpy)[0]
    E_des = np2tensor2d_f64(E_des_py)

    # Get mine
    # E_mine = cv.findEssentialMat(kp1, kp2, K, K)

    # assert_almost_equal_tensor[DType.float64, 16](E_des, E_mine)


def test_triangulate() -> NoneType:
    print("# cv triangulate")
    p = mc.Vector3d(0.1, 0.1, 1)
    K = PinholeCamera(8, 6, 4, 3)
    T1 = SE3.identity()
    T2 = SE3(SO3.identity(), mc.Vector3d(0.2, 0.2, 0.1))

    kp1 = K.project(T1 * p)
    kp2 = K.project(T2 * p)

    kp1_tens = Tensor[DType.float64](2, 1)
    kp1_tens.simd_store(0, kp1)
    kp2_tens = Tensor[DType.float64](2, 1)
    kp2_tens.simd_store(0, kp2)

    p_est = cv.triangulate(K, T1, kp1_tens, K, T2, kp2_tens)

    assert_almost_equal(p, p_est[0].val)


def test_PnP() -> NoneType:
    print("# cv PnP")
    ps = Tensor[DType.float64](15, 3)
    random.randn(ps.data(), 45, 0, 1)
    K = PinholeCamera(8, 6, 4, 3)
    T = SE3(SO3.identity(), mc.Vector3d(0, 0, 1))

    kps = Tensor[DType.float64](15, 2)
    for i in range(ps.dim(0)):
        p = mc.Vector3d(ps[i, 0], ps[i, 1], ps[i, 2])
        kp = K.project(T * p)
        kps[Index(i, 0)] = kp[0]
        kps[Index(i, 1)] = kp[1]

    T_est = cv.PnP(K, kps, ps)

    # assert_almost_equal_tensor[DType.float64, 16](T_est.as_mat(), T.as_mat())


# # Run the test
fn main() raises:
    test_findFundamentalMat()
    test_findEssentialMat()
    test_triangulate()
    test_PnP()
