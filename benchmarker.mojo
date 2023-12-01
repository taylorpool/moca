from python import Python
import benchmark
import random
from utils.index import Index

from src.sfm import SfM
from src.sfm.sfm import gather
import src.sfm.cv as cv
import src.moca as mc
from src.variables import SO3, SE3, PinholeCamera, Landmark


fn randf() -> SIMD[DType.float64, 1]:
    return random.randn_float64(0, 1)


struct MyBenchmarker:
    var f_mojo: fn () capturing raises -> NoneType
    var f_opencv: fn () capturing raises -> NoneType
    var f_cvpy: fn () capturing raises -> NoneType

    var r_mojo: benchmark.Report
    var r_opencv: benchmark.Report
    var r_cvpy: benchmark.Report

    def __init__(
        inout self,
        f_mojo: fn () capturing raises -> NoneType,
        f_opencv: fn () capturing raises -> NoneType,
        f_cvpy: fn () capturing raises -> NoneType,
    ):
        self.f_mojo = f_mojo
        self.f_opencv = f_opencv
        self.f_cvpy = f_cvpy

        self.r_mojo = benchmark.Report()
        self.r_opencv = benchmark.Report()
        self.r_cvpy = benchmark.Report()

    fn run(inout self):
        @parameter
        fn f_mojo():
            try:
                self.f_mojo()
            except e:
                print(e)

        @parameter
        fn f_opencv():
            try:
                self.f_opencv()
            except e:
                print(e)

        @parameter
        fn f_cvpy():
            try:
                self.f_cvpy()
            except e:
                print(e)

        self.r_mojo = benchmark.run[f_mojo](1)
        self.r_opencv = benchmark.run[f_opencv](1)
        self.r_cvpy = benchmark.run[f_cvpy](1)

    fn plot(self, filename: String, title: String = "") raises:
        Python.add_to_path(".")
        let plotter = Python.import_module("plotter").plot
        _ = plotter(
            ["OpenCV", "Mojo", "Python"],
            [self.r_opencv.mean(), self.r_mojo.mean(), self.r_cvpy.mean()],
            filename,
            title,
        )

    fn print(self):
        print("Mojo: ", self.r_mojo.mean())
        print("OpenCV: ", self.r_opencv.mean())
        print("CV.py: ", self.r_cvpy.mean())


fn main() raises:
    let run_eight = False
    let run_triangulate = False
    let run_pnp = False
    let run_recoverpose = True

    # ------------------------- Load test data ------------------------- #
    let K = PinholeCamera(8, 6, 4, 3)
    let T1 = SE3.identity()
    let T2 = SE3(SO3.expmap(mc.Vector3d(0.01, 0.01, 0.01)), mc.Vector3d(1, 0, 0))

    let n = 500
    var kp1 = Tensor[DType.float64](n, 2)
    var kp2 = Tensor[DType.float64](n, 2)
    var pts = Tensor[DType.float64](n, 3)
    for i in range(n):
        let p = mc.Vector3d(randf(), randf(), randf() + 5)
        pts[Index(i, 0)] = p[0]
        pts[Index(i, 1)] = p[1]
        pts[Index(i, 2)] = p[2]

        let k1 = K.project(T1 * p)
        let k2 = K.project(T2 * p)
        kp1[Index(i, 0)] = k1[0]
        kp1[Index(i, 1)] = k1[1]
        kp2[Index(i, 0)] = k2[0]
        kp2[Index(i, 1)] = k2[1]

    var ptspy = mc.tensor2np2d(pts)
    var kp1py = mc.tensor2np2d(kp1)
    var kp2py = mc.tensor2np2d(kp2)
    var Kpy = mc.tensor2np2d(K.as_mat())
    var Ksmallpy = mc.tensor2np2d(K.as_mat(True))
    var T1py = mc.tensor2np2d(T1.as_mat())
    var T2py = mc.tensor2np2d(T2.as_mat())

    # ------------------------- Start benchmarking ------------------------- #
    let np = Python.import_module("numpy")
    let opencv = Python.import_module("cv2")
    Python.add_to_path("src/sfm/")
    let cvpy = Python.import_module("cv_py")

    if run_eight:
        print("------------------------- 8-Point Algorithm -------------------------")

        fn f_mojo() raises:
            let f = cv.findFundamentalMat(kp1, kp2)

        fn f_opencv() raises:
            let f = opencv.findFundamentalMat(kp1py, kp2py, opencv.FM_8POINT)[0]

        fn f_cvpy() raises:
            let f = cvpy.findFundamentalMat(kp1py, kp2py)

        var t = MyBenchmarker(f_mojo, f_opencv, f_cvpy)
        t.run()
        t.print()
        t.plot("fundamental_mat.png", "8-Point Algorithm")

        print()

    if run_triangulate:
        print("------------------------- Triangulate -------------------------")

        fn tri_mojo() raises:
            let f = cv.triangulate(K, T1, kp1, K, T2, kp2)

        fn tri_opencv() raises:
            let f = opencv.triangulatePoints(
                np.matmul(Kpy, T1py), np.matmul(Kpy, T2py), kp1py.T, kp2py.T
            )

        fn tri_cvpy() raises:
            let f = cvpy.triangulate(
                kp1py, kp2py, np.matmul(Kpy, T1py), np.matmul(Kpy, T2py)
            )

        var t = MyBenchmarker(tri_mojo, tri_opencv, tri_cvpy)
        t.run()
        t.print()
        t.plot("triangulate.png", "Triangulate")
        print()

    if run_pnp:
        print("------------------------- PnP -------------------------")

        fn pnp_mojo() raises:
            let f = cv.PnP(K, kp2, pts)

        fn pnp_opencv() raises:
            let f = opencv.solvePnPGeneric(
                ptspy, kp2py, Ksmallpy, None, None, None, False, opencv.SOLVEPNP_EPNP
            )

        fn pnp_cvpy() raises:
            let f = cvpy.PnP(Ksmallpy, kp2py, ptspy)

        var t = MyBenchmarker(pnp_mojo, pnp_opencv, pnp_cvpy)
        t.run()
        t.print()
        t.plot("pnp.png", "PnP")
        print()

    if run_recoverpose:
        print("------------------------- Recover Pose -------------------------")

        fn rp_mojo() raises:
            let E = cv.findEssentialMat(kp1, kp2, K, K)
            let T = cv.recoverPose(E, kp1, kp2, K, K)

        fn rp_opencv() raises:
            let E = opencv.findEssentialMat(kp1py, kp2py, Ksmallpy)[0]
            let T = opencv.recoverPose(E, kp1py, kp2py, Ksmallpy, Ksmallpy)

        fn rp_cvpy() raises:
            let f = cvpy.PnP(Ksmallpy, kp2py, ptspy)

        var t = MyBenchmarker(rp_mojo, rp_opencv, rp_cvpy)
        t.run()
        t.print()
        t.plot("recoverpose.png", "RecoverPose")
        print()
