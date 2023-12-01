from src.sfm import SfM
from src.sfm.sfm import gather
import src.sfm.cv as cv
from python import Python
import benchmark
from time import sleep

import src.moca as mc
from src.variables import SE3, PinholeCamera, Landmark


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
    # ------------------------- Load test data ------------------------- #
    let dir = Path("trex")
    var sfm = SfM(dir)

    try:
        sfm.frontend()
    except e:
        print("Failed to run SfM frontend")
        print(e)

    var pose_pair = (0, 0)
    var cam_pair = (0, 0)
    let first_factors = sfm.scene.get_first_pair(pose_pair)
    let pose1_id = pose_pair.get[0, Int]()
    let pose2_id = pose_pair.get[1, Int]()

    var kp1 = Tensor[DType.float64](0)
    var kp2 = Tensor[DType.float64](0)
    var lm_idx = Tensor[DType.int64](0)
    gather(first_factors, sfm.factors, pose_pair, cam_pair, kp1, kp2, lm_idx)

    var kp1py = mc.tensor2np2d(kp1)
    var kp2py = mc.tensor2np2d(kp2)

    var T1 = SE3.identity()

    # ------------------------- Start benchmarking ------------------------- #
    let opencv = Python.import_module("cv2")
    Python.add_to_path("src/sfm/")
    let cvpy = Python.import_module("cv_py")

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

    print("------------------------- Triangulate -------------------------")

    # fn e_mojo() raises:
    #     let f = cv.findEssentialMat(kp1, kp2)

    # fn e_opencv() raises:
    #     let f = opencv.findEssentialMat(kp1py, kp2py, opencv.FM_8POINT)[0]

    # fn e_cvpy() raises:
    #     let f = cvpy.findEssentialMat(kp1py, kp2py)

    # t = MyBenchmarker(e_mojo, e_opencv, e_cvpy)
    # t.run()
    # t.print()
    # t.plot("essential_mat.png", "Triangulate")
