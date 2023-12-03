from pathlib import Path, cwd
from python import Python
from tensor import TensorShape
from utils.index import Index

from src.set import IntSet
from src.dict import TupleIntDict
import src.moca as mc
import src.util as util
from src.variables.pinhole import PinholeCamera
from src.variables.se3 import SE3
from src.variables.so3 import SO3
from src.variables.landmark import Landmark
from src.sfm.factors import ProjectionFactor
from src.sfm.scene import SceneGraph
import src.sfm.cv as cv


fn pyfloat[type: DType](i: PythonObject) -> SIMD[type, 1]:
    return i.to_float64().cast[type]()


fn pyint(i: PythonObject) -> Int:
    return i.__index__()


fn gather(
    idx: Tensor[DType.int64],
    factors: DynamicVector[ProjectionFactor],
    pair: Tuple[Int, Int],
    inout cam_pair: Tuple[Int, Int],
    inout kp1: Tensor[DType.float64],
    inout kp2: Tensor[DType.float64],
    inout lm_idx: Tensor[DType.int64],
):
    let num_factors = idx.dim(0)
    let num_factors_each: Int = (num_factors / 2).__int__()
    kp1 = Tensor[DType.float64](num_factors_each, 2)
    kp2 = Tensor[DType.float64](num_factors_each, 2)
    lm_idx = Tensor[DType.int64](num_factors_each)
    var kp1_count = 0
    var kp2_count = 0
    var cam1_id = 0
    var cam2_id = 0
    let id_img1 = pair.get[0, Int]()
    let id_img2 = pair.get[1, Int]()

    for i in range(num_factors):
        let id = idx[Index(i)].__int__()
        if id == -1:
            break
        let factor = factors[id]
        if factor.id_pose == id_img1:
            kp1[Index(kp1_count, 0)] = factor.measured[0]
            kp1[Index(kp1_count, 1)] = factor.measured[1]
            lm_idx[kp1_count] = factor.id_lm
            cam1_id = factor.id_cam.__int__()
            kp1_count += 1
        elif factor.id_pose == id_img2:
            kp2[Index(kp2_count, 0)] = factor.measured[0]
            kp2[Index(kp2_count, 1)] = factor.measured[1]
            cam2_id = factor.id_cam.__int__()
            kp2_count += 1
        else:
            print("ERROR: Factor does not belong to either image!", factor.id_pose)

    cam_pair = (cam1_id, cam2_id)


struct SfM:
    var dir_images: Path
    var scene: SceneGraph

    # Storage for entire graph
    var poses: DynamicVector[SE3]
    var landmarks: DynamicVector[Landmark]
    var cameras: DynamicVector[PinholeCamera]
    var factors: DynamicVector[ProjectionFactor]

    # Contains the indices of variables/factors that have been added to the graph
    var active_poses: IntSet
    var active_landmarks: IntSet
    var active_cameras: IntSet
    var active_factors: IntSet

    fn __init__(inout self, dir_images: Path):
        self.dir_images = dir_images
        self.scene = SceneGraph()

        self.poses = DynamicVector[SE3](0)
        self.landmarks = DynamicVector[Landmark](0)
        self.cameras = DynamicVector[PinholeCamera](0)
        self.factors = DynamicVector[ProjectionFactor](0)

        self.active_poses = IntSet(0)
        self.active_landmarks = IntSet(0)
        self.active_cameras = IntSet(0)
        self.active_factors = IntSet(0)

    fn frontend(inout self, force: Bool = False) raises:
        """Run COLMAP as frontend to get all factors."""

        Python.add_to_path("src/sfm/")
        let colmap = Python.import_module("colmap")
        let sfm_data = colmap.frontend(self.dir_images.path)

        let num_poses = sfm_data.num_poses.__index__()
        let num_lm = sfm_data.num_lm.__index__()
        let num_cam = sfm_data.cameras.__len__().__index__()
        let num_factors = sfm_data.factors.__len__().__index__()

        # Fill everything in
        self.poses = DynamicVector[SE3](num_poses)
        self.landmarks = DynamicVector[Landmark](num_lm)
        self.cameras = DynamicVector[PinholeCamera](num_cam)
        self.factors = DynamicVector[ProjectionFactor](num_factors)

        self.active_poses = IntSet(num_poses)
        self.active_landmarks = IntSet(num_lm)
        self.active_cameras = IntSet(num_cam)
        self.active_factors = IntSet(num_factors)

        for _ in range(num_cam):
            self.poses.push_back(SE3.identity())

        for _ in range(num_lm):
            self.landmarks.push_back(Landmark.identity())

        for cam in sfm_data.cameras:
            let cal = SIMD[DType.float64, 4](
                pyfloat[DType.float64](cam.fx),
                pyfloat[DType.float64](cam.fy),
                pyfloat[DType.float64](cam.px),
                pyfloat[DType.float64](cam.py),
            )
            self.cameras.push_back(PinholeCamera(cal))

        for factor in sfm_data.factors:
            let measured = SIMD[DType.float64, 2](
                pyfloat[DType.float64](factor.u), pyfloat[DType.float64](factor.v)
            )
            let id_pose = pyint(factor.id_pose)
            let id_cam = pyint(factor.id_cam)
            let id_lm = pyint(factor.id_lm)
            let f = ProjectionFactor(id_pose, id_cam, id_lm, measured)
            self.factors.push_back(f)

        # Setup SceneGraph tracker
        var id_pairs = TupleIntDict()
        let indices_pair = mc.np2tensor2d_i64(sfm_data.pair_indices)

        let pairs = mc.np2tensor2d_i64(sfm_data.pairs)
        for i in range(pairs.dim(0)):
            let id1 = pairs[Index(i, 0)].__int__()
            let id2 = pairs[Index(i, 1)].__int__()
            id_pairs.add(Tuple(id1, id2), i)

        # make lookup for pairs & count number of factors for each pose
        var lms_per_pose = Tensor[DType.uint64](num_cam)

        for i in range(self.factors.__len__()):
            let f = self.factors[i]
            lms_per_pose[f.id_pose.__int__()] += 1

        self.scene.setup(id_pairs, indices_pair, lms_per_pose)

    fn register_first_pair(inout self):
        # Get first image data
        let next_pair = self.scene.get_first_pair()

        var cam_pair = (0, 0)
        var kp1 = Tensor[DType.float64](0)
        var kp2 = Tensor[DType.float64](0)
        var lm_idx = Tensor[DType.int64](0)
        gather(
            next_pair.factor_idx,
            self.factors,
            (next_pair.id1, next_pair.id2),
            cam_pair,
            kp1,
            kp2,
            lm_idx,
        )
        let cam1_id = cam_pair.get[0, Int]()
        let cam2_id = cam_pair.get[1, Int]()

        # Estimate everything
        let E = cv.findEssentialMat(
            kp1, kp2, self.cameras[cam1_id], self.cameras[cam2_id]
        )
        let pose1 = SE3.identity()
        let pose2 = cv.recoverPose(
            E, kp1, kp2, self.cameras[cam1_id], self.cameras[cam2_id]
        )
        let lms = cv.triangulate(
            self.cameras[cam1_id], pose1, kp1, self.cameras[cam2_id], pose2, kp2
        )

        # Insert everything into graph
        self.active_poses.add(next_pair.id1)
        self.active_poses.add(next_pair.id2)
        self.active_cameras.add(cam1_id)
        self.active_cameras.add(cam2_id)
        for i in range(lm_idx.dim(0)):
            let idx = lm_idx[Index(i)].__int__()
            self.active_landmarks.add(idx)
            self.landmarks[idx] = lms[i]

        for i in range(next_pair.factor_idx.dim(0)):
            self.active_factors.add(next_pair.factor_idx[Index(i)].__int__())

        self.poses[next_pair.id1] = pose1
        self.poses[next_pair.id2] = pose2

    fn register(inout self):
        # TODO: Probably want some kind of cheriality check here after triangulation
        #  - maybe remove those factors in the _init_estimate_lm function?
        #  - might want to do the same for outliers in the _init_estimate_pose function
        let next_pair = self.scene.get_next_pair(self.active_poses)

        # Find which factors have / haven't been added
        var new_lm_factors = IntSet(self.factors.__len__())  # for PnP
        var old_factors = IntSet(self.factors.__len__())  # for Triangulation

        for i in range(next_pair.factor_idx.dim(0)):
            let id = next_pair.factor_idx[Index(i)].__int__()
            let cam_id = self.factors[id].id_cam.__int__()
            let lm_id = self.factors[id].id_lm.__int__()
            if self.active_factors.contains(id):
                old_factors.add(id)
            if not self.active_landmarks.contains(lm_id):
                new_lm_factors.add(id)

            self.active_factors.add(id)
            self.active_cameras.add(cam_id)

        if old_factors.size() < 30:
            print("Not enough new factors to register!")

        # Add in any new poses
        if not self.active_poses.contains(next_pair.id1):
            self._estimate_init_pose(next_pair.id1, old_factors)
        if not self.active_poses.contains(next_pair.id2):
            self._estimate_init_pose(next_pair.id2, old_factors)
        # Add in any new landmarks
        if new_lm_factors.size() > 0:
            self._estimate_init_landmarks(new_lm_factors)
        else:
            print("No new landmarks to add!")

    fn _estimate_init_pose(inout self, pose_id: Int, factor_idx: IntSet):
        """Initialize a pose based on a set of factors with initialized landmarks"""
        # Get data for new factors
        var pts3d = Tensor[DType.float64](factor_idx.size(), 3)
        var pts2d = Tensor[DType.float64](factor_idx.size(), 2)
        var cam_id = 0
        for i in range(factor_idx.size()):
            let id = factor_idx.elements[i]
            let factor = self.factors[id]
            let lm = self.landmarks[factor.id_lm.__int__()]
            pts3d[Index(i, 0)] = lm.val[0]
            pts3d[Index(i, 1)] = lm.val[1]
            pts3d[Index(i, 2)] = lm.val[2]
            pts2d[Index(i, 0)] = factor.measured[0]
            pts2d[Index(i, 1)] = factor.measured[1]
            if factor.id_pose == pose_id:
                cam_id = factor.id_cam.__int__()
        let pose_new = cv.PnP(self.cameras[cam_id], pts2d, pts3d)
        self.poses[pose_id] = pose_new
        self.active_poses.add(pose_id)

    fn _estimate_init_landmarks(inout self, factor_idx: IntSet):
        """Initialize landmarks based on a set of factors with initialized poses"""
        # TODO: This assumes factors are ordered every other... hopefully ture
        # Get data for new factors
        let num_new: Int = (factor_idx.size() / 2).__int__()
        var pts1 = Tensor[DType.float64](num_new, 2)
        var pts2 = Tensor[DType.float64](num_new, 2)
        var lm_id = Tensor[DType.int64](num_new)

        let factor1 = self.factors[factor_idx.elements[0]]
        let K1 = self.cameras[factor1.id_cam.__int__()]
        let T1 = self.poses[factor1.id_pose.__int__()]

        let factor2 = self.factors[factor_idx.elements[1]]
        let K2 = self.cameras[factor2.id_cam.__int__()]
        let T2 = self.poses[factor2.id_pose.__int__()]

        var pose1_count = 0
        var pose2_count = 0
        for i in range(factor_idx.size()):
            let id = factor_idx.elements[i]
            let factor = self.factors[id]

            if factor.id_pose == factor1.id_pose:
                pts1[Index(pose1_count, 0)] = factor.measured[0]
                pts1[Index(pose1_count, 1)] = factor.measured[1]
                lm_id[pose1_count] = factor.id_lm
                pose1_count += 1
            elif factor.id_pose == factor2.id_pose:
                pts2[Index(pose2_count, 0)] = factor.measured[0]
                pts2[Index(pose2_count, 1)] = factor.measured[1]
                pose2_count += 1
            else:
                print("ERROR: Factor does not belong to either image!", factor.id_pose)

        let lm_new = cv.triangulate(K1, T1, pts1, K2, T2, pts2)

        for i in range(num_new):
            let this_id = lm_id[i].__int__()
            self.landmarks[this_id] = lm_new[i]
            self.active_landmarks.add(this_id)

    fn optimize(inout self):
        # TODO: Make sure not optimizing over first pose - it essentially needs a prior on it
        let num_cameras = self.active_cameras.size()
        let num_landmarks = self.active_landmarks.size()
        let intrinsic_dim = 9
        let extrinsic_dim = 7
        let landmark_dim = 3

        var x = Tensor[DType.float64](
            num_cameras * (intrinsic_dim + extrinsic_dim) + num_landmarks * landmark_dim
        )

        let max_iters = 100
        for iter in range(max_iters):
            # Factors to optimize with
            var A = Tensor[DType.float64]()
            var b = Tensor[DType.float64]()
            for i in range(self.active_factors.elements.size):
                let factor = self.factors[i]
                let camera_intrinsic = self.cameras[factor.id_cam.to_int()]
                let camera_extrinsic = self.poses[factor.id_pose.to_int()]
                let landmark = self.landmarks[factor.id_lm.to_int()]
                var H_camera_intrinsic: Tensor[DType.float64]
                var H_camera_extrinsic: Tensor[DType.float64]
                var H_landmark: Tensor[DType.float64]
                let residual = self.factors[i].residual(
                    camera_intrinsic, camera_extrinsic, landmark
                )
                self.factors[i].jacobian(
                    camera_intrinsic,
                    camera_extrinsic,
                    landmark,
                    H_camera_intrinsic,
                    H_camera_extrinsic,
                    H_landmark,
                )

                var Dr_i = Tensor[DType.float64](
                    H_camera_intrinsic.shape()[0],
                    H_camera_intrinsic.shape()[1]
                    + H_camera_extrinsic.shape()[1]
                    + H_landmark.shape()[1],
                )
                b = mc.subtract(b, mc.matT_vec[DType.float64, 2](Dr_i, residual))
                A = mc.add(A, mc.matT_mat(Dr_i, Dr_i))

            var lambd: Float64 = 1e-6
            for lambd_index in range(10):
                let qr = moca.qr_factor(A)
                let step = qr.solve(b)

                x = moca.add(x, step)
                break
                lambd *= 10.0
