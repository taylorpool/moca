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
        let indices_pair = util.np2tensor2d_i64(sfm_data.pair_indices)

        let pairs = util.np2tensor2d_i64(sfm_data.pairs)
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
        var pose_pair = (0, 0)
        var cam_pair = (0, 0)
        let first_factors = self.scene.get_first_pair(pose_pair)
        let pose1_id = pose_pair.get[0, Int]()
        let pose2_id = pose_pair.get[1, Int]()

        var kp1 = Tensor[DType.float64](0)
        var kp2 = Tensor[DType.float64](0)
        var lm_idx = Tensor[DType.int64](0)
        gather(first_factors, self.factors, pose_pair, cam_pair, kp1, kp2, lm_idx)
        let cam1_id = cam_pair.get[0, Int]()
        let cam2_id = cam_pair.get[1, Int]()

        # Estimate everything
        let E = cv.findEssentialMat(
            kp1, kp2, self.cameras[cam1_id], self.cameras[cam2_id]
        )
        let pose1 = SE3.identity()
        let pose2 = cv.recoverPose(E, kp1, kp2)
        let lms = cv.triangulate(
            self.cameras[cam1_id], pose1, kp1, self.cameras[cam2_id], pose2, kp2
        )

        # Insert everything into graph
        self.active_poses.add(pose1_id)
        self.active_poses.add(pose2_id)
        self.active_cameras.add(cam1_id)
        self.active_cameras.add(cam2_id)
        for i in range(lm_idx.dim(0)):
            let idx = lm_idx[Index(i)].__int__()
            self.active_landmarks.add(idx)
            self.landmarks[idx] = lms[i]

        for i in range(first_factors.dim(0)):
            self.active_factors.add(first_factors[Index(i)].__int__())

        self.poses[pose1_id] = pose1
        self.poses[pose2_id] = pose2

    fn register(inout self):
        pass

    fn optimize(inout self):
        # TODO: Make sure not optimizing over first pose - it essentially needs a prior on it
        pass
