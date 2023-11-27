from pathlib import Path, cwd
from python import Python
from tensor import TensorShape
from utils.index import Index

import src.moca as mc
import src.util as util
from src.variables.pinhole import PinholeCamera
from src.variables.se3 import SE3
from src.variables.so3 import SO3
from src.variables.landmark import Landmark


fn pyfloat[type: DType](i: PythonObject) -> SIMD[type, 1]:
    return i.to_float64().cast[type]()


fn pyint(i: PythonObject) -> Int:
    return i.__index__()


@value
@register_passable("trivial")
struct ProjectionFactor:
    var id_pose: Int32  # aka image #
    var id_cam: Int32
    var lm_id: Int32
    var measured: SIMD[DType.float64, 2]


struct SfM:
    var dir_images: Path
    # Storage for entire graph
    var poses: InlinedFixedVector[SE3]
    var landmarks: InlinedFixedVector[Landmark]
    var cameras: InlinedFixedVector[PinholeCamera]
    var factors: InlinedFixedVector[ProjectionFactor]
    # Each row corresponds to a match, with the entries being the factor index #.
    var pair_indices: Tensor[DType.int64]
    # Contains the indices of variables/factors that have been added to the graph
    var active_poses: InlinedFixedVector[UInt64]
    var active_landmarks: InlinedFixedVector[UInt64]
    var active_cameras: InlinedFixedVector[UInt64]
    var active_factors: InlinedFixedVector[UInt64]

    fn __init__(inout self, dir_images: Path):
        self.dir_images = dir_images
        self.poses = InlinedFixedVector[SE3](0)
        self.landmarks = InlinedFixedVector[Landmark](0)
        self.cameras = InlinedFixedVector[PinholeCamera](0)
        self.factors = InlinedFixedVector[ProjectionFactor](0)
        self.pair_indices = Tensor[DType.int64](0)
        self.active_poses = InlinedFixedVector[UInt64](0)
        self.active_landmarks = InlinedFixedVector[UInt64](0)
        self.active_cameras = InlinedFixedVector[UInt64](0)
        self.active_factors = InlinedFixedVector[UInt64](0)

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
        self.poses = InlinedFixedVector[SE3](num_poses)
        self.landmarks = InlinedFixedVector[Landmark](num_lm)
        self.cameras = InlinedFixedVector[PinholeCamera](num_cam)
        self.factors = InlinedFixedVector[ProjectionFactor](num_factors)

        self.active_poses = InlinedFixedVector[UInt64](num_poses)
        self.active_landmarks = InlinedFixedVector[UInt64](num_lm)
        self.active_cameras = InlinedFixedVector[UInt64](num_cam)
        self.active_factors = InlinedFixedVector[UInt64](num_factors)

        for _ in range(num_cam):
            self.poses.append(SE3.identity())

        for _ in range(num_lm):
            self.landmarks.append(Landmark.identity())

        for cam in sfm_data.cameras:
            let cal = SIMD[DType.float64, 4](
                pyfloat[DType.float64](cam.fx),
                pyfloat[DType.float64](cam.fy),
                pyfloat[DType.float64](cam.px),
                pyfloat[DType.float64](cam.py),
            )
            self.cameras.append(PinholeCamera(cal))

        for factor in sfm_data.factors:
            let measured = SIMD[DType.float64, 2](
                pyfloat[DType.float64](factor.u), pyfloat[DType.float64](factor.v)
            )
            let id_pose = pyint(factor.id_pose)
            let id_cam = pyint(factor.id_cam)
            let id_lm = pyint(factor.id_lm)
            self.factors.append(ProjectionFactor(id_pose, id_cam, id_lm, measured))

        self.pair_indices = util.np2tensor2d_i64(sfm_data.pair_indices)

    fn register_first_pair(inout self):
        pass

    fn register(inout self):
        pass

    fn optimize(inout self):
        pass
