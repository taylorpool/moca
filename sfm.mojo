from pathlib import Path, cwd
from python import Python
from tensor import TensorShape
from utils.index import Index

from pinhole import PinholeCamera
from se3 import SE3
from so3 import SO3

# TODO LIST
# - Figure out how to store Image / Matches in SfM
# - SE(3) class for optimization
# - Fix loading binary -> numpy -> Tensor


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
    # Indexed by pose, then list of all factors from that pose
    # Done this way so we can remove poses with high residual later
    var poses: InlinedFixedVector[SE3]
    var landmarks: InlinedFixedVector[SIMD[DType.float64, 4]]
    var cameras: InlinedFixedVector[PinholeCamera]
    var factors: InlinedFixedVector[ProjectionFactor]

    fn __init__(inout self, dir_images: Path):
        self.dir_images = dir_images
        self.poses = InlinedFixedVector[SE3](0)
        self.landmarks = InlinedFixedVector[SIMD[DType.float64, 4]](0)
        self.cameras = InlinedFixedVector[PinholeCamera](0)
        self.factors = InlinedFixedVector[ProjectionFactor](0)

    fn frontend(inout self, force: Bool = False) raises:
        """Run COLMAP as frontend to get all factors."""

        Python.add_to_path(".")
        let colmap = Python.import_module("colmap")
        let sfm_data = colmap.frontend(self.dir_images.path)

        let num_poses = sfm_data.num_poses.__index__()
        let num_lm = sfm_data.num_lm.__index__()
        let num_cam = sfm_data.cameras.__len__().__index__()
        let num_factors = sfm_data.factors.__len__().__index__()

        self.poses = InlinedFixedVector[SE3](num_poses)
        self.landmarks = InlinedFixedVector[SIMD[DType.float64, 4]](num_lm)
        self.cameras = InlinedFixedVector[PinholeCamera](num_cam)
        self.factors = InlinedFixedVector[ProjectionFactor](num_factors)

        # TODO: Fill out pose and landmark vectors

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

    fn register(inout self):
        pass

    fn optimize(inout self):
        pass


fn main() raises:
    let dir_in = "trex"
    var sfm = SfM(Path(dir_in))
    sfm.frontend()
