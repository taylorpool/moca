from pathlib import Path, cwd
from python import Python
from tensor import TensorShape
from memory import memset_zero
from utils.index import Index

from src.set import IntSet
from src.dict import TupleIntDict
import src.moca as mc
import src.util as util
from src.variables import SE3, SO3, PinholeCamera, Landmark
from src.sfm.factors import ProjectionFactor
from src.sfm.scene import SceneGraph
import src.sfm.cv as cv


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

    # var tempSet = IntSet(factors.__len__())

    # print("Gathering factors for pair:", id_img1, id_img2)

    for i in range(num_factors):
        let id = idx[Index(i)].__int__()
        if id == -1:
            break
        let factor = factors[id]
        if factor.id_pose == id_img1:
            kp1[Index(kp1_count, 0)] = factor.measured[0]
            kp1[Index(kp1_count, 1)] = factor.measured[1]
            lm_idx[kp1_count] = factor.id_lm
            # if tempSet.contains(factor.id_lm.__int__()):
            #     print(
            #         "ERROR: Landmark already used!",
            #         factor.id_pose.__int__(),
            #         factor.id_lm.__int__(),
            #     )
            # else:
            #     tempSet.add(factor.id_lm.__int__())
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


alias camera_dim = 4
alias pose_dim = 6
alias landmark_dim = 3
alias factor_dim = 2


fn copy_vector[type: AnyType](src: DynamicVector[type]) -> DynamicVector[type]:
    var dst = DynamicVector[type](src.__len__())
    for i in range(src.__len__()):
        dst.push_back(src[i])
    return dst


@value
struct PartiallyActiveVector[type: AnyType]:
    var dim: Int
    var values: DynamicVector[type]  # indexed by id
    var active_to_id: IntSet  # maps active_idx to id
    var id_to_active: DynamicVector[Int]  # maps id to active_idx

    fn __init__(inout self, dim: Int):
        self.dim = dim
        self.values = DynamicVector[type](0)
        self.active_to_id = IntSet(0)
        self.id_to_active = DynamicVector[Int](0)

    fn __copyinit__(inout self, other: Self):
        self.dim = other.dim
        self.values = copy_vector(other.values)
        self.active_to_id.__copyinit__(other.active_to_id)
        self.id_to_active = copy_vector(other.id_to_active)

    fn resize(inout self, size: Int):
        self.values = DynamicVector[type](size)
        self.active_to_id = IntSet(size)
        self.id_to_active = DynamicVector[Int](size)
        for i in range(size):
            self.id_to_active.push_back(-1)

    fn offset(self) -> Int:
        return self.dim * self.active_to_id.size()

    fn add(inout self, id: Int):
        if not self.active_to_id.contains(id):
            self.active_to_id.add(id)
            self.id_to_active[id] = self.active_to_id.size() - 1

    fn size(self) -> Int:
        return self.active_to_id.size()

    fn total(self) -> Int:
        return self.values.__len__()

    fn __getitem__(self, i: Int) -> type:
        return self.values[i]

    fn __setitem__(inout self, i: Int, value: type):
        self.values[i] = value

    fn contains(self, id: Int) -> Bool:
        return self.active_to_id.contains(id)


@value
struct SfMState:
    var cameras: PartiallyActiveVector[PinholeCamera]
    var poses: PartiallyActiveVector[SE3]
    var landmarks: PartiallyActiveVector[Landmark]

    fn __init__(inout self):
        self.cameras = PartiallyActiveVector[PinholeCamera](camera_dim)
        self.poses = PartiallyActiveVector[SE3](pose_dim)
        self.landmarks = PartiallyActiveVector[Landmark](landmark_dim)

    fn __copyinit__(inout self, other: Self):
        self.cameras.__copyinit__(other.cameras)
        self.poses.__copyinit__(other.poses)
        self.landmarks.__copyinit__(other.landmarks)


alias SfMFactors = PartiallyActiveVector[ProjectionFactor]


fn compute_residual_vec(state: SfMState, factors: SfMFactors) -> Tensor[DType.float64]:
    var residual = Tensor[DType.float64](factors.offset())
    for i in range(factors.active_to_id.size()):
        let factor = factors.values[factors.active_to_id.elements[i]]
        residual.simd_store(
            factors.dim * i,
            factor.residual(
                state.cameras.values[factor.id_cam.__int__()],
                state.poses.values[factor.id_pose.__int__()],
                state.landmarks.values[factor.id_lm.__int__()],
            ),
        )

    return residual


fn compute_residual_jac(state: SfMState, factors: SfMFactors) -> Tensor[DType.float64]:
    let camera_offset = state.cameras.offset()
    let pose_offset = camera_offset + (state.poses.offset() - 6)
    let landmark_offset = pose_offset + state.landmarks.offset()

    var Dr = Tensor[DType.float64](factor_dim * factors.size(), landmark_offset)
    memset_zero(Dr.data(), Dr.num_elements())

    for i in range(factors.size()):
        let factor = factors.values[factors.active_to_id.elements[i]]

        var H_K = Tensor[DType.float64](factor_dim, camera_dim)
        var H_T = Tensor[DType.float64](factor_dim, pose_dim)
        var H_p = Tensor[DType.float64](factor_dim, landmark_dim)
        factor.jacobian(
            state.cameras.values[factor.id_cam.__int__()],
            state.poses.values[factor.id_pose.__int__()],
            state.landmarks.values[factor.id_lm.__int__()],
            H_K,
            H_T,
            H_p,
        )

        let row = factor_dim * i
        mc.copy(
            H_K,
            Dr,
            row,
            camera_dim * state.cameras.id_to_active[factor.id_cam.__int__()],
        )

        if factor.id_pose != state.poses.active_to_id.elements[0]:
            mc.copy(
                H_T,
                Dr,
                row,
                pose_dim * (state.poses.id_to_active[factor.id_pose.__int__()] - 1)
                + camera_offset,
            )

        mc.copy(
            H_p,
            Dr,
            row,
            landmark_dim * state.landmarks.id_to_active[factor.id_lm.__int__()]
            + pose_offset,
        )

    return Dr


fn perturb(borrowed state: SfMState, perturbation: Tensor[DType.float64]) -> SfMState:
    var perturbed_state: SfMState
    perturbed_state.__copyinit__(state)
    let camera_offset = state.cameras.offset()
    let pose_offset = (state.poses.offset() - 6) + camera_offset

    for i in range(state.cameras.size()):
        let id = state.cameras.active_to_id.elements[i]
        let perturbation_index = camera_dim * i
        perturbed_state.cameras.values[id].cal[0] += perturbation[perturbation_index]
        perturbed_state.cameras.values[id].cal[1] += perturbation[
            perturbation_index + 1
        ]
        perturbed_state.cameras.values[id].cal[2] += perturbation[
            perturbation_index + 2
        ]
        perturbed_state.cameras.values[id].cal[3] += perturbation[
            perturbation_index + 3
        ]

    for i in range(1, state.poses.size()):
        let id = state.poses.active_to_id.elements[i]
        let twist_index = pose_dim * (i - 1) + camera_offset
        let twist = mc.Vector6d(
            perturbation[twist_index],
            perturbation[twist_index + 1],
            perturbation[twist_index + 2],
            perturbation[twist_index + 3],
            perturbation[twist_index + 4],
            perturbation[twist_index + 5],
        )
        perturbed_state.poses.values[id] = state.poses.values[id] + twist

    for i in range(state.landmarks.size()):
        let id = state.landmarks.active_to_id.elements[i]
        let index = landmark_dim * i + pose_offset
        perturbed_state.landmarks.values[id].val[0] += perturbation[index]
        perturbed_state.landmarks.values[id].val[1] += perturbation[index + 1]
        perturbed_state.landmarks.values[id].val[2] += perturbation[index + 2]

    return perturbed_state


struct SfM:
    var dir_images: Path
    var scene: SceneGraph

    var state: SfMState
    var factors: SfMFactors
    var lambd: Float64

    fn __init__(inout self, dir_images: Path):
        self.dir_images = dir_images
        self.scene = SceneGraph()
        self.state = SfMState()
        self.factors = SfMFactors(factor_dim)
        self.lambd = 1e-4

    fn frontend(inout self, force: Bool = False) raises:
        """Run COLMAP as frontend to get all factors."""
        print("Running frontend")

        Python.add_to_path("src/sfm/")
        let colmap = Python.import_module("colmap")
        let sfm_data = colmap.frontend(self.dir_images.path)

        let num_poses = sfm_data.num_poses.__index__()
        let num_lm = sfm_data.num_lm.__index__()
        let num_cam = sfm_data.cameras.__len__().__index__()
        let num_factors = sfm_data.factors.__len__().__index__()

        # Fill everything in
        self.state.cameras.resize(num_cam)
        self.state.poses.resize(num_poses)
        self.state.landmarks.resize(num_lm)
        self.factors.resize(num_factors)

        for cam in sfm_data.cameras:
            let cal = SIMD[DType.float64, 4](
                mc.pyfloat[DType.float64](cam.fx),
                mc.pyfloat[DType.float64](cam.fy),
                mc.pyfloat[DType.float64](cam.px),
                mc.pyfloat[DType.float64](cam.py),
            )
            self.state.cameras.values.push_back(PinholeCamera(cal))

        for _ in range(num_poses):
            self.state.poses.values.push_back(SE3.identity())

        for _ in range(num_lm):
            self.state.landmarks.values.push_back(Landmark.identity())

        for factor in sfm_data.factors:
            let measured = SIMD[DType.float64, 2](
                mc.pyfloat[DType.float64](factor.u), mc.pyfloat[DType.float64](factor.v)
            )
            let id_pose = mc.pyint(factor.id_pose)
            let id_cam = mc.pyint(factor.id_cam)
            let id_lm = mc.pyint(factor.id_lm)
            let f = ProjectionFactor(id_pose, id_cam, id_lm, measured)
            self.factors.values.push_back(f)

        # Setup SceneGraph tracker
        var id_pairs = TupleIntDict()
        let indices_pair = mc.np2tensor2d_i64(sfm_data.pair_indices)

        let pairs = mc.np2tensor2d_i64(sfm_data.pairs)
        for i in range(pairs.dim(0)):
            let id1 = pairs[Index(i, 0)].__int__()
            let id2 = pairs[Index(i, 1)].__int__()
            id_pairs.add(Tuple(id1, id2), i)

        # make lookup for pairs & count number of factors for each pose
        var lms_per_pose = Tensor[DType.uint64](num_poses)

        for i in range(self.factors.values.__len__()):
            let f = self.factors.values[i]
            lms_per_pose[f.id_pose.__int__()] += 1

        self.scene.setup(id_pairs, indices_pair, lms_per_pose)
        print()

    fn pairs_left(self) -> Int:
        return self.scene.id_pairs.size()

    fn register_first_pair(inout self):
        # Get first image data
        let next_pair = self.scene.get_first_pair()
        print("Registering pair:", next_pair.id1, next_pair.id2)

        var cam_pair = (0, 0)
        var kp1 = Tensor[DType.float64](0)
        var kp2 = Tensor[DType.float64](0)
        var lm_idx = Tensor[DType.int64](0)
        gather(
            next_pair.factor_idx,
            self.factors.values,
            (next_pair.id1, next_pair.id2),
            cam_pair,
            kp1,
            kp2,
            lm_idx,
        )
        let cam1_id = cam_pair.get[0, Int]()
        let cam2_id = cam_pair.get[1, Int]()

        # Estimate everything
        let result = cv.recoverPoseAndECV(
            kp1,
            kp2,
            self.state.cameras.values[cam1_id],
            self.state.cameras.values[cam2_id],
        )
        let pose1 = SE3.identity()
        let pose2 = result.pose
        let lms = cv.triangulate(
            self.state.cameras.values[cam1_id],
            pose1,
            kp1,
            self.state.cameras.values[cam2_id],
            pose2,
            kp2,
        )

        print("---E RANSAC:", result.num_inliers, "/", kp1.dim(0))

        # Insert all inliers into graph
        self.state.poses.add(next_pair.id1)
        self.state.poses.add(next_pair.id2)
        self.state.poses.values[next_pair.id1] = pose1
        self.state.poses.values[next_pair.id2] = pose2

        self.state.cameras.add(cam1_id)
        self.state.cameras.add(cam2_id)

        for i in range(result.inliers.dim(0)):
            if result.inliers[Index(i)]:
                self.factors.add(next_pair.factor_idx[Index(2 * i)].__int__())
                self.factors.add(next_pair.factor_idx[Index(2 * i + 1)].__int__())
                let lm_idx = lm_idx[Index(i)].__int__()
                self.state.landmarks.add(lm_idx)
                self.state.landmarks.values[lm_idx] = lms[i]
        print()

    fn register(inout self):
        let next_pair = self.scene.get_next_pair(self.state.poses.active_to_id)
        let num_factors_before = self.factors.size()
        print("Registering pair:", next_pair.id1, next_pair.id2)

        # Find which factors have / haven't been added
        var new_lm_factors = IntSet(self.factors.total())  # for PnP
        var new_pose_factors = IntSet(self.factors.total())  # for Triangulation

        for i in range(next_pair.factor_idx.dim(0)):
            let factor_id = next_pair.factor_idx[Index(i)].__int__()
            let cam_id = self.factors[factor_id].id_cam.__int__()
            let lm_id = self.factors[factor_id].id_lm.__int__()
            let pose_id = self.factors[factor_id].id_pose.__int__()
            # Grab any factors with landmarks that haven't been added yet (-> triangulate)
            if not self.state.landmarks.contains(lm_id):
                new_lm_factors.add(factor_id)
            # Grab any factors with poses that haven't been added yet, but landmarks have (-> PnP)
            elif not self.state.poses.contains(pose_id):
                new_pose_factors.add(factor_id)
            # When a pose & landmark have been added, but not together
            # TODO: This should be on, but optimization get more difficult with it on
            # else:
            #     self.factors.add(factor_id)

        let has_both = self.state.poses.contains(
            next_pair.id1
        ) and self.state.poses.contains(next_pair.id2)

        if not has_both and new_pose_factors.size() < 10:
            print("Not enough new factors to register new image!")
            print()
            return

        # These functions add their respective factors in as well
        # Add in any new poses
        if not self.state.poses.contains(next_pair.id1):
            self._estimate_init_pose(next_pair.id1, new_pose_factors)
        if not self.state.poses.contains(next_pair.id2):
            self._estimate_init_pose(next_pair.id2, new_pose_factors)
        # Add in any new landmarks
        if new_lm_factors.size() > 0:
            self._estimate_init_landmarks(new_lm_factors)
        else:
            print("---No new landmarks to add!")

        print("---Added factors:", self.factors.size() - num_factors_before)
        print()

    fn _estimate_init_pose(inout self, pose_id: Int, factor_idx: IntSet):
        """Initialize a pose based on a set of factors with initialized landmarks"""
        # Get data for new factors
        var pts3d = Tensor[DType.float64](factor_idx.size(), 3)
        var pts2d = Tensor[DType.float64](factor_idx.size(), 2)
        var cam_id = 0
        for i in range(factor_idx.size()):
            let id = factor_idx.elements[i]
            let factor = self.factors[id]
            let lm = self.state.landmarks[factor.id_lm.__int__()]
            pts3d[Index(i, 0)] = lm.val[0]
            pts3d[Index(i, 1)] = lm.val[1]
            pts3d[Index(i, 2)] = lm.val[2]
            pts2d[Index(i, 0)] = factor.measured[0]
            pts2d[Index(i, 1)] = factor.measured[1]
            if factor.id_pose == pose_id:
                cam_id = factor.id_cam.__int__()
        let result = cv.PnPCV(self.state.cameras[cam_id], pts2d, pts3d)
        self.state.poses[pose_id] = result.pose
        self.state.poses.add(pose_id)

        self.state.cameras.add(cam_id)

        # Handle outliers
        for i in range(factor_idx.size()):
            let id = factor_idx.elements[i]
            if result.inliers[Index(i)]:
                self.factors.add(id)

        print("---PnP RANSAC:", result.num_inliers, "/", factor_idx.size())

    fn _estimate_init_landmarks(inout self, factor_idx: IntSet):
        """Initialize landmarks based on a set of factors with initialized poses"""
        # Get data for new factors
        let num_new: Int = (factor_idx.size() / 2).__int__()
        var pts1 = Tensor[DType.float64](num_new, 2)
        var pts2 = Tensor[DType.float64](num_new, 2)
        var lm_id = Tensor[DType.int64](num_new)

        let factor1 = self.factors[factor_idx.elements[0]]
        let K1 = self.state.cameras[factor1.id_cam.__int__()]
        let T1 = self.state.poses[factor1.id_pose.__int__()]

        let factor2 = self.factors[factor_idx.elements[1]]
        let K2 = self.state.cameras[factor2.id_cam.__int__()]
        let T2 = self.state.poses[factor2.id_pose.__int__()]

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

        var num_inliers = 0
        for i in range(num_new):
            # outlier check
            let this_id = lm_id[i].__int__()
            let lm = lm_new[i]
            if (
                (T1.inv() * lm.val)[2] > 0
                and (T2.inv() * lm.val)[2] > 0
                # and (lm.val * lm.val).reduce_add() < 100
            ):
                self.state.landmarks[this_id] = lm_new[i]
                self.state.landmarks.add(this_id)

                let id1 = factor_idx.elements[2 * i]
                let id2 = factor_idx.elements[2 * i + 1]
                self.factors.add(id1)
                self.factors.add(id2)

                num_inliers += 1

        print("---Triangulate Good:", num_inliers, "/", num_new)

    fn optimize(
        inout self,
        max_iters: Int = 20,
        abs_tol: Float64 = 1e-2,
        rel_tol: Float64 = 1e-5,
        grad_tol: Float64 = 1e-1,
    ):
        print(
            "Optimize\n -> cameras:",
            self.state.cameras.active_to_id.__str__(),
            "\n -> poses:",
            self.state.poses.size(),
            "\n -> landmarks:",
            self.state.landmarks.size(),
            "\n -> factors:",
            self.factors.size(),
        )
        var abs_error = abs_tol + 1
        var rel_diff = rel_tol + 1
        if self.lambd >= 1e6:
            self.lambd = 1

        for iter in range(max_iters):
            try:
                let np = Python.import_module("numpy")
                let scipy = Python.import_module("scipy")
                let r = compute_residual_vec(self.state, self.factors)
                let Dr = compute_residual_jac(self.state, self.factors)

                let Drpy = mc.tensor2np(Dr)
                let rpy = mc.tensor2np(r)
                let r_norm = np.linalg.norm(rpy)
                print("---R INIT", r_norm)

                let DrT_rpy = -np.matmul(Drpy.T, rpy)
                let grad_norm = np.linalg.norm(DrT_rpy) / rpy.shape[0]
                if grad_norm < grad_tol:
                    print("---Finishing due to grad norm", grad_norm)
                    break

                let DrT_Drpy = np.matmul(Drpy.T, Drpy)

                var step = Tensor[DType.float64](0)
                while self.lambd < 10**6:
                    try:
                        let diag = np.eye(Drpy.shape[1]) * self.lambd
                        let DrT_Drpy_damped = DrT_Drpy + diag

                        let steppy = scipy.linalg.solve(
                            DrT_Drpy_damped,
                            DrT_rpy,
                            False,
                            False,
                            False,
                            False,
                            "pos",
                        )
                        # let steppy = np.linalg.solve(DrT_Drpy, DrT_rpy, )
                        step = mc.np2tensor1d_f64(steppy)
                    except e:
                        print("---Failed to solve!")
                        print(e)
                        self.lambd *= 10.0
                        continue

                    let next_state = perturb(self.state, step)
                    let next_r = compute_residual_vec(next_state, self.factors)
                    let next_r_norm = np.linalg.norm(mc.tensor2np(next_r))

                    # print("next_r_norm:", next_r_norm)
                    if next_r_norm < r_norm:
                        print("---STEPPED", next_r_norm, ", lambda: ", self.lambd)
                        self.state = next_state
                        rel_diff = mc.squared_norm(step)
                        abs_error = mc.pyfloat[DType.float64](r_norm - next_r_norm)
                        self.lambd /= 10.0
                        break
                    else:
                        # print("---try again", self.lambd, r_norm - next_r_norm)
                        self.lambd *= 10.0

                if self.lambd >= 10**6:
                    print("---Failed to find a good step size!")
                    break

            except e:
                print("---Failed to import numpy")
                print(e)

            if rel_diff < rel_tol:
                print("---Finishing due to small step size", rel_diff)
                break

            if abs_error < abs_tol:
                print("---Finishing due to small error", abs_error)
                break

        print()
