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

alias camera_dim = 4
alias pose_dim = 6
alias landmark_dim = 3
alias factor_dim = 2

@value
struct PartiallyActiveVector[type: AnyType]:
    var dim: Int
    var values: DynamicVector[type]
    var active_to_id: IntSet
    var id_to_active: IntSet

    fn __init__(inout self, dim: Int):
        self.dim = dim
        self.values = DynamicVector[type](0)
        self.active_to_id = IntSet(0)
        self.id_to_active = IntSet(0)

    fn offset(self) -> Int:
        return self.dim*self.active_to_id.size()

@value
struct SfMState:
    var cameras: PartiallyActiveVector[PinholeCamera]
    var poses: PartiallyActiveVector[SE3]
    var landmarks: PartiallyActiveVector[Landmark]

    fn __init__(inout self):
        self.cameras = PartiallyActiveVector[PinholeCamera](4)
        self.poses = PartiallyActiveVector[SE3](6)
        self.landmarks = PartiallyActiveVector[Landmark](3)

alias SfMFactors = PartiallyActiveVector[ProjectionFactor]

fn compute_residual_vec(state: SfMState, factors: SfMFactors) -> Tensor[DType.float64]:
    var residual = Tensor[DType.float64](factors.offset()+6)
    for i in range(factors.active_to_id.size()):
        let factor = factors.values[factors.active_to_id[i]]
        residual.simd_store(factors.dim*i, factor.residual(
            state.cameras.values[factor.id_cam],
            state.poses.values[factor.id_pose],
            state.landmarks.values[factor.id_lm]
        ))

    return residual

fn compute_residual_jac(state: SfMState, factors: SfMFactors) -> Tensor[DType.float64]:
    let camera_offset = state.cameras.offset()
    let pose_offset = camera_offset + state.poses.offset()
    let landmark_offset = pose_offset + state.landmarks.offset()

    var Dr = Tensor[DType.float64](factor_dim*state.factors.active_to_id.size(), landmark_offset)

    for i in range(state.factors.active_to_id.size()):
        let factor = state.factors[state.factors.active_to_id[i]]
        var H_K = Tensor[DType.float64](residual_dim, camera_dim)
        var H_T = Tensor[DType.float64](residual_dim, pose_dim)
        var H_p = Tensor[DType.float64](residual_dim, landmark_dim)
        factor.jacobian(
            state.cameras.values[factor.id_cam],
            state.poses.values[factor.id_pose],
            state.landmarks.values[factor.id_lm],
            H_K,
            H_T,
            H_p
        )
        let row = factor_dim*i
        mc.copy(H_K, Dr, row, camera_dim*state.cameras.id_to_active[factor.id_cam])
        let pose_col = pose_dim*state.poses.id_to_active[factor.id_pose] + camera_offset
        if factor.id_pose != 0:
            mc.copy(H_T, Dr, row, pose_dim*state.poses.id_to_active[factor.id_pose] + camera_offset)
        else:
            let rowp1 = row+1
            Dr[row, pose_col] = 0
            Dr[row, pose_col+1] = 0
            Dr[row, pose_col+2] = 0
            Dr[row, pose_col+3] = 0
            Dr[row, pose_col+4] = 0
            Dr[row, pose_col+5] = 0
            Dr[rowp1, pose_col] = 0
            Dr[rowp1, pose_col+1] = 0
            Dr[rowp1, pose_col+2] = 0
            Dr[rowp1, pose_col+3] = 0
            Dr[rowp1, pose_col+4] = 0
            Dr[rowp1, pose_col+5] = 0
        mc.copy(H_p, Dr, row, landmark_dim*state.landmarks.id_to_active[factor.id_lm] + pose_offset)
    return Dr


fn perturb(state: SfMState, perturbation: Tensor[DType.float64]) -> SfMState:
    let perturbed_state = state
    let camera_offset = state.cameras.offset()
    let pose_offset = state.poses.offset() + camera_offset

    for i in range(state.cameras.active_to_id.size()):
        let id = state.cameras.active_to_id[i]
        perturbation_index = camera_dim*i
        perturbed_state.cameras.values[id].cal[0] += perturbation[perturbation_index]
        perturbed_state.cameras.values[id].cal[1] += perturbation[perturbation_index+1]
        perturbed_state.cameras.values[id].cal[2] += perturbation[perturbation_index+2]
        perturbed_state.cameras.values[id].cal[3] += perturbation[perturbation_index+3]

    for i in range(state.poses.active_to_id.size()):
        let id = state.poses.active_to_id[i]
        let twist_index = pose_dim*i + camera_offset
        let twist = mc.Vector6d(
            perturbation[twist_index],
            perturbation[twist_index+1],
            perturbation[twist_index+2],
            perturbation[twist_index+3],
            perturbation[twist_index+4],
            perturbation[twist_index+5]
        )
        perturbed_state.poses.values[id] = state.poses.values[id] + twist

    for i in range(state.landmarks.active_to_id.size()):
        let id = state.landmarks.active_to_id[i]
        let index = landmark_dim*i + pose_offset
        perturbed_state.landmarks.values[id].val[0] += perturbation[index]
        perturbed_state.landmarks.values[id].val[1] += perturbation[index+1]
        perturbed_state.landmarks.values[id].val[2] += perturbation[index+2]


    return perturbed_state

struct SfM:
    var dir_images: Path
    var scene: SceneGraph

    var state: SfMState
    var factors: SfMFactors


    fn __init__(inout self, dir_images: Path):
        self.dir_images = dir_images
        self.scene = SceneGraph()
        self.state = SfMState()
        self.factors = SfMFactors()

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
        self.state.cameras.values = DynamicVector[PinholeCamera](num_cam)
        self.state.poses.values = DynamicVector[SE3](num_poses)
        self.state.landmarks.values = DynamicVector[Landmark](num_lm)
        self.factors.values = DynamicVector[ProjectionFactor](num_factors)

        self.state.cameras.active_to_id = IntSet(num_cam)
        self.state.poses.active_to_id = IntSet(num_poses)
        self.state.landmarks.active_to_id = IntSet(num_lm)
        self.factors.active_to_id = IntSet(num_factors)

        for cam in sfm_data.cameras:
            let cal = SIMD[DType.float64, 4](
                pyfloat[DType.float64](cam.fx),
                pyfloat[DType.float64](cam.fy),
                pyfloat[DType.float64](cam.px),
                pyfloat[DType.float64](cam.py),
            )
            self.state.cameras.values.push_back(PinholeCamera(cal))

        for _ in range(num_cam):
            self.state.poses.values.push_back(SE3.identity())

        for _ in range(num_lm):
            self.state.landmarks.values.push_back(Landmark.identity())


        for factor in sfm_data.factors:
            let measured = SIMD[DType.float64, 2](
                pyfloat[DType.float64](factor.u), pyfloat[DType.float64](factor.v)
            )
            let id_pose = pyint(factor.id_pose)
            let id_cam = pyint(factor.id_cam)
            let id_lm = pyint(factor.id_lm)
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
        self.state.poses.active_to_id.add(next_pair.id1)
        self.state.poses.active_to_id.add(next_pair.id2)
        self.state.poses[next_pair.id1] = pose1
        self.state.poses[next_pair.id2] = pose2

        self.state.cameras.active_to_id.add(cam1_id)
        self.state.cameras.active_to_id.add(cam2_id)

        for i in range(lm_idx.dim(0)):
            let idx = lm_idx[Index(i)].__int__()
            self.state.landmarks.active_to_id.add(idx)
            self.state.landmarks[idx] = lms[i]

        for i in range(next_pair.factor_idx.dim(0)):
            self.factors.active_to_id.add(next_pair.factor_idx[Index(i)].__int__())

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


    fn optimize(inout self, max_iters: Int = 100, abs_tol = 1e-2, rel_tol = 1e-12):
        # TODO: Make sure not optimizing over first pose - it essentially needs a prior on it

        var abs_error = abs_tol + 1
        var rel_diff = rel_tol + 1

        for iter in range(max_iters):
            let r = compute_residual_vec(self.state, self.factors)
            let r_norm = mc.squared_norm(r)
            let Dr = compute_residual_jac(self.state, self.factors)

            let DrT_Dr = mc.matT_mat(Dr, Dr)
            let diags = mc.diag(mc.diag(DrT_Dr))
            let DrT_b = mc.multiply(-1.0, mc.matT_vec(Dr, r))

            var lambd: Float64 = 1e-4
            for lambd_index in range(10):
                let llt = mc.llt_factor(mc.add(DrT_Dr, mc.multiply(lambd, diags)))
                let step = llt.solve(DrT_b)

                let next_state = perturb(self.state, step)
                let next_r = compute_residual_vec(next_state, self.factors)
                let next_r_norm = mc.squared_norm(next_r)
                if next_r_norm < r_norm:
                    self.state = next_state
                    rel_diff = mc.squared_norm(step)
                    abs_error = next_r_norm
                    break
                else:
                    lambd *= 10.0

            if rel_diff < rel_tol or abs_error < abs_tol:
                break
                    

