from src.sfm import SfM
from pathlib import Path
from python import Python
from src.variables import Landmark


fn main() raises:
    let dir = Path("trex8")
    var sfm = SfM(dir)

    sfm.frontend()

    sfm.register_first_pair()
    # sfm.optimize(max_iters=10)

    while sfm.pairs_left() > 0:
        sfm.register()
        sfm.optimize(max_iters=5)

    sfm.optimize(max_iters=40)

    # Save data
    Python.add_to_path(".")
    let plot3d = Python.import_module("plot").plot3d
    let np = Python.import_module("numpy")
    let pts = Python.evaluate("[]")
    let poses = Python.evaluate("[]")
    for i in range(sfm.state.landmarks.size()):
        let id = sfm.state.landmarks.active_to_id.elements[i]
        let lm = sfm.state.landmarks[id]
        _ = pts.append([lm.val[0], lm.val[1], lm.val[2]])

    for i in range(sfm.state.poses.size()):
        let id = sfm.state.poses.active_to_id.elements[i]
        let pose = sfm.state.poses[id]
        _ = poses.append(
            [
                pose.rot.quat[0],
                pose.rot.quat[1],
                pose.rot.quat[2],
                pose.rot.quat[3],
                pose.trans[0],
                pose.trans[1],
                pose.trans[2],
            ]
        )

    _ = np.savez(dir.path + ".npz", pts, poses)

    # _ = plot3d(dir.path + ".png")
