from src.sfm import SfM
from pathlib import Path
from python import Python
from src.variables import Landmark


fn save(filename: String, sfm: SfM) raises:
    # Save data to npz file
    let np = Python.import_module("numpy")
    let pts = Python.evaluate("[]")
    let poses = Python.evaluate("[]")
    for i in range(sfm.state.landmarks.size()):
        let id = sfm.state.landmarks.active_to_id.elements[i]
        let lm = sfm.state.landmarks[id]
        _ = pts.append([id, lm.val[0], lm.val[1], lm.val[2]])

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

    _ = np.savez(filename, pts, poses)


fn main() raises:
    let dir = Path("moose")
    var sfm = SfM(dir)

    sfm.frontend()

    sfm.register_first_pair()
    sfm.optimize(max_iters=5)

    var i = 1
    let i_max = 100
    while sfm.pairs_left() > 0 and i < i_max:
        print(
            "#----------------------------- Pair",
            i,
            ", Pairs left",
            sfm.pairs_left(),
            "-----------------------------#",
        )

        sfm.register()
        sfm.optimize(max_iters=5, grad_tol=5e1)

        if i % 10 == 0:
            save(dir.path + ".npz", sfm)
            print("Saved!\n")
        i += 1

    # print(
    #     "#----------------------------- Final Optimization"
    #     " -----------------------------#"
    # )
    # sfm.lambd = 1e-4
    # sfm.optimize(max_iters=40)
    save(dir.path + ".npz", sfm)
