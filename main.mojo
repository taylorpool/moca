from src.sfm import SfM
from pathlib import Path
from python import Python
from src.variables import Landmark


fn main() raises:
    let dir = Path("trex_small")
    var sfm = SfM(dir)

    sfm.frontend()

    sfm.register_first_pair()
    sfm.optimize()

    # for i in range(1):
    #     print()
    #     sfm.register()
    #     sfm.optimize()
    #     print(sfm.state.poses.active_to_id.__str__())
    #     print(sfm.state.cameras.active_to_id.__str__())
    #     print()

    # Save data
    let np = Python.import_module("numpy")
    let data = Python.evaluate("[]")
    for i in range(sfm.state.landmarks.size()):
        let lm = sfm.state.landmarks[i]
        _ = data.append([lm.val[0], lm.val[1], lm.val[2]])

    _ = np.save("data.npy", data)
